"""
Full Metrics Evaluation for Anomaly Detection Model
Metrics: IoU, Dice, Precision, Recall, F1, Pixel-AUROC, Image-AUROC, AUPRO
"""
import sys
sys.path.insert(0, 'src')

import os
import glob
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Reuse from inference.py
from src.inference import (
    load_checkpoint, 
    defaultdict_from_json, 
    preprocess_image,
    predict_single_tensor,
    min_max_norm,
    show_cam_on_image,
)
from src.models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule

# ===================== CONFIG =====================
CKPT_PATH = "outputs/model/diff-params-ARGS=2/mat_truc/merged-params-best.pt"
TEST_DIR = "datasets/DENSO/mat_truc/test/bad"
GT_DIR = "datasets/DENSO/mat_truc/ground_truth"
OUTPUT_DIR = "outputs/evaluation"
THRESHOLD = 0.5  # For binary mask

# ===================== HELPER FUNCTIONS =====================

def load_model(ckpt_path, device):
    """Load model from checkpoint - reuse load_checkpoint from inference.py"""
    checkpoint = load_checkpoint(ckpt_path, device)
    args = defaultdict_from_json(checkpoint['args'])
    
    # UNet model
    unet = UNetModel(
        args['img_size'][0], args['base_channels'], 
        channel_mults=args['channel_mults'],
        dropout=args["dropout"], 
        n_heads=args["num_heads"], 
        n_head_channels=args["num_head_channels"],
        in_channels=args["channels"]
    ).to(device)
    unet.load_state_dict(checkpoint['unet_model_state_dict'])
    unet.eval()
    
    # Segmentation model
    seg_model = SegmentationSubNetwork(
        in_channels=args["channels"] * 2, 
        out_channels=1
    ).to(device)
    seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
    seg_model.eval()
    
    # Diffusion model
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    ddpm = GaussianDiffusionModel(
        args['img_size'], betas, 
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], 
        noise=args["noise_fn"], 
        img_channels=args["channels"]
    )
    
    return unet, seg_model, ddpm, args

def load_gt_mask(gt_path, img_size):
    """Load ground truth mask"""
    mask = cv2.imread(gt_path, 0)
    if mask is None:
        return np.zeros((img_size[0], img_size[1]), dtype=np.float32)
    mask = cv2.resize(mask, (img_size[1], img_size[0]))
    mask = (mask > 127).astype(np.float32)
    return mask

@torch.no_grad()
def predict(unet, seg_model, ddpm, image_tensor, args, device, threshold=0.5):
    """Run inference - reuse predict_single_tensor from inference.py"""
    result = predict_single_tensor(
        unet, seg_model, ddpm, image_tensor, args, 
        device=device, heatmap_threshold=threshold, return_visualizations=True
    )
    # Get raw mask from heatmap (before thresholding)
    heatmap = result.get("heatmap", None)
    if heatmap is None:
        # Fallback: recalculate from logits
        normal_t = args["eval_normal_t"]
        noiser_t = args["eval_noisier_t"]
        image_tensor = image_tensor.to(device)
        normal_t_tensor = torch.tensor([normal_t], device=device)
        noiser_t_tensor = torch.tensor([noiser_t], device=device)
        _, pred_x0, *_ = ddpm.norm_guided_one_step_denoising_eval(
            unet, image_tensor, normal_t_tensor, noiser_t_tensor, args
        )
        pred_logits = seg_model(torch.cat((image_tensor, pred_x0), dim=1))
        pred_mask = torch.sigmoid(pred_logits)
        heatmap = pred_mask.cpu().numpy()[0, 0]
    
    return heatmap, result

# ===================== METRICS =====================

def compute_iou(pred, gt, threshold=0.5):
    """Intersection over Union"""
    pred_bin = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred_bin * gt)
    union = np.sum(pred_bin) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)

def compute_dice(pred, gt, threshold=0.5):
    """Dice Coefficient"""
    pred_bin = (pred > threshold).astype(np.float32)
    intersection = np.sum(pred_bin * gt)
    return 2 * intersection / (np.sum(pred_bin) + np.sum(gt) + 1e-8)

def compute_precision_recall_f1(pred, gt, threshold=0.5):
    """Precision, Recall, F1"""
    pred_bin = (pred > threshold).astype(np.float32)
    tp = np.sum(pred_bin * gt)
    fp = np.sum(pred_bin * (1 - gt))
    fn = np.sum((1 - pred_bin) * gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

def compute_aupro(pred_masks, gt_masks, max_fpr=0.3):
    """Area Under Per-Region Overlap curve"""
    # Simplified AUPRO calculation
    fprs = []
    pros = []
    
    for threshold in np.linspace(0, 1, 100):
        total_fp = 0
        total_tn = 0
        per_region_overlap = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            pred_bin = (pred > threshold).astype(np.float32)
            
            # FPR for non-defect regions
            non_defect = (gt == 0)
            if np.sum(non_defect) > 0:
                fp = np.sum(pred_bin[non_defect])
                tn = np.sum(non_defect) - fp
                total_fp += fp
                total_tn += tn
            
            # Per-region overlap for defect regions
            if np.sum(gt) > 0:
                overlap = np.sum(pred_bin * gt) / np.sum(gt)
                per_region_overlap.append(overlap)
        
        if total_fp + total_tn > 0:
            fpr = total_fp / (total_fp + total_tn)
        else:
            fpr = 0
        
        if len(per_region_overlap) > 0:
            pro = np.mean(per_region_overlap)
        else:
            pro = 0
        
        fprs.append(fpr)
        pros.append(pro)
    
    # Sort by FPR
    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    pros = np.array(pros)[sorted_indices]
    
    # Compute AUPRO up to max_fpr
    valid_idx = fprs <= max_fpr
    if np.sum(valid_idx) > 1:
        aupro = auc(fprs[valid_idx], pros[valid_idx]) / max_fpr
    else:
        aupro = 0
    
    return aupro

# ===================== MAIN =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {CKPT_PATH}")
    unet, seg_model, ddpm, args = load_model(CKPT_PATH, device)
    img_size = args['img_size']
    channels = args['channels']
    print(f"   Image size: {img_size}, Channels: {channels}")
    
    # Get test images
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    print(f"\nFound {len(test_images)} test images")
    
    # Storage for metrics
    all_pred_masks = []
    all_gt_masks = []
    all_image_scores = []
    all_image_labels = []
    
    ious, dices, precisions, recalls, f1s = [], [], [], [], []
    
    # Run inference
    print("\nRunning inference...")
    for img_path in tqdm(test_images):
        # Get corresponding GT path
        img_name = os.path.basename(img_path)
        gt_name = img_name.replace(".png", "_mask.png")
        gt_path = os.path.join(GT_DIR, gt_name)
        
        # Load image and GT - reuse preprocess_image from inference.py
        image_tensor = preprocess_image(img_path, img_size, channels)
        gt_mask = load_gt_mask(gt_path, img_size)
        
        # Predict - reuse predict_single_tensor from inference.py
        pred_mask, result = predict(unet, seg_model, ddpm, image_tensor, args, device, THRESHOLD)
        
        # Store for global metrics
        all_pred_masks.append(pred_mask)
        all_gt_masks.append(gt_mask)
        
        # Image-level score (top-k mean)
        topk = min(50, pred_mask.size)
        image_score = np.mean(np.sort(pred_mask.flatten())[-topk:])
        all_image_scores.append(image_score)
        all_image_labels.append(1 if gt_mask.sum() > 0 else 0)  # Has defect
        
        # Per-image metrics
        iou = compute_iou(pred_mask, gt_mask, THRESHOLD)
        dice = compute_dice(pred_mask, gt_mask, THRESHOLD)
        prec, rec, f1 = compute_precision_recall_f1(pred_mask, gt_mask, THRESHOLD)
        
        ious.append(iou)
        dices.append(dice)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    # ===================== COMPUTE GLOBAL METRICS =====================
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Pixel-level AUROC
    all_pred_flat = np.concatenate([m.flatten() for m in all_pred_masks])
    all_gt_flat = np.concatenate([m.flatten() for m in all_gt_masks])
    pixel_auroc = roc_auc_score(all_gt_flat, all_pred_flat)
    
    # Image-level AUROC
    image_auroc = roc_auc_score(all_image_labels, all_image_scores)
    
    # Pixel-level AP
    pixel_ap = average_precision_score(all_gt_flat, all_pred_flat)
    
    # AUPRO
    aupro = compute_aupro(all_pred_masks, all_gt_masks)
    
    # Average metrics
    avg_iou = np.mean(ious)
    avg_dice = np.mean(dices)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    # Print results
    print(f"\nPixel-Level Metrics (threshold={THRESHOLD}):")
    print(f"   IoU (Jaccard):     {avg_iou:.4f}")
    print(f"   Dice Coefficient:  {avg_dice:.4f}")
    print(f"   Precision:         {avg_precision:.4f}")
    print(f"   Recall:            {avg_recall:.4f}")
    print(f"   F1 Score:          {avg_f1:.4f}")
    
    print(f"\nAUC Metrics:")
    print(f"   Pixel AUROC:       {pixel_auroc:.4f}")
    print(f"   Image AUROC:       {image_auroc:.4f}")
    print(f"   Pixel AP:          {pixel_ap:.4f}")
    print(f"   AUPRO (FPR≤0.3):   {aupro:.4f}")
    
    # ===================== SAVE RESULTS =====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        "model": CKPT_PATH,
        "test_dir": TEST_DIR,
        "num_images": len(test_images),
        "threshold": THRESHOLD,
        "metrics": {
            "pixel_auroc": float(pixel_auroc),
            "image_auroc": float(image_auroc),
            "pixel_ap": float(pixel_ap),
            "aupro": float(aupro),
            "iou": float(avg_iou),
            "dice": float(avg_dice),
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1": float(avg_f1)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR}/metrics.json")
    
    # ===================== PLOT =====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve (Pixel-level)
    fpr, tpr, _ = roc_curve(all_gt_flat, all_pred_flat)
    axes[0].plot(fpr, tpr, 'b-', label=f'Pixel AUROC = {pixel_auroc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Pixel-Level ROC Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_gt_flat, all_pred_flat)
    axes[1].plot(recall_curve, precision_curve, 'r-', label=f'Pixel AP = {pixel_ap:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Pixel-Level Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_curves.png"), dpi=150)
    print(f"Curves saved to: {OUTPUT_DIR}/metrics_curves.png")
    
    # ===================== VISUALIZE SOME SAMPLES =====================
    # Re-run with visualization for display
    print("\n Generating sample visualizations...")
    n_samples = min(8, len(test_images))
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    
    for i in range(n_samples):
        img_path = test_images[i]
        img_name = os.path.basename(img_path)
        gt_name = img_name.replace(".png", "_mask.png")
        gt_path = os.path.join(GT_DIR, gt_name)
        
        # Re-predict with visualizations from inference.py
        image_tensor = preprocess_image(img_path, img_size, channels)
        gt_mask = load_gt_mask(gt_path, img_size)
        _, result = predict(unet, seg_model, ddpm, image_tensor, args, device, THRESHOLD)
        
        # Get visualizations from inference.py
        original_img = result.get("original_image")
        recon_img = result.get("reconstructed_image")
        anomaly_mask_colored = result.get("anomaly_mask")
        heatmap_overlay = result.get("heatmap_overlay")
        
        # Display: Input | Reconstruction | GT | Prediction | Overlay
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f'Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(gt_mask, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(anomaly_mask_colored)
        axes[i, 3].set_title(f'Pred Mask')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(heatmap_overlay)
        axes[i, 4].set_title(f'Overlay (IoU={ious[i]:.3f})')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
    print(f"Samples saved to: {OUTPUT_DIR}/sample_predictions.png")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()

