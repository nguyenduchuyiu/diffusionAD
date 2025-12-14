import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve,average_precision_score
from sklearn.metrics import roc_auc_score
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import torch.nn as nn
from models import UNetModel, update_ema_params
from models import SegmentationSubNetwork
import torch.nn as nn
from utils import RealIADTestDataset
from models import GaussianDiffusionModel, get_beta_schedule
from math import exp
import torch.nn.functional as F
torch.cuda.empty_cache()
from tqdm import tqdm
import json
import os
from collections import defaultdict
import pandas as pd
import torchvision.utils
import os
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops
import sys
from utils import BinaryFocalLoss

def preprocess_image(image_path, img_size=(256, 256), channels=3):
    image = cv2.imread(image_path)
    if channels == 1:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (img_size[1], img_size[0]))
        image = (image / 255.0).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size[1], img_size[0]))
        image = (image / 255.0)
        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)  # (1, 3, H, W)
    return image

def denormalize_image(tensor_image):
    img = tensor_image.cpu().squeeze(0)
    if img.dim() == 2:  # Grayscale (H, W)
        img_np = img.numpy()
    elif img.shape[0] == 1:  # Grayscale (1, H, W)
        img_np = img.squeeze(0).numpy()
    else:  # RGB (C, H, W)
        img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np + 1) / 2.0
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=1.0, neginf=0.0)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    # Convert grayscale to RGB for display
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np

def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

def load_checkpoint(ckpt_path, device):

    print("checkpoint",ckpt_path)

    from collections import defaultdict
    try:
        torch.serialization.add_safe_globals([defaultdict])
    except Exception:
        pass

    loaded_model = torch.load(ckpt_path, map_location=device, weights_only=False)
    return loaded_model

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def image_transform(image):
     return np.clip(image* 255, 0, 255).astype(np.uint8)
 
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / max_val
    return np.uint8(255 * cam) 


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_max - a_min == 0:
        return np.zeros_like(image)
    return (image-a_min)/(a_max - a_min)

        
def predict(unet_model, seg_model, ddpm, image_tensor, args, device='cpu', heatmap_threshold=0.6):
    """
    Hàm predict giờ nhận vào image_tensor đã được xử lý và args từ model.
    """
    normal_t = args["eval_normal_t"]
    noiser_t = args["eval_noisier_t"]
    
    image_tensor = image_tensor.to(device)

    normal_t_tensor = torch.tensor([normal_t], device=device).repeat(image_tensor.shape[0])
    noiser_t_tensor = torch.tensor([noiser_t], device=device).repeat(image_tensor.shape[0])

    with torch.no_grad():
        _, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier = ddpm.norm_guided_one_step_denoising_eval(unet_model, image_tensor, normal_t_tensor, noiser_t_tensor, args)
        pred_mask_logits = seg_model(torch.cat((image_tensor, pred_x_0_condition), dim=1))
            
    pred_mask = torch.sigmoid(pred_mask_logits)
    out_mask = pred_mask

    # Tính điểm anomaly
    topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
    topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
    image_score = torch.mean(topk_out_mask)

    # --- Visualisation ---
    # Original image: convert from [0, 1] directly to [0, 255] (not [-1, 1])
    img = image_tensor.cpu().squeeze(0)
    if img.shape[0] == 1:  # Grayscale (1, H, W)
        raw_image_np = img.squeeze(0).numpy()
        raw_image = np.clip(raw_image_np * 255.0, 0, 255).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    else:  # RGB (C, H, W)
        raw_image_np = img.permute(1, 2, 0).numpy()
        raw_image = np.clip(raw_image_np * 255.0, 0, 255).astype(np.uint8)
    
    # Reconstructed images: use denormalize (model outputs are in [-1, 1])
    recon_condition = denormalize_image(pred_x_0_condition)
    recon_normal_t = denormalize_image(pred_x_0_normal)
    recon_noisier_t = denormalize_image(pred_x_0_noisier)
    
    # Create heatmap with higher contrast
    mask_data = out_mask[0, 0].cpu().numpy().astype(np.float32)
    mask_data[mask_data < heatmap_threshold] = 0
    
    # Apply contrast enhancement using gamma correction
    gamma = 0.1  # Lower gamma = higher contrast for bright areas
    mask_data_enhanced = np.power(np.clip(mask_data, 0, 1), gamma)
    
    ano_map = cv2.GaussianBlur(mask_data_enhanced, (15, 15), 4)
    ano_map = min_max_norm(ano_map)
    ano_map = np.nan_to_num(ano_map, nan=0.0)
    
    # Use HOT colormap for better visibility (red/yellow/white)
    ano_map_heatmap = cv2.applyColorMap(np.uint8(np.clip(ano_map * 255.0, 0, 255)), cv2.COLORMAP_HOT)
    
    # Create overlay
    raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    ano_map_overlay = show_cam_on_image(raw_image_bgr, ano_map_heatmap)
    ano_map_overlay = cv2.cvtColor(ano_map_overlay, cv2.COLOR_BGR2RGB)
    
    # Hiển thị
    f, axes = plt.subplots(1, 5, figsize=(20, 4))
    f.suptitle(f'Anomaly Score: {image_score:.4f}')

    axes[0].imshow(raw_image)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(recon_condition)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    axes[2].imshow(recon_noisier_t)
    axes[2].set_title('Recon (Noisier)')
    axes[2].axis('off')
    
    # Create enhanced anomaly mask with high contrast
    mask_raw = out_mask[0][0].cpu().numpy().astype(np.float32)
    mask_raw[mask_raw < heatmap_threshold] = 0
    mask_enhanced = np.power(np.clip(mask_raw, 0, 1), 0.3)
    mask_normalized = min_max_norm(mask_enhanced)
    mask_normalized = np.nan_to_num(mask_normalized, nan=0.0)
    mask_stretched = np.uint8(np.clip(mask_normalized * 255.0, 0, 255))
    anomaly_mask_colored = cv2.applyColorMap(mask_stretched, cv2.COLORMAP_HOT)
    anomaly_mask_colored = cv2.cvtColor(anomaly_mask_colored, cv2.COLOR_BGR2RGB)
    
    axes[3].imshow(anomaly_mask_colored)
    axes[3].set_title('Anomaly Mask')
    axes[3].axis('off')

    axes[4].imshow(ano_map_overlay)
    axes[4].set_title('Heatmap Overlay')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

def predict_image(unet_model, seg_model, ddpm, image_path, args, device='cpu', heatmap_threshold=0.6):
    image_tensor = preprocess_image(image_path, img_size=args['img_size'], channels=args['channels'])
    image_tensor = image_tensor.to(device)
    return predict(unet_model, seg_model, ddpm, image_tensor, args, device, heatmap_threshold)

def predict_batch(unet_model, seg_model, ddpm, image_arrays, args, device='cpu', heatmap_threshold=0.6, batch_size=8, progress_callback=None):
    """
    Predict anomalies for a batch of images with parallel processing
    Args:
        unet_model: UNet model
        seg_model: Segmentation model
        ddpm: DDPM model
        image_arrays: List of numpy image arrays
        args: Model arguments
        device: Device to run on
        heatmap_threshold: Threshold for anomaly detection
        batch_size: Batch size for parallel processing
        progress_callback: Optional callback function(progress, status_text) for progress updates
    Returns:
        List of prediction results with inference_time
    """
    results = []
    total_images = len(image_arrays)
    num_batches = (total_images + batch_size - 1) // batch_size
    
    # Process images in batches
    for batch_idx in range(0, len(image_arrays), batch_size):
        batch_images = image_arrays[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        # Update progress
        if progress_callback is not None:
            progress = batch_idx / total_images
            status_text = f'Processing batch {current_batch}/{num_batches} ({len(batch_images)} images)...'
            progress_callback(progress, status_text)
        
        # Preprocess batch - process all images in parallel
        batch_tensors = []
        for image_array in batch_images:
            if len(image_array.shape) == 3:
                # Check if model expects grayscale
                if args.get('channels', 3) == 1:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    image_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                else:
                    image_tensor = torch.from_numpy(np.transpose(image_array.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
            elif len(image_array.shape) == 2:
                image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("Image must be 2D (H, W) or 3D array (H, W, C)")
            
            # Resize to model input size
            image_tensor = torch.nn.functional.interpolate(image_tensor, size=args['img_size'], mode='bilinear', align_corners=False)
            batch_tensors.append(image_tensor)
        
        # Stack into batch tensor - this enables parallel processing on GPU
        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        
        normal_t = args["eval_normal_t"]
        noiser_t = args["eval_noisier_t"]
        
        normal_t_tensor = torch.tensor([normal_t], device=device).repeat(batch_tensor.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=device).repeat(batch_tensor.shape[0])
        
        # Synchronize GPU before timing (for accurate measurement)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time for batch - all images processed in parallel
        inference_start = time.time()
        with torch.no_grad():
            _, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier = ddpm.norm_guided_one_step_denoising_eval(unet_model, batch_tensor, normal_t_tensor, noiser_t_tensor, args)
            pred_mask_logits = seg_model(torch.cat((batch_tensor, pred_x_0_condition), dim=1))
        
        # Synchronize GPU after inference (for accurate timing)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        batch_inference_time = time.time() - inference_start
        
        pred_mask = torch.sigmoid(pred_mask_logits)
        out_mask = pred_mask
        
        # Process each image in batch (post-processing)
        for i in range(batch_tensor.shape[0]):
            # Calculate anomaly score
            topk_out_mask = torch.flatten(out_mask[i], start_dim=1)
            topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
            image_score = torch.mean(topk_out_mask).cpu().item()
            
            # Each image in batch is processed in parallel, so effective latency per image
            # is approximately the batch time (since they run simultaneously on GPU)
            # We record the batch time as the inference time for each image
            per_image_time = batch_inference_time
            
            results.append({
                "anomaly_score": float(image_score),
                "is_anomaly": image_score > heatmap_threshold,
                "inference_time": per_image_time
            })
        
        # Update progress after processing batch
        if progress_callback is not None:
            progress = min((batch_idx + len(batch_images)) / total_images, 1.0)
            status_text = f'Completed batch {current_batch}/{num_batches} ({len(results)}/{total_images} images processed)...'
            progress_callback(progress, status_text)
    
    return results

def predict_single_tensor(unet_model, seg_model, ddpm, image_tensor, args, device='cpu', heatmap_threshold=0.6, return_visualizations=True):
    """
    Predict anomaly for a single image tensor
    """
    normal_t = args["eval_normal_t"]
    noiser_t = args["eval_noisier_t"]
    
    image_tensor = image_tensor.to(device)
    normal_t_tensor = torch.tensor([normal_t], device=device).repeat(image_tensor.shape[0])
    noiser_t_tensor = torch.tensor([noiser_t], device=device).repeat(image_tensor.shape[0])

    with torch.no_grad():
        _, pred_x_0_condition, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noiser_t, pred_x_t_noisier = ddpm.norm_guided_one_step_denoising_eval(unet_model, image_tensor, normal_t_tensor, noiser_t_tensor, args)
        pred_mask_logits = seg_model(torch.cat((image_tensor, pred_x_0_condition), dim=1))
            
    pred_mask = torch.sigmoid(pred_mask_logits)
    out_mask = pred_mask

    # Calculate anomaly score
    topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
    topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
    image_score = torch.mean(topk_out_mask).cpu().item()

    result = {
        "anomaly_score": float(image_score),
        "is_anomaly": image_score > heatmap_threshold
    }
    
    if return_visualizations:
        # Create visualizations
        # Original image: convert from [0, 1] directly to [0, 255] (not [-1, 1])
        img = image_tensor.cpu().squeeze(0)
        if img.shape[0] == 1:  # Grayscale (1, H, W)
            raw_image_np = img.squeeze(0).numpy()
            raw_image = np.clip(raw_image_np * 255.0, 0, 255).astype(np.uint8)
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        else:  # RGB (C, H, W)
            raw_image_np = img.permute(1, 2, 0).numpy()
            raw_image = np.clip(raw_image_np * 255.0, 0, 255).astype(np.uint8)
        
        # Reconstructed images: use denormalize (model outputs are in [-1, 1])
        recon_condition = denormalize_image(pred_x_0_condition)
        recon_noisier_t = denormalize_image(pred_x_0_noisier)
        
        # Create heatmap with higher contrast
        mask_data = out_mask[0, 0].cpu().numpy().astype(np.float32)
        mask_data[mask_data < heatmap_threshold] = 0
        
        # Apply contrast enhancement using gamma correction
        gamma = 0.1  # Lower gamma = higher contrast for bright areas
        mask_data_enhanced = np.power(np.clip(mask_data, 0, 1), gamma)
        
        ano_map = cv2.GaussianBlur(mask_data_enhanced, (15, 15), 4)
        ano_map = min_max_norm(ano_map)
        ano_map = np.nan_to_num(ano_map, nan=0.0)
        
        # Use HOT colormap for better visibility (red/yellow/white)
        ano_map_heatmap = cv2.applyColorMap(np.uint8(np.clip(ano_map * 255.0, 0, 255)), cv2.COLORMAP_HOT)
        
        # Create overlay
        raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        ano_map_overlay = show_cam_on_image(raw_image_bgr, ano_map_heatmap)
        ano_map_overlay = cv2.cvtColor(ano_map_overlay, cv2.COLOR_BGR2RGB)
        
        # Create enhanced anomaly mask with high contrast
        mask_raw = out_mask[0][0].cpu().numpy().astype(np.float32)
        mask_raw[mask_raw < heatmap_threshold] = 0
        mask_enhanced = np.power(np.clip(mask_raw, 0, 1), 0.3)
        mask_normalized = min_max_norm(mask_enhanced)
        mask_normalized = np.nan_to_num(mask_normalized, nan=0.0)
        mask_stretched = np.uint8(np.clip(mask_normalized * 255.0, 0, 255))
        anomaly_mask_colored = cv2.applyColorMap(mask_stretched, cv2.COLORMAP_HOT)
        anomaly_mask_colored = cv2.cvtColor(anomaly_mask_colored, cv2.COLOR_BGR2RGB)
        
        result.update({
            "original_image": raw_image,
            "reconstructed_image": recon_condition,
            "recon_noisier": recon_noisier_t,
            "anomaly_mask": anomaly_mask_colored,
            "heatmap_overlay": ano_map_overlay,
            "heatmap": ano_map
        })
    
    return result

def predict_single_image_array(unet_model, seg_model, ddpm, image_array, args, device='cpu', heatmap_threshold=0.6):
    """
    Predict anomaly for a single image array (numpy array)
    Args:
        unet_model: UNet model
        seg_model: Segmentation model
        ddpm: DDPM model
        image_array: numpy array of shape (H, W, C) with values in [0, 255]
        args: Model arguments
        device: Device to run on
        heatmap_threshold: Threshold for anomaly detection
    Returns:
        Dictionary with prediction results and visualizations
    """
    import time
    
    # Preprocess image
    if len(image_array.shape) == 3:
        # Check if model expects grayscale
        if args.get('channels', 3) == 1:
            # Convert RGB to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            image_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            image_tensor = torch.from_numpy(np.transpose(image_array.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
    elif len(image_array.shape) == 2:
        # Already grayscale
        image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Image must be 2D (H, W) or 3D array (H, W, C)")
    
    # Resize to model input size
    image_tensor = torch.nn.functional.interpolate(image_tensor, size=args['img_size'], mode='bilinear', align_corners=False)
    
    # Measure inference time
    inference_start = time.time()
    result = predict_single_tensor(unet_model, seg_model, ddpm, image_tensor, args, device, heatmap_threshold, return_visualizations=True)
    inference_time = time.time() - inference_start
    
    result['inference_time'] = inference_time
    return result

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # ckpt_path = "outputs/model/diff-params-ARGS=1/PCB5/params-last.pt"
    # image_path = "datasets/RealIAD/PCB5/test/bad/pcb_0123_NG_ZW_C1_20231028122009.jpg"
    ckpt_path = "params-best (1).pt"
    image_path = "datasets/denso_dataset/mat_tru/test/bad/7_2024_9_5_11_25_49_8_P4.png"
    heatmap_threshold = 0.5
    # 1. Load checkpoint và lấy args từ đó
    ckpt_state = load_checkpoint(ckpt_path, device)
    args = ckpt_state['args']
    args = defaultdict_from_json(args)
    print(args)
    
    # 2. Khởi tạo model với args đã load
    unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=args["channels"]
            ).to(device)

    seg_model = SegmentationSubNetwork(in_channels=args["channels"] * 2, out_channels=1).to(device)

    # 3. Khởi tạo DDPM
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    
    ddpm =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=args["channels"]
            )
    
    # 4. Load state dicts vào model
    unet_model.load_state_dict(ckpt_state['unet_model_state_dict'])
    seg_model.load_state_dict(ckpt_state['seg_model_state_dict'])
    unet_model.eval()
    seg_model.eval()


    # 5. Chạy predict
    predict_image(unet_model, seg_model, ddpm, image_path, args, device, heatmap_threshold)