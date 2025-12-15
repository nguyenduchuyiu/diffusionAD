"""Train segmentation model only with frozen diffusion model"""
import torch
import os
import json
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import albumentations as A

from models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule
from utils import RealIADTrainDataset, rand_perlin_2d_np


# ============== AUGMENTATION DATASET ==============

class AugmentedTrainDataset(Dataset):
    """
    Dataset với chiến lược augmentation:
    - Nhóm 1 (mask=1): Tạo lỗi thật (CoarseDropout, DTD/Perlin)
    - Nhóm 2 (mask=0): Tạo giả lỗi/nhiễu môi trường (Glare)
    """
    
    def __init__(self, data_path, classname, img_size, args):
        self.classname = classname
        self.root_dir = os.path.join(data_path, 'train', 'good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        self.channels = args.get("channels", 3)
        
        # Load images
        IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, ext)))
        self.image_paths = sorted(self.image_paths)
        
        # Load anomaly sources (DTD textures)
        self.anomaly_source_paths = []
        anomaly_glob_path = os.path.join(self.anomaly_source_path, "images", "*")
        for ext in IMAGE_EXTENSIONS:
            self.anomaly_source_paths.extend(glob.glob(os.path.join(anomaly_glob_path, ext)))
        self.anomaly_source_paths = sorted(self.anomaly_source_paths)
        
        print(f"AugmentedDataset: {len(self.image_paths)} images, {len(self.anomaly_source_paths)} anomaly sources")
        
        # Augmenters for texture modification
        self.texture_augs = [
            A.RandomGamma(gamma_limit=(50, 200), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ]
        
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, path):
        """Load và resize image"""
        img = cv2.imread(path)
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]))
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]))
        return img
    
    def _get_foreground_mask(self, image_path):
        """Lấy foreground mask từ DISthresh"""
        mask_path = image_path.replace('train', 'DISthresh')
        base = os.path.splitext(os.path.basename(mask_path))[0]
        dir_path = os.path.dirname(mask_path)
        
        for ext in ['.png', '.jpg']:
            p = os.path.join(dir_path, f"{base}_mask{ext}")
            if os.path.exists(p):
                mask = cv2.imread(p, 0)
                mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
                return (mask > 127).astype(np.float32)
        
        # Fallback: full mask
        return np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
    
    def _coarse_dropout(self, image, foreground):
        """
        Nhóm 1: CoarseDropout - Giả lập rỗ khí, sứt mẻ
        Fill: 15-60 (xám tối), Size: 2-8 pixel
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        n_holes = random.randint(5, 20)
        for _ in range(n_holes):
            hole_h = random.randint(1, 6)
            hole_w = random.randint(1, 6)
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            
            # Chỉ đặt trong foreground
            if foreground[y + hole_h//2, x + hole_w//2] > 0.5:
                fill_value = random.randint(15, 60)
                image[y:y+hole_h, x:x+hole_w] = fill_value
                mask[y:y+hole_h, x:x+hole_w] = 1.0
        
        return image, mask
    
    def _dtd_perlin_paste(self, image, foreground):
        """
        Nhóm 1: DTD/Perlin - Giả lập dầu bẩn
        Chỉ dùng texture làm tối đi bề mặt
        """
        h, w = image.shape[:2]
        
        # Generate perlin mask
        scale = 2 ** random.randint(0, 5)
        perlin = rand_perlin_2d_np((h, w), (scale, scale))
        perlin_mask = (perlin > 0.5).astype(np.float32) * foreground
        
        if perlin_mask.sum() < 10:
            return image, np.zeros((h, w), dtype=np.float32)
        
        # Load random DTD texture
        dtd_path = random.choice(self.anomaly_source_paths)
        dtd = self._load_image(dtd_path)
        
        # Apply random aug to make it darker
        aug = random.choice(self.texture_augs)
        dtd = aug(image=dtd)['image']
        
        # Darken the texture
        dtd = (dtd * 0.5).astype(np.uint8)
        
        # Blend
        perlin_mask_3d = np.expand_dims(perlin_mask, axis=2)
        beta = random.uniform(0.3, 0.7)
        blended = image * (1 - perlin_mask_3d) + (beta * dtd + (1-beta) * image) * perlin_mask_3d
        
        return blended.astype(np.uint8), perlin_mask
    
    def _glare_augment(self, image, foreground):
        """
        Nhóm 2: Glare - Giả lập chói sáng (MASK = 0!)
        Fill: 200-255, dùng Gaussian blur circle
        """
        h, w = image.shape[:2]
        
        # Random center trong foreground
        fg_coords = np.where(foreground > 0.5)
        if len(fg_coords[0]) == 0:
            return image, np.zeros((h, w), dtype=np.float32)
        
        idx = random.randint(0, len(fg_coords[0]) - 1)
        cy, cx = fg_coords[0][idx], fg_coords[1][idx]
        
        # Random radius
        radius = random.randint(10, 50)
        
        # Create glare mask (soft circle)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        glare_mask = np.clip(1 - dist / radius, 0, 1).astype(np.float32)
        
        # Apply glare (bright)
        fill_value = random.randint(200, 255)
        glare_mask_3d = np.expand_dims(glare_mask, axis=2)
        blended = image * (1 - glare_mask_3d) + fill_value * glare_mask_3d
        
        # QUAN TRỌNG: Trả về mask = 0 cho glare!
        return blended.astype(np.uint8), np.zeros((h, w), dtype=np.float32)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        original = image.copy()
        foreground = self._get_foreground_mask(image_path)
        
        # Random chọn augmentation strategy
        # 30% normal, 25% coarse_dropout, 25% dtd_perlin, 20% glare
        r = random.random()
        
        if r < 0.3:
            # Normal image - no anomaly
            aug_image = image
            mask = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
            has_anomaly = 0.0
        elif r < 0.55:
            # CoarseDropout (mask=1)
            aug_image, mask = self._coarse_dropout(image.copy(), foreground)
            has_anomaly = 1.0 if mask.sum() > 0 else 0.0
        elif r < 0.80:
            # DTD/Perlin paste (mask=1)
            aug_image, mask = self._dtd_perlin_paste(image.copy(), foreground)
            has_anomaly = 1.0 if mask.sum() > 0 else 0.0
        else:
            # Glare (mask=0!) - Dạy model "tha" cho vết chói sáng
            aug_image, mask = self._glare_augment(image.copy(), foreground)
            has_anomaly = 0.0  # Glare không phải anomaly
        
        # Normalize và chuyển format
        image = original.astype(np.float32) / 255.0
        aug_image = np.array(aug_image).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # Transpose to CHW
        image = np.transpose(image, (2, 0, 1))
        aug_image = np.transpose(aug_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return {
            'image': image,
            'augmented_image': aug_image,
            'anomaly_mask': mask,
            'has_anomaly': np.array([has_anomaly], dtype=np.float32),
            'idx': idx
        }

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return torch.mean(self.alpha * (1-pt)**self.gamma * bce)


def train_seg(args, sub_class, device, ckpt_path):
    # Load frozen diffusion model
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    unet = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
        dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=args["channels"]
    ).to(device)
    unet.load_state_dict(checkpoint['unet_model_state_dict'])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # Use DataParallel for unet if multiple GPUs are available (mostly defensive, since it's frozen)
    if torch.cuda.device_count() > 1:
        unet = torch.nn.DataParallel(unet)

    # Segmentation model (trainable)
    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)
    seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
    # Use DataParallel for segmentation model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        seg_model = torch.nn.DataParallel(seg_model)
    # Diffusion sampler
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    ddpm = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=args["channels"]
    )
    
    # Dataset - sử dụng AugmentedTrainDataset với chiến lược mới
    subclass_path = os.path.join(args["data_root_path"], args['data_name'], sub_class)
    use_new_augment = args.get('use_new_augment', True)  # Default: dùng augment mới
    if use_new_augment:
        dataset = AugmentedTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
    else:
        dataset = RealIADTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
    loader = DataLoader(dataset, batch_size=args['Batch_Size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # Optimizer & Loss
    # Only the parameters of the underlying model are passed to the optimizer
    # (DataParallel exposes a .module attribute)
    optimizer = optim.Adam(
        seg_model.module.parameters() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.parameters(),
        lr=args.get('seg_lr', 1e-4)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    focal_loss = BinaryFocalLoss().to(device)
    smL1_loss = nn.SmoothL1Loss().to(device)
    
    epochs = args['EPOCHS']
    # lấy trong checkpoint
    losses = {
        'total': checkpoint.get('train_loss_list', []),
        'focal': checkpoint.get('train_focal_loss_list', []),
        'smL1': checkpoint.get('train_smL1_loss_list', [])
    }
    best_loss = float('inf')
    
    for epoch in range(epochs):
        seg_model.train()
        epoch_loss, epoch_focal, epoch_smL1 = 0, 0, 0
        
        tbar = tqdm(loader, desc=f'Epoch {epoch}')
        for sample in tbar:
            aug_image = sample['augmented_image'].to(device)
            anomaly_mask = sample['anomaly_mask'].to(device)
            anomaly_label = sample['has_anomaly'].to(device).squeeze()
            
            with torch.no_grad():
                # If unet is DataParallel, call .module for compatibility with state_dict (already handled above)
                unet_forward = unet
                # Don't need .module here because DataParallel respects __call__
                _, pred_x0, *_ = ddpm.norm_guided_one_step_denoising(unet_forward, aug_image, anomaly_label, args)
            
            pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1))
            fl = focal_loss(pred_mask, anomaly_mask)
            sl = smL1_loss(pred_mask, anomaly_mask)
            loss = 5*fl + sl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_focal += fl.item()
            epoch_smL1 += sl.item()
            tbar.set_postfix(loss=f'{epoch_loss/(tbar.n+1):.4f}')
        
        scheduler.step()
        n_batches = len(loader)
        epoch_loss /= n_batches
        epoch_focal /= n_batches
        epoch_smL1 /= n_batches
        losses['total'].append(epoch_loss)
        losses['focal'].append(epoch_focal)
        losses['smL1'].append(epoch_smL1)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
            os.makedirs(save_path, exist_ok=True)
            # Save underlying model's state_dict if using DataParallel
            torch.save({
                'seg_model_state_dict': seg_model.module.state_dict() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.state_dict(),
                'epoch': epoch, 'loss': best_loss
            }, f'{save_path}/seg-best.pt')
            print(f'Best model saved at epoch {epoch}, loss={best_loss:.4f}')
        
        if epoch % 100 == 0:
            save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'seg_model_state_dict': seg_model.module.state_dict() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.state_dict(),
                'epoch': epoch, 'loss': epoch_loss
            }, f'{save_path}/seg-last.pt')
            print(f'Checkpoint saved at epoch {epoch}, loss={epoch_loss:.4f}')
        
        if epoch % 10 == 0:
            plot_losses(losses, sub_class, args)
            
        
    
    plot_losses(losses, sub_class, args, final=True)
    return losses


def plot_losses(losses, sub_class, args, final=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(losses['total']))
    ax.plot(epochs, losses['total'], 'b-', label='Total', linewidth=2)
    ax.plot(epochs, losses['focal'], 'r-', label='Focal')
    ax.plot(epochs, losses['smL1'], 'g-', label='SmoothL1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{sub_class} - Segmentation Training')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    
    save_dir = f'{args["output_path"]}/learning_curves/ARGS={args["arg_num"]}'
    os.makedirs(save_dir, exist_ok=True)
    suffix = '_final' if final else ''
    plt.savefig(f'{save_dir}/{sub_class}_seg{suffix}.png', dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('./args/args1.json', 'r') as f:
        args = json.load(f)
    ckpt_path = "outputs/model/diff-params-ARGS=1/metal_nut/params-last.pt"
    args['arg_num'] = '1'
    args = defaultdict(str, args)
    args['EPOCHS'] = 1800 # can override
    classes = os.listdir(os.path.join(args["data_root_path"], args['data_name']))
    for sub_class in classes:
        print(f"\n=== Training segmentation for {sub_class} ===")
        train_seg(args, sub_class, device, ckpt_path)


if __name__ == '__main__':
    main()

