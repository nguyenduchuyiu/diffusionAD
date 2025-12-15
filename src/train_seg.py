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
# Data augmentation dataset for segmentation anomaly training.
# Generates augmented copies of good samples using three strategies:
# 1. "True anomaly" augmentations (mask=1): structural/realistic defect simulation (e.g., dropout, DTD/Perlin).
# 2. "Fake anomaly" augmentations (mask=0): environmental artifacts (e.g., glare) to teach anomaly ignoring.
class AugmentedTrainDataset(Dataset):
    """
    Dataset for anomaly segmentation augmentation.
    - Group 1 (mask=1): Simulate real defects (CoarseDropout, DTD/Perlin).
    - Group 2 (mask=0): Simulate environmental/fake anomalies (Glare).
    """
    def __init__(self, data_path, classname, img_size, args):
        self.classname = classname
        self.root_dir = os.path.join(data_path, 'train', 'good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.channels = args.get("channels", 3)

        # Gather all image file paths from training "good" images
        IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, ext)))
        self.image_paths = sorted(self.image_paths)

        # Augmenters used to manipulate texture intensity
        self.texture_augs = [
            A.RandomGamma(gamma_limit=(50, 200), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ]

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """Reads and resizes image. Handles grayscale loading if needed."""
        img = cv2.imread(path)
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]))
            img = np.expand_dims(img, axis=2)  # (H, W, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.resize_shape[1], self.resize_shape[0]))
        return img

    def _get_foreground_mask(self, image_path):
        """
        Retrieves a binary foreground mask corresponding to the image.
        If not found, returns full-one mask.
        Expects mask file to be named <img>_mask.png or <img>_mask.jpg
        in "DISthresh" root parallel to "train".
        """
        mask_path = image_path.replace('train', 'DISthresh')
        base = os.path.splitext(os.path.basename(mask_path))[0]
        dir_path = os.path.dirname(mask_path)
        for ext in ['.png', '.jpg']:
            p = os.path.join(dir_path, f"{base}_mask{ext}")
            if os.path.exists(p):
                mask = cv2.imread(p, 0)
                mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
                return (mask > 127).astype(np.float32)
        # If no mask file is found, default to all ones (all foreground)
        return np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)

    def _apply_texture_to_mask(self, image, mask):
        """
        Biến vùng lỗi trơn tuột thành sần sùi (High Frequency Noise).
        Mô phỏng: Rỉ sét, Bụi đóng cục, Sỉ hàn bề mặt nhám.
        
        Tần số cao (Noise) -> Model học đây là LỖI (khác với dầu loang mịn)
        """
        if mask.sum() == 0:
            return image
        
        h, w = image.shape[:2]
        
        # 1. Tạo nhiễu hạt (Grain Noise)
        noise_std = random.randint(10, 30)  # Độ sần sùi
        noise = np.random.normal(0, noise_std, (h, w)).astype(np.float32)
        
        # 2. Tạo nền tối (Base Color) - lỗi thường tối màu
        fill_val = random.randint(20, 80)
        
        # 3. Kết hợp: Lỗi = Nền tối + Nhiễu
        textured_defect = fill_val + noise
        textured_defect = np.clip(textured_defect, 0, 255).astype(np.uint8)
        
        # 4. Áp dụng vào ảnh gốc (chỉ tại vùng mask)
        image_aug = image.copy()
        mask_bool = mask > 0.5
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                image_aug[:, :, c][mask_bool] = textured_defect[mask_bool]
        else:
            image_aug[mask_bool] = textured_defect[mask_bool]
        
        return image_aug

    def _coarse_dropout(self, image, foreground):
        """
        Group 1 - Simulate structural defects.
        Applies various non-rectangular dropout masks (circle, polygon, "blob") at random locations within foreground.
        Returns: (modified image, binary anomaly mask)
        """
        h, w = image.shape[:2]
        original = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        n_holes = random.randint(10, 25)
        for _ in range(n_holes):
            pad = 10
            y = random.randint(pad, h - pad)
            x = random.randint(pad, w - pad)
            if foreground[y, x] < 0.5:
                continue
            fill_value = random.randint(15, 60)
            shape_type = random.choice(['circle', 'poly', 'blob'])
            if shape_type == 'circle':
                radius = random.randint(2, 6)
                cv2.circle(image, (x, y), radius, (fill_value,), -1)
                cv2.circle(mask, (x, y), radius, (1,), -1)
            elif shape_type == 'poly':
                num_pts = random.randint(3, 5)
                pts = []
                for _ in range(num_pts):
                    r_offset = random.randint(3, 8)
                    angle = random.uniform(0, 2 * np.pi)
                    pt_x = int(x + r_offset * np.cos(angle))
                    pt_y = int(y + r_offset * np.sin(angle))
                    pts.append([pt_x, pt_y])
                pts = np.array([pts], dtype=np.int32)
                cv2.fillPoly(image, pts, (fill_value,))
                cv2.fillPoly(mask, pts, (1,))
            elif shape_type == 'blob':
                num_blobs = random.randint(2, 4)
                for _ in range(num_blobs):
                    r_blob = random.randint(2, 5)
                    off_x = random.randint(-4, 4)
                    off_y = random.randint(-4, 4)
                    cv2.circle(image, (x + off_x, y + off_y), r_blob, (fill_value,), -1)
                    cv2.circle(mask, (x + off_x, y + off_y), r_blob, (1,), -1)
        
        # Nhân foreground để tránh tràn ra ngoài
        mask = mask * foreground
        image = self._apply_texture_to_mask(original, mask)

        
        fg_3d = np.expand_dims(foreground, axis=2) if len(image.shape) == 3 else foreground
        image = (original * (1 - fg_3d) + image * fg_3d).astype(np.uint8)
                
        return image, mask

    def _polygon_defect(self, image, foreground):
        """
        Giả lập Sứt mẻ / Bong tróc lớp mạ.
        Vẽ đa giác ngẫu nhiên + TEXTURE SẦN SÙI (High Freq Noise).
        """
        h, w = image.shape[:2]
        original = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        n_defects = random.randint(1, 5)
        
        for _ in range(n_defects):
            fg_coords = np.where(foreground > 0.5)
            if len(fg_coords[0]) == 0:
                break
            idx = random.randint(0, len(fg_coords[0]) - 1)
            cy, cx = fg_coords[0][idx], fg_coords[1][idx]
            
            # Tạo đa giác ngẫu nhiên (Jagged Polygon)
            num_pts = random.randint(3, 6)
            pts = []
            max_radius = random.randint(5, 15)
            
            for _ in range(num_pts):
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(max_radius * 0.5, max_radius)
                pt_x = int(cx + r * np.cos(angle))
                pt_y = int(cy + r * np.sin(angle))
                pts.append([pt_x, pt_y])
            
            pts = np.array([pts], dtype=np.int32)
            
            # CHỈ VẼ VÀO MASK (không tô màu trơn)
            cv2.fillPoly(mask, pts, (1,))
        
        # Nhân foreground để tránh tràn ra ngoài
        mask = mask * foreground
        
        # ĐỔ TEXTURE NHÁM vào vùng mask (High Freq = Lỗi thật)
        image = self._apply_texture_to_mask(original, mask)
        
        # Blend với foreground
        fg_3d = np.expand_dims(foreground, axis=2) if len(image.shape) == 3 else foreground
        image = (original * (1 - fg_3d) + image * fg_3d).astype(np.uint8)
            
        return image, mask

    def _scratch_scar(self, image, foreground):
        """
        Giả lập Vết Hằn (Indentation) / Vết Xước.
        Vẽ đường tối + đường sáng sát nhau -> Tạo hiệu ứng gờ 3D.
        """
        h, w = image.shape[:2]
        original = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(random.randint(1, 3)):
            fg_coords = np.where(foreground > 0.5)
            if len(fg_coords[0]) == 0:
                break
            idx = random.randint(0, len(fg_coords[0]) - 1)
            y, x = fg_coords[0][idx], fg_coords[1][idx]
            
            # Kích thước vết hằn (Dài và Mảnh)
            scar_w = random.randint(15, 50)
            angle = random.uniform(0, 180)
            
            # Tính điểm đầu và cuối
            rad = np.radians(angle)
            pt1 = (int(x - scar_w/2 * np.cos(rad)), int(y - scar_w/2 * np.sin(rad)))
            pt2 = (int(x + scar_w/2 * np.cos(rad)), int(y + scar_w/2 * np.sin(rad)))
            
            # Vẽ đường tối (Shadow) - tạo vết lõm
            cv2.line(image, pt1, pt2, (40,), thickness=2)
            # Vẽ đường sáng (Highlight) lệch 1px - tạo hiệu ứng 3D
            cv2.line(image, (pt1[0]+1, pt1[1]+1), (pt2[0]+1, pt2[1]+1), (180,), thickness=1)
            
            # Mask bao phủ cả vết
            cv2.line(mask, pt1, pt2, (1,), thickness=3)
        
        # Nhân foreground để tránh tràn ra ngoài
        mask = mask * foreground
        fg_3d = np.expand_dims(foreground, axis=2) if len(image.shape) == 3 else foreground
        image = (original * (1 - fg_3d) + image * fg_3d).astype(np.uint8)
            
        return image, mask

    def _glare_augment(self, image, foreground):
        """
        Group 2 - Simulate 'glare' (fake/environmental anomaly).
        Adds soft elliptical glare patches, blurred to simulate light.
        Does NOT update the anomaly mask (returns zeros).
        Returns: (modified image, zero mask)
        """
        h, w = image.shape[:2]
        original = image.copy()
        
        # Create empty layer for glare spots
        glare_layer = np.zeros((h, w), dtype=np.uint8)
        # 1-3 randomly placed ellipses, sampled within foreground
        num_glare = random.randint(1, 3)
        for _ in range(num_glare):
            fg_coords = np.where(foreground > 0.5)
            if len(fg_coords[0]) > 0:
                idx = random.randint(0, len(fg_coords[0]) - 1)
                cy, cx = fg_coords[0][idx], fg_coords[1][idx]
            else:
                cx, cy = random.randint(0, w), random.randint(0, h)
            axis_major = random.randint(20, 80)
            axis_minor = random.randint(10, 40)
            angle = random.randint(0, 180)
            cv2.ellipse(glare_layer, (cx, cy), (axis_major, axis_minor),
                        angle, 0, 360, 255, -1)
        # Blur the glare for smooth edges
        ksize = random.choice([21, 31, 41, 51])
        glare_layer = cv2.GaussianBlur(glare_layer, (ksize, ksize), 0)
        
        # Nhân foreground vào glare layer để chỉ áp dụng trong vùng sản phẩm
        glare_layer = glare_layer.astype(np.float32) * foreground
        
        # Add glare to image (float32)
        image = image.astype(np.float32)
        intensity = random.uniform(0.2, 0.5)
        if len(image.shape) == 3:
            glare_layer_3d = np.expand_dims(glare_layer, axis=2)
            blended = image + (glare_layer_3d * intensity)
        else:
            blended = image + (glare_layer * intensity)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Return image, but mask remains all-zero (mask=0 means "do not learn")
        return blended, np.zeros((h, w), dtype=np.float32)

    def _random_scribble(self, image, foreground):
        """
        Nhóm 1: Lỗi dạng Dây/Sợi (1D curves)
        Mô phỏng: Tóc, sợi vải, nứt chân chim, vết cào rối.
        Dùng Bezier curve ngẫu nhiên.
        Mask = 1
        """
        h, w = image.shape[:2]
        original = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        n_scribbles = random.randint(1, 3)
        
        for _ in range(n_scribbles):
            fg_coords = np.where(foreground > 0.5)
            if len(fg_coords[0]) == 0:
                break
            idx = random.randint(0, len(fg_coords[0]) - 1)
            start_y, start_x = fg_coords[0][idx], fg_coords[1][idx]
            
            # Tạo đường cong ngẫu nhiên bằng nhiều đoạn line
            n_segments = random.randint(5, 15)
            pts = [(start_x, start_y)]
            
            for _ in range(n_segments):
                # Random direction and length
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(5, 20)
                new_x = int(pts[-1][0] + length * np.cos(angle))
                new_y = int(pts[-1][1] + length * np.sin(angle))
                new_x = np.clip(new_x, 0, w - 1)
                new_y = np.clip(new_y, 0, h - 1)
                pts.append((new_x, new_y))
            
            # Màu tối (sợi tóc, vết nứt)
            fill_val = random.randint(20, 60)
            thickness = random.randint(1, 3)
            
            # Vẽ polyline
            pts_array = np.array(pts, dtype=np.int32)
            cv2.polylines(image, [pts_array], False, (fill_val,), thickness)
            cv2.polylines(mask, [pts_array], False, (1,), thickness + 1)
        
        # Nhân foreground
        mask = mask * foreground
        fg_3d = np.expand_dims(foreground, axis=2) if len(image.shape) == 3 else foreground
        image = (original * (1 - fg_3d) + image * fg_3d).astype(np.uint8)
        
        return image, mask

    def _amorphous_defect(self, image, foreground):
        """
        Nhóm 1: Lỗi Vô Định Hình (Amorphous / Splatter)
        Mô phỏng: Bùn bắn, xác côn trùng, sỉ hàn, vệt bẩn đóng cặn.
        Cách làm: Vẽ chùm ellipse + TEXTURE SẦN SÙI (High Freq Noise).
        Mask = 1
        """
        h, w = image.shape[:2]
        original = image.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        n_blobs = random.randint(1, 3)
        
        for _ in range(n_blobs):
            fg_coords = np.where(foreground > 0.5)
            if len(fg_coords[0]) == 0:
                break
            idx = random.randint(0, len(fg_coords[0]) - 1)
            center_y, center_x = fg_coords[0][idx], fg_coords[1][idx]
            
            # Số phần tử trong 1 cụm (tạo hình méo mó)
            n_parts = random.randint(3, 8)
            
            for _ in range(n_parts):
                off_x = random.randint(-10, 10)
                off_y = random.randint(-10, 10)
                
                axes = (random.randint(3, 10), random.randint(3, 10))
                angle = random.randint(0, 360)
                
                pt = (np.clip(center_x + off_x, 0, w-1), np.clip(center_y + off_y, 0, h-1))
                
                # CHỈ VẼ VÀO MASK (không tô màu trơn)
                cv2.ellipse(mask, pt, axes, angle, 0, 360, (1,), -1)
        
        # Nhân foreground
        mask = mask * foreground
        
        # ĐỔ TEXTURE NHÁM vào vùng mask (High Freq = Sỉ hàn, bùn đất)
        image = self._apply_texture_to_mask(original, mask)
        fg_3d = np.expand_dims(foreground, axis=2) if len(image.shape) == 3 else foreground
        image = (original * (1 - fg_3d) + image * fg_3d).astype(np.uint8)
        
        return image, mask

    def _oil_stain(self, image, foreground):
        """
        Nhóm 2: Dầu loang (Soft edge, biên mềm)
        Dùng Perlin noise + blur để tạo vết loang tự nhiên.
        MASK = 0 (Dạy model BỎ QUA)
        """
        h, w = image.shape[:2]
        original = image.copy()
        
        # Tạo perlin noise pattern
        scale = 2 ** random.randint(2, 4)
        perlin = rand_perlin_2d_np((h, w), (scale, scale))
        
        # Threshold để tạo vùng loang
        threshold = random.uniform(0.3, 0.6)
        oil_mask = (perlin > threshold).astype(np.float32) * foreground
        
        # Blur để biên mềm
        ksize = random.choice([11, 21, 31])
        oil_mask = cv2.GaussianBlur(oil_mask, (ksize, ksize), 0)
        
        # Màu dầu (tối hơn một chút, như bóng)
        darken_factor = random.uniform(0.4, 0.6)
        
        image = image.astype(np.float32)
        if len(image.shape) == 3:
            oil_mask_3d = np.expand_dims(oil_mask, axis=2)
            blended = image * (1 - oil_mask_3d * (1 - darken_factor))
        else:
            blended = image * (1 - oil_mask * (1 - darken_factor))
        
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # QUAN TRỌNG: Mask = 0!
        return blended, np.zeros((h, w), dtype=np.float32)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        original = image.copy()
        foreground = self._get_foreground_mask(image_path)

        # Prepare augmented image and anomaly mask (H, W)
        aug_image = image.copy()
        final_mask = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)

        # === CHIẾN THUẬT HOÀN CHỈNH: BAO PHỦ TẤT CẢ TOPOLOGY ===
        
        # --- NHÓM 1: LỖI VẬT LÝ (Mask = 1) ---
        
        # 1. Rỗ khí / Tạp chất (Hình học cơ bản - điểm/blob)
        if random.random() < 0.4:
            aug_image, m = self._coarse_dropout(aug_image, foreground)
            final_mask = np.maximum(final_mask, m)
        
        # 2. Sứt lớn / Bong tróc (Đa giác sắc cạnh)
        if random.random() < 0.3:
            aug_image, m = self._polygon_defect(aug_image, foreground)
            final_mask = np.maximum(final_mask, m)
        
        # 3. Vết Xước / Hằn (Đường kẻ thẳng với 3D effect)
        if random.random() < 0.3:
            aug_image, m = self._scratch_scar(aug_image, foreground)
            final_mask = np.maximum(final_mask, m)
        
        # 4. Lỗi Lạ / Dị vật (Tóc rối + Cục dị dạng) - QUAN TRỌNG
        if random.random() < 0.4:
            if random.random() < 0.5:
                aug_image, m = self._random_scribble(aug_image, foreground)  # Dây dợ, tóc
            else:
                aug_image, m = self._amorphous_defect(aug_image, foreground)  # Cục dị dạng
            final_mask = np.maximum(final_mask, m)
        
        # --- NHÓM 2: NHIỄU MÔI TRƯỜNG (Mask = 0) ---
        
        # 5. Glare (Chói sáng - Elipse mờ)
        if random.random() < 0.5:
            aug_image, _ = self._glare_augment(aug_image, foreground)
        
        # 6. Dầu loang (Perlin mờ) - Dạy model BỎ QUA
        if random.random() < 0.3:
            aug_image, _ = self._oil_stain(aug_image, foreground)

        # Set has_anomaly flag if at least one pixel in anomaly mask
        has_anomaly = 1.0 if final_mask.sum() > 0 else 0.0

        # Preprocess: normalize to [0,1], expand dimension, arrange CHW
        image = original.astype(np.float32) / 255.0
        aug_image = np.array(aug_image).astype(np.float32) / 255.0
        mask = np.expand_dims(final_mask, axis=2)
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
    """
    Focal Loss với pos_weight để xử lý class imbalance cực nặng.
    - pos_weight: Phạt nặng khi bỏ sót pixel lỗi (default=50)
    - gamma: Focus vào hard examples (default=4)
    """
    def __init__(self, alpha=0.5, gamma=4, pos_weight=50.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # BCE với pos_weight để cân bằng class
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


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
    seg_model = SegmentationSubNetwork(in_channels=args["channels"]*2, out_channels=1).to(device)
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
        'total': [],
        'focal': [],
        'smL1': []
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
            
        
    torch.save({
        'seg_model_state_dict': seg_model.module.state_dict() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.state_dict(),
        'epoch': epoch, 'loss': epoch_loss
    }, f'{save_path}/seg-last.pt')
    print(f'Final model saved at epoch {epoch}, loss={epoch_loss:.4f}')
    
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
    ckpt_path = "/kaggle/input/diffusionad/diffusionAD/outputs/model/diff-params-ARGS=2/mat_tru/params-last.pt"
    args['arg_num'] = '2'
    args = defaultdict(str, args)
    args['EPOCHS'] = 10 # can override
    classes = os.listdir(os.path.join(args["data_root_path"], args['data_name']))
    for sub_class in classes:
        print(f"\n=== Training segmentation for {sub_class} ===")
        train_seg(args, sub_class, device, ckpt_path)


if __name__ == '__main__':
    main()

