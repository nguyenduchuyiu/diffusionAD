"""Test augmentation - Hiển thị TẤT CẢ loại lỗi + Combined"""
import sys
sys.path.insert(0, 'src')

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from src.train_seg import AugmentedTrainDataset

# Load args
with open('./args/args2.json', 'r') as f:
    args = json.load(f)
args = defaultdict(str, args)

# Get first subclass
data_path = os.path.join(args["data_root_path"], args['data_name'])
classes = os.listdir(data_path)
sub_class = classes[0]
print(f"Testing with class: {sub_class}")

subclass_path = os.path.join(data_path, sub_class)
dataset = AugmentedTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)

# Lấy 1 ảnh gốc để test
orig_image = dataset._load_image(dataset.image_paths[0])
foreground = dataset._get_foreground_mask(dataset.image_paths[0])

# === PLOT: 7 loại + 1 Combined ===
fig, axes = plt.subplots(8, 3, figsize=(12, 26), constrained_layout=True)

defect_types = [
    # NHÓM 1: LỖI VẬT LÝ (Mask = 1)
    ("CoarseDropout\n(Rỗ khí, Tạp chất)", "_coarse_dropout", True),
    ("Polygon\n(Sứt, Bong nhôm)", "_polygon_defect", True),
    ("Scratch/Scar\n(Vết Hằn, Xước)", "_scratch_scar", True),
    ("Scribble\n(Tóc, Sợi vải, Nứt)", "_random_scribble", True),
    ("Amorphous\n(Bùn, Dị vật, Sỉ hàn)", "_amorphous_defect", True),
    # NHÓM 2: NHIỄU MÔI TRƯỜNG (Mask = 0)
    ("Glare\n(Chói sáng - mask=0)", "_glare_augment", False),
    ("Oil Stain\n(Dầu loang - mask=0)", "_oil_stain", False),
]

# Row 0-6: Từng loại lỗi riêng
for row, (name, method_name, is_defect) in enumerate(defect_types):
    method = getattr(dataset, method_name)
    img_copy = orig_image.copy()
    aug_img, mask = method(img_copy, foreground)
    
    orig_disp = orig_image.astype(np.float32) / 255.0
    aug_disp = aug_img.astype(np.float32) / 255.0
    
    if len(orig_disp.shape) == 3 and orig_disp.shape[2] == 1:
        orig_disp = orig_disp.squeeze()
        aug_disp = aug_disp.squeeze()
    
    axes[row, 0].imshow(orig_disp, cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_title('Original', fontsize=10)
    axes[row, 0].axis('off')
    
    title_color = 'red' if is_defect else 'green'
    axes[row, 1].imshow(aug_disp, cmap='gray', vmin=0, vmax=1)
    axes[row, 1].set_title(f'{name}', fontsize=10, fontweight='bold', color=title_color)
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(mask, cmap='hot', vmin=0, vmax=1)
    mask_pct = (mask.sum() / (mask.shape[0] * mask.shape[1])) * 100
    if mask_pct > 0:
        mask_label = f'Mask ({mask_pct:.2f}%)'
        color = 'red'
    else:
        mask_label = 'Mask = 0 (IGNORE)'
        color = 'green'
    axes[row, 2].set_title(mask_label, fontsize=10, color=color)
    axes[row, 2].axis('off')

# Row 7: COMBINED (Sequential như training)
row = 7
aug_combined = orig_image.copy()
final_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]), dtype=np.float32)

# Áp dụng tuần tự như trong __getitem__
aug_combined, m = dataset._coarse_dropout(aug_combined, foreground)
final_mask = np.maximum(final_mask, m)

aug_combined, m = dataset._polygon_defect(aug_combined, foreground)
final_mask = np.maximum(final_mask, m)

aug_combined, m = dataset._scratch_scar(aug_combined, foreground)
final_mask = np.maximum(final_mask, m)

aug_combined, m = dataset._random_scribble(aug_combined, foreground)
final_mask = np.maximum(final_mask, m)

aug_combined, m = dataset._amorphous_defect(aug_combined, foreground)
final_mask = np.maximum(final_mask, m)

aug_combined, _ = dataset._glare_augment(aug_combined, foreground)
aug_combined, _ = dataset._oil_stain(aug_combined, foreground)

# Display combined
orig_disp = orig_image.astype(np.float32) / 255.0
aug_disp = aug_combined.astype(np.float32) / 255.0
if len(orig_disp.shape) == 3 and orig_disp.shape[2] == 1:
    orig_disp = orig_disp.squeeze()
    aug_disp = aug_disp.squeeze()

axes[row, 0].imshow(orig_disp, cmap='gray', vmin=0, vmax=1)
axes[row, 0].set_title('Original', fontsize=10)
axes[row, 0].axis('off')

axes[row, 1].imshow(aug_disp, cmap='gray', vmin=0, vmax=1)
axes[row, 1].set_title('COMBINED\n(All Sequential)', fontsize=10, fontweight='bold', color='blue')
axes[row, 1].axis('off')

axes[row, 2].imshow(final_mask, cmap='hot', vmin=0, vmax=1)
mask_pct = (final_mask.sum() / (final_mask.shape[0] * final_mask.shape[1])) * 100
axes[row, 2].set_title(f'Combined Mask ({mask_pct:.2f}%)', fontsize=10, color='blue')
axes[row, 2].axis('off')

plt.suptitle(f'Các loại lỗi mô phỏng - Class: {sub_class}', fontsize=14, fontweight='bold')
plt.savefig('augment_test.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
print("Saved to augment_test.png")
plt.show()
