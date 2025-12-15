"""Test augmentation visualization"""
import sys
sys.path.insert(0, 'src')

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from train_seg import AugmentedTrainDataset

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

# Plot 10 samples
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

for i in range(10):
    sample = dataset[i % len(dataset)]
    
    # Original
    img = sample['image'].transpose(1, 2, 0)
    if img.shape[2] == 1:
        img = img.squeeze()
    # Fix: dùng vmin/vmax cố định để không auto-normalize
    axes[i*2].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[i*2].set_title(f'Original {i}')
    axes[i*2].axis('off')
    
    # Augmented + mask overlay
    aug = sample['augmented_image'].transpose(1, 2, 0)
    mask = sample['anomaly_mask'].transpose(1, 2, 0).squeeze()
    has_anomaly = sample['has_anomaly'][0]
    
    if aug.shape[2] == 1:
        aug = aug.squeeze()
    # Fix: dùng vmin/vmax cố định
    axes[i*2+1].imshow(aug, cmap='gray', vmin=0, vmax=1)
    
    # Overlay mask in red
    if mask.sum() > 0:
        axes[i*2+1].imshow(mask, cmap='Reds', alpha=0.4)
    
    label = 'ANOMALY' if has_anomaly > 0 else 'NORMAL'
    
    # Debug: check if non-mask regions are identical
    diff = np.abs(img - aug)
    non_mask_diff = diff[mask < 0.5].mean() if (mask < 0.5).sum() > 0 else 0
    axes[i*2+1].set_title(f'Aug {i} [{label}] diff={non_mask_diff:.4f}')
    axes[i*2+1].axis('off')

plt.tight_layout()
plt.savefig('augment_test.png', dpi=150)
print("Saved to augment_test.png")
plt.show()

