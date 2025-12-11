import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import albumentations as A
from PIL import Image
from torchvision import transforms
import random
import functools
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict, deque
from contextlib import contextmanager

class PerformanceProfiler:
    """
    Performance profiling tool for monitoring training performance
    """
    
    def __init__(self, log_dir="outputs/profiling", window_size=100):
        self.log_dir = log_dir
        self.window_size = window_size
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance metrics
        self.metrics = defaultdict(deque)
        self.timers = {}
        self.counters = defaultdict(int)
        
        # System monitoring
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.gpu_memory = deque(maxlen=window_size)
        self.gpu_utilization = deque(maxlen=window_size)
        
        # Training specific
        self.batch_times = deque(maxlen=window_size)
        self.epoch_times = deque(maxlen=window_size)
        self.loss_values = defaultdict(lambda: deque(maxlen=window_size))
        
    @contextmanager
    def timer(self, name):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.metrics[name].append(elapsed)
            if len(self.metrics[name]) > self.window_size:
                self.metrics[name].popleft()
    
    def log_system_metrics(self):
        """Log current system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        
        # Memory
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            self.gpu_memory.append({
                'allocated': gpu_memory_allocated,
                'reserved': gpu_memory_reserved
            })
            
            # Try to get GPU utilization (requires nvidia-ml-py3)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.append(util.gpu)
            except:
                self.gpu_utilization.append(0)
    
    def log_training_metrics(self, epoch, batch_idx, losses, batch_time=None, epoch_time=None):
        """Log training specific metrics"""
        if batch_time:
            self.batch_times.append(batch_time)
        if epoch_time:
            self.epoch_times.append(epoch_time)
            
        # Log losses
        for loss_name, loss_value in losses.items():
            self.loss_values[loss_name].append(loss_value)
    
    def get_summary_stats(self):
        """Get summary statistics"""
        stats = {}
        
        # System metrics
        if self.cpu_usage:
            stats['cpu'] = {
                'avg': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'current': self.cpu_usage[-1] if self.cpu_usage else 0
            }
        
        if self.memory_usage:
            stats['memory'] = {
                'avg': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'current': self.memory_usage[-1] if self.memory_usage else 0
            }
        
        if self.gpu_memory:
            allocated = [m['allocated'] for m in self.gpu_memory]
            reserved = [m['reserved'] for m in self.gpu_memory]
            stats['gpu_memory'] = {
                'allocated_avg': np.mean(allocated),
                'allocated_max': np.max(allocated),
                'reserved_avg': np.mean(reserved),
                'reserved_max': np.max(reserved)
            }
        
        # Training metrics
        if self.batch_times:
            stats['batch_time'] = {
                'avg': np.mean(self.batch_times),
                'std': np.std(self.batch_times),
                'throughput': 1.0 / np.mean(self.batch_times)  # batches per second
            }
        
        if self.epoch_times:
            stats['epoch_time'] = {
                'avg': np.mean(self.epoch_times),
                'std': np.std(self.epoch_times),
                'total': np.sum(self.epoch_times)
            }
        
        # Custom timers
        for timer_name, times in self.metrics.items():
            if times:
                stats[timer_name] = {
                    'avg': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times)
                }
        
        return stats
    
    def plot_metrics(self, save_path=None, inline=False):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # CPU usage
        if self.cpu_usage:
            axes[0, 0].plot(list(self.cpu_usage))
            axes[0, 0].set_title('CPU Usage (%)')
            axes[0, 0].set_ylabel('Usage %')
            axes[0, 0].grid(True)
        
        # Memory usage
        if self.memory_usage:
            axes[0, 1].plot(list(self.memory_usage))
            axes[0, 1].set_title('Memory Usage (%)')
            axes[0, 1].set_ylabel('Usage %')
            axes[0, 1].grid(True)
        
        # GPU memory
        if self.gpu_memory:
            allocated = [m['allocated'] for m in self.gpu_memory]
            reserved = [m['reserved'] for m in self.gpu_memory]
            axes[0, 2].plot(allocated, label='Allocated')
            axes[0, 2].plot(reserved, label='Reserved')
            axes[0, 2].set_title('GPU Memory (GB)')
            axes[0, 2].set_ylabel('Memory GB')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Batch times
        if self.batch_times:
            axes[1, 0].plot(list(self.batch_times))
            axes[1, 0].set_title('Batch Processing Time')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True)
        
        # Loss values
        if self.loss_values:
            for loss_name, values in self.loss_values.items():
                axes[1, 1].plot(list(values), label=loss_name)
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # GPU utilization
        if self.gpu_utilization:
            axes[1, 2].plot(list(self.gpu_utilization))
            axes[1, 2].set_title('GPU Utilization (%)')
            axes[1, 2].set_ylabel('Utilization %')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if inline:
            # Display inline for Jupyter notebooks
            try:
                from IPython.display import display
                display(fig)
                plt.close()
            except ImportError:
                # Fallback to regular display if IPython not available
                plt.show()
        else:
            # Save to file
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(self.log_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_stats(self, filename=None):
        """Save statistics to JSON file"""
        stats = self.get_summary_stats()
        
        if filename is None:
            filename = os.path.join(self.log_dir, 'performance_stats.json')
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_summary_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'cpu' in stats:
            print(f"CPU Usage: {stats['cpu']['avg']:.1f}% avg, {stats['cpu']['max']:.1f}% max")
        
        if 'memory' in stats:
            print(f"Memory Usage: {stats['memory']['avg']:.1f}% avg, {stats['memory']['max']:.1f}% max")
        
        if 'gpu_memory' in stats:
            print(f"GPU Memory: {stats['gpu_memory']['allocated_avg']:.2f}GB avg, {stats['gpu_memory']['allocated_max']:.2f}GB max")
        
        if 'batch_time' in stats:
            print(f"Batch Time: {stats['batch_time']['avg']:.3f}s avg, {stats['batch_time']['throughput']:.2f} batches/s")
        
        if 'epoch_time' in stats:
            print(f"Epoch Time: {stats['epoch_time']['avg']:.2f}s avg, {stats['epoch_time']['total']:.2f}s total")
        
        print("="*60)
    
    def detect_bottlenecks(self):
        """Detect potential performance bottlenecks"""
        bottlenecks = []
        stats = self.get_summary_stats()
        
        # High CPU usage
        if 'cpu' in stats and stats['cpu']['avg'] > 80:
            bottlenecks.append("High CPU usage detected - consider reducing data preprocessing or num_workers")
        
        # High memory usage
        if 'memory' in stats and stats['memory']['avg'] > 85:
            bottlenecks.append("High memory usage detected - consider reducing batch size or enabling gradient checkpointing")
        
        # High GPU memory usage
        if 'gpu_memory' in stats and stats['gpu_memory']['allocated_avg'] > 8:
            bottlenecks.append("High GPU memory usage - consider reducing batch size or model size")
        
        # Slow batch processing
        if 'batch_time' in stats and stats['batch_time']['avg'] > 2.0:
            bottlenecks.append("Slow batch processing - check data loading pipeline and model efficiency")
        
        # Low GPU utilization
        if self.gpu_utilization and np.mean(self.gpu_utilization) < 50:
            bottlenecks.append("Low GPU utilization - data loading might be the bottleneck")
        
        return bottlenecks

# Decorator for automatic timing
def profile_function(profiler, name=None):
    """Decorator to automatically profile function execution time"""
    def decorator(func):
        func_name = name or func.__name__
        def wrapper(*args, **kwargs):
            with profiler.timer(func_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global profiler instance
global_profiler = None

def get_profiler():
    """Get or create global profiler instance"""
    global global_profiler
    if global_profiler is None:
        global_profiler = PerformanceProfiler()
    return global_profiler

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):  
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):  #shape (256 256) res (16,2))
    delta = (res[0] / shape[0], res[1] / shape[1]) #(1/16,1,128)
    d = (shape[0] // res[0], shape[1] // res[1])  #(16,128)
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1   #delta 为间隔 0:res[0]为上下界。 (256,256,2)

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)    #(17,3)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  #(17,3,2)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1) # (272,384,2)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) #(256,256)
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]]) #(256,256,2)
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]) #(256,256)



texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']


class RealIADTestDataset(Dataset):
    def __init__(self, data_path, classname, img_size):
        # data_path already includes RealIAD/classname, just add test
        self.root_dir = os.path.join(data_path, 'test')
        self.images = sorted(glob.glob(self.root_dir + "/*/*.jpg"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))

        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.transpose(np.array(image).astype(np.float32), (2, 0, 1))
        mask = np.transpose(np.expand_dims(np.array(mask).astype(np.float32), axis=2), (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)

        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/bad_mask')
            mask_file_name = file_name.split(".")[0] + ".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx, 'file_name': [img_path]}
        return sample

class RealIADTrainDataset(Dataset):
    def __init__(self, data_path, classname, img_size, args):
        self.classname = classname
        self.root_dir = os.path.join(data_path, 'train', 'good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        # Hỗ trợ nhiều định dạng ảnh, không chỉ .jpg
        IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, ext)))
        self.image_paths = sorted(self.image_paths)

        self.anomaly_source_paths = []
        anomaly_glob_path = os.path.join(self.anomaly_source_path, "images", "*")
        for ext in IMAGE_EXTENSIONS:
            self.anomaly_source_paths.extend(glob.glob(os.path.join(anomaly_glob_path, ext)))
        self.anomaly_source_paths = sorted(self.anomaly_source_paths)
        
        # Cache for preprocessed data
        self._image_cache = {}
        self._thresh_cache = {}
        self._cache_enabled = True
        
        print(f"Dataset initialized with {len(self.image_paths)} training images and {len(self.anomaly_source_paths)} anomaly sources")

        # Giữ nguyên các augmenters - chuyển sang albumentations
        self.augmenters = [
            A.RandomGamma(gamma_limit=(50, 200), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=0, p=1.0),
            A.Solarize(threshold=128, p=1.0),
            A.Posterize(num_bits=4, p=1.0),
            A.InvertImg(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Equalize(mode='cv', by_channels=True, p=1.0),
            A.Rotate(limit=45, p=1.0)
        ]
        self.rot_transform = A.Rotate(limit=90, p=1.0)

    def __len__(self):
        return len(self.image_paths)
    
    def get_foreground(self, image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        # Thêm _mask vào tên file
        base_name = os.path.basename(foreground_path)
        name_without_ext = os.path.splitext(base_name)[0]
        mask_name = f"{name_without_ext}_mask.jpg"
        foreground_path = os.path.join(os.path.dirname(foreground_path), mask_name)
        return foreground_path

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        selected_augs = [self.augmenters[i] for i in aug_ind]
        aug = A.Compose(selected_augs)
        return aug

    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  
                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot_transform(image=perlin_noise)['image']
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)  

                msk = (object_perlin).astype(np.float32) 
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
                
            
            if self.classname in texture_list: # only DTD
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)['image']
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else: # DTD and self-augmentation
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  # >0.5 is DTD 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)['image']
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else: #self-augmentation
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)['image']
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)
        
    def _load_and_cache_image(self, image_path):
        """Load and cache preprocessed image"""
        if self._cache_enabled and image_path in self._image_cache:
            return self._image_cache[image_path]
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        if self._cache_enabled:
            self._image_cache[image_path] = image
            
        return image
    
    def _load_and_cache_thresh(self, thresh_path):
        """Load and cache threshold mask"""
        if self._cache_enabled and thresh_path in self._thresh_cache:
            return self._thresh_cache[thresh_path]
        
        thresh = cv2.imread(thresh_path, 0)
        
        # Xử lý trường hợp file mask không tồn tại
        if thresh is None:
            print(f"Warning: Mask file not found: {thresh_path}")
            # Tạo mask trắng (toàn bộ là foreground)
            thresh = np.ones((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8) * 255
        else:
            thresh = cv2.resize(thresh, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        thresh = np.array(thresh).astype(np.float32) / 255.0
        
        if self._cache_enabled:
            self._thresh_cache[thresh_path] = thresh
            
        return thresh

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Use cached loading
        image = self._load_and_cache_image(image_path)
        cv2_image = image.copy()
        image = np.array(image).astype(np.float32) / 255.0

        thresh_path = self.get_foreground(image_path)
        thresh = self._load_and_cache_thresh(thresh_path)

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly = self.perlin_synthetic(image, thresh, anomaly_path, cv2_image, thresh_path)

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        sample = {'image': image, "anomaly_mask": anomaly_mask, 'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        return sample