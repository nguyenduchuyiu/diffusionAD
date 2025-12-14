import torch
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import os

# ===== CONFIG =====
img_a_path = "a.png"
img_b_path = "b.png"
video_path = "diffusion_bridge.mp4"

# TỐI ƯU DUNG LƯỢNG
MAX_SIZE = 480       # Giới hạn cạnh nhỏ nhất là 480px (quan trọng để giảm size)
fps = 30             # Giảm fps xuống 15 (đủ cho mắt thường)
duration_per_phase = 3.0 # Giảm xuống 3s mỗi pha (6s quá dài sinh ra nhiều frame rác)

def load_gray_fill_box(path, target_res=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    img = Image.open(path).convert("L")
    if target_res is not None:
        target_w, target_h = target_res
        orig_w, orig_h = img.size
        scale = max(target_w / orig_w, target_h / orig_h)
        resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        left = (resized_w - target_w) // 2
        top = (resized_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    
    x = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x)

def add_gaussian_noise(x, sigma):
    if sigma <= 0: return x
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0, 1)

def to_uint8(x):
    return (x.numpy() * 255).astype(np.uint8)

# ===== Main Process =====
try:
    print("--- Bắt đầu xử lý ---")
    img_a_raw = Image.open(img_a_path)
    img_b_raw = Image.open(img_b_path)
    w_a, h_a = img_a_raw.size
    w_b, h_b = img_b_raw.size

    # 1. TÍNH TOÁN KÍCH THƯỚC MỚI (Tối ưu size)
    # Lấy kích thước nhỏ nhất của 2 ảnh
    base_w = min(w_a, w_b)
    base_h = min(h_a, h_b)
    
    # Tính tỷ lệ scale down nếu ảnh lớn hơn MAX_SIZE
    scale = min(1.0, MAX_SIZE / min(base_w, base_h))
    
    new_w = int(base_w * scale)
    new_h = int(base_h * scale)

    # Làm tròn chia hết cho 16 (cho codec video)
    target_w = (new_w // 16) * 16
    target_h = (new_h // 16) * 16
    
    target_res = (target_w, target_h)
    print(f"Kích thước video gốc: {base_w}x{base_h}")
    print(f"Kích thước sau tối ưu: {target_w}x{target_h} (Scale: {scale:.2f})")

    xa = load_gray_fill_box(img_a_path, target_res)
    xb = load_gray_fill_box(img_b_path, target_res)

    frames = []
    n_frames_phase = int(fps * duration_per_phase)
    max_sigma = 3.0

    # Phase 1: A -> Noise
    for i in range(n_frames_phase):
        progress = i / (n_frames_phase - 1)
        sigma = max_sigma * (progress ** 2) 
        frames.append(to_uint8(add_gaussian_noise(xa, sigma)))

    # Phase 2: Noise -> B
    for i in range(n_frames_phase):
        progress = i / (n_frames_phase - 1)
        sigma = max_sigma * (1 - progress ** 2)
        frames.append(to_uint8(add_gaussian_noise(xb, sigma)))

    # Frame kết thúc
    final_frame = to_uint8(xb)
    for _ in range(int(fps * 1.0)): # Giữ 1s cuối
        frames.append(final_frame)

    print(f"Tổng số frames: {len(frames)}")

    # 2. SAVE VIDEO (Ưu tiên MP4 vì nén tốt hơn GIF 100 lần với noise)
    try:
        # Cài đặt quality=5 (mặc định 5, range 0-10) để cân bằng đẹp/nhẹ
        imageio.mimsave(video_path, frames, fps=fps, quality=7)        
        print(f"✅ Đã lưu MP4 thành công: {video_path}")
        print("💡 Mẹo: File MP4 sẽ nhẹ hơn GIF rất nhiều.")
    except Exception as e:
        print(f"⚠️ Không lưu được MP4 (Lỗi: {e})")
        print("🔄 Đang chuyển sang chế độ lưu GIF tối ưu...")
        
        gif_path = video_path.replace(".mp4", ".gif")
        
        # Tối ưu cho GIF:
        # quantizer='nq': Dùng thuật toán NeuQuant để giảm bảng màu thông minh -> Giảm size
        imageio.mimsave(gif_path, frames, fps=fps, loop=0) 
        print(f"✅ Đã lưu GIF: {gif_path}")
        print("⚠️ Lưu ý: GIF chứa nhiễu (noise) sẽ luôn nặng hơn bình thường.")

except ImportError:
    print("❌ Thiếu thư viện! Hãy chạy: pip install imageio-ffmpeg")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")