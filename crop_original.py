import os
from PIL import Image

# Đường dẫn thư mục input và output
input_dir = "datasets/DENSO/mat_truc/ground_truth1"
output_dir = "datasets/DENSO/mat_truc/ground_truth"
# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Vùng crop (left, top, right, bottom)
crop_box = (751, 0, 3072, 2048)

# Duyệt qua toàn bộ file trong thư mục input
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Mở và crop ảnh
        img = Image.open(input_path)
        cropped_img = img.crop(crop_box)

        # Lưu ảnh đã crop vào thư mục output
        cropped_img.save(output_path)

        print(f"Đã xử lý: {filename}")

print("Hoàn tất! Ảnh đã được lưu trong:", output_dir)