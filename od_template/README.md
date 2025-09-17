# YOLOv8 Object Detection Template

Template chuyên dụng cho Object Detection / Defect Detection sử dụng YOLOv8 - Thiết kế tối ưu cho hackathon và phát triển nhanh.

## Quick Start

### Cách 1: Sử dụng script (Khuyến nghị)

```bash
# Make executable
chmod +x run.sh

# Training
./run.sh train              # Docker training
./run.sh local-train        # Local training

# Run services
./run.sh api               # API server
./run.sh demo              # Demo UI
./run.sh all               # API + Demo

# Build
./run.sh build             # Build Docker images
```

### Cách 2: Docker Compose

```bash
# Training
docker-compose --profile training up train

# Services
docker-compose up api       # API server
docker-compose up demo      # Demo UI
docker-compose up api demo  # Both
```

### Cách 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Training
cd src && python train.py

# API
cd api && python app.py

# Demo
cd demo && streamlit run app.py
```

## Cấu trúc dữ liệu YOLO

```
data/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img3.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img3.txt
        └── ...
```

### Format annotation YOLO

Mỗi file `.txt` chứa các dòng annotation theo format:
```
class_id center_x center_y width height
```

Trong đó:
- `class_id`: ID của class (bắt đầu từ 0)
- `center_x, center_y`: Tọa độ trung tâm của bounding box (normalized 0-1)
- `width, height`: Kích thước bounding box (normalized 0-1)

Ví dụ:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

## Cấu hình Training

### Config file (config.yaml)

```yaml
data_dir: 'data'
model: 'yolov8n.pt'  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
epochs: 50
batch_size: 16
lr: 0.01
input_size: 640
save_path: 'best_yolo.pt'
confidence: 0.25
iou: 0.45
```

### Supported Models

- **YOLOv8n**: Nano - Nhanh nhất, nhẹ nhất
- **YOLOv8s**: Small - Cân bằng tốc độ và độ chính xác
- **YOLOv8m**: Medium - Độ chính xác cao hơn
- **YOLOv8l**: Large - Độ chính xác rất cao
- **YOLOv8x**: Extra Large - Độ chính xác tối đa

### Training Arguments

```bash
python src/train.py
# Sử dụng config.yaml để cấu hình
```

## API Endpoints

- `GET /` - Health check
- `GET /classes` - Get class names
- `POST /predict` - Single image detection
  - Parameters: `confidence`, `iou`, `draw_boxes`
- `POST /predict_batch` - Batch detection
- `GET /annotated_image/{path}` - Download annotated image

### API Usage Examples

```bash
# Single image detection
curl -X POST "http://localhost:8000/predict?confidence=0.5&draw_boxes=true" \
     -F "file=@image.jpg"

# Batch detection
curl -X POST "http://localhost:8000/predict_batch?confidence=0.3" \
     -F "files=@img1.jpg" -F "files=@img2.jpg"
```

## Demo Features

- **Single Image Detection**: Upload và detect objects trong 1 ảnh
- **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- **Interactive Parameters**: Điều chỉnh confidence và IoU threshold
- **Visualization**: Hiển thị bounding boxes và confidence scores
- **Statistics**: Thống kê detection results
- **Export Results**: Download kết quả dưới dạng CSV

## Docker Services

- **api**: FastAPI server (port 8000)
- **demo**: Streamlit UI (port 8501)
- **train**: Training service (run once)

## Performance & Evaluation

### Metrics được sử dụng:
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Độ chính xác
- **Recall**: Độ bao phủ

### Inference Speed:
- **YOLOv8n**: ~1ms per image (GPU)
- **YOLOv8s**: ~2ms per image (GPU)
- **YOLOv8m**: ~5ms per image (GPU)

## Quick Tips cho Hackathon

### 1. Chuẩn bị data nhanh
```bash
# Tạo cấu trúc thư mục
mkdir -p data/{images,labels}/{train,val}

# Sử dụng tools annotation:
# - LabelImg (desktop)
# - Roboflow (online)
# - CVAT (web-based)
```

### 2. Training nhanh (3-4 giờ trên Colab)
```bash
# Sử dụng YOLOv8n cho tốc độ
./run.sh local-train

# Hoặc training với pretrained weights
python src/train.py  # Auto-download pretrained weights
```

### 3. Test ngay lập tức
```bash
# Chạy demo để test model
./run.sh demo

# Hoặc dùng YOLO CLI (cực nhanh)
yolo predict model=best_yolo.pt source=test_image.jpg
```

### 4. Deploy production
```bash
# Chạy API + Demo
./run.sh all

# Hoặc chỉ API
./run.sh api
```

## Troubleshooting

### Lỗi thường gặp:

1. **Dataset structure không đúng**
   - Kiểm tra cấu trúc thư mục images/ và labels/
   - Đảm bảo tên file ảnh và label giống nhau

2. **Annotation format sai**
   - Kiểm tra format YOLO (5 số trên mỗi dòng)
   - Đảm bảo coordinates đã normalize (0-1)

3. **Out of memory**
   - Giảm batch_size trong config.yaml
   - Sử dụng model nhỏ hơn (yolov8n)

4. **Model không load được**
   - Kiểm tra đường dẫn model file
   - Đảm bảo đã training xong

### Debug commands:

```bash
# Kiểm tra dataset
python -c "from ultralytics import YOLO; YOLO().val(data='data/dataset.yaml')"

# Test inference
python -c "from ultralytics import YOLO; YOLO('best_yolo.pt').predict('test.jpg')"

# Validate model
yolo val model=best_yolo.pt data=data/dataset.yaml
```

## Use Cases

### 1. Defect Detection (Phát hiện lỗi sản phẩm)
- Detect scratches, dents, cracks trên sản phẩm
- Quality control trong manufacturing
- Surface inspection

### 2. Object Detection (Phát hiện đối tượng)
- Security surveillance
- Inventory management  
- Traffic monitoring

### 3. Medical Imaging
- Detect abnormalities in X-rays
- Tumor detection
- Medical equipment detection

## Performance Benchmarks

| Model | Size | mAP50 | Speed (ms) | Params |
|-------|------|-------|------------|--------|
| YOLOv8n | 6MB | 37.3 | 0.99 | 3.2M |
| YOLOv8s | 22MB | 44.9 | 1.20 | 11.2M |
| YOLOv8m | 52MB | 50.2 | 1.83 | 25.9M |
| YOLOv8l | 87MB | 52.9 | 2.39 | 43.7M |
| YOLOv8x | 136MB | 53.9 | 3.53 | 68.2M |

---

**Happy Detecting! 🎯**

Được tối ưu cho hackathon - từ zero đến production trong vài giờ!