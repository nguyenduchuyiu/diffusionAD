# 🏆 CV Hackathon Template

Template tổng quát cho Computer Vision hackathon - Thiết kế để dễ tùy chỉnh cho bất kỳ task nào.

## 🚀 Quick Start

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
cd src && python train.py --data_dir ../data --save_path ../best_model.pth

# API
cd api && python app.py

# Demo
cd demo && streamlit run app.py
```

## 📁 Cấu trúc dữ liệu

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    └── class2/
```

## 🔧 Tùy chỉnh

### Training Arguments

```bash
python src/train.py \
  --data_dir data \
  --model resnet18 \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001 \
  --input_size 224 \
  --freeze_backbone \
  --save_path best_model.pth
```

### Supported Models
- `resnet18`, `resnet34`, `resnet50`
- `efficientnet_b0`
- `mobilenet_v2`

## 📊 API Endpoints

- `GET /` - Health check
- `GET /classes` - Get class names
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction

## 🖥️ Demo Features

- Single image upload & prediction
- Batch processing
- Confidence visualization
- Results download

## 🐳 Docker Services

- **api**: FastAPI server (port 8000)
- **demo**: Streamlit UI (port 8501)
- **train**: Training service (run once)

## ⚡ Quick Tips

1. **Chuẩn bị data**: Tổ chức theo folder structure
2. **Training nhanh**: Sử dụng `./run.sh local-train`
3. **Test ngay**: `./run.sh demo` để test model
4. **Deploy**: `./run.sh all` để chạy API + Demo

---

**Happy Hacking! 🚀**
