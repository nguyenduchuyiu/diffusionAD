# Anomaly Detection Template

A comprehensive template for anomaly detection using Autoencoder and Isolation Forest approaches.

## Features

- **Autoencoder**: Reconstruction-based anomaly detection with difference heatmaps
- **Isolation Forest**: Feature-based anomaly detection using pretrained models (ResNet/CLIP)
- **AUROC Evaluation**: Comprehensive evaluation metrics
- **Heatmap Visualization**: Visual explanation of anomalies
- **FastAPI**: Production-ready API endpoints
- **Streamlit Demo**: Interactive web interface
- **Docker Support**: Easy deployment and reproducibility

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate anomaly-detection
```

### 2. Prepare Data

Organize your data in the following structure:

```
datasets/
└── RealIAD
    └── PCB5
        ├── DISthresh
        │   └── good
        ├── ground_truth
        │   └── bad_mask
        ├── test
        │   ├── bad
        │   └── good
        └── train
            └── good
```

### 3. Train Models

```bash
# Train both models
python src/train.py

# Train specific model
# Edit config.yaml to set method: 'autoencoder' or 'isolation_forest'
```

### 4. Run Applications

#### Option A: Using Docker (Recommended)

```bash
# Build and run all services
docker-compose up --build

# Run specific services
docker-compose --profile api up      # API only
docker-compose --profile demo up     # Demo only
docker-compose --profile training up # Training only
```

#### Option B: Manual Setup

```bash
# Start both API and demo
./run.sh

# Or start individually
uvicorn api.app:app --host 0.0.0.0 --port 8000  # API
streamlit run demo/app.py --server.port 8501    # Demo
```

## Usage

### Training

Configure training parameters in `config.yaml`:

```yaml
method: 'both'           # 'autoencoder', 'isolation_forest', or 'both'
epochs: 50
batch_size: 32
lr: 0.001
latent_dim: 128
contamination: 0.1       # Expected anomaly proportion
```

### Inference

#### Python API

```python
from src.inference import AutoencoderInference, IsolationForestInference

# Autoencoder
detector = AutoencoderInference('best_autoencoder.pth')
result = detector.predict('path/to/image.jpg', return_heatmap=True)

# Isolation Forest
detector = IsolationForestInference('best_isolation_forest.pkl')
result = detector.predict('path/to/image.jpg')

print(f"Anomaly Score: {result['anomaly_score']}")
print(f"Is Anomaly: {result['is_anomaly']}")
```

#### REST API

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "method=autoencoder" \
  -F "return_heatmap=true"

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "method=isolation_forest"
```

#### Web Interface

Access the Streamlit demo at `http://localhost:8501` for:
- Single image analysis with visual results
- Batch processing
- Interactive model comparison

## Model Details

### Autoencoder

- **Architecture**: Convolutional autoencoder with encoder-decoder structure
- **Loss**: MSE reconstruction loss
- **Anomaly Score**: Reconstruction error (MSE between input and output)
- **Visualization**: Difference heatmaps showing reconstruction errors

### Isolation Forest

- **Features**: Extracted from pretrained ResNet models
- **Algorithm**: Unsupervised outlier detection
- **Anomaly Score**: Isolation score (lower = more anomalous)
- **Scalability**: Efficient for large datasets

## Configuration

### Data Configuration
```yaml
data_dir: 'data'
input_size: 224
input_channels: 3
```

### Training Configuration
```yaml
method: 'both'
epochs: 50
batch_size: 32
lr: 0.001
```

### Model-Specific Settings
```yaml
# Autoencoder
latent_dim: 128

# Isolation Forest
feature_extractor: 'resnet18'
contamination: 0.1
```

## API Endpoints

### Main Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction
- `GET /models/info` - Model information

### Request/Response Format

```json
{
  "method": "autoencoder",
  "anomaly_score": 0.0234,
  "is_anomaly": false,
  "confidence": 0.0234,
  "original_image": "data:image/png;base64,...",
  "reconstructed_image": "data:image/png;base64,...",
  "heatmap": "data:image/png;base64,..."
}
```

## Evaluation Metrics

- **AUROC**: Area Under ROC Curve
- **Precision-Recall Curve**: For imbalanced datasets
- **Confusion Matrix**: Classification performance
- **Score Distribution**: Visual analysis of anomaly scores

## File Structure

```
ad_template/
├── src/
│   ├── models.py          # Model definitions
│   ├── train.py           # Training script
│   ├── inference.py       # Inference utilities
│   └── utils.py           # Helper functions
├── api/
│   └── app.py             # FastAPI application
├── demo/
│   └── app.py             # Streamlit demo
├── data/                  # Data directory
├── runs/                  # Training outputs
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose
└── run.sh                # Startup script
```

## Tips for Best Results

### Data Preparation
1. Ensure training data contains only normal samples
2. Use sufficient validation data with both normal and anomaly samples
3. Balance the validation set if possible

### Hyperparameter Tuning
1. Adjust `contamination` based on expected anomaly rate
2. Tune `latent_dim` for autoencoder complexity
3. Experiment with different feature extractors

### Threshold Calibration
1. Use validation data to find optimal thresholds
2. Consider precision-recall trade-offs
3. Adjust based on business requirements

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size
2. **Poor performance**: Check data quality and balance
3. **Model not loading**: Verify file paths and model compatibility

### Performance Optimization

1. Use GPU for training: Set device in config
2. Optimize batch size for your hardware
3. Use appropriate number of workers for data loading

## License

This template is provided as-is for educational and research purposes.
