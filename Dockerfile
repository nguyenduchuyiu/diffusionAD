# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY demo/ ./demo/
COPY config.yaml .
COPY run.sh .

# Copy models if they exist
COPY best_*.pth ./
COPY best_*.pkl ./

# Make run script executable
RUN chmod +x run.sh

# Create directories
RUN mkdir -p data/train/normal data/val/normal data/test/normal data/test/anomaly runs uploads results

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["./run.sh"]
