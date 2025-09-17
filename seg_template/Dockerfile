# Base stage (ổn định, đủ lib cho CV)
FROM python:3.9-slim-bullseye as base

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY demo/ ./demo/

# API stage
FROM base as api
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Demo stage
FROM base as demo
EXPOSE 8501
CMD ["streamlit", "run", "demo/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Training stage
FROM base as train
CMD ["python", "src/train.py", "--help"]
