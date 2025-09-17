#!/bin/bash

# Quick run script for CV Hackathon Template

case $1 in
  "train")
    echo "🚀 Starting training..."
    docker-compose --profile training up train
    ;;
  "api")
    echo "🔌 Starting API server..."
    docker-compose up api
    ;;
  "demo")
    echo "🖥️ Starting demo UI..."
    docker-compose up demo
    ;;
  "all")
    echo "🚀 Starting API + Demo..."
    docker-compose up api demo
    ;;
  "build")
    echo "🔨 Building images..."
    docker-compose build
    ;;
  "local-train")
    echo "🏃 Training locally..."
    cd src && python train.py --data_dir ../data --save_path ../best_model.pth --epochs 20 --batch_size 32
    ;;
  "local-api")
    echo "🔌 Starting local API..."
    cd api && python app.py
    ;;
  "local-demo")
    echo "🖥️ Starting local demo..."
    cd demo && streamlit run app.py
    ;;
  *)
    echo "CV Hackathon Template"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Docker commands:"
    echo "  train     - Train model in container"
    echo "  api       - Start API server"
    echo "  demo      - Start demo UI"
    echo "  all       - Start API + Demo"
    echo "  build     - Build Docker images"
    echo ""
    echo "Local commands:"
    echo "  local-train - Train locally"
    echo "  local-api   - Start API locally"
    echo "  local-demo  - Start demo locally"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train"
    echo "  ./run.sh all"
    echo "  ./run.sh local-train"
    ;;
esac
