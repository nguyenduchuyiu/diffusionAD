#!/bin/bash

# Anomaly Detection Application Runner
echo "Starting Anomaly Detection Application..."

# Check if models exist
if [ ! -f "best_autoencoder.pth" ] && [ ! -f "best_isolation_forest.pkl" ]; then
    echo "Warning: No trained models found. Please train models first."
    echo "To train models, run: python src/train.py"
fi

# Function to start API
start_api() {
    echo "Starting FastAPI server..."
    cd /app && uvicorn api.app:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
}

# Function to start demo
start_demo() {
    echo "Starting Streamlit demo..."
    cd /app && streamlit run demo/app.py --server.address 0.0.0.0 --server.port 8501 &
    DEMO_PID=$!
}

# Function to cleanup
cleanup() {
    echo "Shutting down services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$DEMO_PID" ]; then
        kill $DEMO_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handling
trap cleanup SIGTERM SIGINT

# Start services based on environment variable or default to both
case "${SERVICE_MODE:-both}" in
    "api")
        start_api
        wait $API_PID
        ;;
    "demo")
        start_demo
        wait $DEMO_PID
        ;;
    "both"|*)
        start_api
        start_demo
        
        echo "Services started:"
        echo "- FastAPI: http://localhost:8000"
        echo "- Streamlit: http://localhost:8501"
        echo ""
        echo "Press Ctrl+C to stop all services"
        
        # Wait for either process to finish
        wait -n
        cleanup
        ;;
esac
