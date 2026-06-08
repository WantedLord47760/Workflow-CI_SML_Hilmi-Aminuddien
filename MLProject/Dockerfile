# ============================================================
# Dockerfile - Heart Disease Prediction API
# Penulis: Hilmi Aminuddien
# Base: Python 3.12 slim (menggantikan MLflow auto-generated Dockerfile)
# ============================================================

FROM python:3.12-slim

LABEL maintainer="Hilmi Aminuddien"
LABEL description="Heart Disease Prediction Model - MLflow Serving"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install sistem dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (layer caching)
COPY requirements_serve.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_serve.txt

# Copy model artifacts dan script serving
COPY . .

# Expose port
EXPOSE 5001

# Environment variables
ENV MLFLOW_TRACKING_URI=mlruns
ENV MODEL_NAME=heart-disease-model

# Jalankan MLflow model serving
# Model path akan di-set saat container di-run via environment variable
CMD ["sh", "-c", "mlflow models serve -m $MODEL_PATH -p 5001 -h 0.0.0.0 --env-manager=local --no-conda"]
