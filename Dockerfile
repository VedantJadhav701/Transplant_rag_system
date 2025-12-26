# Medical RAG API - GPU-Enabled Docker Image
# ===========================================
# Base image with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Medical RAG Team"
LABEL description="Production-ready Medical Transplant RAG API with GPU support"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip and install uv
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY rag_config.toml .

# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SECRET_KEY="change-this-in-production"
ENV CHROMA_TELEMETRY="false"
ENV ANONYMIZED_TELEMETRY="False"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
