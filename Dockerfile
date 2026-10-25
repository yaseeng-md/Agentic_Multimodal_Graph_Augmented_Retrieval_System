# ============================================================================
# AMG (Agentic Multimodal Graph-Augmented Retrieval System) - Dockerfile
# ============================================================================

FROM python:3.12-slim

# Prevent interactive prompts and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RUNNING_IN_DOCKER=true

WORKDIR /app

# Install system dependencies
# - poppler-utils: required for pdf2image / PDF rendering
# - libgl1, libglib2.0-0: required for OpenCV and visual operations
# - curl: for container healthchecks
# - git, build-essential: for compiling C/C++ dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configurable PyTorch wheel index (defaults to CUDA 12.4; override with https://download.pytorch.org/whl/cpu for CPU-only builds)
ARG TORCH_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Copy and install python dependencies
COPY requirements.txt /app/requirements.txt

# Install PyTorch with selected hardware acceleration index, followed by application requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --extra-index-url ${TORCH_EXTRA_INDEX_URL} && \
    pip install -r requirements.txt --extra-index-url ${TORCH_EXTRA_INDEX_URL}

# Copy the rest of the application source code
COPY . /app

# Ensure runtime directories exist
RUN mkdir -p /app/data /root/.cache/huggingface

# Healthcheck testing the /health endpoint
HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default command starts the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
