# Use official PyTorch image with CUDA and build tools
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Avoid Python bytecode, unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Optional: enable pip wheel cache for speed
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Then copy the rest of your code
COPY . .

# Set default command (edit if needed)
CMD ["python", "main.py"]