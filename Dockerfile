# Use a lightweight version of Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies for OpenCV
# 'libgl1' is the correct package for newer Debian versions (fixes the build error)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip first to avoid installation issues
RUN pip install --no-cache-dir --upgrade pip

# 3. Install CPU-only PyTorch FIRST.
# This prevents downloading 2GB+ of Nvidia drivers, fixing the timeout issue.
RUN pip install --no-cache-dir --timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install AI & Networking Dependencies
# Added 'transformers' and 'pillow' for the Live Captioning (FastVLM) feature
RUN pip install --no-cache-dir --timeout=1000 \
    opencv-python-headless \
    redis \
    websockets \
    numpy \
    ultralytics \
    transformers \
    pillow

# Copy EVERYTHING from our project folder into the container
COPY . .