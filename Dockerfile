# Use a lightweight version of Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies for OpenCV
# FIX: Changed 'libgl1-mesa-glx' to 'libgl1' because the old package 
# does not exist in the newest Debian version.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# 3. Install CPU-only PyTorch FIRST.
RUN pip install --no-cache-dir --timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Ultralytics and other dependencies
RUN pip install --no-cache-dir --timeout=1000 opencv-python-headless redis websockets numpy ultralytics

# Copy EVERYTHING from our project folder into the container's /app folder
COPY . .