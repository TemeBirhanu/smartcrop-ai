# Use a small official Python image
FROM python:3.11-slim

LABEL maintainer="you"

# Keep Python output unbuffered and send Matplotlib cache to /tmp (writable on Spaces)
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860

# Install system deps needed by OpenCV, ffmpeg and ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files (FastAPI-specific if present)
COPY ai/requirements_fastapi.txt ./ai/requirements_fastapi.txt
COPY requirements.txt ./requirements.txt

# Install Python deps: prefer ai/requirements_fastapi.txt if available, otherwise fall back
RUN python -m pip install --upgrade pip setuptools wheel && \
    (pip install --no-cache-dir -r ./ai/requirements_fastapi.txt || pip install --no-cache-dir -r ./requirements.txt)

# Copy the full project
COPY . .

# Expose HF Spaces default web port
EXPOSE 7860

# Use uvicorn to run the FastAPI app; allow PORT override by the environment
CMD ["sh", "-c", "uvicorn ai.predict_server_fastapi:app --host 0.0.0.0 --port ${PORT:-7860}"]