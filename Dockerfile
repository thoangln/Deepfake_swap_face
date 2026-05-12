# ============================================================
# Deepfake Face Swap Demo — Dockerfile
# Base: python:3.11-slim (stable với insightface + onnxruntime)
# ============================================================

FROM python:3.11-slim

# --- Metadata ---
LABEL maintainer="deepfake-demo"
LABEL description="Real-time face swap via InsightFace + inswapper_128"

# --- System dependencies ---
# build-essential: cung cấp g++, gcc, make — cần để compile insightface Cython extension
#   (insightface build wheel từ source, có Cython file mesh_core_cython.cpp)
# libgl1: thư viện OpenGL stub — cần để cv2 link trong Linux headless (không cần GPU thật)
# libglib2.0-0: GLib runtime — dependency của libGL
# curl: dùng cho health check
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Python dependencies ---
# Copy requirements trước (Docker layer cache: chỉ re-install khi requirements thay đổi)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Fix ml_dtypes conflict với onnx 1.19
    && pip install --no-cache-dir "ml_dtypes>=0.5.0"

# --- Application code ---
COPY app/     ./app/
COPY static/  ./static/
COPY models/  ./models/

# --- Port ---
# Port 7777 — tránh conflict với OrbStack (8000, 8080, 5000, 9000)
EXPOSE 7777

# --- Health check ---
# Docker sẽ gọi endpoint này mỗi 30s để kiểm tra container còn healthy không
# --start-period=60s: chờ 60s đầu cho InsightFace load model xong (buffalo_l ~400MB)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7777/status || exit 1

# --- Entrypoint ---
# --host 0.0.0.0: lắng nghe tất cả interface trong container (bắt buộc để host truy cập được)
# --workers 1: chỉ 1 worker vì FaceSwapper singleton giữ model trong RAM — multi-worker sẽ load nhiều lần
# Không dùng --reload trong production/Docker
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7777", \
     "--workers", "1"]
