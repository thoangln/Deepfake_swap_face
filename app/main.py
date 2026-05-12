"""
FastAPI Server — Deepfake Web Demo

Endpoints:
  GET  /           → serve static/index.html
  POST /upload-source → nhận ảnh source face (multipart form)
  GET  /status     → kiểm tra trạng thái server + source face
  WS   /ws         → WebSocket: nhận webcam frame → trả swapped frame

Giải thích kiến trúc:
  - FastAPI: web framework async của Python, xử lý nhiều request đồng thời hiệu quả
  - StaticFiles: mount thư mục static/ để browser có thể lấy index.html trực tiếp
  - WebSocket: connection hai chiều (full-duplex) giữ liên tục, khác HTTP request/response
    → Phù hợp stream video frame vì không cần tạo connection mới cho mỗi frame
"""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.face_swapper import FaceSwapper

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent  # thư mục gốc project
MODEL_PATH = str(BASE_DIR / "models" / "inswapper_128.onnx")
STATIC_DIR = BASE_DIR / "static"

# Kiểm tra model tồn tại trước khi start
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model tại: {MODEL_PATH}\n"
        "Hãy đặt file inswapper_128.onnx vào thư mục models/"
    )

# --- Khởi tạo Face Swap Engine (singleton, load 1 lần khi start server) ---
# Model được load vào RAM/ANE ngay khi import module này
# Lần đầu chạy: InsightFace sẽ download buffalo_l (~400MB) vào ~/.insightface/models/
print("[INFO] Đang load Face Swap Engine... (lần đầu có thể mất 1-2 phút)")
face_swapper = FaceSwapper(model_path=MODEL_PATH)
print("[INFO] Face Swap Engine ready!")

# --- FastAPI App ---
app = FastAPI(title="Deepfake Web Demo", version="1.0.0")

# Mount thư mục static/ để serve HTML/CSS/JS
# URL /static/... → file static/...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve trang chính index.html"""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/status")
async def status():
    """Kiểm tra trạng thái server và source face"""
    return JSONResponse({
        "server": "running",
        "source_face_loaded": face_swapper.has_source_face,
        "model": "inswapper_128.onnx"
    })


@app.get("/metrics")
async def metrics():
    """
    Trả về timing breakdown của từng step trong pipeline (rolling avg 30 frames).
    Dùng để phân tích bottleneck và tối ưu performance.
    """
    return JSONResponse(face_swapper.get_metrics())


@app.post("/upload-source")
async def upload_source(file: UploadFile = File(...)):
    """
    Nhận ảnh source face qua multipart form upload.

    Giải thích:
      - UploadFile: FastAPI tự parse multipart/form-data, expose file-like object
      - await file.read(): đọc toàn bộ bytes của file (async, không block event loop)
      - face_swapper.set_source_face(): detect + extract ArcFace embedding, lưu lại

    Validation:
      - Chỉ chấp nhận image files (MIME type bắt đầu bằng "image/")
      - Trả 400 nếu không detect được mặt trong ảnh
    """
    # Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPEG, PNG...)")

    image_bytes = await file.read()

    # Giới hạn size 10MB để tránh ảnh quá lớn
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn, tối đa 10MB")

    success = face_swapper.set_source_face(image_bytes)
    if not success:
        raise HTTPException(
            status_code=422,
            detail="Không phát hiện được khuôn mặt trong ảnh. Hãy dùng ảnh rõ mặt, nhìn thẳng."
        )

    return JSONResponse({"status": "ok", "message": "Source face đã được load thành công!"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint xử lý video frame streaming.

    Protocol:
      - Client gửi: binary frame (JPEG bytes từ canvas.toBlob())
      - Server trả: binary frame đã swap (JPEG bytes)
        hoặc JSON text {"error": "..."} nếu có lỗi

    Giải thích flow:
      1. await websocket.accept(): hoàn thành WebSocket handshake
      2. Loop: receive_bytes() block cho đến khi nhận được frame tiếp theo
      3. face_swapper.process_frame() chạy đồng bộ (CPU/ANE compute)
      4. send_bytes() gửi kết quả về client
      5. WebSocketDisconnect: client đóng tab/connection → thoát loop bình thường
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    print(f"[WS] Client kết nối: {websocket.client}")

    try:
        while True:
            # Nhận binary frame từ browser (blocking async)
            frame_bytes = await websocket.receive_bytes()

            # run_in_executor: chạy CPU-bound swap trong thread pool
            # → không block asyncio event loop → server vẫn responsive với requests khác
            # Nếu không dùng run_in_executor, toàn bộ server bị freeze trong lúc swap
            result = await loop.run_in_executor(None, face_swapper.process_frame, frame_bytes)

            if result is None:
                # Chưa set source face hoặc không detect được mặt → báo client
                await websocket.send_json({
                    "type": "warning",
                    "message": "Chưa upload source face hoặc không phát hiện mặt trong frame"
                })
            else:
                # Gửi frame đã swap về client (binary)
                await websocket.send_bytes(result)

    except WebSocketDisconnect:
        print(f"[WS] Client ngắt kết nối: {websocket.client}")
    except Exception as e:
        print(f"[WS] Lỗi: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
