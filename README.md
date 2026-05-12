# Deepfake Face Swap — Local Web Demo

Real-time face swap trên webcam qua browser. Backend Python, không cần GPU.

---

## Architecture

```
Browser (Webcam)
    │  getUserMedia() → canvas frame (JPEG)
    │  WebSocket binary frames
    ▼
FastAPI Server (port 7777)
    │  /ws WebSocket endpoint
    ▼
InsightFace Engine
    ├── SCRFD (det_10g.onnx)       → detect face regions
    ├── ArcFace (w600k_r50.onnx)   → extract 512-dim embedding
    └── INSwapper (inswapper_128)  → blend source → target face
    ▼
Swapped frame (JPEG) → WebSocket → Browser canvas
```

## Models

| Model | File | Vai trò |
|---|---|---|
| **SCRFD** | `det_10g.onnx` | Face detection — xác định vị trí mặt trong frame (bounding box) |
| **ArcFace ResNet50** | `w600k_r50.onnx` | Face recognition — chuyển khuôn mặt thành vector 512 chiều (embedding) |
| **3D Landmark** | `1k3d68.onnx` | 68 điểm landmark 3D cho alignment chính xác |
| **2D Landmark** | `2d106det.onnx` | 106 điểm landmark 2D chi tiết |
| **Gender/Age** | `genderage.onnx` | Phân loại giới tính + tuổi (metadata) |
| **INSwapper 128** | `inswapper_128.onnx` | Core swap model — inject source embedding vào target face region |

> **buffalo_l** (SCRFD + ArcFace + Landmark + GenderAge): tự động download lần đầu vào `~/.insightface/models/` (~275MB)
>
> **inswapper_128.onnx**: phải download thủ công và đặt vào `models/` (~500MB)
> → Download tại: [InsightFace Releases v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7)

---

## Yêu cầu

- macOS (Apple Silicon M1/M2/M3 hoặc Intel)
- Python 3.11+ **hoặc** Docker / OrbStack
- Webcam
- File `inswapper_128.onnx` trong thư mục `models/`

---

## Cách chạy

### Option A — Local (Python venv)

```bash
# 1. Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Đảm bảo model tồn tại
ls models/inswapper_128.onnx

# 4. Khởi động server
uvicorn app.main:app --host 127.0.0.1 --port 7777 --reload

# 5. Mở browser
open http://localhost:7777
```

> Lần đầu khởi động: InsightFace tự download `buffalo_l` (~275MB), chờ 1-2 phút.

### Option B — Docker (khuyến nghị)

```bash
# Build và chạy
docker compose up --build

# Chạy background
docker compose up --build -d

# Xem logs
docker compose logs -f

# Dừng
docker compose down
```

Truy cập: **http://localhost:7777**

> Buffalo_l models được cache vào Docker named volume `insightface-models` — không cần download lại khi restart.

---

## Sử dụng

1. **Upload Source Face**: click "📁 Upload Source Face" → chọn ảnh khuôn mặt muốn swap sang
   - Ảnh rõ mặt, nhìn thẳng, 1 người, tốt nhất là ảnh portrait
2. **Bắt đầu Swap**: nhấn "▶ Bắt đầu Swap" → allow quyền camera
3. **Xem kết quả**: cột phải hiển thị frame đã swap realtime

---

## Cấu trúc project

```
Deep fake research/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, WebSocket handler, HTTP endpoints
│   └── face_swapper.py   # InsightFace engine (FaceSwapper class)
├── static/
│   └── index.html        # Frontend: webcam capture + WebSocket client
├── models/
│   └── inswapper_128.onnx  # Face swap model (download thủ công)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Serve trang web `index.html` |
| `GET` | `/status` | Health check + trạng thái source face |
| `POST` | `/upload-source` | Upload ảnh source face (multipart/form-data) |
| `WS` | `/ws` | WebSocket: nhận binary frame → trả swapped frame |

### POST /upload-source

```bash
curl -X POST http://localhost:7777/upload-source \
  -F "file=@/path/to/face.jpg"
```

Response: `{"status": "ok", "message": "Source face đã được load thành công!"}`

---

## Performance (M1 Mac)

| Mode | FPS | Ghi chú |
|---|---|---|
| CPU (M1) | 5–12 FPS | Chạy local, unified memory |
| Docker (M1) | 3–8 FPS | Overhead container + linux/arm64 emulation |
| CPU (Intel) | 1–4 FPS | Chậm hơn do x86 |

> **Tip tăng FPS:** Giảm `det_size` từ `(640,640)` xuống `(320,320)` trong `face_swapper.py` line 36 — detect nhanh hơn nhưng accuracy giảm.

---

## Troubleshooting

**`inswapper_128.onnx not found`**
→ Đặt file vào `models/inswapper_128.onnx`

**`Không phát hiện được khuôn mặt`**
→ Dùng ảnh rõ mặt, nhìn thẳng, đủ sáng, tối thiểu 200x200px

**`WebSocket error` trên browser**
→ Kiểm tra server đang chạy: `curl http://localhost:7777/status`

**Docker build lỗi `libGL`**
→ Đã handle bằng `libgl1` package trong Dockerfile

**FPS thấp (< 3)**
→ Giảm resolution webcam trong `index.html`: đổi `width: 640` thành `width: 320`

---

## License & Ethics

- **InsightFace models**: [Non-commercial research license](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- **inswapper_128.onnx**: Chỉ dùng cho mục đích nghiên cứu, học tập
- **Không** dùng để tạo nội dung gây hiểu lầm, deepfake người thật mà không có sự đồng ý
