# Deepfake Research — Face Swap Demo

> Tài liệu cho buổi sharing nội bộ.
> Demo: real-time face swap qua webcam trên browser, chạy hoàn toàn local.

---

## Mục lục

1. [Deepfake là gì?](#1-deepfake-là-gì)
2. [Demo overview](#2-demo-overview)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Khái niệm cốt lõi](#4-khái-niệm-cốt-lõi)
5. [AI Models](#5-ai-models)
6. [Performance Profiling & Optimization](#6-performance-profiling--optimization)
7. [Vấn đề & giải pháp](#7-vấn-đề--giải-pháp)
8. [Tuneable Parameters](#8-tuneable-parameters)
9. [Cách chạy](#9-cách-chạy)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. Deepfake là gì?

**Deepfake** = Deep Learning + Fake — dùng neural network tổng hợp media trông thật nhưng là giả.

| Loại | Mô tả | Ví dụ |
|---|---|---|
| **Face Swap** | Thay mặt A lên video người B | ← *project này* |
| **Face Reenactment** | Điều khiển biểu cảm theo video nguồn | First Order Motion |
| **Voice Cloning** | Clone giọng nói từ vài giây audio | VALL-E, Bark |

---

## 2. Demo overview

```
┌──────────────────────────────────────────────────────────┐
│                  http://localhost:7777                    │
│                                                          │
│  ┌───────────────────┐    ┌───────────────────┐         │
│  │   Webcam gốc      │    │  Face Swap Output │         │
│  │   (mirrored)      │    │  (realtime)       │         │
│  └───────────────────┘    └───────────────────┘         │
│                                                          │
│  [📁 Upload Source Face]  [▶ Bắt đầu]  [■ Dừng]        │
│                                                          │
│  ⏱ Pipeline Timing Panel                                │
│  (decode → detect → swap → encode)                      │
└──────────────────────────────────────────────────────────┘
```

**Flow người dùng:**
1. Mở browser → `localhost:7777`
2. Upload ảnh có mặt người muốn swap sang (source face)
3. Bấm "Bắt đầu Swap" → webcam bật, mặt trong webcam bị thay bằng source face

**Chạy hoàn toàn local** — không gửi data ra internet, không cần GPU.

---

## 3. Kiến trúc hệ thống

### Tech Stack

```
Frontend   │  HTML5 + Vanilla JS (Canvas API, WebSocket, getUserMedia)
───────────┼──────────────────────────────────────────────────────────
Backend    │  Python 3.11 · FastAPI · uvicorn (ASGI)
           │  InsightFace 0.7.3 · ONNX Runtime 1.20.1 · OpenCV
───────────┼──────────────────────────────────────────────────────────
AI Models  │  SCRFD (detection) · ArcFace (embedding) · inswapper (swap)
───────────┼──────────────────────────────────────────────────────────
Infra      │  Docker + Docker Compose · Port 7777 · Named volume
```

### System Flow

```
 BROWSER                              FASTAPI SERVER
 ───────                              ──────────────
 Webcam getUserMedia()
    │
    ├─ Canvas capture (mirrored)
    │  480×360, JPEG quality=0.55
    │
    ├──── WebSocket binary ──────────► receive_bytes()
    │                                    │
    │                                    ▼  run_in_executor
    │                                 ┌─────────────────────┐
    │                                 │   FaceSwapper        │
    │                                 │   ① decode JPEG      │
    │                                 │   ② SCRFD detect     │
    │                                 │   ③ inswapper swap   │
    │                                 │   ④ JPEG encode      │
    │                                 └─────────────────────┘
    │                                    │
    ◄──── swapped JPEG binary ──────────┘
    │
    ├─ drawImage lên output canvas
    └─ gửi frame tiếp (request-response loop)
```

### Source Face Setup (1 lần duy nhất)

```
Upload ảnh → SCRFD detect → ArcFace embedding [512-dim] → emap projection → lưu RAM (cached)
```

---

## 4. Khái niệm cốt lõi

### 4.1 ONNX — "Docker Image cho AI Model"

```
PyTorch / TensorFlow  ─── export ───►  .onnx file (platform-agnostic)
                                            │
                                       ONNX Runtime (CPU / GPU / ANE)
```

Format trung gian chạy model AI trên bất kỳ hardware nào mà không cần framework gốc.

### 4.2 Face Embedding — "DNA kỹ thuật số" của khuôn mặt

```
Khuôn mặt  ──►  ArcFace  ──►  [0.12, -0.87, 0.34, ..., 0.91]  (512 số)
```

- Hai mặt giống nhau → vector gần nhau (cosine similarity cao)
- Nắm bắt đặc trưng bản chất (hình mắt, tỷ lệ mũi) **độc lập** góc độ, ánh sáng
- `inswapper_128` dùng embedding để "in" đặc trưng source vào target, giữ nguyên pose/ánh sáng target

### 4.3 WebSocket vs HTTP

| | HTTP | WebSocket |
|---|---|---|
| Connection | Tạo mới mỗi request | 1 connection liên tục |
| Overhead/frame | ~800 bytes header | ~2 bytes header |
| Hướng | Client → Server | Full-duplex |

Ở 10 FPS, HTTP tạo **600 connections/phút**. WebSocket giữ **1 connection** duy nhất.

### 4.4 Backpressure — giữ latency cố định

```
❌ setInterval (gửi liên tục):           ✅ Request-Response:
  t=0    gửi frame 1                       t=0    gửi frame 1
  t=100  gửi frame 2 (server chưa xong)   t=200  nhận → gửi frame 2  (200ms)
  t=200  gửi frame 3 → buffer tăng        t=400  nhận → gửi frame 3  (200ms)
  → latency tăng vô hạn!                  → latency cố định ✓
```

Client chỉ gửi frame mới **sau khi nhận response** → không bao giờ tạo backlog.

### 4.5 `run_in_executor` — async + CPU-bound

```python
# ❌ Block event loop → server đóng băng
result = face_swapper.process_frame(frame_bytes)

# ✅ Offload sang thread pool → event loop vẫn xử lý request khác
result = await loop.run_in_executor(None, face_swapper.process_frame, frame_bytes)
```

---

## 5. AI Models

### Pipeline tổng quan

```
SOURCE (upload 1 lần):   SCRFD → ArcFace → embedding 512-dim → emap cache → RAM

WEBCAM (mỗi frame):     SCRFD → face crop → inswapper(cached_latent, crop) → blend → output
```

### 5.1 SCRFD — Face Detection (`det_10g.onnx`)

| | |
|---|---|
| **Nhiệm vụ** | Tìm vị trí khuôn mặt trong frame |
| **Output** | Bounding box + 5 landmarks (mắt, mũi, miệng) |
| **Input size** | Frame resize về 320×320 |
| **Đặc điểm** | Nhanh hơn YOLO cho face, cân bằng tốc độ/accuracy |

### 5.2 ArcFace — Face Recognition (`w600k_r50.onnx`)

| | |
|---|---|
| **Nhiệm vụ** | Chuyển mặt thành vector 512 chiều |
| **Input** | Ảnh mặt crop + align 112×112 |
| **Training** | WebFace600K — 600K người, 5M ảnh |

### 5.3 inswapper_128 — Face Swap (`inswapper_128.onnx`, ~500MB)

| | |
|---|---|
| **Input** | Source latent (projected) + target face crop (128×128) |
| **Output** | Swapped face (128×128) |
| **Post-process** | Alpha blend kết quả vào frame gốc bằng soft mask |

**Bên trong `INSwapper.get()`** (insightface source code):

```
① face_align.norm_crop2() → align target face → 128×128 crop
② cv2.dnn.blobFromImage()  → NCHW float32 blob (/255)
③ np.dot(embedding, emap)  → project source latent (chúng ta cache bước này)
④ session.run()             → ONNX inference → swapped 128×128
⑤ paste_back=True?
   → 3× warpAffine full image
   → erosion + dilation (morphological ops)
   → 2× GaussianBlur full image
   → float32 alpha blending toàn ảnh
```

> **Phát hiện quan trọng:** insightface document gọi step ⑤ là "Poisson blending"
> nhưng đọc source code (`inswapper.py` line 60-101) cho thấy thực tế là
> **alpha mask blending** (soft mask + `cv2.GaussianBlur`), không phải `cv2.seamlessClone`.

### Models không dùng

| Model | Mục đích | Lý do bỏ |
|---|---|---|
| `1k3d68.onnx` | 3D Landmark 68 điểm | Face swap không cần |
| `2d106det.onnx` | 2D Landmark 106 điểm | Face swap không cần |
| `genderage.onnx` | Đoán giới tính/tuổi | Không cần |

→ `allowed_modules=["detection","recognition"]` bỏ 3/5 models, giảm **~40% thời gian load**.

---

## 6. Performance Profiling & Optimization

### 6.1 Timing — phát hiện bottleneck

Tích hợp `time.perf_counter()` vào 4 bước pipeline, rolling average 30 frames.

**Kết quả đo ban đầu (Docker, CPU-only, M1 Mac, chưa optimize):**

```
┌──────────────────────┬────────────┬────────────┬────────┐
│ Step                 │  Avg (ms)  │  Max (ms)  │   %    │
├──────────────────────┼────────────┼────────────┼────────┤
│ ① Decode + ColorConv │      0.6   │       2    │  0.03% │
│ ② SCRFD Detection    │    265.1   │     396    │ 11.4%  │
│ ③ inswapper + Blend  │  2,067.2   │   2,736    │ 88.5%  │ ← BOTTLENECK
│ ④ Encode JPEG        │      1.0   │       9    │  0.04% │
├──────────────────────┼────────────┼────────────┼────────┤
│ TOTAL                │  2,333.9   │            │ 0.4 FPS│
└──────────────────────┴────────────┴────────────┴────────┘
```

**Kết luận:** `inswapper + Blend` chiếm **88.5%** tổng thời gian → focus optimize step này.

### 6.2 Phân tích insightface source code

Đọc `insightface/model_zoo/inswapper.py` (101 dòng) → phát hiện `paste_back=True` chạy rất nhiều OpenCV operations trên full-image mỗi frame:

```
paste_back=True gồm:
  ├── diff computation (float32 subtraction, abs, mean)
  ├── cv2.invertAffineTransform
  ├── 3× cv2.warpAffine trên full image (480×360 × 3 lần!)
  ├── np.where + bbox calculation
  ├── cv2.erode (kernel ~mask_size/10)
  ├── cv2.dilate (kernel 2×2)
  ├── 2× cv2.GaussianBlur trên full image
  └── float32 alpha blending toàn ảnh
```

→ Nhiều operation thừa có thể thay thế bằng cách tự handle blending.

### 6.3 Tối ưu đã áp dụng

| # | Tối ưu | Chi tiết | Impact |
|---|---|---|---|
| 1 | **Custom lightweight blend** | Bypass `paste_back=True`. Dùng `paste_back=False` + 2× warpAffine + pre-computed soft mask + ROI-only blend | Giảm 6+ ops → 3 ops |
| 2 | **Cache emap projection** | `np.dot(embedding, emap) + norm` tính 1 lần khi upload, không mỗi frame | ~1ms/frame |
| 3 | **ONNX graph optimization** | `ORT_ENABLE_ALL` (constant folding, operator fusion, memory planning) | ~10-30% inference |
| 4 | **Pre-computed blend mask** | Soft feathered mask 128×128 tạo 1 lần init, reuse mỗi frame | Bỏ erosion+dilation+blur |

**So sánh trước/sau:**

```
TRƯỚC (insightface default):              SAU (custom):
─────────────────────────────              ─────────────
3× warpAffine full image                  1× warpAffine face 128×128 → full
  + diff mask computation                 1× warpAffine pre-computed mask
  + cv2.erode                             ROI-only alpha blend (chỉ vùng mặt ~150×150)
  + cv2.dilate
  + 2× GaussianBlur full image            → Bỏ: erosion, dilation, diff, blur,
  + full-image alpha blend                   full-image operations
```

### 6.4 Giới hạn CPU

ONNX neural net inference (`session.run()`) trên 128×128 input chiếm phần lớn step ③.
Đây là **giới hạn phần cứng** — encoder-decoder 500MB trên CPU không thể nhanh hơn.

| Hardware | Step ③ ước tính | FPS |
|---|---|---|
| CPU M1 (hiện tại) | ~300-500ms | 1-3 |
| NVIDIA GPU (CUDA) | ~10-30ms | 15-30 |
| NVIDIA GPU + TensorRT | ~5-15ms | 30-60 |

→ Để realtime (>15 FPS), **bắt buộc cần GPU**.
Trên CPU, tối ưu code-level chỉ cải thiện phần post-processing (~10-20% tổng).

---

## 7. Vấn đề & giải pháp

| # | Vấn đề | Root Cause | Giải pháp |
|---|---|---|---|
| 1 | Docker build `g++ not found` | insightface compile Cython | `build-essential` trong Dockerfile |
| 2 | `ml_dtypes` AttributeError | onnx 1.19 cần ml_dtypes≥0.5.0 | Pin `ml_dtypes>=0.5.0` |
| 3 | CoreML `I/O error` trên M1 | buffalo_l opset không compatible | `CPUExecutionProvider` only |
| 4 | Miss detect khi xoay mặt | `det_thresh=0.5` quá cao | `det_thresh=0.35` |
| 5 | Output nhấp nháy | SCRFD miss 1-2 frame | Temporal smoothing `FALLBACK_FRAMES=6` |
| 6 | Latency tăng dần | `setInterval` tạo WS backlog | Request-Response loop |
| 7 | Server freeze | `process_frame()` block event loop | `run_in_executor` |
| 8 | Ảnh bị đảo ngược | Webcam default selfie-mode | CSS `scaleX(-1)` + canvas mirror |
| 9 | Docker no space | Docker disk cache đầy | `docker system prune -af` |
| 10 | Swap 2s/frame | `paste_back=True`: 6+ OpenCV ops/frame | Custom lightweight blend |

---

## 8. Tuneable Parameters

| Thông số | Giá trị | Giảm → | Tăng → |
|---|---|---|---|
| `det_thresh` | 0.35 | Khó detect mặt | Nhiều false positive |
| `det_size` | 320×320 | Nhanh hơn, miss mặt nhỏ | Chậm hơn, detect tốt hơn |
| `FALLBACK_FRAMES` | 6 | Responsive khi mặt mất | Mượt khi xoay nhanh |
| JPEG quality server | 80 | Nhỏ hơn, mờ hơn | Rõ hơn, to hơn |
| Webcam resolution | 480×360 | Nhanh hơn | Chậm hơn, rõ hơn |
| JPEG quality client | 0.55 | Latency thấp | Latency cao |
| Docker memory | 4G | Có thể OOM | Tốn RAM host |

---

## 9. Cách chạy

### Docker (khuyến nghị)

```bash
# Download inswapper_128.onnx → thư mục models/

docker compose up --build -d    # Build & chạy
open http://localhost:7777      # Mở browser

docker compose logs -f          # Xem logs (có [PERF] timing)
curl localhost:7777/metrics     # Xem timing JSON
```

### Local (Python 3.11+)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 7777 --reload
```

### Cấu trúc project

```
Deep fake research/
├── app/
│   ├── main.py           # FastAPI server + WebSocket
│   └── face_swapper.py   # Face swap engine (optimized)
├── static/
│   └── index.html        # Frontend UI + metrics panel
├── models/
│   └── inswapper_128.onnx  # (~500MB, download thủ công)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── DEEPFAKE_RESEARCH.md
```

---

## 10. Key Takeaways

### AI/ML
- **Model lớn ≠ tốt hơn** — `det_size=320` đủ cho webcam, nhanh 4x so với 640
- **Chỉ load model cần thiết** — bỏ 3/5 sub-models → giảm 40%
- **ONNX provider phải test thực tế** — "hardware hỗ trợ" ≠ "chạy được"
- **Đọc source code thư viện** — insightface gọi "Poisson blend" nhưng thực tế là alpha mask. Hiểu internals mới optimize được
- **Profile trước khi optimize** — timing cho thấy 88.5% ở swap, không phải detect/encode

### Real-time Streaming
- **Backpressure bắt buộc** cho mọi producer-consumer pipeline
- **`run_in_executor`** = pattern chuẩn cho CPU-bound + async Python
- **WebSocket >> HTTP** cho video streaming

### Performance
- **Đo trước, optimize sau** — đừng đoán bottleneck
- **CPU có giới hạn cứng** — neural net inference cần GPU để realtime
- **Post-processing tối ưu được** — pre-compute, cache, ROI-only
- **Đừng wrap library blindly** — bypass default behavior khi cần

### Docker
- `build-essential` cho Cython extensions
- Named volume cho model cache
- Monitor disk space khi làm AI models (~500MB+)

---

*License: InsightFace models — Non-commercial research use only.*
*Sử dụng có trách nhiệm. Không dùng để tạo nội dung gây hiểu lầm.*
