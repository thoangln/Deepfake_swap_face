# Deepfake Research: Chạm Ngõ AI Face Swap

> Dành cho buổi technical sharing. Tập trung vào các khái niệm cốt lõi, tư duy thiết kế hệ thống và bài học kỹ thuật khi tích hợp AI models vào ứng dụng thực tế. Tối giản code, nhấn mạnh vào flow và design.

---

## 1. Deepfake là gì? Góc nhìn tổng quan
**Deepfake** là sự kết hợp giữa **Deep Learning** và **Fake** media, sử dụng mạng nề-ron nhân tạo để tổng hợp/thao túng hình ảnh, âm thanh, video sống động như thật.

**Các phân nhánh phổ biến:**
1. **Face Swap (Đổi mặt):** Lấy mặt của người A đắp lên người B trong video/ảnh (Phạm vi của project này).
2. **Face Reenactment (Khắc họa biểu cảm):** Điều khiển cử chỉ miệng, mắt, đầu của một bức ảnh tĩnh theo video của người khác (First Order Motion).
3. **Voice Cloning / Lip Sync:** Sao chép giọng nói chỉ từ vài giây audio mẫu và đồng bộ khẩu hình môi.

---

## 2. Nhập môn các thuật ngữ & Khái niệm cốt lõi

Khi tiếp cận quy trình tích hợp AI Deepfake, chúng ta sẽ bắt gặp các khái niệm sau:

*   **ONNX (Open Neural Network Exchange):** Được ví như "Docker Image dành cho AI". Nó là định dạng chuẩn giúp export model AI từ môi trường huấn luyện (PyTorch/TensorFlow) và dùng chung để chạy suy luận (Inference) trên đa dạng nền tảng phần cứng (CPU, GPU, Apple Neural Engine) mà không cần cài đặt lại môi trường gốc cồng kềnh.
*   **Face Embedding (Latent Space):** Bản "DNA kỹ thuật số" của khuôn mặt. Mạng AI sẽ nén ảnh khuôn mặt thành một vector (ví dụ mảng 512 con số). Các số này mô tả cấu trúc tự nhiên (mắt, mũi, xương hàm) bất biến và độc lập với ánh sáng hay góc nghiêng.
*   **Bounding Box & Landmarks:** Box là khung chữ nhật bao quanh một khuôn mặt. Landmarks là các điểm tọa độ chi tiết (mắt, mũi, khóe miệng) dùng để tính toán góc nghiêng và căn chỉnh mặt (Face Alignment).
*   **Alpha Blending & ROI (Region of Interest):** Lớp phủ mềm làm mờ (Feather Mask) giúp trộn mượt viền khuôn mặt nhân tạo vào ảnh thật. Chỉ áp dụng tính toán hình ảnh tại đúng một vùng nhỏ (ROI) thay vì xử lý trên toàn bộ hình ảnh lớn để tiết kiệm tối đa tài nguyên.

---

## 3. Kiến trúc Hệ thống & Luồng xử lý (Flows)

Dự án hỗ trợ hai luồng tương tác với người dùng với các mô hình quản lý process riêng biệt:

### 3.1. Luồng Real-time Face Swap (Webcam)
Đặc thù của luồng này là cần **Độ trễ rất thấp (Low Latency)**.
*   **Giao thức:** Sử dụng **WebSocket** duy trì 1 kết nối liên tục, tránh overhead sinh header/connection tốn kém của HTTP khi truyền video ở tần số 10-20 FPS.
*   **Backpressure Pattern (Request-Response Loop):** 
    Thay vì bắt Client gửi liên tục, Client sẽ gửi frame 1 $\rightarrow$ Ném vào Executor xử lý song song $\rightarrow$ Server trả kết quả $\rightarrow$ Client vẽ lên UI xong MỚI gửi tiếp frame 2. Đảm bảo server không bao giờ bị nghẽn (backlog) làm tăng dồn latency.

### 3.2. Luồng Video Batch Processing (File MP4)
Đặc thù tốn cực kỳ nhiều thời gian xử lý từng khung hình, có thể gây đứt connection/timeout.
*   **Giao thức:** **HTTP Polling** kết hợp Background Tasks.
*   **Flow & Graceful Shutdown:**
    1. Client nạp file video. Server cấp `job_id` và bắt tay xử lý ngầm (Background Task).
    2. API `/video-status`: Trả về tiến độ hiện tại để Client tạo thanh Progress bar qua Polling.
    3. Trạng thái Cancel (Dừng): Nếu Client bấm "Dừng", gọi API Cancel truyền tín hiệu Thread Event để ngắt ngầm process OpenCV an toàn, tránh waste CPU. Kết thúc sớm luồng chèn âm thanh (FFMPEG).

---

## 4. Bóc tách AI Models (InsightFace)

Hệ thống kết hợp nhiều model nhẹ chạy **hoàn toàn offline/local** để bảo vệ dữ liệu khuôn mặt:

1. **SCRFD (Face Detection):** Tìm khoanh vùng vị trí khuôn mặt. Cực kỳ nhanh, phù hợp quét camera real-time.
2. **ArcFace (Face Recognition):** Trích xuất nhận diện (Face Embedding 512 số). Đặc biệt, khối lượng tính toán này chỉ dùng **1 lần duy nhất** khi người dùng tải ảnh đích lên, và lưu cache lại trên RAM (tránh tính lại thừa thãi).
3. **INSwapper (Face Swap):** Mạng sinh nội dung thực hiện trích xuất mặt người và đắp đè đặc trưng nhận diện vào.

> *Mẹo tối ưu Load:* Thư viện InsightFace sẽ mặc định load tất tần tật các thành phần như nhận diện Tuổi, Giới tính, 3D Mesh. Chủ động filter lược bỏ để chỉ nạp Detection + Recognition giúp rút ngắn **~40% thời gian** khởi động ứng dụng ban đầu.

---

## 5. Cải thiện Hiệu năng thực chiến (Performance Optimizations)

Đây là các khía cạnh tốn chi phí rực rỡ nhất để đưa dự án từ chỗ "Swap chờ gãy cổ" thành "Swap Real-time mượt".

### 5.1. Bài toán Môi trường: Docker VM vs. Local Native 🔥 (BƯỚC NGOẶT)
*   **Vấn đề:** Chạy hệ thống trên Docker ở Mac có độ trễ cực cao, FPS lẹt đẹt. Trong khi đó, phần cứng Mac (chip M-series) nổi tiếng mạnh mẽ.
*   **Nguyên nhân:** Docker Desktop cấu trúc bên dưới trên Mac và Windows là chạy thông qua một máy ảo Linux (Linux VM). ONNX Runtime bị cô lập và chỉ thấy nhân CPU x86/ARM giả lập, **không thể tiếp cập Neural Engine (ANE)** hay GPU.
*   **Giải pháp:** Gỡ Docker -> Setup môi trường **chạy Local trực tiếp qua Python ảo (venv)**. Lúc này, hệ thống kích hoạt thành công provider tăng tốc: **`CoreMLExecutionProvider` (Native Apple Silicon)**. Độ trễ tụt đáng kinh ngạc, xử lý khung hình siêu tốc và mở khóa mức FPS kỳ vọng.

### 5.2. Thuật toán Frame-skipping (Tiết kiệm xử lý)
*   **Vấn đề:** Không phải khâu nào sinh ra cũng bắt buộc chạy trên định mức 100%. Model AI "Dò tìm mặt" (Detection) rất ngốn tài nguyên và không cần chạy mỗi milisecond (đầu người ít giật cục).
*   **Giải pháp - Áp dụng Cơ chế Theo vết (Tracking) + Cache:** Chỉ yêu cầu tìm mặt **sau mỗi 5 khung hình**. Các khung hình liền kề ở giữa sử dụng tái lại kết quả tọa độ của box vừa detect, chỉ tịnh tiến rất ít. Kéo giảm mạnh mẽ mức tải lên CPU tổng thể hệ thống.

### 5.3. Custom Lightweight Blend (Cắt chi phí bọc gói)
*   **Sự thật về thư viện mở:** Đọc sâu source code OpenCV của base model phát hiện các hàm ghép nối xử lý tràn lan làm mịn trên **Toàn khung hình 480x360 pixel** (Erosion, Dilation, Gaussian Blur) gây tụt giảm hiệu suất tàn bạo trên các thiết bị không có Card đồ họa mạnh.
*   **Giải pháp (Bypass):** Chủ động can thiệp vào tham số chạy `paste_back=False`. Thay thế cơ chế mặc định bằng luồng tự tạo: tính toán cắt ghép **vừa khít trong vùng kích cỡ 128x128 pixel quanh khuôn mặt**, lưu bộ đệm các viền mờ (Feather Cache). Kết quả là nhả cả chục task dư thừa từ thư viện cho cấu hình phổ thông.

---

## 6. Bài học kinh nghiệm Đắt giá (Key Takeways)

1. **Hiểu rõ giới hạn Tầng ảo hóa (Virtualized Environments):** Đừng vội vàng đổ lỗi do AI/Codebase khi dùng Docker mà thấy chậm. Rào cản truy cập phần cứng phân luồng cấp thấp (NPU/GPU/CoreML) từ bên trong Docker VM là vô cùng gian truân. Đối chiếu test Local là một quy trình bắt buộc trong ứng dụng AI thời gian thực.
2. **Measure First, Optimize Later (Đo lường trước, Tối ưu sau):** Tránh Optimize bằng cảm tính. Thực thi việc cắm timing/Profiling cho 4 bước (Decode, Detect, Swap, Encode) đã dẫn lối thẳng đến bước Swap tốn lượng tài nguyên khổng lồ nhất (chiếm 80%) thay vì Detect.
3. **Đừng tin mù quáng Framework (Bypass Defaults):** Tool mã nguồn mở được viết ra để có độ "bao phủ/trơn tru an toàn nhất", không phải "thời gian thực" nhất. Hiểu luồng rễ bên dưới cho phép lược bỏ tính toán dư là cách hệ thống bùng nổ hiệu năng.
4. **Kiểm soát Graceful Shutdown:** Xử lý media lớn là bài toán Background Worker. Cần thiết kế tốt các tín hiệu hủy luồng an toàn (Event Signals) cho phép break các vòng lặp tính toán nặng giúp giải phóng RAM và tránh kẹp server treo triền miên.

---

## 7. Setup System (TL;DR)

**Chạy Native trên hệ thống Local (Sử dụng CoreML - KHUYÊN DÙNG ĐỂ DEMO PERFORMANCE):**
```bash
# MacOS: Bắt buộc Python 3.11.x để tương thích mượt dependency ml-dtypes & numpy
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Server
uvicorn app.main:app --host 127.0.0.1 --port 7778 --reload
# -> Truy cập: http://localhost:7778
```

**Chạy hệ thống cấp Container (Docker - Fallback CPU):**
```bash
docker compose up --build -d
# Cổng expose tại 7777 thay vì 7778
```

---

## 8. Chrome Extension — Face Swap trên Google Meet

Mở rộng demo từ bản web sang **Chrome Extension** cho phép swap face trực tiếp trong cuộc gọi Google Meet.

### 8.1. Kiến trúc Extension

```
┌─────────────────────────────────────────────────────────┐
│  Chrome Extension (MV3)                                 │
│                                                         │
│  ┌──────────┐    messages    ┌──────────────────────┐  │
│  │  Popup   │ ◄────────────► │  Content Script      │  │
│  │ (UI/UX)  │                │  (inject vào Meet)   │  │
│  └──────────┘                └──────────┬───────────┘  │
│                                         │               │
│                              Override getUserMedia()    │
│                              Hook RTCPeerConnection     │
│                                         │               │
└─────────────────────────────────────────┼───────────────┘
                                          │ WebSocket
                                          ▼
                              ┌──────────────────────┐
                              │  FastAPI Backend      │
                              │  (local:7778)         │
                              │  SCRFD → inswapper    │
                              └──────────────────────┘
```

### 8.2. Cơ chế hoạt động (Key Concepts)

1. **Override `getUserMedia()`:** Content script chạy ở `document_start`, ghi đè hàm lấy webcam TRƯỚC khi Google Meet gọi. Khi Meet yêu cầu camera → ta vẫn lấy stream thật nhưng trả về một stream giả (canvas).
2. **Hook `RTCPeerConnection.addTrack()`:** Bắt tất cả video sender của WebRTC, dùng `sender.replaceTrack()` để thay video track thật bằng canvas track đã swap.
3. **Canvas Pipeline:** Webcam thật → hidden `<video>` → capture lên `<canvas>` → gửi qua WebSocket → nhận frame đã swap → vẽ lại lên canvas → `captureStream()` trả cho Meet.
4. **Backpressure giữ nguyên:** Cùng mô hình Request-Response — chỉ gửi frame mới sau khi nhận response từ server.

### 8.3. Cách cài đặt & sử dụng

```bash
# 1. Chạy backend server (bắt buộc)
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 7778 --reload

# 2. Load extension vào Chrome
#    → chrome://extensions → Bật "Developer mode"
#    → "Load unpacked" → chọn thư mục chrome-extension/

# 3. Mở Google Meet → Click icon extension
#    → Upload source face → Bấm "Bắt đầu Swap"
```
