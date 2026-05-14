"""
Face Swap Engine — InsightFace + inswapper_128.onnx (Optimized)

Tối ưu so với bản gốc:
  1. Bypass insightface paste_back=True (3× warpAffine + erosion + dilation + 2× blur)
     → Custom lightweight blend: 1× warpAffine + pre-computed soft mask + ROI-only blend
  2. Cache emap-projected latent (np.dot + norm tính 1 lần khi set source, không mỗi frame)
  3. ONNX session với graph optimization level ALL
  4. Pre-computed 128×128 soft blend mask (reuse mỗi frame)
"""

import os
import time
import threading
import collections
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class FaceSwapper:
    def __init__(self, model_path: str):
        providers = ["CPUExecutionProvider"]

        # ── Face Analyzer: SCRFD (detection) + ArcFace (recognition) ────────
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition"]
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.35)

        # ── Optimized ONNX session for inswapper ────────────────────────────
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 0   # auto-detect (M1: 8 cores)
        sess_opts.inter_op_num_threads = 1   # sequential node execution

        # Thử CoreMLExecutionProvider (M1 ANE/GPU, macOS local chỉ — Docker Linux tự bỏ qua)
        # Nếu CoreML không khả dụng, ORT tự fallback sang CPUExecutionProvider
        swap_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self._swap_session = onnxruntime.InferenceSession(
            model_path, sess_opts, providers=swap_providers
        )
        _active = self._swap_session.get_providers()
        if "CoreMLExecutionProvider" in _active:
            print("[INFO] inswapper: dùng CoreMLExecutionProvider (M1 ANE/GPU) ✓")
        else:
            print("[INFO] inswapper: CPUExecutionProvider (CoreML không khả dụng hoặc đang trong Docker)")
        self._input_names = [inp.name for inp in self._swap_session.get_inputs()]
        self._output_names = [out.name for out in self._swap_session.get_outputs()]

        # Extract emap matrix from ONNX graph (for source latent projection)
        model_proto = onnx.load(model_path)
        self._emap = numpy_helper.to_array(model_proto.graph.initializer[-1])
        del model_proto  # free ~500MB protobuf

        # ── Pre-computed soft blend mask 128×128 (reused every frame) ───────
        self._blend_mask_128 = self._make_blend_mask(128)

        # ── State ───────────────────────────────────────────────────────────
        self._source_face = None
        self._source_latent: np.ndarray | None = None  # cached emap projection
        self._lock = threading.Lock()

        # Temporal smoothing
        self._last_swapped: bytes | None = None
        self._no_face_streak: int = 0
        self.FALLBACK_FRAMES: int = 6

        # Frame-skip detection: chỉ chạy SCRFD mỗi N frame, các frame giữa dùng cache
        # → Tiết kiệm ~265ms × (N-1)/N mỗi frame trung bình trên CPU
        self.DETECT_INTERVAL: int = 5   # Chạy SCRFD full detection mỗi 5 frame
        self._detect_frame_idx: int = 0
        self._cached_faces: list = []   # Cache kết quả SCRFD gần nhất

        # --- Timing metrics (rolling window 30 frames) ---
        _w = 30
        self._t_decode   = collections.deque(maxlen=_w)
        self._t_detect   = collections.deque(maxlen=_w)
        self._t_swap     = collections.deque(maxlen=_w)
        self._t_encode   = collections.deque(maxlen=_w)
        self._t_total    = collections.deque(maxlen=_w)
        self._frame_count = 0

    @staticmethod
    def _make_blend_mask(size: int) -> np.ndarray:
        """
        Tạo soft feathered mask cho face blending.
        Mask = 1.0 ở center, fade ra 0.0 ở edges.
        Pre-compute 1 lần, reuse mỗi frame (thay vì tính erosion+dilation+blur mỗi lần).
        """
        mask = np.ones((size, size), dtype=np.float32)
        border = max(size // 20, 4)  # ~6px cho 128
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        k = size // 8  # ~16 cho 128
        k = k if k % 2 == 1 else k + 1  # kernel phải lẻ
        mask = cv2.GaussianBlur(mask, (k, k), k // 3)
        return mask

    def set_source_face(self, image_bytes: bytes) -> bool:
        """
        Nhận ảnh source (bytes), extract face embedding + pre-compute emap latent.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = self.face_analyzer.get(img_rgb)
        if not faces:
            return False

        # Chọn face lớn nhất
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        with self._lock:
            self._source_face = faces[0]

            # ── Pre-compute emap-projected latent (1 lần, không mỗi frame) ──
            # Tiết kiệm np.dot(512, emap_dim) + norm mỗi frame
            latent = faces[0].normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self._emap)
            latent /= np.linalg.norm(latent)
            self._source_latent = latent.astype(np.float32)

        return True

    def _fast_swap_face(self, img: np.ndarray, target_face) -> np.ndarray:
        """
        Optimized face swap — bypass insightface paste_back.

        Thay vì insightface's get(paste_back=True) gồm:
          - 3× warpAffine trên full image
          - erosion + dilation (morphological ops)
          - 2× GaussianBlur trên full image
          - float32 diff computation

        Ta chỉ dùng:
          - 1× warpAffine cho face
          - 1× warpAffine cho pre-computed mask
          - ROI-only alpha blend (chỉ blend vùng mặt, không cả ảnh)
        """
        # 1. Align & crop target face → 128×128
        aimg, M = face_align.norm_crop2(img, target_face.kps, 128)

        # 2. Prepare ONNX input blob (NCHW, float32, /255)
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True
        )

        # 3. ONNX inference (source_latent đã được pre-compute trong set_source_face)
        pred = self._swap_session.run(
            self._output_names,
            {self._input_names[0]: blob, self._input_names[1]: self._source_latent}
        )[0]

        # 4. Post-process: NCHW → HWC, denormalize, RGB→BGR
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        # 5. Warp swapped face + pre-computed mask back to full image coords
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        face_warped = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0)
        mask_warped = cv2.warpAffine(self._blend_mask_128, IM, (w, h), borderValue=0)

        # 6. ROI-only alpha blend — chỉ xử lý vùng chứa mặt, không cả ảnh
        rows = np.any(mask_warped > 0.01, axis=1)
        cols = np.any(mask_warped > 0.01, axis=0)
        if not rows.any():
            return img

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        mask_roi = mask_warped[y1:y2+1, x1:x2+1, np.newaxis]
        result = img.copy()
        result[y1:y2+1, x1:x2+1] = (
            mask_roi * face_warped[y1:y2+1, x1:x2+1] +
            (1 - mask_roi) * img[y1:y2+1, x1:x2+1]
        ).astype(np.uint8)

        return result

    def process_frame(self, frame_bytes: bytes) -> bytes | None:
        with self._lock:
            source_face = self._source_face
            source_latent = self._source_latent

        if source_face is None or source_latent is None:
            return None

        t0 = time.perf_counter()

        # ── Step 1: Decode + Downscale + Color convert ──────────────────────
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t1 = time.perf_counter()

        # ── Step 2: SCRFD Face Detection (với Frame-Skip) ────────────────────
        # Chỉ chạy detection đầy đủ mỗi DETECT_INTERVAL frame.
        # Các frame ở giữa reuse cached_faces từ lần detect trước:
        #   - Nếu mặt không di chuyển nhiều (< 1/10 frame interval ≈ 30ms), sai số rất nhỏ
        #   - Khi cache rỗng (chưa có face), force detect ngay
        self._detect_frame_idx += 1
        run_detection = (
            self._detect_frame_idx % self.DETECT_INTERVAL == 1
            or len(self._cached_faces) == 0
        )

        if run_detection:
            target_faces = self.face_analyzer.get(frame_rgb)
            self._cached_faces = target_faces
        else:
            target_faces = self._cached_faces

        t2 = time.perf_counter()

        if not target_faces:
            with self._lock:
                self._no_face_streak += 1
                if self._last_swapped is not None and self._no_face_streak <= self.FALLBACK_FRAMES:
                    return self._last_swapped
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return buf.tobytes()

        # ── Step 3: Optimized Face Swap ──────────────────────────────────────
        result = frame_rgb
        for target_face in target_faces:
            result = self._fast_swap_face(result, target_face)

        t3 = time.perf_counter()

        # ── Step 4: Color convert + JPEG Encode ──────────────────────────────
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result_bytes = buf.tobytes()

        t4 = time.perf_counter()

        # ── Record timings (ms) ───────────────────────────────────────────────
        with self._lock:
            self._t_decode.append((t1 - t0) * 1000)
            self._t_detect.append((t2 - t1) * 1000)
            self._t_swap.append((t3 - t2) * 1000)
            self._t_encode.append((t4 - t3) * 1000)
            self._t_total.append((t4 - t0) * 1000)
            self._frame_count += 1

            # In log mỗi 30 frame để không spam
            if self._frame_count % 30 == 0:
                def avg(q): return sum(q) / len(q) if q else 0
                print(
                    f"[PERF] decode={avg(self._t_decode):.1f}ms "
                    f"detect={avg(self._t_detect):.1f}ms "
                    f"swap={avg(self._t_swap):.1f}ms "
                    f"encode={avg(self._t_encode):.1f}ms "
                    f"total={avg(self._t_total):.1f}ms "
                    f"fps={1000/avg(self._t_total):.1f}"
                )

            self._last_swapped = result_bytes
            self._no_face_streak = 0

        return result_bytes

    def get_metrics(self) -> dict:
        """Trả về timing metrics hiện tại (ms, rolling average 30 frames)."""
        def avg(q): return round(sum(q) / len(q), 2) if q else 0
        def mx(q):  return round(max(q), 2) if q else 0

        with self._lock:
            total_avg = avg(self._t_total)
            return {
                "frames_processed": self._frame_count,
                "fps_avg": round(1000 / total_avg, 1) if total_avg > 0 else 0,
                "steps_ms": {
                    "decode_colorconv": {"avg": avg(self._t_decode), "max": mx(self._t_decode)},
                    "scrfd_detection":  {"avg": avg(self._t_detect), "max": mx(self._t_detect)},
                    "inswapper_blend":  {"avg": avg(self._t_swap),   "max": mx(self._t_swap)},
                    "encode_jpeg":      {"avg": avg(self._t_encode),  "max": mx(self._t_encode)},
                    "total":            {"avg": total_avg,            "max": mx(self._t_total)},
                },
                "bottleneck": max(
                    ["decode_colorconv", "scrfd_detection", "inswapper_blend", "encode_jpeg"],
                    key=lambda k: avg({
                        "decode_colorconv": self._t_decode,
                        "scrfd_detection":  self._t_detect,
                        "inswapper_blend":  self._t_swap,
                        "encode_jpeg":      self._t_encode,
                    }[k])
                ) if self._frame_count > 0 else "no_data",
            }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """
        Swap face trên từng frame của video MP4.

        Args:
            input_path:        Đường dẫn file MP4 đầu vào.
            output_path:       Đường dẫn file MP4 đầu ra (có audio gốc).
            progress_callback: Callable(frames_done, frames_total) — cập nhật tiến độ.
            cancel_event:      threading.Event — set() để dừng xử lý sớm.

        Returns:
            True nếu hoàn thành, False nếu bị cancel.

        Pipeline:
            1. cv2.VideoCapture đọc từng frame
            2. _fast_swap_face() — bỏ qua JPEG encode/decode (nhanh hơn process_frame)
            3. cv2.VideoWriter ghi ra file tạm (không có audio)
            4. ffmpeg mux audio gốc vào video đã swap → output_path
        """
        import subprocess
        import tempfile

        with self._lock:
            source_latent = self._source_latent

        if source_latent is None:
            raise RuntimeError("Chưa upload source face")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {input_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ghi ra file tạm (không có audio) — dùng mp4v codec (cross-platform)
        tmp_video = output_path + ".tmp.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

        cancelled = False
        try:
            done = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                faces = self.face_analyzer.get(frame_rgb)

                if faces:
                    result = frame_rgb
                    with self._lock:
                        lat = self._source_latent
                    for face in faces:
                        # Gọi trực tiếp _fast_swap_face để tránh JPEG encode/decode overhead
                        # Tạm thời set _source_latent nếu khác (không cần vì đã read ở trên)
                        result = self._fast_swap_face_with_latent(result, face, lat)
                    frame_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                writer.write(frame_bgr)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        finally:
            cap.release()
            writer.release()

        # Nếu bị cancel → dọn dẹp file tạm và trả về False
        if cancelled:
            import os as _os
            if _os.path.exists(tmp_video):
                _os.remove(tmp_video)
            return False

        # Mux audio từ video gốc vào video đã swap bằng ffmpeg
        # -y: overwrite output, -loglevel error: suppress verbose output
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", tmp_video,        # video đã swap (không có audio)
            "-i", input_path,       # video gốc (lấy audio track)
            "-c:v", "copy",         # giữ nguyên video stream
            "-c:a", "aac",          # encode audio sang AAC
            "-map", "0:v:0",        # video từ stream 0
            "-map", "1:a:0?",       # audio từ stream 1 (? = optional, nếu video gốc không có audio thì bỏ qua)
            "-shortest",            # cắt theo stream ngắn hơn
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            import os
            if os.path.exists(tmp_video):
                os.remove(tmp_video)

        return True

    def _fast_swap_face_with_latent(
        self, img: np.ndarray, target_face, latent: np.ndarray
    ) -> np.ndarray:
        """
        Giống _fast_swap_face() nhưng nhận latent trực tiếp thay vì đọc từ self._source_latent.
        Dùng trong process_video() để tránh acquire lock mỗi frame.
        """
        from insightface.utils import face_align as _face_align
        aimg, M = _face_align.norm_crop2(img, target_face.kps, 128)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)

        pred = self._swap_session.run(
            self._output_names,
            {self._input_names[0]: blob, self._input_names[1]: latent}
        )[0]

        img_fake  = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake  = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        face_warped = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0)
        mask_warped = cv2.warpAffine(self._blend_mask_128, IM, (w, h), borderValue=0)

        rows = np.any(mask_warped > 0.01, axis=1)
        cols = np.any(mask_warped > 0.01, axis=0)
        if not rows.any():
            return img

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        mask_roi = mask_warped[y1:y2+1, x1:x2+1, np.newaxis]
        result = img.copy()
        result[y1:y2+1, x1:x2+1] = (
            mask_roi * face_warped[y1:y2+1, x1:x2+1] +
            (1 - mask_roi) * img[y1:y2+1, x1:x2+1]
        ).astype(np.uint8)

        return result

    @property
    def has_source_face(self) -> bool:
        with self._lock:
            return self._source_face is not None
