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

        self._swap_session = onnxruntime.InferenceSession(
            model_path, sess_opts, providers=providers
        )
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
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t1 = time.perf_counter()

        # ── Step 2: SCRFD Face Detection ─────────────────────────────────────
        target_faces = self.face_analyzer.get(frame_rgb)

        t2 = time.perf_counter()

        if not target_faces:
            with self._lock:
                self._no_face_streak += 1
                if self._last_swapped is not None and self._no_face_streak <= self.FALLBACK_FRAMES:
                    return self._last_swapped
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()

        # ── Step 3: Optimized Face Swap ──────────────────────────────────────
        result = frame_rgb
        for target_face in target_faces:
            result = self._fast_swap_face(result, target_face)

        t3 = time.perf_counter()

        # ── Step 4: Color convert + JPEG Encode ──────────────────────────────
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
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

    @property
    def has_source_face(self) -> bool:
        with self._lock:
            return self._source_face is not None
