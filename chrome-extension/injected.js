/**
 * injected.js — chạy trong MAIN WORLD của trang Google Meet
 *
 * Được inject bởi content.js qua <script src>.
 * Có quyền truy cập trực tiếp vào các API của trang:
 *   - navigator.mediaDevices.getUserMedia (override thật sự)
 *   - RTCPeerConnection (không cần trong proxy approach)
 *
 * Giao tiếp với content.js (Isolated World) qua window.postMessage.
 */
(function () {
  "use strict";

  if (window.__faceswap_injected__) return; // tránh inject nhiều lần
  window.__faceswap_injected__ = true;

  // ── State ──
  let serverUrl   = null;
  let ws          = null;
  let swapActive  = false;
  let waitingForResponse = false;

  let realVideo     = null;
  let outputCanvas  = null;
  let outputCtx     = null;
  let outputStream  = null;
  let captureCanvas = null;
  let captureCtx    = null;
  let lastSwappedImg = null;
  let renderRAF     = null;
  let proxyReady    = false;
  let proxySetupPromise = null;  // guard chống concurrent setupProxy

  let CANVAS_W    = 1280;  // sẽ cập nhật theo resolution thật của camera
  let CANVAS_H    = 720;
  const JPEG_QUALITY = 0.88;

  // ══════════════════════════════════════════════════════════════════════════
  //  Override getUserMedia — chạy trong MAIN WORLD nên thật sự override được
  // ══════════════════════════════════════════════════════════════════════════
  const _orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  navigator.mediaDevices.getUserMedia = async function (constraints) {
    // KHÔNG override constraints — để camera chạy resolution tự nhiên
    const stream = await _orig(constraints || {});
    if (!constraints?.video) return stream;

    console.log("[FaceSwap injected] getUserMedia intercepted ✓");
    try {
      await setupProxy(stream);
      return buildMixedStream(stream);
    } catch (e) {
      // Fallback: proxy thất bại → trả stream thật để Meet vẫn có camera (không face swap)
      console.warn("[FaceSwap injected] Proxy failed, fallback raw stream:", e);
      return stream;
    }
  };
  // Cũng override prototype → bắt mọi call pattern (Workspace Meet có thể dùng prototype trực tiếp)
  MediaDevices.prototype.getUserMedia = navigator.mediaDevices.getUserMedia;

  // ══════════════════════════════════════════════════════════════════════════
  //  Setup proxy pipeline (1 lần duy nhất)
  // ══════════════════════════════════════════════════════════════════════════
  async function setupProxy(realStream) {
    if (proxyReady) {
      if (realVideo) {
        realVideo.srcObject = realStream;
        realVideo.play().catch(() => {});
      }
      return;
    }
    // Concurrency guard: Meet gọi getUserMedia nhiều lần → chỉ chạy setup 1 lần
    // Không có guard → 2 setupProxy chạy song song → conflict realVideo/outputCanvas
    if (proxySetupPromise) { await proxySetupPromise; return; }
    let _done;
    proxySetupPromise = new Promise(r => _done = r);

    try {
      // Hidden video đọc webcam thật
      realVideo = document.createElement("video");
      realVideo.setAttribute("autoplay", "");
      realVideo.setAttribute("playsinline", "");
      realVideo.setAttribute("muted", "");
      realVideo.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;";
      document.documentElement.appendChild(realVideo);
      realVideo.srcObject = realStream;
      realVideo.play().catch((e) => console.warn("[FaceSwap injected] video.play:", e));

      // Output canvas → Meet nhận stream này (default 1280×720)
      outputCanvas = document.createElement("canvas");
      outputCanvas.width  = CANVAS_W;
      outputCanvas.height = CANVAS_H;
      outputCanvas.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;";
      document.documentElement.appendChild(outputCanvas);
      outputCtx = outputCanvas.getContext("2d");

      // Capture canvas (gửi WS)
      captureCanvas = document.createElement("canvas");
      captureCanvas.width  = CANVAS_W;
      captureCanvas.height = CANVAS_H;
      captureCtx = captureCanvas.getContext("2d");

      // Tạo stream NGAY — KHÔNG chờ canplay
      // Nếu await canplay (1500ms) → getUserMedia block quá lâu → Workspace Meet timeout → camera fail
      outputStream = outputCanvas.captureStream(30);

      // Spoof canvas track → trông như camera thật
      // Workspace Meet kiểm tra track.label + getSettings() + getCapabilities() để validate
      const realTrack   = realStream.getVideoTracks()[0];
      const canvasTrack = outputStream.getVideoTracks()[0];
      if (realTrack && canvasTrack) {
        const rs = realTrack.getSettings();
        canvasTrack.getSettings    = () => ({ ...rs, width: CANVAS_W, height: CANVAS_H });
        canvasTrack.getCapabilities = realTrack.getCapabilities
          ? () => realTrack.getCapabilities()
          : canvasTrack.getCapabilities;
        try {
          Object.defineProperty(canvasTrack, "label",
            { get: () => realTrack.label, configurable: true });
        } catch (_) {}
      }

      proxyReady = true;
      startRenderLoop();
      console.log("[FaceSwap injected] Proxy pipeline ready", CANVAS_W + "x" + CANVAS_H);
      postToContent("PROXY_READY", {});

      // Async (không block): điều chỉnh kích thước canvas khi video thật có data
      // Render loop sẽ tự vẽ frame khi readyState >= 2
      new Promise((resolve) => {
        if (realVideo.readyState >= 2) { resolve(); return; }
        realVideo.addEventListener("canplay", resolve, { once: true });
        setTimeout(resolve, 3000);
      }).then(() => {
        const vw = realVideo.videoWidth;
        const vh = realVideo.videoHeight;
        if (vw && vh && (vw !== CANVAS_W || vh !== CANVAS_H)) {
          CANVAS_W = vw; CANVAS_H = vh;
          outputCanvas.width  = vw; outputCanvas.height  = vh;
          captureCanvas.width = vw; captureCanvas.height = vh;
          if (realTrack && canvasTrack) {
            const rs2 = realTrack.getSettings();
            canvasTrack.getSettings = () => ({ ...rs2, width: CANVAS_W, height: CANVAS_H });
          }
        }
        if (realVideo.readyState >= 2) {
          outputCtx.drawImage(realVideo, 0, 0, CANVAS_W, CANVAS_H);
        }
      });

    } finally {
      _done();
    }
  }

  function buildMixedStream(realStream) {
    const mixed = new MediaStream();
    realStream.getAudioTracks().forEach((t) => mixed.addTrack(t));
    outputStream.getVideoTracks().forEach((t) => mixed.addTrack(t));
    return mixed;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Render Loop
  // ══════════════════════════════════════════════════════════════════════════
  function startRenderLoop() {
    if (renderRAF) return;

    function render() {
      renderRAF = requestAnimationFrame(render);
      if (!realVideo || realVideo.readyState < 2) return;

      if (swapActive && lastSwappedImg) {
        // Vẽ swapped frame — KHÔNG mirror vì Meet đã tự CSS-mirror self-preview
        outputCtx.drawImage(lastSwappedImg, 0, 0, CANVAS_W, CANVAS_H);
      } else {
        // Passthrough — KHÔNG mirror, để Meet tự xử lý mirror cho self-view
        outputCtx.drawImage(realVideo, 0, 0, CANVAS_W, CANVAS_H);
      }
    }
    render();
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  WebSocket
  // ══════════════════════════════════════════════════════════════════════════
  function connectWS() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;
    const wsUrl = serverUrl.replace(/^http/, "ws") + "/ws";
    console.log("[FaceSwap injected] Connecting WS:", wsUrl);
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[FaceSwap injected] WS connected ✓");
      waitingForResponse = false;
      sendNextFrame();
    };

    ws.onmessage = (event) => {
      waitingForResponse = false;
      if (!event?.data) { if (swapActive) sendNextFrame(); return; }

      if (typeof event.data === "string") {
        if (swapActive) sendNextFrame();
        return;
      }

      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url  = URL.createObjectURL(blob);
      const img  = new Image();
      img.onload = () => {
        if (lastSwappedImg?._url) URL.revokeObjectURL(lastSwappedImg._url);
        img._url = url;
        lastSwappedImg = img;
        if (swapActive) sendNextFrame();
      };
      img.onerror = () => { URL.revokeObjectURL(url); if (swapActive) sendNextFrame(); };
      img.src = url;
    };

    ws.onerror = () => postToContent("SWAP_ERROR", { error: "WebSocket lỗi kết nối" });
    ws.onclose = () => {
      waitingForResponse = false;
      if (swapActive) setTimeout(connectWS, 2000);
    };
  }

  function sendNextFrame() {
    if (!swapActive || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (waitingForResponse) return;
    if (!realVideo || realVideo.readyState < 2) { setTimeout(sendNextFrame, 200); return; }

    // Capture raw (không mirror) — backend nhận natural frame, trả natural frame
    // Meet sẽ tự mirror self-preview qua CSS
    captureCtx.drawImage(realVideo, 0, 0, CANVAS_W, CANVAS_H);

    captureCanvas.toBlob((blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
      waitingForResponse = true;
      blob.arrayBuffer().then((buf) => ws.send(buf));
    }, "image/jpeg", JPEG_QUALITY);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Start / Stop
  // ══════════════════════════════════════════════════════════════════════════
  async function startSwap(url) {
    serverUrl = url;
    swapActive = true;
    lastSwappedImg = null;

    // Nếu proxy chưa sẵn sàng (Meet chưa gọi getUserMedia qua override)
    // → tự lấy webcam và setup proxy
    if (!proxyReady) {
      console.log("[FaceSwap injected] Proxy not ready, acquiring webcam directly...");
      try {
        const stream = await _orig({ video: true, audio: false });
        await setupProxy(stream);
      } catch (e) {
        console.error("[FaceSwap injected] getUserMedia failed:", e);
        postToContent("SWAP_ERROR", { error: "Không lấy được webcam: " + e.message });
        swapActive = false;
        return;
      }
    }

    connectWS();
    console.log("[FaceSwap injected] Swap STARTED");
  }

  function stopSwap() {
    swapActive = false;
    lastSwappedImg = null;
    if (ws) { ws.close(); ws = null; }
    console.log("[FaceSwap injected] Swap STOPPED");
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Message bridge: content.js → injected.js (via CustomEvent)
  // ══════════════════════════════════════════════════════════════════════════
  window.addEventListener("__faceswap_to_page__", (e) => {
    const { action, payload } = e.detail || {};
    if (action === "START_SWAP") startSwap(payload.serverUrl);
    else if (action === "STOP_SWAP") stopSwap();
  });

  function postToContent(type, payload) {
    window.dispatchEvent(new CustomEvent("__faceswap_to_content__", {
      detail: { type, payload }
    }));
  }

  console.log("[FaceSwap injected] Main world script loaded ✓");
})();
