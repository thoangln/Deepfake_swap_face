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

  const CANVAS_W    = 640;
  const CANVAS_H    = 480;
  const JPEG_QUALITY = 0.88; // tăng chất lượng JPEG

  // ══════════════════════════════════════════════════════════════════════════
  //  Override getUserMedia — chạy trong MAIN WORLD nên thật sự override được
  // ══════════════════════════════════════════════════════════════════════════
  const _orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  navigator.mediaDevices.getUserMedia = async function (constraints) {
    const stream = await _orig(constraints);
    if (!constraints?.video) return stream;

    console.log("[FaceSwap injected] getUserMedia intercepted ✓");
    await setupProxy(stream);
    return buildMixedStream(stream);
  };

  // ══════════════════════════════════════════════════════════════════════════
  //  Setup proxy pipeline (1 lần duy nhất)
  // ══════════════════════════════════════════════════════════════════════════
  async function setupProxy(realStream) {
    if (proxyReady) {
      // Update real stream source nếu stream đổi
      if (realVideo) {
        realVideo.srcObject = realStream;
        await realVideo.play().catch(() => {});
      }
      return;
    }

    // Hidden video đọc webcam thật
    realVideo = document.createElement("video");
    realVideo.setAttribute("autoplay", "");
    realVideo.setAttribute("playsinline", "");
    realVideo.setAttribute("muted", "");
    realVideo.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;";
    document.documentElement.appendChild(realVideo);
    realVideo.srcObject = realStream;
    await realVideo.play().catch((e) => console.warn("[FaceSwap injected] video.play:", e));

    // Output canvas → Meet nhận stream này
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

    outputStream = outputCanvas.captureStream(30);
    proxyReady = true;

    startRenderLoop();
    console.log("[FaceSwap injected] Proxy pipeline ready", CANVAS_W + "x" + CANVAS_H);
    postToContent("PROXY_READY", {});
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
