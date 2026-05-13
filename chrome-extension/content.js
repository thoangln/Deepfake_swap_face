/**
 * content.js — Isolated World bridge
 *
 * Chạy trong Chrome Extension Isolated World — có quyền truy cập chrome.* API
 * nhưng KHÔNG thể override các API của trang (window, navigator...).
 *
 * Vai trò:
 *  1. Inject injected.js vào MAIN WORLD (trang thật) để override getUserMedia
 *  2. Relay messages: popup → chrome.runtime → content.js → window.postMessage → injected.js
 *  3. Relay ngược: injected.js → window.postMessage → content.js → chrome.runtime → popup
 */
(function () {
  "use strict";

  // ── 1. Inject script vào MAIN WORLD ──
  const script = document.createElement("script");
  script.src = chrome.runtime.getURL("injected.js");
  script.onload = () => {
    console.log("[FaceSwap bridge] injected.js loaded into main world ✓");
    script.remove();
  };
  script.onerror = (e) => console.error("[FaceSwap bridge] Failed to inject:", e);
  (document.head || document.documentElement).appendChild(script);

  // ── 2. Relay: injected.js → chrome.runtime (popup) ──
  window.addEventListener("__faceswap_to_content__", (e) => {
    const { type, payload } = e.detail || {};
    try {
      chrome.runtime.sendMessage({ type, ...payload });
    } catch (err) {
      // Extension context may be invalidated (e.g. after reload), ignore
    }
  });

  // ── 3. Relay: chrome.runtime (popup) → injected.js ──
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "START_SWAP" || msg.action === "STOP_SWAP") {
      window.dispatchEvent(new CustomEvent("__faceswap_to_page__", {
        detail: { action: msg.action, payload: { serverUrl: msg.serverUrl } }
      }));
      sendResponse({ ok: true });
    }
  });

  console.log("[FaceSwap bridge] Content script (isolated world) ready ✓");
})();


(function () {
  "use strict";

  // ── State ──
  let serverUrl   = null;
  let ws          = null;
  let swapActive  = false;
  let waitingForResponse = false;

  // Proxy stream elements (tạo 1 lần khi getUserMedia được gọi đầu tiên)
  let realVideo     = null;   // hidden <video> chứa webcam thật
  let outputCanvas  = null;   // canvas mà Meet nhận stream từ đây
  let outputCtx     = null;
  let outputStream  = null;   // canvas.captureStream() — trả cho Meet
  let captureCanvas = null;   // canvas tạm để capture + gửi WS
  let captureCtx    = null;
  let renderRAF     = null;   // requestAnimationFrame id
  let lastSwappedImg = null;  // Image object chứa frame swap mới nhất

  const CANVAS_W = 640;
  const CANVAS_H = 480;
  const JPEG_QUALITY = 0.55;

  // ══════════════════════════════════════════════════════════════════════════
  //  getUserMedia Override — LUÔN trả proxy canvas stream
  // ══════════════════════════════════════════════════════════════════════════
  const _origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  navigator.mediaDevices.getUserMedia = async function (constraints) {
    const stream = await _origGetUserMedia(constraints);

    // Chỉ intercept khi có video track
    if (!constraints?.video) return stream;

    console.log("[FaceSwap] getUserMedia intercepted:", constraints.video);

    // ── Tạo proxy pipeline (1 lần duy nhất) ──
    if (!outputCanvas) {
      // Hidden video element đọc webcam thật
      realVideo = document.createElement("video");
      realVideo.setAttribute("autoplay", "");
      realVideo.setAttribute("playsinline", "");
      realVideo.setAttribute("muted", "");
      realVideo.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;";
      document.documentElement.appendChild(realVideo);

      // Output canvas — đây là nguồn stream cho Meet
      outputCanvas = document.createElement("canvas");
      outputCanvas.width  = CANVAS_W;
      outputCanvas.height = CANVAS_H;
      outputCanvas.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;";
      document.documentElement.appendChild(outputCanvas);
      outputCtx = outputCanvas.getContext("2d");

      // Capture canvas (để gửi WS, reuse)
      captureCanvas = document.createElement("canvas");
      captureCanvas.width  = CANVAS_W;
      captureCanvas.height = CANVAS_H;
      captureCtx = captureCanvas.getContext("2d");

      // Canvas stream 30fps
      outputStream = outputCanvas.captureStream(30);

      console.log("[FaceSwap] Proxy pipeline created (canvas", CANVAS_W + "x" + CANVAS_H + ")");
    }

    // Gán webcam stream thật cho hidden video
    realVideo.srcObject = stream;
    await realVideo.play().catch((e) => console.warn("[FaceSwap] video play:", e));

    // Bắt đầu render loop (passthrough khi chưa swap)
    startRenderLoop();

    // ── Build mixed stream: audio gốc + video từ canvas ──
    const mixed = new MediaStream();
    // Giữ audio tracks nguyên vẹn
    stream.getAudioTracks().forEach((t) => mixed.addTrack(t));
    // Video track từ canvas proxy
    outputStream.getVideoTracks().forEach((t) => mixed.addTrack(t));

    console.log("[FaceSwap] Returning proxy stream to Meet (audio:", mixed.getAudioTracks().length, "video:", mixed.getVideoTracks().length + ")");
    return mixed;
  };

  // ══════════════════════════════════════════════════════════════════════════
  //  Render Loop — liên tục vẽ lên outputCanvas
  //  - Passthrough mode: vẽ webcam thật
  //  - Swap mode: vẽ frame đã swap (từ lastSwappedImg)
  // ══════════════════════════════════════════════════════════════════════════
  function startRenderLoop() {
    if (renderRAF) return; // đã chạy rồi

    function render() {
      renderRAF = requestAnimationFrame(render);

      if (!realVideo || realVideo.readyState < 2) return; // HAVE_CURRENT_DATA

      if (swapActive && lastSwappedImg) {
        // Swap mode: vẽ ảnh đã swap
        outputCtx.drawImage(lastSwappedImg, 0, 0, CANVAS_W, CANVAS_H);
      } else {
        // Passthrough: vẽ webcam thật (mirror)
        outputCtx.save();
        outputCtx.translate(CANVAS_W, 0);
        outputCtx.scale(-1, 1);
        outputCtx.drawImage(realVideo, 0, 0, CANVAS_W, CANVAS_H);
        outputCtx.restore();
      }
    }

    render();
    console.log("[FaceSwap] Render loop started");
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  WebSocket — gửi frame tới backend, nhận swapped frame
  // ══════════════════════════════════════════════════════════════════════════
  function connectWebSocket() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;

    const wsUrl = serverUrl.replace(/^http/, "ws") + "/ws";
    console.log("[FaceSwap] Connecting WS:", wsUrl);
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[FaceSwap] WebSocket connected ✓");
      waitingForResponse = false;
      sendNextFrame();
    };

    ws.onmessage = (event) => {
      waitingForResponse = false;

      if (!event || !event.data) {
        if (swapActive) sendNextFrame();
        return;
      }

      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);
          console.warn("[FaceSwap] Server:", msg.message || msg);
        } catch { /* ignore */ }
        if (swapActive) sendNextFrame();
        return;
      }

      // Binary: swapped JPEG → tạo Image để render loop vẽ
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url  = URL.createObjectURL(blob);
      const img  = new Image();
      img.onload = () => {
        // Cập nhật lastSwappedImg → render loop sẽ vẽ frame này
        if (lastSwappedImg && lastSwappedImg._blobUrl) {
          URL.revokeObjectURL(lastSwappedImg._blobUrl);
        }
        img._blobUrl = url;
        lastSwappedImg = img;
        // Backpressure: gửi frame tiếp sau khi nhận response
        if (swapActive) sendNextFrame();
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        if (swapActive) sendNextFrame();
      };
      img.src = url;
    };

    ws.onerror = (err) => {
      console.error("[FaceSwap] WS error:", err);
      chrome.runtime.sendMessage({ type: "SWAP_ERROR", error: "WebSocket lỗi" });
    };

    ws.onclose = () => {
      console.log("[FaceSwap] WebSocket closed");
      waitingForResponse = false;
      if (swapActive) setTimeout(connectWebSocket, 2000);
    };
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Capture & Send — lấy frame từ webcam, gửi qua WS
  // ══════════════════════════════════════════════════════════════════════════
  function sendNextFrame() {
    if (!swapActive || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (waitingForResponse) return;
    if (!realVideo || realVideo.readyState < 2) {
      setTimeout(sendNextFrame, 200);
      return;
    }

    // Vẽ webcam lên capture canvas (mirror)
    captureCtx.save();
    captureCtx.translate(CANVAS_W, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(realVideo, 0, 0, CANVAS_W, CANVAS_H);
    captureCtx.restore();

    captureCanvas.toBlob(
      (blob) => {
        if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
        waitingForResponse = true;
        blob.arrayBuffer().then((buf) => ws.send(buf));
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Start / Stop
  // ══════════════════════════════════════════════════════════════════════════
  function startSwap(url) {
    serverUrl = url;
    swapActive = true;
    lastSwappedImg = null;
    connectWebSocket();
    console.log("[FaceSwap] Swap STARTED");
  }

  function stopSwap() {
    swapActive = false;
    lastSwappedImg = null;
    if (ws) { ws.close(); ws = null; }
    console.log("[FaceSwap] Swap STOPPED — passthrough mode");
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Message listener từ popup
  // ══════════════════════════════════════════════════════════════════════════
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "START_SWAP") {
      startSwap(msg.serverUrl);
      sendResponse({ ok: true });
    } else if (msg.action === "STOP_SWAP") {
      stopSwap();
      sendResponse({ ok: true });
    }
  });

  console.log("[FaceSwap] Content script loaded — proxy mode ✓");
})();
