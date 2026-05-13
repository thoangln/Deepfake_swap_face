/**
 * popup.js — Chrome Extension Popup Controller
 *
 * Quản lý:
 *  - Kết nối tới backend server (kiểm tra /status)
 *  - Upload source face (/upload-source)
 *  - Gửi lệnh start/stop swap tới content script qua chrome.tabs.sendMessage
 *  - Hiển thị metrics polling từ /metrics
 */

const $ = (sel) => document.querySelector(sel);

// ── Elements ──
const statusDot    = $("#statusDot");
const statusText   = $("#statusText");
const serverUrl    = $("#serverUrl");
const btnConnect   = $("#btnConnect");
const uploadZone   = $("#uploadZone");
const fileInput    = $("#fileInput");
const facePreview  = $("#facePreview");
const btnStart     = $("#btnStart");
const btnStop      = $("#btnStop");
const metricFps    = $("#metricFps");
const metricLatency = $("#metricLatency");
const metricDetect = $("#metricDetect");
const metricSwap   = $("#metricSwap");

let connected = false;
let swapping  = false;
let metricsInterval = null;

// ── Init: load saved state ──
chrome.storage.local.get(["serverUrl", "swapping"], (data) => {
  if (data.serverUrl) serverUrl.value = data.serverUrl;
  if (data.swapping)  swapping = data.swapping;
  checkServer();
});

// ── Server Connection ──
btnConnect.addEventListener("click", () => {
  chrome.storage.local.set({ serverUrl: serverUrl.value });
  checkServer();
});

async function checkServer() {
  const base = serverUrl.value.replace(/\/$/, "");
  try {
    const res = await fetch(`${base}/status`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    connected = true;
    statusDot.classList.add("connected");
    statusText.textContent = data.source_face_loaded
      ? "✅ Server sẵn sàng (có source face)"
      : "🟡 Server sẵn sàng (chưa có source face)";

    if (data.source_face_loaded) {
      uploadZone.classList.add("has-face");
      btnStart.disabled = false;
    }

    // Nếu đang swap → cập nhật UI
    if (swapping) {
      btnStart.disabled = true;
      btnStop.disabled = false;
      statusDot.classList.add("swapping");
      startMetricsPolling();
    }
  } catch {
    connected = false;
    statusDot.classList.remove("connected", "swapping");
    statusText.textContent = "❌ Không kết nối được server";
    btnStart.disabled = true;
    btnStop.disabled = true;
  }
}

// ── Source Face Upload ──
uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Preview
  const reader = new FileReader();
  reader.onload = (ev) => {
    facePreview.src = ev.target.result;
    facePreview.classList.add("show");
  };
  reader.readAsDataURL(file);

  // Upload to backend
  const base = serverUrl.value.replace(/\/$/, "");
  const form = new FormData();
  form.append("file", file);

  try {
    statusText.textContent = "⏳ Đang upload source face...";
    const res = await fetch(`${base}/upload-source`, { method: "POST", body: form });
    if (res.ok) {
      uploadZone.classList.add("has-face");
      statusText.textContent = "✅ Source face loaded!";
      btnStart.disabled = false;
    } else {
      const err = await res.json();
      statusText.textContent = `❌ ${err.detail || "Upload thất bại"}`;
    }
  } catch {
    statusText.textContent = "❌ Không gửi được tới server";
  }
});

// ── Start/Stop Swap ──
btnStart.addEventListener("click", async () => {
  const base = serverUrl.value.replace(/\/$/, "");
  // Lấy tab Google Meet đang active
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url?.includes("meet.google.com")) {
    statusText.textContent = "⚠ Hãy mở Google Meet trước!";
    return;
  }

  // Gửi lệnh START tới content script
  chrome.tabs.sendMessage(tab.id, {
    action: "START_SWAP",
    serverUrl: base,
  });

  swapping = true;
  chrome.storage.local.set({ swapping: true });
  btnStart.disabled = true;
  btnStop.disabled = false;
  statusDot.classList.add("swapping");
  statusText.textContent = "🔄 Đang swap face...";
  startMetricsPolling();
});

btnStop.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) {
    chrome.tabs.sendMessage(tab.id, { action: "STOP_SWAP" });
  }

  swapping = false;
  chrome.storage.local.set({ swapping: false });
  btnStart.disabled = false;
  btnStop.disabled = true;
  statusDot.classList.remove("swapping");
  statusText.textContent = "⏹ Đã dừng swap";
  stopMetricsPolling();
});

// ── Metrics Polling ──
function startMetricsPolling() {
  stopMetricsPolling();
  pollMetrics();
  metricsInterval = setInterval(pollMetrics, 2000);
}

function stopMetricsPolling() {
  if (metricsInterval) clearInterval(metricsInterval);
  metricsInterval = null;
}

async function pollMetrics() {
  const base = serverUrl.value.replace(/\/$/, "");
  try {
    const res = await fetch(`${base}/metrics`, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    const steps = data.steps_ms || {};
    metricFps.textContent     = `${data.fps_avg || 0}`;
    metricLatency.textContent = `${steps.total?.avg || 0}ms`;
    metricDetect.textContent  = `${steps.scrfd_detection?.avg || 0}ms`;
    metricSwap.textContent    = `${steps.inswapper_blend?.avg || 0}ms`;
  } catch { /* ignore */ }
}

// ── Listen for messages from content script ──
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SWAP_ERROR") {
    statusText.textContent = `❌ ${msg.error}`;
    swapping = false;
    chrome.storage.local.set({ swapping: false });
    btnStart.disabled = false;
    btnStop.disabled = true;
    statusDot.classList.remove("swapping");
    stopMetricsPolling();
  }
});
