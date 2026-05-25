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
  // const script = document.createElement("script");
  // script.src = chrome.runtime.getURL("injected.js");
  // script.onload = () => {
  //   console.log("[FaceSwap bridge] injected.js loaded into main world ✓");
  //   script.remove();
  // };
  // script.onerror = (e) => console.error("[FaceSwap bridge] Failed to inject:", e);
  // (document.head || document.documentElement).appendChild(script);

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

