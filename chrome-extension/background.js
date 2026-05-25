/**
 * background.js — Service Worker
 *
 * Vai trò: relay message giữa popup ↔ content script khi cần,
 * và quản lý lifecycle extension.
 */

// Khi extension được install/update
chrome.runtime.onInstalled.addListener(() => {
  console.log("[FaceSwap BG] Extension installed/updated");
  // Set default server URL
  chrome.storage.local.get("serverUrl", (data) => {
    if (!data.serverUrl) {
      chrome.storage.local.set({ serverUrl: "http://127.0.0.1:3636" });
    }
  });
});

// Relay messages nếu cần (popup ↔ content script)
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Forward SWAP_ERROR from content script tới popup (nếu popup đang mở)
  if (msg.type === "SWAP_ERROR") {
    // Popup sẽ tự nhận qua chrome.runtime.onMessage
    return;
  }
});
