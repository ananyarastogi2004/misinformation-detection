// background.js — service worker
// Handles right-click context menu "Fact Check with ClaimSense"

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "claimsense-check",
    title: "Fact Check with ClaimSense",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "claimsense-check" && info.selectionText) {
    // Store the selected text so popup can read it on open
    chrome.storage.local.set(
      { pendingClaim: info.selectionText.trim(), source: tab.url },
      () => {
        // Open the popup via a new tab pointing to popup.html (fallback)
        // chrome.action.openPopup() requires user gesture in MV3;
        // right-click IS a user gesture so this works.
        chrome.action.openPopup().catch(() => {
          // Fallback for browsers that don't support openPopup
          chrome.tabs.create({ url: chrome.runtime.getURL("popup.html") });
        });
      }
    );
  }
});
