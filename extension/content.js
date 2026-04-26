// content.js — runs on every page
// Detects text selection and shows a floating "Fact Check" button.

(function () {
  "use strict";

  const MIN_LENGTH = 15;   // minimum chars to trigger the button
  const MAX_LENGTH = 500;  // longer selections are probably not single claims

  let floatBtn = null;
  let hideTimer = null;

  // ── Create the floating button once ──────────────────────────────────
  function createButton() {
    const btn = document.createElement("div");
    btn.id = "__claimsense_btn__";
    btn.innerHTML = `
      <span style="font-size:13px;margin-right:4px;">🔍</span>
      <span>Fact Check</span>
    `;
    Object.assign(btn.style, {
      position:       "fixed",
      display:        "none",
      alignItems:     "center",
      gap:            "4px",
      padding:        "6px 12px",
      background:     "#1a1a2e",
      color:          "#ffffff",
      fontSize:       "13px",
      fontFamily:     "system-ui, sans-serif",
      fontWeight:     "500",
      borderRadius:   "20px",
      boxShadow:      "0 4px 16px rgba(0,0,0,0.25)",
      cursor:         "pointer",
      zIndex:         "2147483647",
      border:         "1px solid rgba(255,255,255,0.15)",
      transition:     "opacity 0.15s ease",
      userSelect:     "none",
      whiteSpace:     "nowrap",
    });

    btn.addEventListener("mouseenter", () => {
      btn.style.background = "#16213e";
    });
    btn.addEventListener("mouseleave", () => {
      btn.style.background = "#1a1a2e";
    });
    btn.addEventListener("mousedown", (e) => {
      e.preventDefault();  // prevent losing the selection
    });
    btn.addEventListener("click", handleFactCheck);

    document.body.appendChild(btn);
    return btn;
  }

  // ── Position the button near the selection ────────────────────────────
  function showButton(x, y, selectedText) {
    if (!floatBtn) floatBtn = createButton();

    // Store the claim text on the button
    floatBtn._selectedText = selectedText;

    // Position: slightly above the cursor, clamped to viewport
    const OFFSET = 12;
    const bw = 130;   // approximate button width
    const left = Math.min(Math.max(x - bw / 2, 8), window.innerWidth - bw - 8);
    const top  = Math.max(y - 44 - OFFSET, 8);

    floatBtn.style.left    = left + "px";
    floatBtn.style.top     = top  + "px";
    floatBtn.style.display = "flex";
    floatBtn.style.opacity = "1";

    // Auto-hide after 4 s of inactivity
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideButton, 4000);
  }

  function hideButton() {
    if (floatBtn) floatBtn.style.display = "none";
  }

  // ── Handle the fact-check click ───────────────────────────────────────
  function handleFactCheck() {
    const text = floatBtn._selectedText || "";
    if (!text) return;

    hideButton();

    // Save claim to storage; popup reads it on open
    chrome.storage.local.set(
      { pendingClaim: text, source: window.location.href },
      () => {
        // Visual feedback on the page
        showToast("Opening ClaimSense…");
      }
    );
  }

  // ── Lightweight toast notification ───────────────────────────────────
  function showToast(msg) {
    const toast = document.createElement("div");
    toast.textContent = msg;
    Object.assign(toast.style, {
      position:     "fixed",
      bottom:       "24px",
      right:        "24px",
      background:   "#1a1a2e",
      color:        "#fff",
      padding:      "10px 18px",
      borderRadius: "10px",
      fontSize:     "13px",
      fontFamily:   "system-ui, sans-serif",
      zIndex:       "2147483647",
      boxShadow:    "0 4px 12px rgba(0,0,0,0.3)",
      opacity:      "0",
      transition:   "opacity 0.2s ease",
    });
    document.body.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = "1"; });
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 2500);
  }

  // ── Selection listener ────────────────────────────────────────────────
  document.addEventListener("mouseup", (e) => {
    // Small delay so selection is finalised
    setTimeout(() => {
      const sel  = window.getSelection();
      const text = sel ? sel.toString().trim() : "";

      if (text.length >= MIN_LENGTH && text.length <= MAX_LENGTH) {
        const range = sel.getRangeAt(0);
        const rect  = range.getBoundingClientRect();
        showButton(
          rect.left + rect.width / 2,
          rect.top + window.scrollY,   // use pageY equivalent
          text
        );
      } else {
        hideButton();
      }
    }, 50);
  });

  // Hide on scroll or click elsewhere
  document.addEventListener("scroll",   hideButton, { passive: true });
  document.addEventListener("mousedown", (e) => {
    if (floatBtn && e.target !== floatBtn && !floatBtn.contains(e.target)) {
      hideButton();
    }
  });
})();
