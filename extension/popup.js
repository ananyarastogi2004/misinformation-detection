"use strict";

const API_URL = "http://127.0.0.1:8000/fact-check/";
const MAX_HISTORY = 10;

// ── DOM refs ──────────────────────────────────────────────────────────
const inputEl        = document.getElementById("inputText");
const charCountEl    = document.getElementById("charCount");
const checkBtn       = document.getElementById("checkBtn");
const btnLabel       = document.getElementById("btnLabel");
const btnSpinner     = document.getElementById("btnSpinner");
const clearBtn       = document.getElementById("clearBtn");
const historyBtn     = document.getElementById("historyBtn");
const resultSection  = document.getElementById("resultSection");
const emptyState     = document.getElementById("emptyState");
const errorBox       = document.getElementById("errorBox");
const errorMsg       = document.getElementById("errorMsg");
const historyDrawer  = document.getElementById("historyDrawer");
const historyList    = document.getElementById("historyList");
const clearHistoryBtn= document.getElementById("clearHistoryBtn");

// ── Verdict config ─────────────────────────────────────────────────────
const VERDICT_CONFIG = {
  "TRUE":         { cls: "v-true",        emoji: "✅", label: "TRUE" },
  "LIKELY TRUE":  { cls: "v-likely-true",  emoji: "🟢", label: "LIKELY TRUE" },
  "UNCERTAIN":    { cls: "v-uncertain",    emoji: "⚠️", label: "UNCERTAIN" },
  "LIKELY FALSE": { cls: "v-likely-false", emoji: "🟠", label: "LIKELY FALSE" },
  "FALSE":        { cls: "v-false",        emoji: "❌", label: "FALSE" },
  "NON_FACTUAL":  { cls: "v-non-factual",  emoji: "💬", label: "SARCASM / OPINION" },
  "NOT_A_CLAIM":  { cls: "v-not-claim",   emoji: "❓", label: "NOT A CLAIM" },
};

// ── On load: check for pre-filled claim from content.js ───────────────
document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.local.get(["pendingClaim"], ({ pendingClaim }) => {
    if (pendingClaim && pendingClaim.trim()) {
      inputEl.value = pendingClaim.trim();
      updateCharCount();
      // Clear it so the next popup open starts fresh
      chrome.storage.local.remove("pendingClaim");
      // Auto-run
      runFactCheck();
    }
  });

  renderHistory();
});

// ── Char counter ───────────────────────────────────────────────────────
inputEl.addEventListener("input", updateCharCount);
function updateCharCount() {
  const n = inputEl.value.length;
  charCountEl.textContent = `${n} / 500`;
  charCountEl.style.color = n > 450 ? "#e74c3c" : "#aaa";
}

// ── Clear ──────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  inputEl.value = "";
  updateCharCount();
  showEmpty();
});

// ── Check button ───────────────────────────────────────────────────────
checkBtn.addEventListener("click", runFactCheck);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runFactCheck();
});

// ── History toggle ─────────────────────────────────────────────────────
historyBtn.addEventListener("click", () => {
  historyDrawer.classList.toggle("hidden");
});
clearHistoryBtn.addEventListener("click", () => {
  chrome.storage.local.set({ history: [] }, renderHistory);
});

// ─────────────────────────────────────────────────────────────────────
// Core fact-check function
// ─────────────────────────────────────────────────────────────────────
async function runFactCheck() {
  const text = inputEl.value.trim();
  if (!text || text.length < 10) {
    showError("Please enter at least 10 characters.");
    return;
  }

  setLoading(true);
  hideAll();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    displayResult(data);
    saveHistory(text, data);

  } catch (e) {
    const msg = e.message.includes("Failed to fetch")
      ? "Cannot reach ClaimSense backend. Make sure the server is running (uvicorn main:app)."
      : e.message;
    showError(msg);
  } finally {
    setLoading(false);
  }
}

// ─────────────────────────────────────────────────────────────────────
// Render result
// ─────────────────────────────────────────────────────────────────────
function displayResult(data) {
  resultSection.classList.remove("hidden");

  const verdict = data.verdict || "UNCERTAIN";
  const cfg     = VERDICT_CONFIG[verdict] || VERDICT_CONFIG["UNCERTAIN"];
  const conf    = Math.round((data.confidence || 0) * 100);

  // ── Verdict badge ──────────────────────────────────────────────────
  const badge = document.getElementById("verdictBadge");
  badge.textContent  = `${cfg.emoji} ${cfg.label}`;
  badge.className    = `verdict-badge ${cfg.cls}`;

  document.getElementById("confValue").textContent = `${conf}%`;
  document.getElementById("confBar").style.width   = `${conf}%`;
  document.getElementById("verdictNote").textContent = data.note || "";

  // ── Context pills ──────────────────────────────────────────────────
  const ctx     = data.context || {};
  const emotion = ctx.emotion || {};
  const sarcasm = ctx.sarcasm || {};
  const flags   = ctx.flags || [];

  const emotionPill  = document.getElementById("emotionPill");
  const sarcasmPill  = document.getElementById("sarcasmPill");
  const flagPill     = document.getElementById("flagPill");

  const emotionMap = {
    joy: "😄", anger: "😠", disgust: "🤢",
    fear: "😨", sadness: "😢", surprise: "😮", neutral: "😐"
  };
  emotionPill.textContent = `${emotionMap[emotion.label] || "😐"} ${emotion.label || "neutral"}`;

  if (sarcasm.is_sarcastic) {
    sarcasmPill.textContent = "⚡ sarcasm detected";
    sarcasmPill.style.background = "#fff5f5";
    sarcasmPill.style.color      = "#c0392b";
    sarcasmPill.style.borderColor = "#fcc";
  } else {
    sarcasmPill.textContent = "✓ not sarcastic";
  }

  if (flags.length > 0) {
    flagPill.textContent = "⚑ " + flags.join(", ").replace(/_/g, " ");
    flagPill.classList.remove("hidden");
  } else {
    flagPill.classList.add("hidden");
  }

  // ── Agreement bar ──────────────────────────────────────────────────
  const ag = data.agreement || {};
  const total = (ag.supports || 0) + (ag.neutral || 0) + (ag.refutes || 0);
  if (total > 0) {
    const sp = ((ag.supports || 0) / total * 100).toFixed(1) + "%";
    const np = ((ag.neutral  || 0) / total * 100).toFixed(1) + "%";
    const rp = ((ag.refutes  || 0) / total * 100).toFixed(1) + "%";
    document.getElementById("agreeSupport").style.width = sp;
    document.getElementById("agreeNeutral").style.width = np;
    document.getElementById("agreeRefute").style.width  = rp;
  }

  // ── Gemini analysis ────────────────────────────────────────────────
  const gemini      = data.gemini_analysis || {};
  const geminiBox   = document.getElementById("geminiBox");
  const geminiSrcs  = document.getElementById("geminiSources");

  geminiBox.textContent = gemini.raw || "No Gemini analysis available.";

  // Show web sources if Gemini used search grounding
  const sources = gemini.sources || [];
  if (sources.length > 0) {
    geminiSrcs.classList.remove("hidden");
    geminiSrcs.innerHTML = '<div class="section-label" style="margin-top:8px">Gemini sources</div>';
    sources.slice(0, 5).forEach(s => {
      const a = document.createElement("a");
      a.className = "gemini-source-link";
      a.href      = s.url;
      a.target    = "_blank";
      a.rel       = "noopener";
      a.textContent = s.title || s.url;
      geminiSrcs.appendChild(a);
    });
  } else {
    geminiSrcs.classList.add("hidden");
  }

  // ── Evidence cards ─────────────────────────────────────────────────
  const evidence = data.evidence || [];
  document.getElementById("evidenceCount").textContent = evidence.length;
  const list = document.getElementById("evidenceList");
  list.innerHTML = "";

  evidence.forEach(ev => {
    const nliLabel   = ev.nli_label || "neutral";
    const nliCls     = nliLabel === "supports" ? "nli-supports"
                     : nliLabel === "refutes"  ? "nli-refutes"
                     : "nli-neutral";
    const credLabel  = ev.credibility_label || "";
    const credStar   = { "fact-checker": "★★★", "high": "★★☆", "medium": "★☆☆", "low": "☆☆☆" }[credLabel] || "";
    const conf       = ev.confidence ? Math.round(ev.confidence * 100) + "%" : "";
    const dateStr    = ev.date
      ? new Date(ev.date).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })
      : "";

    const card = document.createElement("div");
    card.className = "ev-card";
    card.innerHTML = `
      <div class="ev-header">
        <span class="ev-source">${esc(ev.source || "Unknown")}</span>
        <span class="ev-nli ${nliCls}">${nliLabel}</span>
      </div>
      ${ev.title ? `<div class="ev-title">${esc(ev.title)}</div>` : ""}
      <div class="ev-content">${esc(ev.content || "")}</div>
      <div class="ev-footer">
        <span class="ev-cred">${credStar} ${credLabel}${dateStr ? " · " + dateStr : ""}${conf ? " · " + conf : ""}</span>
        ${ev.url ? `<a class="ev-link" href="${esc(ev.url)}" target="_blank" rel="noopener">Source ↗</a>` : ""}
      </div>
    `;
    list.appendChild(card);
  });
}

// ─────────────────────────────────────────────────────────────────────
// History helpers
// ─────────────────────────────────────────────────────────────────────
function saveHistory(claim, data) {
  chrome.storage.local.get(["history"], ({ history = [] }) => {
    const entry = {
      claim:   claim.substring(0, 120),
      verdict: data.verdict,
      conf:    Math.round((data.confidence || 0) * 100),
      ts:      Date.now()
    };
    const updated = [entry, ...history].slice(0, MAX_HISTORY);
    chrome.storage.local.set({ history: updated });
  });
}

function renderHistory() {
  chrome.storage.local.get(["history"], ({ history = [] }) => {
    if (!historyList) return;
    historyList.innerHTML = "";

    if (history.length === 0) {
      historyList.innerHTML = '<p style="color:#aaa;font-size:12px;text-align:center;padding:16px">No checks yet.</p>';
      return;
    }

    history.forEach(h => {
      const cfg  = VERDICT_CONFIG[h.verdict] || VERDICT_CONFIG["UNCERTAIN"];
      const time = new Date(h.ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
      const date = new Date(h.ts).toLocaleDateString("en-IN", { day: "numeric", month: "short" });

      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <div class="history-claim">${esc(h.claim)}</div>
        <div class="history-meta">
          <span>${cfg.emoji} ${cfg.label} · ${h.conf}%</span>
          <span>${date}, ${time}</span>
        </div>
      `;
      item.addEventListener("click", () => {
        inputEl.value = h.claim;
        updateCharCount();
        historyDrawer.classList.add("hidden");
      });
      historyList.appendChild(item);
    });
  });
}

// ─────────────────────────────────────────────────────────────────────
// UI state helpers
// ─────────────────────────────────────────────────────────────────────
function setLoading(on) {
  checkBtn.disabled     = on;
  btnLabel.textContent  = on ? "Analyzing…" : "Check Claim";
  btnSpinner.classList.toggle("hidden", !on);
}

function hideAll() {
  resultSection.classList.add("hidden");
  emptyState.classList.add("hidden");
  errorBox.classList.add("hidden");
}

function showEmpty() {
  hideAll();
  emptyState.classList.remove("hidden");
}

function showError(msg) {
  hideAll();
  errorBox.classList.remove("hidden");
  errorMsg.textContent = msg;
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
