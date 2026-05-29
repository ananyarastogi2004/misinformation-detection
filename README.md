<div align="center">

# 🔍 ClaimSense

### AI-Powered Cross-Platform Misinformation Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3-FF6F00?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Major Project — B.Tech CSE-AI, Indira Gandhi Delhi Technical University for Women**

*Ananya Atri · Ananya Rastogi · Ashmita Sharma · Ishika*

*Supervisor: Rahul Sachdeva, Assistant Professor, AI&DS*

---

[Overview](#-overview) · [Architecture](#-system-architecture) · [Setup](#-setup) · [Running](#-running-the-server) · [API](#-api-reference) · [Extension](#-browser-extension) · [Evaluation](#-evaluation--ablation-study) · [Results](#-results) · [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

ClaimSense is an end-to-end, AI-powered fact-checking system that automatically detects and verifies factual claims in online content. It is designed as the backend engine for a Chrome browser extension that enables real-time misinformation detection on any website or social media platform.

### Key Contributions

- **9-stage modular pipeline** combining local transformer models with live web-search grounding
- **Sarcasm and emotion-aware gating** — the system detects sarcastic or highly emotional inputs and refuses to fact-check them (addressing a core limitation of existing systems)
- **Gemini 2.0 Flash with Google Search grounding** for real-time evidence retrieval beyond static datasets
- **5-class verdict system** (TRUE / LIKELY TRUE / UNCERTAIN / LIKELY FALSE / FALSE) instead of the binary or 3-class systems common in prior work
- **Hybrid fusion** of three independent verification signals: NLI (45%), Gemini web-search (35%), and source agreement (20%)
- **Temporal decay weighting** to penalise outdated evidence
- **Source credibility scoring** across 40+ news and fact-checking organisations including Indian outlets

### Performance (50-claim evaluation)

| Metric | Score |
|--------|-------|
| Overall Accuracy | **66.0%** |
| Macro F1 | **61.8%** |
| FALSE Precision | **100%** |
| NOT_A_CLAIM Recall | **100%** |
| Temporal Claim Accuracy | **62.5%** |

---

## 🏗 System Architecture

```
User Input (text)
       │
       ▼
┌─────────────────────┐
│  1. Context Module  │  Emotion detection + Sarcasm detection
│  (gating layer)     │  → Skips pipeline if sarcastic/subjective
└──────────┬──────────┘
           │ claim-worthy only
           ▼
┌─────────────────────┐
│  2. Claim Detector  │  DeBERTa-v3 (fine-tuned, 92.7% accuracy)
│  (local model)      │  → Binary: is this a verifiable fact?
└──────────┬──────────┘
           │ yes
           ▼
┌─────────────────────┐
│  3. Evidence        │  Google Fact Check API
│     Retrieval       │  + NewsAPI
│                     │  + Wikipedia (fallback)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. Credibility     │  40+ sources scored (PTI=0.90, BBC=0.93,
│     Scoring         │  Factly=0.97, Reuters=0.95 ...)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. FAISS Ranking   │  all-MiniLM-L6-v2 embeddings
│  (semantic search)  │  Cosine similarity + temporal decay weights
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  6. NLI Verification│  facebook/bart-large-mnli
│  (local inference)  │  Claim-specific zero-shot classification
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  7. Source Agreement│  Weighted stance aggregation
│     Analysis        │  using NLI labels (not similarity scores)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  8. Gemini Analysis │  Gemini 2.0 Flash + Google Search grounding
│  (live web search)  │  Real-time evidence from live web
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  9. Hybrid Fusion   │  NLI(45%) + Gemini(35%) + Agreement(20%)
│     + 5-class       │  with controversy penalty
│     Verdict         │
└─────────────────────┘
           │
           ▼
    Final Output JSON
    (verdict + confidence + evidence + sources)
```

![Data Flow Architecture](https://github.com/ananyarastogi2004/misinformation-detection/blob/main/extension-demo/dataflow%20diagram.png)

---

## ⚙️ Setup

### Prerequisites

- Python 3.9 or higher
- 4 GB RAM minimum (8 GB recommended — DeBERTa runs on CPU)
- API keys for 4 free-tier services (see below)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/claimsense.git
cd claimsense
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install takes 5–10 minutes. PyTorch (~2 GB) and sentence-transformers are included.
> For faster install on CPU-only machines:
> ```bash
> pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_FACT_CHECK_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxx
NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Where to get each key (all free tier):**

| Key | Source | Free Tier Limit |
|-----|--------|-----------------|
| `HUGGINGFACE_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | 1000 req/day |
| `GOOGLE_FACT_CHECK_API_KEY` | [console.cloud.google.com](https://console.cloud.google.com) → Enable "Fact Check Tools API" | 1000 req/day |
| `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) | 100 req/day |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 1500 req/day, 15 RPM |

### 5. Verify the model is in place

The trained DeBERTa-v3 claim detection model must exist at:

```
models/claim_extractor_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── added_tokens.json
└── spm.model
```

---

## 🚀 Running the Server

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Production (stable, no reload)
uvicorn main:app --host 127.0.0.1 --port 8000
```

Successful startup looks like:

```
🔄 Initializing Claim Extractor...
✅ DeBERTa Claim Extractor loaded successfully!
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

> The server takes 10–20 seconds to start while DeBERTa loads into memory.

### Interactive API documentation

Once running, open your browser at:

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

### `POST /fact-check/`

Verify a factual claim through the full 9-stage pipeline.

**Request:**
```json
{
  "text": "India is the most populous country in the world."
}
```

**Parameters:**

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `text` | string | Yes | 10–1000 characters |

**Ablation mode** (for research/testing — disable one component):
```
POST /fact-check/?disable=gemini
POST /fact-check/?disable=nli
POST /fact-check/?disable=agreement
```

**Response:**
```json
{
  "input": "India is the most populous country in the world.",
  "verdict": "TRUE",
  "confidence": 0.858,
  "explanation": "Hybrid fusion — NLI(45%) + Gemini(35%) + Agreement(20%)",
  "context": {
    "emotion": { "label": "neutral", "confidence": 0.71 },
    "sarcasm": { "is_sarcastic": false, "confidence": 0.99 },
    "flags": []
  },
  "evidence": [
    {
      "source": "FACTLY",
      "title": "...",
      "content": "...",
      "url": "https://...",
      "date": null,
      "similarity_score": 0.789,
      "nli_label": "supports",
      "confidence": 0.94,
      "credibility": 0.97,
      "credibility_label": "fact-checker",
      "time_weight": 0.75
    }
  ],
  "agreement": {
    "supports": 3,
    "refutes": 0,
    "neutral": 2,
    "agreement_score": 0.763,
    "controversy_score": 0.0
  },
  "gemini_analysis": {
    "raw": "Verdict: TRUE\nConfidence: HIGH\nReason: ...",
    "sources": [],
    "search_used": true
  }
}
```

**Verdict classes:**

| Verdict | Meaning |
|---------|---------|
| `TRUE` | Strong evidence supports the claim |
| `LIKELY TRUE` | Evidence mostly supports, some uncertainty |
| `UNCERTAIN` | Insufficient or conflicting evidence |
| `LIKELY FALSE` | Evidence mostly contradicts the claim |
| `FALSE` | Strong evidence refutes the claim |
| `NON_FACTUAL` | Sarcasm or highly emotional input detected |
| `NOT_A_CLAIM` | Opinion, question, or greeting — not fact-checkable |

**Quick test with curl:**
```bash
curl -X POST http://127.0.0.1:8000/fact-check/ \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"The COVID-19 vaccine contains microchips.\"}"
```

---

## 🌐 Browser Extension

The Chrome extension allows you to select any text on any webpage and fact-check it instantly.

### Loading the extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project

### Usage

**Method 1 — Select text on any page:**
- Highlight any text (15–500 characters)
- A floating **"🔍 Fact Check"** button appears near your selection
- Click it — the popup opens with results automatically

**Method 2 — Right-click menu:**
- Select text on any page
- Right-click → **"Fact Check with ClaimSense"**
- The popup opens with the selected text pre-filled

**Method 3 — Click the extension icon:**
- Click the ClaimSense icon in the Chrome toolbar
- Type or paste any claim into the text box
- Press **Check Claim** or **Ctrl+Enter**

> **Important:** The backend server must be running (`uvicorn main:app ...`) for the extension to work. The extension connects to `http://127.0.0.1:8000`.

### Extension features

- Colour-coded 5-class verdict badge (green → red)
- Confidence progress bar
- Emotion and sarcasm pills
- Source agreement bar (supports / neutral / refutes)
- Gemini AI analysis with web source links
- Collapsible evidence cards with credibility stars and NLI labels
- History of last 10 checks (click any to re-run)
- Works on: news sites, Wikipedia, LinkedIn, Reddit, and most static pages

![Extension demo](https://github.com/ananyarastogi2004/misinformation-detection/blob/main/extension-demo/extension-new.png)

---

## 🧪 Evaluation & Ablation Study

All evaluation scripts and datasets are in the `tests/` folder.

### Test dataset

`tests/test_claims.json` — 50 manually curated claims with ground-truth labels:

| Class | Count | Examples |
|-------|-------|---------|
| TRUE | 15 | Chandrayaan-3 launch, India UN membership, GST date |
| FALSE | 15 | Moon landing conspiracy, 5G cancer, Einstein failed maths |
| UNCERTAIN | 10 | AI job displacement, Bitcoin reserve currency |
| NON_FACTUAL | 5 | Sarcastic statements about politicians, inequality |
| NOT_A_CLAIM | 5 | Questions, greetings, personal opinions |

### Running evaluations

**The server must be running in a separate terminal before running any evaluation.**

**Quick sanity check (5 claims, ~30 seconds):**
```bash
python tests/evaluate.py --claims tests/test_claims.json --output tests/results/ --limit 5
```

**Full 50-claim evaluation (~8 minutes):**
```bash
python tests/evaluate.py --mode full --claims tests/test_claims.json --output tests/results/ --delay 6
```

**Ablation — Gemini disabled (no API quota used, ~2 minutes):**
```bash
python tests/evaluate.py --mode ablation --disable gemini --claims tests/test_claims.json --output tests/results/
```

**Ablation — NLI disabled (~8 minutes):**
```bash
python tests/evaluate.py --mode ablation --disable nli --claims tests/test_claims.json --output tests/results/ --delay 6
```

**Ablation — Agreement disabled (~8 minutes):**
```bash
python tests/evaluate.py --mode ablation --disable agreement --claims tests/test_claims.json --output tests/results/ --delay 6
```

**Full ablation study — all 4 configurations (~35 minutes):**
```bash
python tests/evaluate.py --mode ablation --claims tests/test_claims.json --output tests/results/ --delay 6
```

**All evaluations together:**
```bash
python tests/evaluate.py --mode both --claims tests/test_claims.json --output tests/results/ --delay 6
```

### Output files (saved to `tests/results/`)

| File | Contents |
|------|----------|
| `eval_full_TIMESTAMP.csv` | Per-claim results: verdict, confidence, correct/wrong |
| `eval_ablation_gemini_TIMESTAMP.csv` | Same with Gemini disabled |
| `eval_ablation_nli_TIMESTAMP.csv` | Same with NLI disabled |
| `eval_ablation_agreement_TIMESTAMP.csv` | Same with Agreement disabled |
| `ablation_comparison.json` | Summary table comparing all 4 configurations |

### Ablation weight redistribution

When a component is disabled, its weight is redistributed proportionally:

| Configuration | NLI | Gemini | Agreement |
|--------------|-----|--------|-----------|
| Full pipeline | 45% | 35% | 20% |
| Gemini disabled | 65% | 0% | 35% |
| NLI disabled | 0% | 60% | 40% |
| Agreement disabled | 56% | 44% | 0% |

---

## 📊 Results

### Full pipeline — 50 claims

```
Accuracy : 66.0%   Macro F1 : 61.8%   Correct : 33/50

Class          Prec    Rec     F1   Support
─────────────────────────────────────────────
TRUE          0.700  0.538  0.609    13/15
FALSE         1.000  0.706  0.828    12/17  ← 100% precision
UNCERTAIN     0.421  0.800  0.552     8/10
NON_FACTUAL   1.000  0.200  0.333     1/5
NOT_A_CLAIM   0.625  1.000  0.769    5/5   ← 100% recall
```

### Error taxonomy

| Error Type | Count | Root Cause |
|-----------|-------|-----------|
| API timeout → UNCERTAIN | 4 | Gemini search on complex topics |
| Claim detector rejects factual claim | 3 | DeBERTa trained on political debates, weak on biographical facts |
| Evidence retrieval failure | 5 | Wikipedia query too generic |
| Wrong direction (TRUE↔FALSE) | 1 | F10 "US has 52 states" |
| Sarcasm not caught by pipeline | 4 | DeBERTa filters sarcasm as opinion before sarcasm gate runs |

### Notable findings

- **FALSE precision = 1.000** — the system never falsely accuses a true statement of being misinformation
- **NOT_A_CLAIM recall = 1.000** — all opinions, questions, and greetings correctly filtered before evidence retrieval
- **Temporal accuracy = 62.5%** — the temporal drift problem (2022 fact-checks about India population) reduces this score
- **22% timeout rate** — indicates Gemini free-tier latency on complex geopolitical/economic claims

### Ablation study — component contribution (20 claims)

| Configuration | Accuracy | Drop vs Full |
|---|---|---|
| Full Pipeline | 70.0% | — |
| Agreement disabled | 55.0% | −15.0% |
| Gemini disabled | 60.0% | −10.0% |
| NLI disabled | 10.0% | −60.0% |

**Finding:** NLI is the primary verdict-generating signal (60% contribution).
Gemini and Agreement function as amplifiers. Without NLI, the system
collapses even when other components are active.

---

## 📁 Project Structure

```
fake-news-detector/
├── main.py                          # FastAPI app entry point
├── requirements.txt                 # All dependencies
├── .env                             # API keys (not committed)
│
├── app/
│   └── config.py                    # Settings and thresholds
│
├── api/
│   └── routes/
│       └── fact_check.py            # POST /fact-check/ endpoint
│
├── models/
│   ├── claim_detector.py            # DeBERTa inference wrapper
│   ├── claim_extractor.py           # Model loader
│   ├── nli_model.py                 # NLI via HuggingFace API
│   ├── emotion_model.py             # Emotion classification
│   ├── sarcasm_model.py             # Sarcasm detection
│   └── claim_extractor_model/       # Trained DeBERTa weights (local)
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
│
├── services/
│   ├── pipeline.py                  # 9-stage orchestrator
│   ├── context_service.py           # Emotion + sarcasm gating
│   ├── retrieval_service.py         # Google Fact Check + News + Wikipedia
│   ├── ranking_service.py           # FAISS semantic ranking
│   ├── credibility_service.py       # Source credibility scoring
│   ├── verification_service.py      # NLI + temporal weighting
│   ├── agreement_service.py         # Multi-source stance aggregation
│   └── gemini_service.py            # Gemini 2.0 Flash + search grounding
│
├── utils/
│   ├── http_client.py               # Authenticated HTTP client
│   ├── logger.py                    # Logging setup
│   ├── preprocessing.py             # Text cleaning
│   └── constants.py                 # Shared constants
│
├── extension/                       # Chrome browser extension
│   ├── manifest.json                # Extension manifest (MV3)
│   ├── popup.html                   # Extension UI
│   ├── popup.css                    # Styles
│   ├── popup.js                     # UI logic + API calls
│   ├── content.js                   # Page text selection detection
│   └── background.js                # Service worker + context menu
│
└── tests/
    ├── test_claims.json             # 50-claim ground-truth evaluation set
    ├── evaluate.py                  # Full evaluation + ablation study script
    └── results/                     # Generated CSV/JSON result files
        ├── eval_full_*.csv
        ├── eval_ablation_gemini_*.csv
        └── ablation_comparison.json
```

---

## 🛠 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` inside `(venv)` |
| `FileNotFoundError: claim_extractor_model` | The `models/claim_extractor_model/` folder with DeBERTa weights must exist |
| Wikipedia returns 403 | Fixed in `http_client.py` — User-Agent header is set automatically |
| Gemini 503 / overloaded | Free-tier model under load — the service retries with backoff (1s → 2s → 4s) automatically |
| Gemini 429 / quota exceeded | Free tier resets every 24 hours. Run `--disable gemini` ablation tests in the meantime |
| Extension shows "Cannot reach backend" | Start the server first: `uvicorn main:app --host 127.0.0.1 --port 8000` |
| Extension button doesn't appear on Twitter/X | Dynamic rendering — use the toolbar popup instead. Works reliably on static news sites |
| `422 Unprocessable Entity` | Input text is under 10 or over 1000 characters |
| Evaluation timeouts on complex claims | Increase timeout: `python tests/evaluate.py ... --delay 8` |

---

## 📚 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.110.0 | REST API framework |
| `uvicorn` | 0.29.0 | ASGI server |
| `torch` | 2.2.2 | Local model inference |
| `transformers` | 4.40.1 | DeBERTa tokenizer and model |
| `sentence-transformers` | 2.6.1 | FAISS embedding generation |
| `faiss-cpu` | 1.8.0 | Semantic similarity search |
| `sentencepiece` | 0.2.0 | DeBERTa tokenization |
| `requests` | 2.31.0 | HTTP client for external APIs |
| `pydantic` | 2.6.4 | Request/response validation |

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ at IGDTUW · 2025–26

</div>
