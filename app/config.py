import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    # ── API Keys ──────────────────────────────────────────────────────
    HUGGINGFACE_API_KEY:       str = os.getenv("HUGGINGFACE_API_KEY", "")
    GOOGLE_FACT_CHECK_API_KEY: str = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
    NEWS_API_KEY:              str = os.getenv("NEWS_API_KEY", "")
    GEMINI_API_KEY:            str = os.getenv("GEMINI_API_KEY", "")

    # ── HuggingFace model endpoints ───────────────────────────────────
    EMOTION_MODEL_URL: str = (
        "https://router.huggingface.co/hf-inference/models/"
        "j-hartmann/emotion-english-distilroberta-base"
    )
    SARCASM_MODEL_URL: str = (
        "https://router.huggingface.co/hf-inference/models/"
        "helinivan/english-sarcasm-detector"
    )
    NLI_MODEL_URL: str = (
        "https://router.huggingface.co/hf-inference/models/"
        "facebook/bart-large-mnli"
    )

    # ── Timeouts ──────────────────────────────────────────────────────
    # CHANGED: was 15s — too short now that Gemini does live web searches
    # (search grounding adds ~5–10s latency on top of generation).
    # HF cold-start can also take 20s on free tier.
    # Set to 30s for HF calls; Gemini service uses its own 30s timeout.
    REQUEST_TIMEOUT: int = 45

    # ── Evidence limits ───────────────────────────────────────────────
    MAX_EVIDENCE: int = 10          # raised from 5; pipeline still uses top-5

    # ── Thresholds ────────────────────────────────────────────────────
    SARCASM_THRESHOLD:  float = 0.60
    EMOTION_THRESHOLD:  float = 0.70

    # ── 5-class verdict thresholds (mirrors pipeline.py) ─────────────
    VERDICT_TRUE_THRESHOLD:         float =  0.40
    VERDICT_LIKELY_TRUE_THRESHOLD:  float =  0.15
    VERDICT_LIKELY_FALSE_THRESHOLD: float = -0.15
    VERDICT_FALSE_THRESHOLD:        float = -0.40

    # ── Paths ─────────────────────────────────────────────────────────
    BASE_DIR:       str = os.getcwd()
    CACHE_DIR:      str = os.path.join(os.getcwd(), "data/cache")
    EMBEDDINGS_DIR: str = os.path.join(os.getcwd(), "data/embeddings")


# Singleton
settings = Settings()