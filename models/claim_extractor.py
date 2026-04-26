from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "claim_extractor_model")

print("\n🔄 Initializing Claim Extractor...")

# ==============================
# Check Model Folder Exists
# ==============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Claim model folder NOT found at: {MODEL_PATH}")

print(f"📁 Found model directory: {MODEL_PATH}")
print("📄 Files:", os.listdir(MODEL_PATH))


# ==============================
# Select Device
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙ Using device: {DEVICE}")


# ==============================
# Load Tokenizer + Model
# ==============================
try:
    print("🔍 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        fix_mistral_regex=True
    )

    print("🔍 Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    ).to(DEVICE)

    model.eval()
    print("✅ DeBERTa Claim Extractor loaded successfully!\n")

except Exception as e:
    print("❌ ERROR loading model:", str(e))
    raise e


# ==============================
# Inference Function
# ==============================
def is_claim(text: str):
    """
    Runs claim detection on input text using the local model.
    """

    if not text or not isinstance(text, str):
        return {"is_claim": False, "confidence": 0.0}

    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        return {
            "is_claim": bool(pred_idx.item()),
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return {
            "is_claim": False,
            "confidence": 0.0,
            "error": str(e)
        }