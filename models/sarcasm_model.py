from typing import Dict, List

from utils.http_client import http_client
from app.config import settings


# Map model labels to human-readable labels
label_map = {
    "LABEL_0": "not_sarcastic",
    "LABEL_1": "sarcastic"
}


class SarcasmModel:
    def __init__(self):
        self.model_url = settings.SARCASM_MODEL_URL
        self.threshold = settings.SARCASM_THRESHOLD

    def predict(self, text: str) -> Dict:
        """
        Detect sarcasm in text

        Returns:
        {
            "is_sarcastic": bool,
            "confidence": float,
            "label": str
        }
        """

        payload = {"inputs": text}

        response = http_client.post(self.model_url, payload)

        # 🚨 Error handling
        if isinstance(response, dict) and "error" in response:
            return {
                "is_sarcastic": False,
                "confidence": 0.0,
                "label": "unknown",
                "error": response["error"]
            }

        try:
            # HF response format may vary
            # Usually: [[{label: "LABEL_0", score: 0.9}, ...]]
            predictions: List[Dict] = response[0]

            # Sort by score
            predictions_sorted = sorted(
                predictions,
                key=lambda x: x["score"],
                reverse=True
            )

            top_pred = predictions_sorted[0]

            raw_label = top_pred.get("label", "")
            confidence = top_pred.get("score", 0.0)

            # Map the model label to the requested human-readable label_map
            mapped_label = label_map.get(raw_label)

            if not mapped_label:
                # Fallback heuristics for different label formats
                low = raw_label.lower()
                if "sarcasm" in low or low in ["1", "label_1", "sarcastic"]:
                    mapped_label = "sarcastic"
                elif "not" in low or low in ["0", "label_0", "not_sarcastic"]:
                    mapped_label = "not_sarcastic"
                else:
                    mapped_label = low or "unknown"

            is_sarcastic = (mapped_label == "sarcastic") and confidence >= self.threshold

            return {
                "is_sarcastic": is_sarcastic,
                "confidence": round(confidence, 4),
                "label": mapped_label,
                "raw_label": raw_label,
                "all_predictions": predictions_sorted
            }

        except Exception as e:
            return {
                "is_sarcastic": False,
                "confidence": 0.0,
                "label": "unknown",
                "error": str(e)
            }


# Singleton instance
sarcasm_model = SarcasmModel()