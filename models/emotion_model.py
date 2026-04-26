from typing import Dict, List

from utils.http_client import http_client
from app.config import settings


class EmotionModel:
    def __init__(self):
        self.model_url = settings.EMOTION_MODEL_URL

    def predict(self, text: str) -> Dict:
        """
        Predict emotion from text using HF model
        Returns:
        {
            "label": str,
            "confidence": float,
            "all_emotions": List[Dict]
        }
        """

        payload = {"inputs": text}

        response = http_client.post(self.model_url, payload)

        # 🚨 Error handling
        if isinstance(response, dict) and "error" in response:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_emotions": [],
                "error": response["error"]
            }

        try:
            # HF returns: [[{label, score}, ...]]
            emotions: List[Dict] = response[0]

            # Sort by confidence
            emotions_sorted = sorted(
                emotions,
                key=lambda x: x["score"],
                reverse=True
            )

            top_emotion = emotions_sorted[0]

            return {
                "label": top_emotion["label"],
                "confidence": round(top_emotion["score"], 4),
                "all_emotions": emotions_sorted
            }

        except Exception as e:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "all_emotions": [],
                "error": str(e)
            }


# Singleton instance
emotion_model = EmotionModel()