from typing import Dict
from models.claim_extractor import is_claim  # 👈 your old file


class ClaimDetector:
    def __init__(self):
        pass

    def predict(self, text: str) -> Dict:
        """
        Uses local trained DeBERTa model
        """
        result = is_claim(text)

        return {
            "is_claim": result.get("is_claim", False),
            "confidence": result.get("confidence", 0.0)
        }


# Singleton
claim_detector = ClaimDetector()