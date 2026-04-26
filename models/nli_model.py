from typing import Dict
from utils.http_client import http_client
from utils.logger import logger
from app.config import settings


class NLIModel:
    def __init__(self):
        self.model_url = settings.NLI_MODEL_URL

    def predict(self, claim: str, evidence: str) -> Dict:
        """
        Zero-shot NLI using facebook/bart-large-mnli via HuggingFace
        Inference API.

        Uses claim-specific candidate labels so the model reasons about
        this exact claim rather than generic entailment categories.
        """

        # Truncate very long evidence to avoid token limit errors
        evidence_trimmed = evidence[:512] if len(evidence) > 512 else evidence

        payload = {
            "inputs": evidence_trimmed,
            "parameters": {
                "candidate_labels": [
                    f"The claim '{claim[:120]}' is true",
                    f"The claim '{claim[:120]}' is false",
                    "Not enough information"
                ]
            }
        }

        response = http_client.post(self.model_url, payload)

        # CHANGED: removed the debug print("🧠 NLI raw response:", response)
        # that was flooding the terminal on every evidence item.
        # Errors are still logged.

        if isinstance(response, dict) and "error" in response:
            logger.warning(f"NLI error for claim '{claim[:60]}': {response['error']}")
            return {"label": "neutral", "confidence": 0.0}

        try:
            if isinstance(response, list):
                sorted_preds = sorted(response, key=lambda x: x["score"], reverse=True)
                top   = sorted_preds[0]
                label = top["label"].lower()
                score = top["score"]

                if "true" in label:
                    final = "supports"
                elif "false" in label:
                    final = "refutes"
                else:
                    final = "neutral"

                return {
                    "label":      final,
                    "confidence": round(float(score), 4)
                }

            # Unexpected response shape
            logger.warning(f"Unexpected NLI response format: {type(response)}")
            return {"label": "neutral", "confidence": 0.0}

        except Exception as e:
            logger.error(f"NLI parsing error: {e}")
            return {"label": "neutral", "confidence": 0.0}


# Singleton
nli_model = NLIModel()