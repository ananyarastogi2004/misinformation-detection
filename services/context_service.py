from typing import Dict

from models.emotion_model import emotion_model
from models.sarcasm_model import sarcasm_model


class ContextService:
    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict:
        """
        Perform context-aware analysis:
        - Emotion detection
        - Sarcasm detection
        - Generate flags for pipeline decisions
        """

        emotion_result = emotion_model.predict(text)
        sarcasm_result = sarcasm_model.predict(text)

        flags = []

        # 🚨 Rule 1: Sarcasm detection
        if sarcasm_result.get("is_sarcastic", False):
            flags.append("sarcasm_detected")

        # 🚨 Rule 2: High emotional bias
        emotion_label = emotion_result.get("label", "").lower()
        emotion_conf = emotion_result.get("confidence", 0.0)

        if emotion_label in ["anger", "disgust", "fear"] and emotion_conf > 0.6:
            flags.append("high_emotional_bias")

        # 🚨 Rule 3: Strong opinion (optional extension)
        if emotion_label in ["joy", "surprise"] and emotion_conf > 0.8:
            flags.append("strong_subjective_tone")

        return {
            "emotion": emotion_result,
            "sarcasm": sarcasm_result,
            "flags": flags
        }

    def should_skip_fact_check(self, context: Dict) -> bool:
        """
        Decide whether to skip fact-checking
        """

        return "sarcasm_detected" in context.get("flags", [])


# Singleton instance
context_service = ContextService()