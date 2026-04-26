from typing import List, Dict


class AgreementService:
    """
    BUG FIX: Previous version used similarity_score >= 0.5 to classify
    evidence as 'supports' — completely wrong. Semantic similarity measures
    how RELEVANT the evidence is, not whether it agrees with the claim.
    A Newschecker article titled 'India Population Claim Is FALSE' with
    high similarity was being counted as supporting evidence.

    FIX: Now uses the nli_label field set by the NLI model ('supports',
    'refutes', 'neutral') which actually captures stance.

    ADDED: Weighted agreement using evidence confidence scores so that
    high-confidence NLI predictions count more than low-confidence ones.
    """

    def __init__(self):
        pass

    def analyze(self, evidence_list: List[Dict]) -> Dict:
        supports_score = 0.0
        refutes_score = 0.0
        neutral_score = 0.0

        supports_count = 0
        refutes_count = 0
        neutral_count = 0

        for e in evidence_list:
            # BUG FIX: use nli_label, not similarity_score
            label = e.get("nli_label", "neutral")
            conf = e.get("confidence", 0.5)

            if label == "supports":
                supports_score += conf
                supports_count += 1
            elif label == "refutes":
                refutes_score += conf
                refutes_count += 1
            else:
                neutral_score += conf
                neutral_count += 1

        total_score = supports_score + refutes_score + neutral_score

        # Weighted agreement: what fraction of weighted evidence supports?
        agreement_score = supports_score / total_score if total_score > 0 else 0.0

        # Controversy score: how much disagreement exists between sources?
        controversy = 0.0
        if total_score > 0:
            controversy = min(supports_score, refutes_score) / total_score

        return {
            "supports": supports_count,
            "refutes": refutes_count,
            "neutral": neutral_count,
            "supports_score": round(supports_score, 4),
            "refutes_score": round(refutes_score, 4),
            "agreement_score": round(agreement_score, 4),
            "controversy_score": round(controversy, 4),
        }


# Singleton
agreement_service = AgreementService()
