from typing import Dict

from services.context_service import context_service
from models.claim_detector import claim_detector
from services.retrieval_service import retrieval_service
from services.ranking_service import ranking_service
from services.verification_service import verification_service
from services.credibility_service import credibility_service
from services.gemini_service import gemini_service
from services.agreement_service import agreement_service


# ──────────────────────────────────────────────────────────────────────
# NEW: 5-class verdict instead of 3.
# This adds nuance for the research paper and real-world usefulness.
# TRUE / LIKELY TRUE / UNCERTAIN / LIKELY FALSE / FALSE
# ──────────────────────────────────────────────────────────────────────

def _five_class_verdict(score: float) -> str:
    if score >  0.40:  return "TRUE"
    if score >  0.15:  return "LIKELY TRUE"
    if score > -0.15:  return "UNCERTAIN"
    if score > -0.40:  return "LIKELY FALSE"
    return "FALSE"


def _parse_gemini(text: str):
    """Extract verdict and confidence hint from Gemini's structured output."""
    t = text.lower()
    if "verdict: true" in t:
        gem_v, gem_c = "TRUE", 0.85
    elif "verdict: false" in t:
        gem_v, gem_c = "FALSE", 0.85
    elif "verdict: uncertain" in t:
        gem_v, gem_c = "UNCERTAIN", 0.50
    else:
        gem_v, gem_c = "UNCERTAIN", 0.40

    # Adjust by Gemini's own stated confidence
    if "confidence: high" in t:
        gem_c = min(gem_c + 0.10, 0.95)
    elif "confidence: low" in t:
        gem_c = max(gem_c - 0.15, 0.20)

    return gem_v, gem_c


def _to_score(verdict: str) -> int:
    return {"TRUE": 1, "LIKELY TRUE": 1, "FALSE": -1, "LIKELY FALSE": -1}.get(verdict, 0)


class FactCheckPipeline:

    def run(self, text: str) -> Dict:

        # ── Step 1: Context (sarcasm + emotion) ─────────────────────────
        context = context_service.analyze(text)

        if context_service.should_skip_fact_check(context):
            return {
                "input": text,
                "verdict": "NON_FACTUAL",
                "verdict_detail": "Sarcasm detected — cannot fact-check reliably.",
                "confidence": 0.0,
                "context": context,
                "message": "Input appears sarcastic or highly subjective."
            }

        # ── Step 2: Claim detection ──────────────────────────────────────
        claim_result = claim_detector.predict(text)
        if not claim_result.get("is_claim", False):
            return {
                "input": text,
                "verdict": "NOT_A_CLAIM",
                "verdict_detail": "Input is not a verifiable factual statement.",
                "confidence": claim_result.get("confidence", 0.0),
                "context": context,
                "message": "Not a factual, check-worthy claim."
            }

        # ── Step 3: Evidence retrieval ───────────────────────────────────
        evidence = retrieval_service.retrieve(text)
        if not evidence:
            return {
                "input": text, "verdict": "UNCERTAIN",
                "verdict_detail": "No evidence sources found.",
                "confidence": 0.0, "context": context,
                "message": "No relevant evidence found."
            }

        # ── Step 4: FAISS ranking ────────────────────────────────────────
        ranked_evidence = ranking_service.rank(text, evidence)
        if not ranked_evidence:
            return {
                "input": text, "verdict": "UNCERTAIN",
                "verdict_detail": "Evidence ranking failed.",
                "confidence": 0.0, "context": context,
                "message": "Evidence ranking failed."
            }

        # ── Step 5: Credibility scoring ──────────────────────────────────
        ranked_evidence = credibility_service.attach_scores(ranked_evidence)

        # ── Step 6: NLI verification ────────────────────────────────────
        verification = verification_service.verify(text, ranked_evidence)
        if not verification or "verdict" not in verification:
            return {
                "input": text, "verdict": "UNCERTAIN",
                "verdict_detail": "NLI verification failed.",
                "confidence": 0.0, "context": context,
                "message": "Verification failed."
            }

        # ── Step 7: Agreement analysis ───────────────────────────────────
        # BUG FIX in agreement_service: now uses nli_label, not similarity
        agreement = agreement_service.analyze(verification.get("evidence", []))

        # ── Step 8: Gemini (live web search grounding) ───────────────────
        top3 = ranked_evidence[:3]
        background_text = " ".join(
            e.get("content", "") for e in top3 if e.get("content")
        )
        gemini_result = gemini_service.fact_check(text, background_text)
        if not gemini_result.get("raw"):
            gemini_result["raw"] = "No Gemini analysis available."

        # ── Step 9: Hybrid decision fusion ──────────────────────────────
        nli_verdict  = verification["verdict"]
        nli_conf     = verification.get("confidence", 0.0)
        gem_verdict, gem_conf = _parse_gemini(gemini_result.get("raw", ""))

        # agreement_score now correctly reflects NLI-label-based stance
        ag_score = agreement.get("agreement_score", 0.0)
        controversy = agreement.get("controversy_score", 0.0)

        nli_score = _to_score(nli_verdict) * nli_conf
        gem_score = _to_score(gem_verdict) * gem_conf

        # Weights: NLI 45% | Gemini 35% | Agreement 20%
        # Gemini weight raised (35%) because it now has live web search
        final_score = (
            0.45 * nli_score +
            0.35 * gem_score +
            0.20 * ag_score
        )

        # Controversy penalty: if sources conflict, reduce confidence
        final_score *= (1.0 - 0.3 * controversy)

        final_verdict    = _five_class_verdict(final_score)
        final_confidence = round(min(abs(final_score), 1.0), 4)

        return {
            "input":          text,
            "verdict":        final_verdict,
            "confidence":     final_confidence,
            "time_aware":     verification.get("time_aware", False),
            "note":           verification.get("note", ""),
            "context":        context,
            "evidence":       verification.get("evidence", []),
            "agreement":      agreement,
            "gemini_analysis": gemini_result,
            "explanation": (
                "Verdict from hybrid fusion: NLI (45%) + "
                "Gemini web-search (35%) + Source agreement (20%)"
            )
        }


# Singleton
fact_check_pipeline = FactCheckPipeline()
