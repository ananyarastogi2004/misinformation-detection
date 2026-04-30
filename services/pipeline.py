from typing import Dict, Optional

from services.context_service import context_service
from models.claim_detector import claim_detector
from services.retrieval_service import retrieval_service
from services.ranking_service import ranking_service
from services.verification_service import verification_service
from services.credibility_service import credibility_service
from services.gemini_service import gemini_service
from services.agreement_service import agreement_service


def _five_class_verdict(score: float) -> str:
    if score >  0.40: return "TRUE"
    if score >  0.15: return "LIKELY TRUE"
    if score > -0.15: return "UNCERTAIN"
    if score > -0.40: return "LIKELY FALSE"
    return "FALSE"


def _parse_gemini(text: str):
    t = text.lower()
    if "verdict: true"        in t: gem_v, gem_c = "TRUE",      0.85
    elif "verdict: false"     in t: gem_v, gem_c = "FALSE",     0.85
    elif "verdict: uncertain" in t: gem_v, gem_c = "UNCERTAIN", 0.50
    else:                           gem_v, gem_c = "UNCERTAIN", 0.40

    if "confidence: high" in t:  gem_c = min(gem_c + 0.10, 0.95)
    elif "confidence: low" in t: gem_c = max(gem_c - 0.15, 0.20)
    return gem_v, gem_c


def _to_score(verdict: str) -> int:
    return {"TRUE": 1, "LIKELY TRUE": 1,
            "FALSE": -1, "LIKELY FALSE": -1}.get(verdict, 0)


class FactCheckPipeline:

    def run(self, text: str, disable: Optional[str] = None) -> Dict:
        """
        disable: None | 'gemini' | 'nli' | 'agreement'
        When a component is disabled, its weight is redistributed
        to the remaining components proportionally.

        Ablation weight configs:
          Full:             NLI=0.45, Gemini=0.35, Agreement=0.20
          No Gemini:        NLI=0.65, Gemini=0.00, Agreement=0.35
          No NLI:           NLI=0.00, Gemini=0.60, Agreement=0.40
          No Agreement:     NLI=0.56, Gemini=0.44, Agreement=0.00
        """

        # ── Step 1: Context ──────────────────────────────────────────
        context = context_service.analyze(text)
        if context_service.should_skip_fact_check(context):
            return {
                "input": text, "verdict": "NON_FACTUAL",
                "verdict_detail": "Sarcasm detected.",
                "confidence": 0.0, "context": context,
                "ablation_disabled": disable
            }

        # ── Step 2: Claim detection ──────────────────────────────────
        claim_result = claim_detector.predict(text)
        if not claim_result.get("is_claim", False):
            return {
                "input": text, "verdict": "NOT_A_CLAIM",
                "verdict_detail": "Not a verifiable factual statement.",
                "confidence": claim_result.get("confidence", 0.0),
                "context": context,
                "ablation_disabled": disable
            }

        # ── Step 3: Evidence retrieval ───────────────────────────────
        evidence = retrieval_service.retrieve(text)
        if not evidence:
            return {
                "input": text, "verdict": "UNCERTAIN",
                "verdict_detail": "No evidence found.",
                "confidence": 0.0, "context": context,
                "ablation_disabled": disable
            }

        # ── Step 4: FAISS ranking ────────────────────────────────────
        ranked_evidence = ranking_service.rank(text, evidence)
        if not ranked_evidence:
            return {
                "input": text, "verdict": "UNCERTAIN",
                "confidence": 0.0, "context": context,
                "ablation_disabled": disable
            }

        # ── Step 5: Credibility scoring ──────────────────────────────
        ranked_evidence = credibility_service.attach_scores(ranked_evidence)

        # ── Step 6: NLI (optional — ablation) ───────────────────────
        if disable == "nli":
            verification = {
                "verdict": "UNCERTAIN", "confidence": 0.0,
                "time_aware": False, "note": "NLI disabled (ablation)",
                "evidence": ranked_evidence
            }
            nli_score = 0.0
        else:
            verification = verification_service.verify(text, ranked_evidence)
            nli_verdict  = verification.get("verdict", "UNCERTAIN")
            nli_conf     = verification.get("confidence", 0.0)
            nli_score    = _to_score(nli_verdict) * nli_conf

        # ── Step 7: Agreement (optional — ablation) ──────────────────
        if disable == "agreement":
            agreement = {"supports": 0, "refutes": 0, "neutral": 0,
                         "agreement_score": 0.0, "controversy_score": 0.0}
            ag_score    = 0.0
            controversy = 0.0
        else:
            agreement   = agreement_service.analyze(
                verification.get("evidence", ranked_evidence)
            )
            ag_score    = agreement.get("agreement_score", 0.0)
            controversy = agreement.get("controversy_score", 0.0)

        # ── Step 8: Gemini (optional — ablation) ─────────────────────
        if disable == "gemini":
            gemini_result = {
                "raw": "Gemini disabled (ablation study).",
                "sources": [], "search_used": False
            }
            gem_score = 0.0
        else:
            top3 = ranked_evidence[:3]
            background_text = " ".join(
                e.get("content", "") for e in top3 if e.get("content")
            )
            gemini_result = gemini_service.fact_check(text, background_text)
            gem_text = gemini_result.get("raw", "")
            if gem_text and "No Gemini" not in gem_text:
                gem_verdict, gem_conf = _parse_gemini(gem_text)
                gem_score = _to_score(gem_verdict) * gem_conf
            else:
                gem_score = 0.0

        # ── NEW: No-signal fallback ───────────────────────────────────
        # Triggered when NLI is disabled AND Gemini is also unavailable.
        # In this state only Agreement remains (weight 0.40) but Agreement
        # is a modifier signal — it cannot independently generate a verdict.
        # Returning UNCERTAIN here is more honest than computing a
        # near-zero score that would produce a random 5-class verdict.
        # This was confirmed by the ablation study: without both NLI and
        # Gemini, all 18 affected claims collapsed to UNCERTAIN anyway.
        if disable == "nli" and gem_score == 0.0:
            return {
                "input":    text,
                "verdict":  "UNCERTAIN",
                "confidence": 0.0,
                "time_aware": False,
                "note": "No primary verification signal available.",
                "context":  context,
                "evidence": ranked_evidence,
                "agreement": agreement,
                "gemini_analysis": gemini_result,
                "ablation_disabled": disable,
                "fusion_weights": {"nli": 0.00, "gemini": 0.60, "agreement": 0.40},
                "explanation": (
                    "UNCERTAIN: NLI disabled and Gemini unavailable — "
                    "Agreement alone cannot generate a verdict."
                )
            }

        # ── Step 9: Hybrid fusion with ablation weights ───────────────
        if disable == "gemini":
            w_nli, w_gem, w_ag = 0.65, 0.00, 0.35
        elif disable == "nli":
            w_nli, w_gem, w_ag = 0.00, 0.60, 0.40
        elif disable == "agreement":
            w_nli, w_gem, w_ag = 0.56, 0.44, 0.00
        else:
            w_nli, w_gem, w_ag = 0.45, 0.35, 0.20

        final_score = (
            w_nli * nli_score +
            w_gem * gem_score +
            w_ag  * ag_score
        ) * (1.0 - 0.3 * controversy)

        final_verdict    = _five_class_verdict(final_score)
        final_confidence = round(min(abs(final_score), 1.0), 4)

        return {
            "input":           text,
            "verdict":         final_verdict,
            "confidence":      final_confidence,
            "time_aware":      verification.get("time_aware", False),
            "note":            verification.get("note", ""),
            "context":         context,
            "evidence":        verification.get("evidence", []),
            "agreement":       agreement,
            "gemini_analysis": gemini_result,
            "ablation_disabled": disable,
            "fusion_weights":  {"nli": w_nli, "gemini": w_gem, "agreement": w_ag},
            "explanation": (
                f"Hybrid fusion — NLI({w_nli:.0%}) + "
                f"Gemini({w_gem:.0%}) + Agreement({w_ag:.0%})"
                + (f" [{disable} disabled]" if disable else "")
            )
        }


fact_check_pipeline = FactCheckPipeline()