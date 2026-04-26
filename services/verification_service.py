from typing import List, Dict
from datetime import datetime, timezone

from models.nli_model import nli_model


class VerificationService:
    def __init__(self):
        pass

    def verify(self, claim: str, evidence_list: List[Dict]) -> Dict:
        results = []
        now = datetime.now(timezone.utc)

        for item in evidence_list:
            evidence_text = item.get("content", "")
            evidence_date = item.get("date")
            source_type   = item.get("type", "news")

            if not evidence_text:
                continue

            nli_result = nli_model.predict(claim, evidence_text)

            # ── Temporal weighting ──────────────────────────────────────
            time_weight = 1.0

            if evidence_date:
                if evidence_date.tzinfo is None:
                    evidence_date = evidence_date.replace(tzinfo=timezone.utc)
                age_days = (now - evidence_date).days

                if age_days < 30:
                    time_weight = 1.2      # very recent: boost
                elif age_days < 365:
                    time_weight = 1.0
                elif age_days < 730:
                    time_weight = 0.80
                else:
                    time_weight = 0.60     # 2+ years old: significant penalty
            else:
                # FIX: no date means we cannot verify recency.
                # Fact-check articles from 2022 with no date field were getting
                # full weight (1.0) even though they may be outdated.
                # Google Fact Check API entries without dates get 0.75 —
                # they're authoritative but we can't confirm they're current.
                if source_type == "google_fact_check":
                    time_weight = 0.75
                else:
                    time_weight = 0.85

            # ── Source type weight ──────────────────────────────────────
            source_weight = item.get("weight", 1.0)

            # ── Semantic similarity weight ──────────────────────────────
            similarity     = item.get("similarity_score", 0)
            sim_weight     = 0.5 + similarity   # range 0.5 – 1.5

            # ── Combined weighted confidence ────────────────────────────
            weighted_conf = (
                nli_result["confidence"]
                * time_weight
                * source_weight
                * sim_weight
            )
            weighted_conf = min(weighted_conf, 1.0)

            results.append({
                "source":           item.get("source"),
                "title":            item.get("title"),
                "content":          evidence_text,
                "url":              item.get("url"),
                "date":             evidence_date,
                "similarity_score": similarity,
                "nli_label":        nli_result["label"],
                "confidence":       round(weighted_conf, 4),
                "time_weight":      round(time_weight, 2),   # visible in output for debugging
            })

        if not results:
            return {"verdict": "UNCERTAIN", "confidence": 0.0, "evidence": []}

        results_sorted = sorted(results, key=lambda x: x["confidence"], reverse=True)
        top_k = results_sorted[:3]

        supports_score = sum(r["confidence"] for r in top_k if r["nli_label"] == "supports")
        refutes_score  = sum(r["confidence"] for r in top_k if r["nli_label"] == "refutes")
        avg_confidence = sum(r["confidence"] for r in top_k) / len(top_k)

        if supports_score > refutes_score:
            verdict = "TRUE"
        elif refutes_score > supports_score:
            verdict = "FALSE"
        else:
            verdict = "UNCERTAIN"

        return {
            "verdict":    verdict,
            "confidence": round(avg_confidence, 4),
            "time_aware": True,
            "note":       "Verdict uses time, source credibility, and semantic relevance",
            "evidence":   results_sorted
        }


verification_service = VerificationService()