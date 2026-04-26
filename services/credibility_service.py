from typing import Dict, List


# ─────────────────────────────────────────────────────
# IMPROVED: expanded from 6 to 40+ sources
# Added Indian news sources (appeared in actual output)
# Added dedicated fact-check sites with higher scores
# Lowered unknown-source default from 0.6 → 0.35
# ─────────────────────────────────────────────────────

FACT_CHECK_SITES = {
    # Dedicated fact-checkers get the highest bonus
    "factly": 0.97,
    "altnews": 0.96,
    "boomlive": 0.95,
    "snopes": 0.95,
    "factcheck.org": 0.95,
    "politifact": 0.94,
    "fullfact": 0.94,
    "newschecker": 0.93,
    "vishvasnews": 0.90,
    "logically": 0.90,
    "lead stories": 0.89,
    "afp fact check": 0.92,
    "reuters fact check": 0.93,
    "ap fact check": 0.93,
}

INTERNATIONAL_SOURCES = {
    "reuters": 0.95,
    "associated press": 0.95,
    "ap news": 0.95,
    "bbc": 0.93,
    "the guardian": 0.90,
    "new york times": 0.90,
    "nyt": 0.90,
    "washington post": 0.89,
    "the economist": 0.91,
    "al jazeera": 0.87,
    "france 24": 0.86,
    "dw": 0.86,
    "abc news": 0.85,
    "nbc news": 0.85,
    "cnn": 0.82,
}

INDIAN_SOURCES = {
    # Added: these sources appeared in actual ClaimSense output
    "hindustan times": 0.85,
    "times of india": 0.84,
    "the hindu": 0.88,
    "ndtv": 0.85,
    "pti": 0.90,          # Press Trust of India — wire service
    "ani": 0.88,           # Asian News International — wire service
    "india today": 0.83,
    "the wire": 0.82,
    "scroll": 0.81,
    "the print": 0.82,
    "mint": 0.84,
    "business standard": 0.84,
    "economic times": 0.83,
    "deccan herald": 0.80,
    "tribune india": 0.79,
}

ENCYCLOPEDIC = {
    "wikipedia": 0.80,    # Useful but editable; treat as medium
    "britannica": 0.92,
}

# Any source not in the lists above gets this score
# CHANGED: was 0.6 (too generous for tabloids) → 0.35
DEFAULT_SCORE = 0.35
UNKNOWN_NEWS_SCORE = 0.50  # Generic news site, no strong signal either way


class CredibilityService:

    def __init__(self):
        # Merge all dicts into one lookup table (lowercase keys)
        self.lookup: Dict[str, float] = {}
        for d in [FACT_CHECK_SITES, INTERNATIONAL_SOURCES,
                  INDIAN_SOURCES, ENCYCLOPEDIC]:
            for k, v in d.items():
                self.lookup[k.lower()] = v

        # Keep the category sets for labelling
        self.fact_check_keys = set(FACT_CHECK_SITES.keys())

    def score(self, source: str) -> float:
        if not source:
            return DEFAULT_SCORE

        src = source.lower().strip()

        # Exact or substring match
        for key, val in self.lookup.items():
            if key in src:
                return val

        # Heuristic: if it ends in a news TLD pattern, give medium score
        if any(src.endswith(x) for x in [".in", ".co.in"]):
            return UNKNOWN_NEWS_SCORE

        return DEFAULT_SCORE

    def label(self, source: str) -> str:
        """Return a human-readable credibility tier."""
        s = self.score(source)
        if s >= 0.93:
            return "fact-checker"
        elif s >= 0.85:
            return "high"
        elif s >= 0.70:
            return "medium"
        elif s >= 0.50:
            return "low"
        return "unknown"

    def attach_scores(self, evidence_list: List[Dict]) -> List[Dict]:
        for item in evidence_list:
            src = item.get("source", "")
            item["credibility"] = self.score(src)
            item["credibility_label"] = self.label(src)
        return evidence_list


# Singleton
credibility_service = CredibilityService()
