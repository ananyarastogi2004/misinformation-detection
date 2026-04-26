import time
import requests
from typing import Dict
from app.config import settings


class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        )
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt: str, use_search: bool = True,
                 retries: int = 3) -> Dict:
        payload: Dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 500,
            }
        }
        if use_search:
            payload["tools"] = [{"google_search": {}}]

        # Retry with exponential backoff for 503 (model overloaded)
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=30
                )

                print(f"Gemini status: {response.status_code}"
                      f"{' (attempt ' + str(attempt+1) + ')' if attempt else ''}")

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return {"text": "", "sources": []}

                    parts = candidates[0]["content"].get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()

                    sources = []
                    grounding = candidates[0].get("groundingMetadata", {})
                    for chunk in grounding.get("groundingChunks", []):
                        web = chunk.get("web", {})
                        if web.get("uri"):
                            sources.append({
                                "title": web.get("title", ""),
                                "url":   web.get("uri", "")
                            })

                    return {"text": text, "sources": sources}

                # 503 = model overloaded → wait and retry
                if response.status_code == 503:
                    wait = 2 ** attempt          # 1s, 2s, 4s
                    print(f"⏳ Gemini overloaded, retrying in {wait}s…")
                    time.sleep(wait)
                    continue

                # 429 = quota hit → wait longer
                if response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"⏳ Gemini quota hit, retrying in {wait}s…")
                    time.sleep(wait)
                    continue

                print("❌ Gemini error:", response.text[:300])
                return {"text": "", "sources": [], "error": response.text}

            except requests.exceptions.Timeout:
                print(f"⏳ Gemini timeout (attempt {attempt+1})")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return {"text": "", "sources": [], "error": "Timeout"}
            except Exception as e:
                print("❌ Gemini exception:", str(e))
                return {"text": "", "sources": [], "error": str(e)}

        return {"text": "", "sources": [], "error": "Max retries exceeded"}

    def fact_check(self, claim: str, background_evidence: str) -> Dict:
        prompt = f"""You are a professional fact-checking assistant with access to Google Search.

CLAIM TO VERIFY:
"{claim}"

BACKGROUND EVIDENCE (already retrieved — may be outdated):
{background_evidence}

IMPORTANT: Some background evidence may be from 2022 or earlier. If the claim
relates to something that changes over time (population rankings, records,
leadership positions), prioritise the most RECENT sources from your search.

INSTRUCTIONS:
1. Search Google for the most recent and authoritative sources on this claim.
2. For claims about India, prioritise: PTI, NDTV, Hindustan Times, The Hindu, India Today.
3. For records or rankings, check the most recent UN/government data.
4. Note if the claim was false at some past point but became true later.

OUTPUT FORMAT (strict — complete all three lines):
Verdict: TRUE | FALSE | UNCERTAIN
Confidence: HIGH | MEDIUM | LOW
Reason: [2-3 sentences citing specific sources and key facts]
"""

        result = self.generate(prompt, use_search=True)

        if not result.get("text"):
            # Fallback without search
            fallback_prompt = (
                f'Fact-check this claim based only on the provided evidence.\n'
                f'Claim: "{claim}"\n'
                f'Evidence: {background_evidence}\n\n'
                f'Note: Evidence may be from 2022 or earlier — apply caution.\n\n'
                f'Output:\nVerdict: TRUE|FALSE|UNCERTAIN\n'
                f'Confidence: HIGH|MEDIUM|LOW\nReason: ...'
            )
            fallback = self.generate(fallback_prompt, use_search=False)
            return {
                "raw": fallback.get("text", "No Gemini analysis available."),
                "sources": [],
                "search_used": False
            }

        return {
            "raw": result["text"],
            "sources": result.get("sources", []),
            "search_used": True
        }


gemini_service = GeminiService()