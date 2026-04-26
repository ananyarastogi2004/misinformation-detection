import requests
import time
from typing import Dict, Any, Optional

from app.config import settings


class HTTPClient:
    def __init__(self):
        self.timeout = settings.REQUEST_TIMEOUT
        # BUG FIX: these headers are now actually passed into every request
        self.default_headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        # Wikipedia requires a User-Agent or it returns 403
        self.wiki_headers = {
            "User-Agent": "ClaimSense/2.0 (major-project fact-checker)"
        }

    # ─────────────────────────────────────────
    # POST  (HuggingFace inference endpoints)
    # ─────────────────────────────────────────
    def post(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        retries: int = 2
    ) -> Dict:
        # BUG FIX: merge default_headers with any caller-supplied headers
        merged = {**self.default_headers, **(headers or {})}

        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=merged,       # was never passed before
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()

                # HuggingFace cold-start: model loading
                if response.status_code == 503:
                    wait = int(response.headers.get("X-Wait-For-Model", 20))
                    print(f"⏳ HF model loading… retrying in {min(wait, 20)}s")
                    time.sleep(min(wait, 20))
                    continue

                print(f"❌ HTTP {response.status_code}:", response.text[:200])
                return {"error": f"HTTP {response.status_code}", "details": response.text}

            except requests.exceptions.Timeout:
                if attempt < retries:
                    time.sleep(2)
                    continue
                return {"error": "Request timeout"}
            except Exception as e:
                print("❌ HTTP Exception:", str(e))
                return {"error": str(e)}

        return {"error": "Max retries exceeded"}

    # ─────────────────────────────────────────
    # GET  (APIs, Wikipedia, NewsAPI, etc.)
    # ─────────────────────────────────────────
    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_wiki_agent: bool = False
    ) -> Dict:
        # BUG FIX: Wikipedia 403 - always send User-Agent on wiki calls
        base = self.wiki_headers if use_wiki_agent else {}
        merged = {**base, **(headers or {})}

        try:
            response = requests.get(
                url,
                headers=merged,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()

            print(f"❌ GET {response.status_code}:", response.text[:200])
            return {"error": f"HTTP {response.status_code}", "details": response.text}

        except Exception as e:
            print("❌ GET Exception:", str(e))
            return {"error": str(e)}


# Singleton
http_client = HTTPClient()
