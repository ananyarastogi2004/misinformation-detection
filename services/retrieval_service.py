from typing import List, Dict
from datetime import datetime, timezone

from utils.http_client import http_client
from app.config import settings


class RetrievalService:
    def __init__(self):
        self.news_api_key = settings.NEWS_API_KEY
        self.google_api_key = settings.GOOGLE_FACT_CHECK_API_KEY

    def _parse_date(self, date_str: str):
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def fetch_google_fact_check(self, query: str) -> List[Dict]:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": query, "key": self.google_api_key, "pageSize": 5}

        response = http_client.get(url, params=params)
        if "error" in response:
            return []

        results = []
        for item in response.get("claims", []):
            claim_text = item.get("text", "")
            for review in item.get("claimReview", []):
                rating = review.get("textualRating", "").strip()

                # FIX: build a content string that gives the NLI model
                # the fact-checker's verdict, not just the raw claim text.
                # Previously: content = claim_text  (NLI only saw the claim)
                # Now: content tells NLI what the fact-checker concluded.
                if rating:
                    content = f"{claim_text}. Fact-checker verdict: {rating}."
                else:
                    content = claim_text

                results.append({
                    "source":  review.get("publisher", {}).get("name"),
                    "title":   review.get("title"),
                    "content": content,
                    "url":     review.get("url"),
                    "date":    None,
                    "type":    "google_fact_check",
                    "weight":  1.2,
                    "rating":  rating,
                })
        return results

    def fetch_wikipedia(self, query: str) -> List[Dict]:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query", "list": "search",
            "srsearch": query, "format": "json", "srlimit": 3
        }

        search_response = http_client.get(
            search_url, params=search_params, use_wiki_agent=True
        )
        if "error" in search_response:
            return []

        results = []
        for item in search_response.get("query", {}).get("search", []):
            page_title = item.get("title", "").replace(" ", "_")
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
            summary = http_client.get(summary_url, use_wiki_agent=True)
            if "error" in summary:
                continue
            extract = summary.get("extract", "")
            if not extract:
                continue
            results.append({
                "source":  "Wikipedia",
                "title":   summary.get("title"),
                "content": extract[:500],
                "url":     summary.get("content_urls", {}).get("desktop", {}).get("page"),
                "date":    None,
                "type":    "wiki",
                "weight":  1.0,
            })
        return results

    def fetch_news(self, query: str) -> List[Dict]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query, "apiKey": self.news_api_key,
            "language": "en", "pageSize": 5, "sortBy": "relevancy"
        }

        response = http_client.get(url, params=params)
        if "error" in response:
            return []

        results = []
        for article in response.get("articles", []):
            results.append({
                "source":  article.get("source", {}).get("name"),
                "title":   article.get("title"),
                "content": article.get("description"),
                "url":     article.get("url"),
                "date":    self._parse_date(article.get("publishedAt")),
                "type":    "news",
                "weight":  0.8,
            })
        return results

    def retrieve(self, query: str) -> List[Dict]:
        google_results = self.fetch_google_fact_check(query)
        wiki_results   = self.fetch_wikipedia(query)
        news_results   = self.fetch_news(query)

        combined = google_results + wiki_results + news_results
        combined = [item for item in combined if item.get("content")]
        return combined[:10]


retrieval_service = RetrievalService()