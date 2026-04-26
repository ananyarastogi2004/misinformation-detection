from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RankingService:
    def __init__(self):
        # Load embedding model once
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = 384

    def _normalize(self, vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def rank(self, claim: str, evidence_list: List[Dict]) -> List[Dict]:
        """
        Rank evidence using FAISS + semantic similarity
        """

        if not evidence_list:
            return []

        texts = [item.get("content", "") for item in evidence_list]

        # Remove empty entries
        valid_pairs = [(i, t) for i, t in enumerate(texts) if t]
        if not valid_pairs:
            return []

        indices, valid_texts = zip(*valid_pairs)

        # Encode
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        embeddings = self._normalize(embeddings)

        # Build FAISS index
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        # Encode claim
        claim_vec = self.model.encode([claim], convert_to_numpy=True)
        claim_vec = self._normalize(claim_vec)

        # Search
        scores, idxs = index.search(claim_vec, k=min(5, len(valid_texts)))

        ranked_results = []

        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
            original_idx = indices[idx]
            item = evidence_list[original_idx]

            item["similarity_score"] = float(score)
            item["rank"] = rank + 1

            ranked_results.append(item)

        return ranked_results


# Singleton
ranking_service = RankingService()