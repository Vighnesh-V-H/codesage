from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalCandidate:
    id: str | int
    text: str
    metadata: dict[str, object]
    semantic_score: float
    lexical_score: float
    structural_score: float
    combined_score: float


class HybridRetriever:
    def __init__(
        self,
        lexical_weight: float = 0.35,
        structural_weight: float = 0.15,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.structural_weight = structural_weight

    def rank(self, query: str, points: list[object]) -> list[RetrievalCandidate]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [
                RetrievalCandidate(
                    id=getattr(point, "id", 0),
                    text=str((point.payload or {}).get("text", "")),
                    metadata={k: v for k, v in (point.payload or {}).items() if k != "text"},
                    semantic_score=float(getattr(point, "score", 0.0) or 0.0),
                    lexical_score=0.0,
                    structural_score=0.0,
                    combined_score=float(getattr(point, "score", 0.0) or 0.0),
                )
                for point in points
            ]

        ranked: list[RetrievalCandidate] = []
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            metadata = {k: v for k, v in payload.items() if k != "text"}
            semantic_score = float(getattr(point, "score", 0.0) or 0.0)
            lexical_score = self._lexical_score(query_tokens, text, metadata)
            structural_score = self._structural_score(query_tokens, metadata)
            combined_score = (
                0.45 * semantic_score
                + 1.1 * lexical_score
                + 0.7 * structural_score
            )
            ranked.append(
                RetrievalCandidate(
                    id=getattr(point, "id", 0),
                    text=text,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    structural_score=structural_score,
                    combined_score=combined_score,
                )
            )

        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        return ranked

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(token) >= 2
        ]

    def _lexical_score(
        self,
        query_tokens: list[str],
        text: str,
        metadata: dict[str, object],
    ) -> float:
        haystacks = [text]
        for value in metadata.values():
            if isinstance(value, str):
                haystacks.append(value)
        flattened = self._tokenize(" ".join(haystacks))
        if not flattened:
            return 0.0

        counts = Counter(flattened)
        unique_query = set(query_tokens)
        overlap = sum(counts.get(token, 0) for token in unique_query)
        return min(1.0, overlap / max(len(unique_query), 1))

    def _structural_score(
        self,
        query_tokens: list[str],
        metadata: dict[str, object],
    ) -> float:
        values: list[str] = []
        for key in ("symbol", "file_path", "node_type", "parent", "language"):
            value = metadata.get(key)
            if isinstance(value, str):
                values.append(value)

        if not values:
            return 0.0

        joined = self._tokenize(" ".join(values))
        if not joined:
            return 0.0

        unique_query = set(query_tokens)
        score = 0.0
        for token in unique_query:
            if token in set(joined):
                score += 1.0
        return min(1.0, score / max(len(unique_query), 1))
