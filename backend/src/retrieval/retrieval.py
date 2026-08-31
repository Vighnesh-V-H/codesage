from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

SYNONYM_MAP = {
    "auth": {"auth", "authentication", "login", "signin", "token"},
    "login": {"login", "signin", "auth", "authentication"},
    "db": {"db", "database", "storage", "connection", "sql"},
    "config": {"config", "settings", "environment", "options"},
    "api": {"api", "endpoint", "route", "handler"},
    "model": {"model", "schema", "entity", "type"},
}


@dataclass(slots=True)
class RetrievalCandidate:
    id: str | int
    text: str
    metadata: dict[str, object]
    semantic_score: float
    lexical_score: float
    structural_score: float
    metadata_score: float
    combined_score: float


class HybridRetriever:
    def __init__(
        self,
        semantic_weight: float = 0.2,
        lexical_weight: float = 0.7,
        structural_weight: float = 0.08,
        metadata_weight: float = 0.02,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.structural_weight = structural_weight
        self.metadata_weight = metadata_weight

    def rank(self, query: str, points: list[object]) -> list[RetrievalCandidate]:
        expanded_tokens = self._expand_query_tokens(query)
        if not expanded_tokens:
            return [
                RetrievalCandidate(
                    id=getattr(point, "id", 0),
                    text=str((point.payload or {}).get("text", "")),
                    metadata={
                        k: v
                        for k, v in (point.payload or {}).items()
                        if k != "text"
                    },
                    semantic_score=float(getattr(point, "score", 0.0) or 0.0),
                    lexical_score=0.0,
                    structural_score=0.0,
                    metadata_score=0.0,
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
            lexical_score = self._lexical_score(expanded_tokens, text, metadata)
            structural_score = self._structural_score(expanded_tokens, metadata)
            metadata_score = self._metadata_score(expanded_tokens, metadata)
            combined_score = (
                self.semantic_weight * semantic_score
                + self.lexical_weight * lexical_score
                + self.structural_weight * structural_score
                + self.metadata_weight * metadata_score
            )
            ranked.append(
                RetrievalCandidate(
                    id=getattr(point, "id", 0),
                    text=text,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    structural_score=structural_score,
                    metadata_score=metadata_score,
                    combined_score=combined_score,
                )
            )

        ranked = self._reciprocal_rank_fusion(ranked)
        ranked.sort(key=lambda item: item.combined_score, reverse=True)
        return ranked

    def _expand_query_tokens(self, query: str) -> list[str]:
        tokens = self._tokenize(query)
        expanded = set(tokens)
        for token in list(tokens):
            for alias, variants in SYNONYM_MAP.items():
                if token == alias or token in variants:
                    expanded.update(variants)
                    expanded.add(alias)
        return sorted(expanded)

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

    def _metadata_score(
        self,
        query_tokens: list[str],
        metadata: dict[str, object],
    ) -> float:
        text_values: list[str] = []
        for key in ("file_path", "symbol", "parent"):
            value = metadata.get(key)
            if isinstance(value, str):
                text_values.append(value)

        if not text_values:
            return 0.0

        flattened = self._tokenize(" ".join(text_values))
        if not flattened:
            return 0.0

        unique_query = set(query_tokens)
        overlap = sum(1 for token in unique_query if token in set(flattened))
        return min(1.0, overlap / max(len(unique_query), 1))

    def _reciprocal_rank_fusion(
        self,
        ranked: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        semantic_order = sorted(
            ranked,
            key=lambda item: item.semantic_score,
            reverse=True,
        )
        lexical_order = sorted(
            ranked,
            key=lambda item: item.lexical_score,
            reverse=True,
        )
        structural_order = sorted(
            ranked,
            key=lambda item: item.structural_score,
            reverse=True,
        )

        ranks = {candidate.id: 0 for candidate in ranked}
        for rank, candidate in enumerate(semantic_order, start=1):
            ranks[candidate.id] += 1 / (60 + rank)
        for rank, candidate in enumerate(lexical_order, start=1):
            ranks[candidate.id] += 1 / (60 + rank)
        for rank, candidate in enumerate(structural_order, start=1):
            ranks[candidate.id] += 1 / (60 + rank)

        for candidate in ranked:
            candidate.combined_score = (
                self.semantic_weight * candidate.semantic_score
                + self.lexical_weight * candidate.lexical_score
                + self.structural_weight * candidate.structural_score
                + self.metadata_weight * candidate.metadata_score
                + ranks[candidate.id]
            )
        return ranked
