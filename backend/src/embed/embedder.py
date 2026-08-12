from __future__ import annotations

from typing import Literal

from openai import OpenAI

from src.chunking.limits import MAX_CHUNK_TOKENS, split_text_by_tokens, token_count
from src.chunking.models import Chunk
from src.core.config import config

BATCH_TOKEN_BUDGET = 7500
BATCH_ITEM_LIMIT = 64


class EmbeddingService:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def get_embeddings(
        self,
        chunks: str | Chunk | list[str | Chunk],
        input_type: Literal["query", "passage"] = "passage",
        model: str | None = None,
        dimensions: int = 1536,
    ) -> list[float] | list[list[float]]:
        is_single = isinstance(chunks, (str, Chunk))
        chunk_list = [chunks] if is_single else chunks
        resolved_model = model or config.EMBEDDING_MODEL

        texts = [c.text if isinstance(c, Chunk) else c for c in chunk_list]
        texts = self._normalize_texts(texts)

        embeddings = self._request_embeddings_batched(
            texts=texts,
            input_type=input_type,
            model=resolved_model,
            dimensions=dimensions,
        )
        return embeddings[0] if is_single else embeddings

    def _normalize_texts(self, texts: list[str]) -> list[str]:
        normalized: list[str] = []
        for text in texts:
            if not text or not text.strip():
                continue
            if token_count(text) <= MAX_CHUNK_TOKENS:
                normalized.append(text)
                continue
            normalized.extend(split_text_by_tokens(
                text, max_tokens=MAX_CHUNK_TOKENS))
        return normalized

    def _request_embeddings_batched(
        self,
        texts: list[str],
        input_type: Literal["query", "passage"],
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for batch in self._make_batches(texts):
            all_embeddings.extend(
                self._request_embeddings(
                    texts=batch,
                    input_type=input_type,
                    model=model,
                    dimensions=dimensions,
                )
            )

        return all_embeddings

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            t = token_count(text)

            if t > MAX_CHUNK_TOKENS:
                for piece in split_text_by_tokens(text, max_tokens=MAX_CHUNK_TOKENS):
                    if not current:
                        current = [piece]
                        current_tokens = token_count(piece)
                        continue
                    if current_tokens + token_count(piece) > BATCH_TOKEN_BUDGET:
                        batches.append(current)
                        current = [piece]
                        current_tokens = token_count(piece)
                        continue
                    current.append(piece)
                    current_tokens += token_count(piece)
                continue

            exceeds_token_budget = current and (
                current_tokens + t > BATCH_TOKEN_BUDGET)
            exceeds_item_limit = len(current) >= BATCH_ITEM_LIMIT

            if exceeds_token_budget or exceeds_item_limit:
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(text)
            current_tokens += t

        if current:
            batches.append(current)

        return batches

    def _request_embeddings(
        self,
        texts: list[str],
        input_type: Literal["query", "passage"],
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            encoding_format="float",
            extra_body={
                "input_type": input_type,
                "truncate": "END",
            },
        )
        return [item.embedding for item in response.data]
