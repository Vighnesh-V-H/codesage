from __future__ import annotations

from typing import Literal

from openai import OpenAI

from src.chunking.models import Chunk
from src.core.config import config


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
        """
        Generate embeddings for one or more chunks or plain strings.
        """

        is_single = isinstance(chunks, (str, Chunk))
        chunk_list = [chunks] if is_single else chunks
        resolved_model = model or config.EMBEDDING_MODEL

        texts = [
            c.text if isinstance(c, Chunk) else c
            for c in chunk_list
        ]

        embeddings = self._request_embeddings(
            texts=texts,
            input_type=input_type,
            model=resolved_model,
            dimensions=dimensions,
        )

        return embeddings[0] if is_single else embeddings

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
                "truncate": "NONE",
            },
        )
        return [item.embedding for item in response.data]
