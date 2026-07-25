from __future__ import annotations

from openai import OpenAI

from chunking.models import Chunk
from core.config import config


class EmbeddingService:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def get_embeddings(
        self,
        chunks: Chunk | list[Chunk],
        model: str = config.EMBEDDING_MODEL,
        dimensions: int = config.EMBEDDING_DIMENSIONS,
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings for one or more chunks.
        """

        is_single = isinstance(chunks, Chunk)
        chunk_list = [chunks] if is_single else chunks

        texts = [chunk.text for chunk in chunk_list]

        response = self.client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "NONE",
            },
        )

        embeddings = [item.embedding for item in response.data]

        return embeddings[0] if is_single else embeddings
