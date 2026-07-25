from __future__ import annotations

from collections.abc import Iterable

from chunking.models import Chunk
from embed.embedder import EmbeddingService
from qdrant.qdrant import QdrantService


class VectorStore:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service

    def store_embeddings(
        self,
        collection_name: str,
        chunks: Iterable[Chunk],
    ) -> None:
        """
        Embed the chunks and store them in Qdrant.
        """

        chunk_list = list(chunks)

        if not chunk_list:
            return

        embeddings = self.embedding_service.get_embeddings(chunk_list)

        points = self.qdrant_service.build_points(
            chunks=chunk_list,
            embeddings=embeddings,
        )

        self.qdrant_service.upsert_points(
            collection_name=collection_name,
            points=points,
        )
