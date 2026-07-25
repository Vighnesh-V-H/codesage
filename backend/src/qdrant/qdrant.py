from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from chunking.models import Chunk
from core.config import config


class QdrantService:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = config.EMBEDDING_DIMENSIONS,
    ) -> None:
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            on_disk_payload=True,
        )

    def build_points(
        self,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
    ) -> list[PointStruct]:
        points: list[PointStruct] = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                    },
                )
            )

        return points

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )