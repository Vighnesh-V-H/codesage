from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from src.chunking.models import Chunk

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def create_collection(
        self,
        collection_name: str,
        vector_size: int | None = None,
    ) -> None:
        logger.info(
            "Checking collection '%s'.",
            collection_name,
        )

        exists = self.client.collection_exists(collection_name=collection_name)

        if exists:
            logger.info(
                "Collection '%s' already exists.",
                collection_name,
            )
            return

        if vector_size is None:
            logger.error(
                "Cannot create collection '%s': vector_size is None.",
                collection_name,
            )
            raise ValueError("vector_size cannot be None")

        logger.info(
            "Creating collection '%s' (vector_size=%d).",
            collection_name,
            vector_size,
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            on_disk_payload=True,
        )

        logger.info(
            "Collection '%s' created successfully.",
            collection_name,
        )

    def build_points(
        self,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
    ) -> list[PointStruct]:
        logger.info("Building Qdrant points.")

        points: list[PointStruct] = []

        embedding_count = len(embeddings)
        logger.info("Received %d embeddings.", embedding_count)

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = dict(chunk.metadata)

            payload = {
                "text": chunk.text,
                **metadata,
            }

            # Verify payload can be serialized before sending to Qdrant.
            try:
                json.dumps(payload)
            except TypeError:
                logger.exception(
                    "Payload for point %d is not JSON serializable.",
                    idx,
                )
                raise

            point = PointStruct(
                id=metadata.get("hash", idx),
                vector=embedding,
                payload=payload,
            )

            if idx == 0:
                logger.info("First point summary:")
                logger.info("  id=%s", point.id)
                logger.info("  vector_dimension=%d", len(embedding))
                logger.info(
                    "  payload_keys=%s",
                    sorted(payload.keys()),
                )

            points.append(point)

        logger.info(
            "Built %d Qdrant points.",
            len(points),
        )

        if len(points) != embedding_count:
            logger.warning(
                "Built %d points from %d embeddings. "
                "The chunks iterable may have produced fewer items.",
                len(points),
                embedding_count,
            )

        return points

    def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> None:
        logger.info(
            "Starting upsert into '%s'.",
            collection_name,
        )

        point_count = len(points)

        logger.info(
            "Points to insert: %d",
            point_count,
        )

        if point_count == 0:
            logger.warning(
                "No points supplied. Skipping upsert."
            )
            return

        first_point = points[0]

        try:
            collection = self.client.get_collection(collection_name)

            logger.info(
                "Fetched collection metadata for '%s'.",
                collection_name,
            )

            try:
                expected_dimension = collection.config.params.vectors.size

                logger.info(
                    "Collection vector dimension: %s",
                    expected_dimension,
                )

                logger.info(
                    "Embedding vector dimension: %d",
                    len(first_point.vector),
                )

                if expected_dimension != len(first_point.vector):
                    logger.error(
                        "Vector dimension mismatch. "
                        "Collection=%s Embedding=%d",
                        expected_dimension,
                        len(first_point.vector),
                    )
            except AttributeError:
                logger.warning(
                    "Unable to inspect collection vector configuration. "
                    "This may be due to a different qdrant-client version."
                )

            logger.info(
                "First point id: %s",
                first_point.id,
            )

            logger.info(
                "First point payload keys: %s",
                sorted(first_point.payload.keys()),
            )

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

            logger.info(
                "Successfully upserted %d points into '%s'.",
                point_count,
                collection_name,
            )

        except UnexpectedResponse as exc:
            logger.exception(
                "Qdrant rejected the upsert request."
            )

            logger.error(
                "HTTP status: %s",
                getattr(exc, "status_code", "unknown"),
            )

            logger.error(
                "Response body: %s",
                getattr(exc, "content", "unavailable"),
            )

            raise

        except Exception:
            logger.exception(
                "Unexpected error during Qdrant upsert."
            )
            raise

    def query_points(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[ScoredPoint]:
        logger.info(
            "Querying '%s' (top_k=%d, vector_dimension=%d).",
            collection_name,
            top_k,
            len(query_vector),
        )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )

        logger.info(
            "Query returned %d points.",
            len(response.points),
        )

        return response.points
