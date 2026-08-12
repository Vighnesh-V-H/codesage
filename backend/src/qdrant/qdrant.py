from __future__ import annotations

import json
import logging
import uuid
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

            try:
                json.dumps(payload)
            except TypeError:
                logger.exception(
                    "Payload for point %d is not JSON serializable.",
                    idx,
                )
                raise

            point_id = self._coerce_point_id(
                metadata.get("hash") or metadata.get("content_hash"),
                idx,
            )
            point = PointStruct(
                id=point_id,
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

    @staticmethod
    def _coerce_point_id(raw_id, fallback_idx):
        if isinstance(raw_id, int) and 0 <= raw_id < 2**64:
            return raw_id

        if isinstance(raw_id, str):
            normalized = raw_id.strip()
            if not normalized:
                return fallback_idx
            try:
                return str(uuid.UUID(normalized))
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_DNS, normalized))

        return fallback_idx

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

            batches = self._split_upsert_batches(points)
            for batch_index, batch in enumerate(batches, start=1):
                logger.info(
                    "Upserting Qdrant batch %d/%d (%d points).",
                    batch_index,
                    len(batches),
                    len(batch),
                )
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
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

    @staticmethod
    def _point_payload_size(point: PointStruct) -> int:
        payload = point.payload or {}
        if hasattr(point, "model_dump"):
            data = point.model_dump(exclude_none=True, mode="json")
        elif hasattr(point, "dict"):
            data = point.dict(exclude_none=True)
        else:
            data = {"id": point.id, "vector": point.vector, "payload": payload}
        return len(json.dumps({"id": point.id, "vector": point.vector, "payload": payload}, separators=(",", ":")).encode("utf-8"))

    @classmethod
    def _split_upsert_batches(cls, points: list[PointStruct], max_payload_bytes: int = 24 * 1024 * 1024) -> list[list[PointStruct]]:
        batches: list[list[PointStruct]] = []
        current: list[PointStruct] = []
        current_size = 0

        for point in points:
            point_size = cls._point_payload_size(point)
            if current and current_size + point_size > max_payload_bytes:
                batches.append(current)
                current = []
                current_size = 0

            current.append(point)
            current_size += point_size

        if current:
            batches.append(current)

        return batches

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
