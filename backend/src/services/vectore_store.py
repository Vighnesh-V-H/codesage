from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from src.chunking.models import Chunk
from src.embed.embedder import EmbeddingService
from src.qdrant.qdrant import QdrantService
from src.retrieval.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.hybrid_retriever = HybridRetriever()

    def store_embeddings(
        self,
        collection_name: str,
        chunks: Iterable[Chunk],
        model: str | None = None,
    ) -> None:
        logger.info(
            "Starting embedding pipeline for collection '%s'.",
            collection_name,
        )

        chunk_list = list(chunks)
        chunk_count = len(chunk_list)

        logger.info(
            "Received %d chunks for embedding.",
            chunk_count,
        )

        if chunk_count == 0:
            logger.warning(
                "No chunks supplied. Skipping embedding pipeline."
            )
            return

        logger.info(
            "Generating embeddings using model '%s'.",
            model,
        )

        embeddings = self.embedding_service.get_embeddings(
            chunk_list,
            model=model,
            input_type="passage",
        )

        embedding_count = len(embeddings)

        logger.info(
            "Generated %d embeddings.",
            embedding_count,
        )

        if embedding_count == 0:
            logger.warning(
                "Embedding service returned no embeddings."
            )
            return

        if embedding_count != chunk_count:
            logger.warning(
                "Chunk count (%d) differs from embedding count (%d).",
                chunk_count,
                embedding_count,
            )

        vector_dimension = len(embeddings[0])

        logger.info(
            "Embedding vector dimension: %d",
            vector_dimension,
        )

        logger.info(
            "Ensuring Qdrant collection '%s' exists.",
            collection_name,
        )

        self.qdrant_service.create_collection(
            collection_name=collection_name,
            vector_size=vector_dimension,
        )

        logger.info(
            "Building Qdrant PointStruct objects."
        )

        points = self.qdrant_service.build_points(
            chunks=chunk_list,
            embeddings=embeddings,
        )

        logger.info(
            "Built %d PointStruct objects.",
            len(points),
        )

        logger.info(
            "Uploading points to Qdrant."
        )

        self.qdrant_service.upsert_points(
            collection_name=collection_name,
            points=points,
        )

        logger.info(
            "Embedding pipeline completed successfully for '%s'.",
            collection_name,
        )

    def retrieve_results(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        model: str | None = None,
    ) -> dict[str, list[Any]]:
        logger.info(
            "Starting retrieval from collection '%s'.",
            collection_name,
        )

        logger.info(
            "Embedding query using model '%s'.",
            model,
        )

        query_embedding = self.embedding_service.get_embeddings(
            query,
            model=model,
            input_type="query",
        )

        logger.info(
            "Query embedding dimension: %d",
            len(query_embedding),
        )

        logger.info(
            "Searching Qdrant (top_k=%d).",
            top_k,
        )

        results = self.qdrant_service.query_points(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=top_k,
        )

        ranked = self.hybrid_retriever.rank(query, results)

        retrieved_ids: list[Any] = []
        retrieved_texts: list[str] = []
        retrieved_metadata: list[dict[str, Any]] = []
        similarity_scores: list[float] = []
        lexical_scores: list[float] = []
        structural_scores: list[float] = []
        metadata_scores: list[float] = []
        combined_scores: list[float] = []

        for candidate in ranked[:top_k]:
            retrieved_ids.append(candidate.id)
            retrieved_texts.append(candidate.text)
            retrieved_metadata.append(candidate.metadata)
            similarity_scores.append(candidate.semantic_score)
            lexical_scores.append(candidate.lexical_score)
            structural_scores.append(candidate.structural_score)
            metadata_scores.append(candidate.metadata_score)
            combined_scores.append(candidate.combined_score)

        logger.info(
            "Successfully processed %d retrieved results.",
            len(retrieved_ids),
        )

        return {
            "retrieved_ids": retrieved_ids,
            "retrieved_texts": retrieved_texts,
            "retrieved_metadata": retrieved_metadata,
            "similarity_scores": similarity_scores,
            "lexical_scores": lexical_scores,
            "structural_scores": structural_scores,
            "metadata_scores": metadata_scores,
            "combined_scores": combined_scores,
        }
