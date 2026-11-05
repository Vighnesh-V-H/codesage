# from collections.abc import Iterable
# from typing import Any

# from src.chunking.models import Chunk
# from src.embed.embedder import EmbeddingService
# from src.qdrant.qdrant import QdrantService


# class VectorStore:
#     def __init__(
#         self,
#         embedding_service: EmbeddingService,
#         qdrant_service: QdrantService,
#     ) -> None:
#         self.embedding_service = embedding_service
#         self.qdrant_service = qdrant_service

#     def store_embeddings(
#         self,
#         collection_name: str,
#         chunks: Iterable[Chunk],
#         model: str | None = None,
#     ) -> None:
#         """
#         Embed the chunks and store them in Qdrant.
#         """

#         chunk_list = list(chunks)

#         if not chunk_list:
#             return

#         embeddings = self.embedding_service.get_embeddings(
#             chunk_list,
#             model=model,
#             input_type="passage",
#         )

#         self.qdrant_service.create_collection(
#             collection_name=collection_name,
#             vector_size=len(embeddings[0]),
#         )

#         points = self.qdrant_service.build_points(
#             chunks=chunk_list,
#             embeddings=embeddings,
#         )

#         self.qdrant_service.upsert_points(
#             collection_name=collection_name,
#             points=points,
#         )

#     def retrieve_results(
#         self,
#         query: str,
#         collection_name: str,
#         top_k: int = 5,
#         model: str | None = None,
#     ) -> dict[str, list[Any]]:
#         """
#         Retrieve nearest chunks and return chunk-aligned fields.
#         """
#         query_embedding = self.embedding_service.get_embeddings(
#             query,
#             model=model,
#             input_type="query",
#         )
#         results = self.qdrant_service.query_points(
#             collection_name=collection_name,
#             query_vector=query_embedding,
#             top_k=top_k,
#         )

#         retrieved_ids: list[Any] = []
#         retrieved_texts: list[str] = []
#         retrieved_metadata: list[dict[str, Any]] = []
#         similarity_scores: list[float] = []

#         for result in results:
#             payload = result.payload or {}
#             retrieved_ids.append(result.id)
#             retrieved_texts.append(str(payload.get("text", "")))
#             retrieved_metadata.append(
#                 {k: v for k, v in payload.items() if k != "text"}
#             )
#             similarity_scores.append(result.score)

#         return {
#             "retrieved_ids": retrieved_ids,
#             "retrieved_texts": retrieved_texts,
#             "retrieved_metadata": retrieved_metadata,
#             "similarity_scores": similarity_scores,
#         }

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.chunking.models import Chunk
from src.core.config import config
from src.embed.embedder import EmbeddingService
from src.qdrant.qdrant import QdrantService

logger = logging.getLogger(__name__)

_RERANK_CANDIDATE_LIMIT = 50
_LEXICAL_CACHE_TTL_SECONDS = 60
_SCROLL_PAGE_SIZE = 1000
_MAX_LEXICAL_DOCUMENTS = 50_000
_RRF_K = 60


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|[\u4e00-\u9fff]|\d+", text.lower())


class VectorStore:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        reranker: Any | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self._reranker = reranker
        self._lexical_corpus: dict[str, tuple[float, list[Any], list[list[str]], BM25Okapi | None]] = {}

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker

    def store_embeddings(
        self,
        collection_name: str,
        chunks: Iterable[Chunk],
        model: str | None = None,
    ) -> None:
        """
        Embed the chunks and store them in Qdrant.
        """
        logger.info(
            "Starting embedding pipeline for collection '%s'.",
            collection_name,
        )

        chunk_list = list(chunks)
        # Invalidate in-process lexical snapshots when this store updates a collection.
        self._lexical_corpus.pop(collection_name, None)

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
            raise ValueError(
                f"Embedding count ({embedding_count}) does not match "
                f"chunk count ({chunk_count}); refusing to misalign stored data"
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
        """
        Retrieve nearest chunks and return chunk-aligned fields.
        """
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

        # Dense candidates are fused with BM25 lexical rankings using RRF.
        candidate_limit = min(max(top_k * 10, _RERANK_CANDIDATE_LIMIT), 200)
        dense_results = self.qdrant_service.query_points(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=candidate_limit,
        )
        cached = self._lexical_corpus.get(collection_name)
        if cached is None or time.monotonic() - cached[0] >= _LEXICAL_CACHE_TTL_SECONDS:
            lexical_points = []
            offset = None
            while len(lexical_points) < _MAX_LEXICAL_DOCUMENTS:
                page, offset = self.qdrant_service.scroll_points(
                    collection_name,
                    limit=min(_SCROLL_PAGE_SIZE, _MAX_LEXICAL_DOCUMENTS - len(lexical_points)),
                    offset=offset,
                )
                lexical_points.extend(point for point in page if (point.payload or {}).get("text"))
                if offset is None or not page:
                    break
            tokenized_documents = [_tokens(str((point.payload or {}).get("text", ""))) for point in lexical_points]
            bm25 = BM25Okapi(tokenized_documents) if tokenized_documents else None
            cached = (time.monotonic(), lexical_points, tokenized_documents, bm25)
            self._lexical_corpus[collection_name] = cached
        _, lexical_points, tokenized_documents, bm25 = cached

        lexical_ranked = []
        query_tokens = _tokens(query)
        if bm25 is not None and query_tokens:
            lexical_scores = bm25.get_scores(query_tokens)
            query_terms = set(query_tokens)
            lexical_ranked = sorted(
                (
                    (point, float(score))
                    for index, (point, score) in enumerate(zip(lexical_points, lexical_scores))
                    if query_terms.intersection(tokenized_documents[index])
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:candidate_limit]

        fused: dict[Any, tuple[Any, float]] = {}
        for rank, point in enumerate(dense_results, start=1):
            fused[point.id] = (point, 1.0 / (_RRF_K + rank))
        for rank, (point, _) in enumerate(lexical_ranked, start=1):
            # Prefer the dense ScoredPoint on overlap; Qdrant scroll returns Records
            # without a similarity score, while both representations share the ID.
            existing = fused.get(point.id)
            if existing is None:
                fused[point.id] = (point, 1.0 / (_RRF_K + rank))
            else:
                fused[point.id] = (existing[0], existing[1] + 1.0 / (_RRF_K + rank))
        candidates = sorted(fused.values(), key=lambda item: item[1], reverse=True)[:max(top_k, _RERANK_CANDIDATE_LIMIT)]
        rerank_inputs = [[query, str((point.payload or {}).get("text", ""))] for point, _ in candidates]
        rerank_scores = self._get_reranker().predict(rerank_inputs) if rerank_inputs else []
        reranked = sorted(zip(candidates, rerank_scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        results = [item[0][0] for item in reranked]
        final_scores = {point.id: float(score) for (point, _), score in zip(candidates, rerank_scores)}

        logger.info("Hybrid retrieval fused %d candidates and returned %d reranked results.", len(fused), len(results))

        retrieved_ids: list[Any] = []
        retrieved_texts: list[str] = []
        retrieved_metadata: list[dict[str, Any]] = []
        similarity_scores: list[float] = []

        for result in results:
            payload = result.payload or {}

            retrieved_ids.append(result.id)
            retrieved_texts.append(str(payload.get("text", "")))
            retrieved_metadata.append(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                }
            )
            similarity_scores.append(final_scores[result.id])

        logger.info(
            "Successfully processed %d retrieved results.",
            len(retrieved_ids),
        )

        return {
            "retrieved_ids": retrieved_ids,
            "retrieved_texts": retrieved_texts,
            "retrieved_metadata": retrieved_metadata,
            "similarity_scores": similarity_scores,
        }
