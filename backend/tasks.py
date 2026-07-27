from pathlib import Path

from celery_app import celery_app

from src.core.config import config
from src.utils.repo import get_repo_id
from src.services.vectore_store import VectorStore
from src.qdrant.qdrant import QdrantService
from src.embed.embedder import EmbeddingService
from src.chunking.hash_chunk import dedup_stream
from src.chunking.chunker import ASTChunker
from src.services.language_detector import detect_language
from src.services.file_walker import walk_repository

from qdrant_client import QdrantClient
from openai import OpenAI


def _build_services():
    """Lazily build service objects for the worker process."""
    openai_client = OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.AI_API_BASE_URL,
        timeout=30.0,
    )
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    embedding_service = EmbeddingService(client=openai_client)
    qdrant_service = QdrantService(client=qdrant_client)
    vector_store = VectorStore(
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
    )
    chunker = ASTChunker()
    return vector_store, chunker


_services = None


def _get_services():
    global _services
    if _services is None:
        _services = _build_services()
    return _services


@celery_app.task(bind=True)
def embed_repo_task(self, folder_path: str):
    """Clone-free embedding: chunks, deduplicates, and stores vectors for a local folder."""
    vector_store, chunker = _get_services()

    repo_path = Path(folder_path)
    repo_id = get_repo_id(folder_path)

    all_chunks = []
    for file_path in walk_repository(repo_path):
        lang = detect_language(file_path)

        if lang.value == "unknown":
            continue
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        all_chunks.extend(chunker.chunk_file(
            source, lang.value, str(file_path)))

    deduped = list(dedup_stream(iter(all_chunks)))

    if not deduped:
        return {
            "repo_id": repo_id,
            "chunks_embedded": 0,
            "message": "No embeddable code chunks found.",
        }

    vector_store.store_embeddings(
        collection_name=repo_id,
        chunks=deduped,
        model=config.EMBEDDING_MODEL,
    )

    return {
        "repo_id": repo_id,
        "chunks_embedded": len(deduped),
        "message": f"Embedded {len(deduped)} chunks into collection '{repo_id}'.",
    }
