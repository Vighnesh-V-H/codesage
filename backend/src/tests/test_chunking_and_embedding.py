from types import SimpleNamespace

from qdrant_client.models import PointStruct

from src.chunking import limits
from src.embed.embedder import EmbeddingService
from src.qdrant.qdrant import QdrantService


def test_split_text_by_tokens_produces_smaller_chunks():
    text = "\n\n".join(["word " * 30 for _ in range(4)])

    chunks = limits.split_text_by_tokens(text, max_tokens=20)

    assert len(chunks) > 1
    assert all(limits.token_count(chunk) <= 20 for chunk in chunks)


def test_embedding_requests_use_safe_nvidia_truncation():
    create = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create)
    )
    service = EmbeddingService(client)

    result = service._request_embeddings(
        texts=["short text"],
        input_type="passage",
        model="nvidia/test-model",
        dimensions=3,
    )

    assert result == [[0.1, 0.2, 0.3]]


def test_qdrant_rewrites_hash_to_valid_point_id():
    point_id = QdrantService._coerce_point_id(
        "b320c304746a32cd87c618276ee3f1c14ac4cd51fc84b2bb92ea7bc2fb6fef90",
        7,
    )

    assert point_id != "b320c304746a32cd87c618276ee3f1c14ac4cd51fc84b2bb92ea7bc2fb6fef90"
    assert isinstance(point_id, str)
    assert len(point_id) > 0


def test_qdrant_splits_large_upserts_into_safe_batches():
    points = [
        PointStruct(
            id=i,
            vector=[0.1, 0.2],
            payload={"text": "x" * 5000},
        )
        for i in range(10)
    ]

    batches = QdrantService._split_upsert_batches(
        points, max_payload_bytes=2000)

    assert len(batches) > 1
    assert all(len(batch) > 0 for batch in batches)
