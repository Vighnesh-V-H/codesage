from types import SimpleNamespace

from src.chunking import limits
from src.embed.embedder import EmbeddingService


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
