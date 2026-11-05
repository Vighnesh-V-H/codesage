from types import SimpleNamespace

from src.services.vectore_store import VectorStore


class FakeEmbeddingService:
    def get_embeddings(self, *args, **kwargs):
        return [0.1, 0.2]


class FakeQdrantService:
    def __init__(self, points):
        self.points = points
        self.scroll_calls = 0

    def query_points(self, **kwargs):
        return []

    def scroll_points(self, collection_name, limit, offset):
        self.scroll_calls += 1
        if offset is not None:
            return [], None
        return self.points, None


class FakeReranker:
    def __init__(self):
        self.inputs = []

    def predict(self, pairs):
        self.inputs = pairs
        return list(range(len(pairs)))


def _record(point_id, text):
    # Scroll records intentionally have no `score` attribute.
    return SimpleNamespace(id=point_id, payload={"text": text, "path": f"{point_id}.py"})


def test_hybrid_retrieval_returns_lexical_records_and_reranker_scores():
    qdrant = FakeQdrantService(
        [_record("match", "function handles unique token"), _record("other", "unrelated contents")]
    )
    reranker = FakeReranker()
    store = VectorStore(FakeEmbeddingService(), qdrant, reranker=reranker)

    results = store.retrieve_results("unique token", "repo", top_k=5)

    assert results["retrieved_ids"] == ["match"]
    assert results["similarity_scores"] == [0.0]
    assert [pair[1] for pair in reranker.inputs] == ["function handles unique token"]


def test_top_k_above_default_rerank_limit_is_honored():
    points = [_record(str(index), f"query matching document {index}") for index in range(60)]
    reranker = FakeReranker()
    store = VectorStore(FakeEmbeddingService(), FakeQdrantService(points), reranker=reranker)

    results = store.retrieve_results("query matching", "repo", top_k=60)

    assert len(results["retrieved_ids"]) == 60


def test_lexical_corpus_and_bm25_index_are_cached_then_refreshed():
    qdrant = FakeQdrantService([_record("match", "query match")])
    store = VectorStore(FakeEmbeddingService(), qdrant, reranker=FakeReranker())

    store.retrieve_results("query", "repo")
    store.retrieve_results("query", "repo")
    assert qdrant.scroll_calls == 1

    cached = store._lexical_corpus["repo"]
    store._lexical_corpus["repo"] = (0.0, cached[1], cached[2], cached[3])
    store.retrieve_results("query", "repo")
    assert qdrant.scroll_calls == 2
