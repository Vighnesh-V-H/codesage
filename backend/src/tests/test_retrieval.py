from types import SimpleNamespace

from src.retrieval.rag import RAGPipeline
from src.retrieval.retrieval import HybridRetriever


def test_hybrid_retriever_prefers_lexical_matches():
    retriever = HybridRetriever()
    points = [
        SimpleNamespace(
            id=1,
            score=0.2,
            payload={
                "text": "user login authentication flow",
                "symbol": "login",
                "file_path": "src/auth.py",
                "node_type": "function",
            },
        ),
        SimpleNamespace(
            id=2,
            score=0.6,
            payload={
                "text": "database connection setup",
                "symbol": "connect",
                "file_path": "src/db.py",
                "node_type": "function",
            },
        ),
    ]

    ranked = retriever.rank("login authentication", points)

    assert ranked[0].id == 1
    assert ranked[0].lexical_score >= ranked[1].lexical_score


def test_rag_pipeline_builds_context_from_retrieved_snippets():
    class FakeStore:
        def retrieve_results(self, query, collection_name, top_k, model=None):
            return {
                "retrieved_texts": [
                    "def login_user():\n    return True",
                    "def connect_db():\n    return db",
                ],
                "retrieved_metadata": [
                    {"file_path": "src/auth.py", "symbol": "login_user"},
                    {"file_path": "src/db.py", "symbol": "connect_db"},
                ],
            }

    pipeline = RAGPipeline(FakeStore())
    result = pipeline.run("repo-demo", "login user", top_k=2)

    assert result["query"] == "login user"
    assert "src/auth.py::login_user" in result["context"]
    assert "def login_user" in result["context"]


def test_vector_store_uses_hybrid_ranking_for_results():
    class FakeEmbeddingService:
        def get_embeddings(self, query, model=None, input_type="query"):
            return [0.1, 0.2, 0.3]

    class FakeQdrantService:
        def query_points(self, collection_name, query_vector, top_k):
            return [
                SimpleNamespace(
                    id=1,
                    score=0.15,
                    payload={
                        "text": "authentication login module",
                        "symbol": "login",
                        "file_path": "src/auth.py",
                        "node_type": "function",
                    },
                ),
                SimpleNamespace(
                    id=2,
                    score=0.95,
                    payload={
                        "text": "database connection config",
                        "symbol": "connect",
                        "file_path": "src/db.py",
                        "node_type": "function",
                    },
                ),
            ]

    from src.services.vectore_store import VectorStore

    store = VectorStore(FakeEmbeddingService(), FakeQdrantService())
    result = store.retrieve_results("login user", "demo-collection", top_k=2)

    assert result["retrieved_ids"][0] == 1
    assert "lexical_scores" in result
    assert "combined_scores" in result
