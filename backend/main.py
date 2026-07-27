from src.errors.repo_clone_errors import RepoCloneError
from src.utils.repo import get_repo_id
from src.services.vectore_store import VectorStore
from src.qdrant.qdrant import QdrantService
from src.embed.embedder import EmbeddingService
from src.services.clone_repo import RepoCloneService
from src.core.config import config
from tasks import embed_repo_task
from qdrant_client import QdrantClient
from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path
from celery.exceptions import OperationalError


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


app = FastAPI(title="CodeSage API", version="0.1.0")

openai_client = OpenAI(
    api_key=config.EMBEDDING_API_KEY,
    base_url=config.AI_API_BASE_URL,
)
qdrant_client = QdrantClient(url=config.QDRANT_URL)

embedding_service = EmbeddingService(client=openai_client)
qdrant_service = QdrantService(client=qdrant_client)
vector_store = VectorStore(
    embedding_service=embedding_service,
    qdrant_service=qdrant_service,
)
clone_service = RepoCloneService()


class CloneRequest(BaseModel):
    repo_url: str


class CloneResponse(BaseModel):
    repo_id: str
    path: str
    message: str


class EmbedRequest(BaseModel):
    folder_path: str


class EmbedResponse(BaseModel):
    repo_id: str
    chunks_embedded: int
    message: str


class RetrieveRequest(BaseModel):
    repo_url: str
    query: str
    top_k: int = 5


@app.post("/clone", response_model=CloneResponse)
def clone_repo(req: CloneRequest):
    """Shallow-clone a public Git repository."""
    try:
        repo_path = clone_service.clone(req.repo_url)
    except RepoCloneError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CloneResponse(
        repo_id=get_repo_id(req.repo_url),
        path=str(repo_path),
        message="Repository cloned successfully.",
    )


@app.post("/embed")
def embed_repo(req: EmbedRequest):
    folder = Path(req.folder_path)
    try:
        task = embed_repo_task.delay(req.folder_path)
    except OperationalError as e:
        raise HTTPException(
            status_code=503, detail=f"Task queue unavailable: {e}")

    return {"job_id": task.id, "status": "queued"}


@app.get("/embed/{job_id}")
def get_embed_status(job_id: str):
    """Poll the status of an embedding job."""
    result = embed_repo_task.AsyncResult(job_id)
    if result.state == "PENDING":
        return {"status": "pending"}
    if result.state == "STARTED":
        return {"status": "started"}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    if result.state == "SUCCESS":
        return {"status": "done", **result.result}
    return {"status": result.state.lower()}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieve the most relevant code chunks for a query."""
    repo_id = get_repo_id(req.repo_url)
    collection_name = repo_id

    if not qdrant_client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found. Embed the repo first.",
        )

    results = vector_store.retrieve_results(
        query=req.query,
        collection_name=collection_name,
        top_k=req.top_k,
    )

    return results
