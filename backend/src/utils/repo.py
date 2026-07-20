import hashlib
def get_repo_id(repo_url: str) -> str:
    normalized = repo_url.strip().lower().removesuffix(".git")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
