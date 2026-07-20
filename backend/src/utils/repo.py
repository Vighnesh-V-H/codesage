from pathlib import Path

import hashlib
def get_repo_id(repo_url: str) -> str:
    normalized = repo_url.strip().lower().removesuffix(".git")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

def is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        chunk = f.read(4096)

    return b"\0" in chunk