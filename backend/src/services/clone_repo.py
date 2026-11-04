from pathlib import Path
import subprocess
import shutil
from src.utils.repo import get_repo_id
from src.errors.repo_clone_errors import InvalidRepositoryError, PrivateRepositoryError, RepoCloneError

BACKEND_ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = BACKEND_ROOT / "tmp"

class RepoCloneService:
    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or TMP_DIR
        self.workspace.mkdir(parents=True, exist_ok=True)

    def clone(self, repo_url: str) -> Path:
        repo_id = get_repo_id(repo_url)
        repo_path = self.workspace / repo_id

        if repo_path.exists():
            if not (repo_path / ".git").is_dir():
                raise RepoCloneError(
                    f"Clone destination exists but is not a Git repository: {repo_path}"
                )
            try:
                origin = subprocess.run(
                    ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if origin.rstrip("/").removesuffix(".git").lower() != repo_url.strip().rstrip("/").removesuffix(".git").lower():
                    raise RepoCloneError(
                        "Clone destination is associated with a different repository URL."
                    )
                subprocess.run(
                    ["git", "-C", str(repo_path), "pull", "--ff-only"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return repo_path
            except subprocess.CalledProcessError as e:
                raise RepoCloneError(
                    f"Unable to refresh existing repository clone: {(e.stderr or '').strip()}"
                ) from e

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--filter=blob:none",
                    repo_url,
                    str(repo_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return repo_path

        except subprocess.CalledProcessError as e:
            error = (e.stderr or "").lower()

            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)

            if (
                "authentication failed" in error
                or "could not read username" in error
                or "repository not found" in error
            ):
                raise PrivateRepositoryError(
                    "Repository is private or requires authentication."
                )

            if "not appear to be a git repository" in error:
                raise InvalidRepositoryError(
                    "The provided URL is not a valid Git repository."
                )

            if "could not resolve host" in error:
                raise RepoCloneError("Unable to reach the Git server.")

            raise RepoCloneError(
                f"Failed to clone repository.\n{e.stderr.strip()}"
            )