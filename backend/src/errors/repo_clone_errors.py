class RepoCloneError(Exception):
    """Base exception for clone failures."""


class PrivateRepositoryError(RepoCloneError):
    """Raised when the repository is private or requires authentication."""


class RepositoryNotFoundError(RepoCloneError):
    """Raised when the repository does not exist."""


class InvalidRepositoryError(RepoCloneError):
    """Raised when the URL is not a valid git repository."""
