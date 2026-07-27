# %%

IGNORE_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".cache",
    ".venv",
    "venv",
    "env",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
}

IGNORE_FILES = {
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".DS_Store",
    "Thumbs.db",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
}

IGNORE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".pyc",
    ".pyo",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".md",
    ".sh",
    ".dockerfile",
}

# %%
from pathlib import Path
import os

MAX_FILE_SIZE = 1_000_000  


def walk_repository(root: Path):
    root = Path(root)

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            path = Path(current_root) / file

            if file in IGNORE_FILES:
                continue

            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue

            if path.suffix.lower() in IGNORE_EXTENSIONS:
                continue

            if path.stat().st_size > MAX_FILE_SIZE:
                continue

            yield path

# %%
from pathlib import Path

def print_repository_files(root):
    count = 0

    for path in walk_repository(Path(root)):
        print(path)
        count += 1

    print(f"\nTotal files found: {count}")

# %%
print_repository_files("../../backend/src/chunking")

# %%



