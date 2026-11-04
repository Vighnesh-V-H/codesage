from pathlib import Path
import os
from src.constants.file_extensions import SOURCE_EXTENSIONS, IGNORE_DIRS, IGNORE_EXTENSIONS, IGNORE_FILES
from src.utils.repo import is_binary

MAX_FILE_SIZE = 1_000_000  


def walk_repository(root: Path):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Repository root is not an accessible directory: {root}")

    for current_root, dirs, files in os.walk(root, onerror=lambda _error: None):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            path = Path(current_root) / file

            if file in IGNORE_FILES:
                continue


            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue

            if path.suffix.lower() in IGNORE_EXTENSIONS:
                continue

            try:
                if is_binary(path) or path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            yield path