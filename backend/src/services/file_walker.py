from pathlib import Path
import os
from constants.file_extensions import SOURCE_EXTENSIONS, IGNORE_DIRS, IGNORE_EXTENSIONS ,IGNORE_FILES
from utils.repo import is_binary

MAX_FILE_SIZE = 1_000_000  


def walk_repository(root: Path):
    root = Path(root)

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            path = Path(current_root) / file

            if file in IGNORE_FILES:
                continue

            if is_binary(path):
                continue

            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue

            if path.suffix.lower() in IGNORE_EXTENSIONS:
                continue

            if path.stat().st_size > MAX_FILE_SIZE:
                continue

            yield path