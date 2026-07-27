from pathlib import Path

from src.constants.langauges import (
    EXTENSION_MAP,
    Language,
    SHEBANG_MAP,
    SPECIAL_FILES,
)


def detect_language(path: str | Path) -> Language:
    """
    Detect the programming language of a file.
    Detection order:
    1. Special filenames (Dockerfile, Makefile, etc.)
    2. File extension
    3. Shebang (#!/usr/bin/env python)
    """

    path = Path(path)

    if path.name in SPECIAL_FILES:
        return SPECIAL_FILES[path.name]

    language = EXTENSION_MAP.get(path.suffix.lower())
    if language:
        return language

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()

        if first_line.startswith("#!"):
            for interpreter, language in SHEBANG_MAP.items():
                if interpreter in first_line:
                    return language
    except OSError:
        pass

    return Language.UNKNOWN