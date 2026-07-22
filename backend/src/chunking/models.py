from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Chunk:
    text: str
    metadata: dict[str, Any]