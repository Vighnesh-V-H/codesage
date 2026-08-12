import re

from transformers import AutoTokenizer

EMBED_MODEL_ID = "nvidia/llama-nemotron-embed-1b-v2"
MAX_CHUNK_TOKENS = 1500

try:
    _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
except Exception:
    _tokenizer = None

_FALLBACK_CHARS_PER_TOKEN = 3.2


def token_count(text: str) -> int:
    if _tokenizer is not None:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return int(len(text) / _FALLBACK_CHARS_PER_TOKEN) + 1


def _split_overlong_line(line: str, max_tokens: int) -> list[str]:
    if not line.strip():
        return []

    words = re.split(r"\s+", line.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = max(1, token_count(word))
        if current and current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_text_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS):
    if not text or not text.strip():
        return []

    if token_count(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0
            continue

        if token_count(line) <= max_tokens:
            line_tokens = token_count(line) + 1
            if current and current_tokens + line_tokens > max_tokens:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0
            current.append(line)
            current_tokens += line_tokens
            continue

        for piece in _split_overlong_line(line, max_tokens):
            piece_tokens = token_count(piece) + 1
            if current and current_tokens + piece_tokens > max_tokens:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0
            current.append(piece)
            current_tokens += piece_tokens

    if current:
        chunks.append("\n".join(current))

    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
