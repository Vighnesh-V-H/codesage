import hashlib


def normalize_for_hash(text: str) -> str:
    return " ".join(text.split())


def dedup_stream(chunks_iter):
    seen = set()
    for c in chunks_iter:
        h = hashlib.sha256(normalize_for_hash(c.text).encode("utf8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        c.metadata["content_hash"] = h
        c.metadata["hash"] = h
        yield c