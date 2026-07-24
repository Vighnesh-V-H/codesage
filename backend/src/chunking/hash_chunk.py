import hashlib

def normalize_for_hash(text: str) -> str:
    return " ".join(text.split())

def dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen = {}
    unique = []
    for c in chunks:
        h = hashlib.sha256(normalize_for_hash(c["text"]).encode("utf8")).hexdigest()
        if h not in seen:
            seen[h] = c
            c["content_hash"] = h
            unique.append(c)
        else:
            seen[h].setdefault("duplicate_of", []).append(c["file_path"])
    return unique