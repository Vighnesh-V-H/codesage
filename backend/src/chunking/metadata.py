import hashlib


def build_metadata(
    *,
    file_path,
    language,
    node,
    symbol,
    parent,
    node_type,
    text,
):
    return {
        "file_path": file_path,
        "language": language,
        "symbol": symbol,
        "parent": parent,
        "node_type": node_type,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "hash": hashlib.sha256(text.encode()).hexdigest(),
    }