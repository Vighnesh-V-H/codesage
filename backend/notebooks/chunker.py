# %%
CHUNK_NODE_TYPES = {
    "python": {
        "class_definition",
        "function_definition",
    },
    "javascript": {
        "class_declaration",
        "function_declaration",
        "method_definition",
        "generator_function_declaration",
    },
    "typescript": {
        "class_declaration",
        "function_declaration",
        "method_definition",
        "generator_function_declaration",
        "interface_declaration",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
}

CLASS_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
    },
    "go": {"type_declaration"},
}

# %%
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Chunk:
    text: str
    metadata: dict[str, Any]

# %%
from tree_sitter import Language, Parser

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_go


LANGUAGES = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "java": Language(tree_sitter_java.language()),
    "go": Language(tree_sitter_go.language()),
}


class ParserFactory:
    def __init__(self):
        self._cache = {}

    def get(self, language: str) -> Parser:
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        if language not in self._cache:
            self._cache[language] = Parser(LANGUAGES[language])

        return self._cache[language]

# %%
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

# %%
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
        yield c

# %%

class ASTChunker:
    def __init__(self):
        self.factory = ParserFactory()

    def chunk_file(self, source_code: str, language: str, file_path: str):
        if language not in CHUNK_NODE_TYPES:
            return

        parser = self.factory.get(language)
        source = source_code.encode("utf8")
        tree = parser.parse(source)

        chunk_nodes = CHUNK_NODE_TYPES[language]
        class_nodes = CLASS_NODE_TYPES[language]

        def node_name(node):
            name = node.child_by_field_name("name")
            return source[name.start_byte:name.end_byte].decode() if name else None

        def visit(node, parent=None):
            if node.type in chunk_nodes:
                text = source[node.start_byte:node.end_byte].decode()
                symbol = node_name(node)
                yield Chunk(
                    text=text,
                    metadata=build_metadata(
                        file_path=file_path, language=language, node=node,
                        symbol=symbol, parent=parent, node_type=node.type, text=text,
                    )
                )
                next_parent = symbol if node.type in class_nodes else parent
                for child in node.children:
                    yield from visit(child, next_parent)
                return

            for child in node.children:
                yield from visit(child, parent)

        yield from visit(tree.root_node)
        yield from self.extract_module_level(tree.root_node, source, chunk_nodes, file_path, language)

    def extract_module_level(
        self,
        root,
        source,
        chunk_nodes,
        file_path,
        language,
    ):
        chunks = []

        cursor = 0

        for child in root.children:
            if child.type not in chunk_nodes:
                continue

            if child.start_byte > cursor:
                text = source[
                    cursor:child.start_byte
                ].decode().strip()

                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            metadata={
                                "file_path": file_path,
                                "language": language,
                                "node_type": "module_level",
                                "symbol": None,
                                "parent": None,
                            }
                        )
                    )

            cursor = child.end_byte

        if cursor < len(source):
            text = source[cursor:].decode().strip()

            if text:
                chunks.append(
                   Chunk(
                       text=text,
                       metadata={
                           "file_path": file_path,
                           "language": language,
                           "node_type": "module_level",
                           "symbol": None,
                           "parent": None,
                       }
                   )
                )

        return chunks

# %%



