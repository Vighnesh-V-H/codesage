from src.chunking.config import CHUNK_NODE_TYPES, CLASS_NODE_TYPES
from src.chunking.metadata import build_metadata
from src.chunking.parser_factory import ParserFactory
from src.chunking.models import Chunk
from src.chunking.limits import token_count, split_text_by_tokens, MAX_CHUNK_TOKENS

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

        def make_chunk(text, node, symbol, parent, node_type):
            return Chunk(
                text=text,
                metadata=build_metadata(
                    file_path=file_path, language=language, node=node,
                    symbol=symbol, parent=parent, node_type=node_type, text=text,
                )
            )

        def visit(node, parent=None):
            if node.type in chunk_nodes:
                text = source[node.start_byte:node.end_byte].decode()
                symbol = node_name(node)
                next_parent = symbol if node.type in class_nodes else parent

                if token_count(text) > MAX_CHUNK_TOKENS:
                    yielded_child = False
                    for child in node.children:
                        if child.type in chunk_nodes:
                            yielded_child = True
                        yield from visit(child, next_parent)

                    if not yielded_child:
                        for piece in split_text_by_tokens(text):
                            yield make_chunk(piece, node, symbol, parent, node.type)
                    return

                yield make_chunk(text, node, symbol, parent, node.type)
                for child in node.children:
                    yield from visit(child, next_parent)
                return

            for child in node.children:
                yield from visit(child, parent)

        yield from visit(tree.root_node)
        yield from self.extract_module_level(tree.root_node, source, chunk_nodes, file_path, language)

    def extract_module_level(self, root, source, chunk_nodes, file_path, language):
        chunks = []
        cursor = 0

        def emit(text):
            for piece in split_text_by_tokens(text):
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata={
                            "file_path": file_path,
                            "language": language,
                            "node_type": "module_level",
                            "symbol": None,
                            "parent": None,
                        }
                    )
                )

        for child in root.children:
            if child.type not in chunk_nodes:
                continue

            if child.start_byte > cursor:
                text = source[cursor:child.start_byte].decode().strip()
                if text:
                    emit(text)

            cursor = child.end_byte

        if cursor < len(source):
            text = source[cursor:].decode().strip()
            if text:
                emit(text)

        return chunks