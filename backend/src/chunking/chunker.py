from .config import CHUNK_NODE_TYPES, CLASS_NODE_TYPES
from .metadata import build_metadata
from .parser_factory import ParserFactory
from .models import Chunk


class ASTChunker:
    def __init__(self):
        self.factory = ParserFactory()

    def chunk_file(
        self,
        source_code: str,
        language: str,
        file_path: str,
    ):
        if language not in CHUNK_NODE_TYPES:
            return []

        parser = self.factory.get(language)

        source = source_code.encode("utf8")

        tree = parser.parse(source)

        chunks = []

        chunk_nodes = CHUNK_NODE_TYPES[language]
        class_nodes = CLASS_NODE_TYPES[language]

        def node_name(node):
            name = node.child_by_field_name("name")
            if name is None:
                return None

            return source[
                name.start_byte:name.end_byte
            ].decode()

        def visit(node, parent=None):
            if node.type in chunk_nodes:
                text = source[
                    node.start_byte:node.end_byte
                ].decode()

                symbol = node_name(node)

                chunks.append(
                    Chunk(
                        text=text,
                        metadata=build_metadata(
                            file_path=file_path,
                            language=language,
                            node=node,
                            symbol=symbol,
                            parent=parent,
                            node_type=node.type,
                            text=text,
                        )
                    )
                )

                next_parent = parent

                if node.type in class_nodes:
                    next_parent = symbol

                for child in node.children:
                    visit(child, next_parent)

                return

            for child in node.children:
                visit(child, parent)

        visit(tree.root_node)

        chunks.extend(
            self.extract_module_level(
                tree.root_node,
                source,
                chunk_nodes,
                file_path,
                language,
            )
        )

        return chunks

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