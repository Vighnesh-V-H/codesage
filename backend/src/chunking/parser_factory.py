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