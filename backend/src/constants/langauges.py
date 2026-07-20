from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"

    HTML = "html"
    CSS = "css"
    SCSS = "scss"

    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"

    SQL = "sql"

    MARKDOWN = "markdown"
    TEXT = "text"

    UNKNOWN = "unknown"


EXTENSION_MAP = {
    # Python
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,

    # JavaScript
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,

    # TypeScript
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,

    # Java
    ".java": Language.JAVA,

    # Go
    ".go": Language.GO,

    # Rust
    ".rs": Language.RUST,

    # C
    ".c": Language.C,
    ".h": Language.C,

    # C++
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".hh": Language.CPP,

    # C#
    ".cs": Language.CSHARP,

    # PHP
    ".php": Language.PHP,

    # Ruby
    ".rb": Language.RUBY,

    # Swift
    ".swift": Language.SWIFT,

    # Kotlin
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,

    # Web
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".css": Language.CSS,
    ".scss": Language.SCSS,

    # Config
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".xml": Language.XML,
    ".toml": Language.TOML,

    # Database
    ".sql": Language.SQL,

    # Documentation
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".txt": Language.TEXT,
}


SPECIAL_FILES = {
    "Dockerfile": Language.TEXT,
    "Makefile": Language.TEXT,
    "CMakeLists.txt": Language.TEXT,

    ".gitignore": Language.TEXT,
    ".dockerignore": Language.TEXT,
    ".editorconfig": Language.TEXT,
    ".env": Language.TEXT,

    "package.json": Language.JSON,
    "tsconfig.json": Language.JSON,
    "pyproject.toml": Language.TOML,
    "requirements.txt": Language.TEXT,
}


SHEBANG_MAP = {
    "python": Language.PYTHON,
    "python3": Language.PYTHON,
    "node": Language.JAVASCRIPT,
    "ruby": Language.RUBY,
    "bash": Language.TEXT,
    "sh": Language.TEXT,
    "zsh": Language.TEXT,
}