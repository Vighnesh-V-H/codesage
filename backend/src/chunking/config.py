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