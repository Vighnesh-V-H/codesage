from __future__ import annotations

from typing import Any


class RAGPipeline:
    def __init__(self, vector_store: Any) -> None:
        self.vector_store = vector_store

    def run(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        model: str | None = None,
    ) -> dict[str, Any]:
        results = self.vector_store.retrieve_results(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            model=model,
        )
        context = self._build_context(results)
        return {
            "query": query,
            "collection_name": collection_name,
            "top_k": top_k,
            "context": context,
            "results": results,
        }

    def _build_context(self, results: dict[str, Any]) -> str:
        lines: list[str] = []
        for index, text in enumerate(results.get("retrieved_texts", []), start=1):
            metadata = results.get("retrieved_metadata", [])[index - 1]
            file_path = str(metadata.get("file_path", "unknown"))
            symbol = metadata.get("symbol") or metadata.get("node_type") or "snippet"
            snippet = text.strip()
            if len(snippet) > 1500:
                snippet = f"{snippet[:1500].rstrip()}..."
            lines.append(f"[{index}] {file_path}::{symbol}\n{snippet}")
        return "\n\n".join(lines)
