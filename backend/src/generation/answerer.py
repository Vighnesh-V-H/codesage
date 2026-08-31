from __future__ import annotations

from typing import Any

from src.core.config import config


class AnswerGenerator:
    def __init__(self, client: Any, model: str | None = None) -> None:
        self.client = client
        self.model = model or config.AGENTIC_MODEL

    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineering assistant. "
                        "Answer using only the provided repository context. "
                        "Be precise, cite file paths in the answer, and state "
                        "uncertainty when the context is insufficient."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"Question:\n{query}\n\n"
            "Repository context:\n"
            f"{context}\n\n"
            "Provide a concise but accurate answer grounded in this context. "
            "Include explicit file references like `src/foo.py` when relevant."
        )
