from src.generation.answerer import AnswerGenerator


def test_answer_generator_builds_prompt_and_returns_answer():
    class FakeMessage:
        content = "Use the login helper in src/auth.py."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return FakeResponse()

    generator = AnswerGenerator(client=type("FakeClient", (), {"chat": FakeChat})())

    answer = generator.generate(
        "How do users authenticate?",
        "src/auth.py::login_user\ndef login_user():\n    return True",
    )

    assert "login helper" in answer
    assert "src/auth.py" in answer or "login_user" in answer


def test_answer_generator_uses_default_model_from_config():
    generator = AnswerGenerator(client=type("FakeClient", (), {})())

    assert generator.model
