import os


def pytest_configure() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-openai-api-key")
