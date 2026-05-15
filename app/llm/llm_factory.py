from typing import Any

from app.config import AgentSettings, load_settings
from app.llm.deepseek_client import build_deepseek_llm


def create_deepseek_llm(
    settings: AgentSettings | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> Any:
    resolved_settings = settings or load_settings()
    return build_deepseek_llm(
        resolved_settings,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
