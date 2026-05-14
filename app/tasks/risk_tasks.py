from typing import Any

from app.crews.config_loader import crewai_task_config
from app.schemas.agent_outputs import RiskAgentOutput
from app.tasks.guardrails import validate_risk_output

try:
    from crewai import Task
except ModuleNotFoundError:
    Task = None


def create_risk_task(agent: Any) -> Any:
    _require_crewai()
    return Task(
        config=crewai_task_config("risk_task"),
        agent=agent,
        output_pydantic=RiskAgentOutput,
        guardrail=validate_risk_output,
        guardrail_max_retries=3,
    )


def _require_crewai() -> None:
    if Task is None:
        raise RuntimeError("CrewAI is required to create Risk Task")
