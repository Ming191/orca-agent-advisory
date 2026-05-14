import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import AgentSettings, load_settings
from app.crews.advisory_crew import AdvisorySpecialistCrew
from app.llm.llm_factory import create_deepseek_llm
from app.schemas.decision import SingleSymbolDecision
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle
from app.validators.output_repair import parse_model_output

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool
except ModuleNotFoundError:
    Agent = None
    Crew = None
    Process = None
    Task = None
    BaseTool = object


COMMON_AGENT_RULES = (
    "You must use only the user request and facts retrieved through assigned read-only tools. "
    "Do not invent missing financial metrics. If a field is missing, return UNAVAILABLE and "
    "explain the limitation. Every numerical claim must map to a source_ref or input field."
)


@dataclass
class CrewRunArtifacts:
    manager_agent: Any
    specialist_agents: list[Any]
    tasks: list[Any]
    crew: Any


@dataclass
class HierarchicalCrewRunner:
    settings: AgentSettings = field(default_factory=load_settings)
    llm_factory: Callable[[AgentSettings], Any] = create_deepseek_llm
    verbose: bool = False
    last_artifacts: CrewRunArtifacts | None = None

    def run(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
        *,
        output_model: type[SingleSymbolDecision] = SingleSymbolDecision,
    ) -> SingleSymbolDecision:
        artifacts = self.build_crew(request, tool_results)
        raw_result = artifacts.crew.kickoff(
            inputs={
                "request": request.model_dump(mode="json"),
                "request_json": request.model_dump_json(),
            }
        )
        return parse_model_output(raw_result, output_model)

    def build_crew(
        self,
        request: AdvisoryDecisionRequest,
        tool_results: ToolResultBundle,
    ) -> CrewRunArtifacts:
        _require_crewai()
        llm = self.llm_factory(self.settings)
        tools = build_mocked_upstream_tools(tool_results)

        manager_agent = Agent(
            role="Investment Advisory Manager",
            goal=(
                "Coordinate specialist agents and synthesize grounded market, sentiment, valuation, "
                "and risk signals into an explainable investment decision."
            ),
            backstory=(
                f"{COMMON_AGENT_RULES} You are the lead investment reasoning agent. You never invent "
                "financial metrics. You resolve conflicts and return structured JSON."
            ),
            llm=llm,
            allow_delegation=True,
            verbose=self.verbose,
        )

        specialist_crew = AdvisorySpecialistCrew(
            llm=llm,
            tools=tools,
            manager_agent=manager_agent,
            verbose=self.verbose,
        )
        specialist_agents = specialist_crew.specialist_agents()
        specialist_tasks = specialist_crew.specialist_tasks()
        tasks = specialist_tasks + [_build_manager_task(request, specialist_tasks)]
        crew = Crew(
            agents=specialist_agents,
            tasks=tasks,
            manager_agent=manager_agent,
            process=Process.hierarchical,
            verbose=self.verbose,
        )

        artifacts = CrewRunArtifacts(
            manager_agent=manager_agent,
            specialist_agents=specialist_agents,
            tasks=tasks,
            crew=crew,
        )
        self.last_artifacts = artifacts
        return artifacts


def build_mocked_upstream_tools(tool_results: ToolResultBundle) -> dict[str, Any]:
    return {
        "market_features": _StaticTool(
            name="MarketFeatureTool",
            description="Read-only mocked market feature snapshot lookup.",
            bundle_field="market_features",
            tool_results=tool_results,
        ),
        "ml_predictions": _StaticTool(
            name="MlPredictionTool",
            description="Read-only mocked machine learning prediction lookup.",
            bundle_field="ml_predictions",
            tool_results=tool_results,
        ),
        "sentiment_snapshot": _StaticTool(
            name="NewsSentimentTool",
            description="Read-only mocked news sentiment snapshot lookup.",
            bundle_field="sentiment_snapshot",
            tool_results=tool_results,
        ),
        "valuation_snapshot": _StaticTool(
            name="FundamentalsTool",
            description="Read-only mocked fundamentals and valuation lookup.",
            bundle_field="valuation_snapshot",
            tool_results=tool_results,
        ),
        "risk_snapshot": _StaticTool(
            name="RiskFeatureTool",
            description="Read-only mocked risk feature lookup.",
            bundle_field="risk_snapshot",
            tool_results=tool_results,
        ),
        "portfolio_snapshot": _StaticTool(
            name="PortfolioTool",
            description="Read-only mocked portfolio snapshot lookup.",
            bundle_field="portfolio_snapshot",
            tool_results=tool_results,
        ),
    }


class _StaticTool(BaseTool):
    name: str
    description: str
    bundle_field: str
    tool_results: ToolResultBundle

    def _run(self, query: str = "") -> str:
        result = getattr(self.tool_results, self.bundle_field)
        if result is None:
            return json.dumps(
                {
                    "tool": self.name,
                    "status": "UNAVAILABLE",
                    "query": query,
                    "error_message": f"{self.bundle_field} was not provided",
                }
            )
        return result.model_dump_json()


def _build_manager_task(request: AdvisoryDecisionRequest, specialist_tasks: list[Any]) -> Any:
    request_fingerprint = _stable_hash(request.model_dump(mode="json"))
    return Task(
        description=(
            "Synthesize all specialist outputs into final advisory JSON. Respect decision_mode, "
            "user constraints, citations, limitations, and the not_financial_advice=true boundary. "
            f"Request fingerprint: {request_fingerprint}."
        ),
        expected_output="Final advisory response JSON matching the Pydantic decision schema.",
        context=specialist_tasks,
    )


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_crewai() -> None:
    if Agent is None or Crew is None or Process is None or Task is None:
        raise RuntimeError("CrewAI is required for the hierarchical crew runner")
