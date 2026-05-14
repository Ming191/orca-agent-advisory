import json
from typing import Any

from pydantic import ValidationError

from app.schemas.agent_outputs import (
    MarketDataAgentOutput,
    RiskAgentOutput,
    SentimentAgentOutput,
    ValuationAgentOutput,
)
from app.schemas.enums import AgentStatus, RiskLabel, SentimentLabel, ValuationLabel


def validate_market_data_output(result: Any) -> tuple[bool, Any]:
    output = _extract_output(result, MarketDataAgentOutput)
    if output is None:
        return False, "Market data output must match MarketDataAgentOutput."
    if output.status == AgentStatus.ERROR and not output.missing_fields:
        return False, "ERROR market data output must identify missing_fields."
    if not output.market_signals and output.status != AgentStatus.ERROR:
        return False, "Non-error market data output must include market_signals."
    return True, result


def validate_sentiment_output(result: Any) -> tuple[bool, Any]:
    output = _extract_output(result, SentimentAgentOutput)
    if output is None:
        return False, "Sentiment output must match SentimentAgentOutput."
    if output.status == AgentStatus.SKIPPED:
        if output.sentiment_label != SentimentLabel.UNAVAILABLE:
            return False, "Skipped sentiment output must use sentiment_label=UNAVAILABLE."
        if not output.limitations:
            return False, "Skipped sentiment output must include limitations."
    return True, result


def validate_valuation_output(result: Any) -> tuple[bool, Any]:
    output = _extract_output(result, ValuationAgentOutput)
    if output is None:
        return False, "Valuation output must match ValuationAgentOutput."
    if output.status == AgentStatus.SKIPPED:
        if output.valuation_label != ValuationLabel.UNKNOWN:
            return False, "Skipped valuation output must use valuation_label=UNKNOWN."
        if not output.limitations:
            return False, "Skipped valuation output must include limitations."
        if output.valuation_drivers:
            return False, "Skipped valuation output cannot include valuation_drivers."
    return True, result


def validate_risk_output(result: Any) -> tuple[bool, Any]:
    output = _extract_output(result, RiskAgentOutput)
    if output is None:
        return False, "Risk output must match RiskAgentOutput."
    if output.status == AgentStatus.ERROR and not output.limitations:
        return False, "ERROR risk output must include limitations."
    if output.risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL} and output.confidence_cap > 0.65:
        return False, "High or critical risk output must cap confidence at 0.65 or lower."
    return True, result


def _extract_output(result: Any, model_type: type[Any]) -> Any | None:
    if isinstance(result, model_type):
        return result

    pydantic_output = getattr(result, "pydantic", None)
    if isinstance(pydantic_output, model_type):
        return pydantic_output

    raw_output = getattr(result, "raw", None)
    if raw_output is None:
        return None

    try:
        if isinstance(raw_output, str):
            return model_type.model_validate_json(raw_output)
        if isinstance(raw_output, dict):
            return model_type.model_validate(raw_output)
        return model_type.model_validate(json.loads(str(raw_output)))
    except (json.JSONDecodeError, TypeError, ValidationError):
        return None
