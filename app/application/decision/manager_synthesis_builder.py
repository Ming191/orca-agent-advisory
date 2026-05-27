from app.application.specialists import analyze_market_data
from app.application.specialists import analyze_risk
from app.application.specialists import analyze_sentiment
from app.application.specialists import analyze_valuation
from app.application.decision.decision_helpers import agent_limitations, portfolio_summary, tool_citations, unique
from app.schemas.agent_outputs import AgentOutputBundle
from app.schemas.decision import DecisionRationale, PortfolioAllocation
from app.schemas.enums import (
    DecisionMode,
    FactorWeight,
    PortfolioAction,
    Recommendation,
    RiskLabel,
    SentimentLabel,
    SignalStance,
    ValuationLabel,
)
from app.schemas.manager_outputs import ManagerSynthesisOutput
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import ToolResultBundle


def run_specialist_analysis(request: AdvisoryDecisionRequest, tool_results: ToolResultBundle) -> AgentOutputBundle:
    """Legacy pure helper for tests; runtime uses hierarchical crew orchestration."""
    return AgentOutputBundle(
        market_data_agent=analyze_market_data(request, tool_results),
        sentiment_agent=analyze_sentiment(request, tool_results),
        valuation_agent=analyze_valuation(request, tool_results),
        risk_agent=analyze_risk(request, tool_results),
    )


def build_deterministic_manager_synthesis(
    request: AdvisoryDecisionRequest,
    tool_results: ToolResultBundle,
    agent_outputs: AgentOutputBundle,
) -> ManagerSynthesisOutput:
    """Legacy pure helper for tests; runtime must not call deterministic synthesis."""
    if request.decision_mode == DecisionMode.PORTFOLIO_RECOMMENDATION:
        return _portfolio_synthesis(request, tool_results, agent_outputs)
    return _single_symbol_synthesis(request, tool_results, agent_outputs)


def _single_symbol_synthesis(request: AdvisoryDecisionRequest, tool_results: ToolResultBundle, agent_outputs: AgentOutputBundle) -> ManagerSynthesisOutput:
    symbol = request.symbols[0]
    recommendation = _derive_recommendation(agent_outputs)
    market_stance = _dominant_market_stance(agent_outputs)
    valuation_label = _valuation_label(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label
    rationale = [
        DecisionRationale(factor="market_signal", stance=market_stance or SignalStance.NEUTRAL, weight=FactorWeight.HIGH, explanation=agent_outputs.market_data_agent.summary),
        DecisionRationale(factor="risk", stance=_risk_stance(risk_label), weight=FactorWeight.HIGH, explanation=", ".join(agent_outputs.risk_agent.risk_factors) or agent_outputs.risk_agent.summary),
    ]
    if sentiment_label not in {None, SentimentLabel.UNAVAILABLE}:
        rationale.append(DecisionRationale(factor="sentiment", stance=_sentiment_stance(sentiment_label), weight=FactorWeight.MEDIUM, explanation=agent_outputs.sentiment_agent.summary if agent_outputs.sentiment_agent is not None else "Sentiment unavailable."))
    if valuation_label not in {None, ValuationLabel.UNKNOWN}:
        rationale.append(DecisionRationale(factor="valuation", stance=_valuation_stance(valuation_label), weight=FactorWeight.MEDIUM, explanation=agent_outputs.valuation_agent.summary if agent_outputs.valuation_agent is not None else "Valuation unavailable."))
    return ManagerSynthesisOutput(
        summary=f"{symbol} draft recommendation is {recommendation.value} based on specialist market, sentiment, valuation, and risk evidence.",
        time_horizon=request.user_context.investment_horizon,
        proposed_recommendation=recommendation,
        decision_rationale=rationale,
        supporting_signals=_supporting_signals(agent_outputs),
        conflicting_signals=_draft_conflicts(agent_outputs),
        risk_warnings=agent_outputs.risk_agent.risk_factors,
        limitations=agent_limitations(agent_outputs),
        data_citations=tool_citations(tool_results),
    )


def _portfolio_synthesis(request: AdvisoryDecisionRequest, tool_results: ToolResultBundle, agent_outputs: AgentOutputBundle) -> ManagerSynthesisOutput:
    min_cash = float(request.user_context.custom_constraints.get("min_cash_weight", 0.0))
    cash_weight = min_cash if request.user_context.allow_cash_position else 0.0
    remaining_weight = max(0.0, 100.0 - cash_weight)
    per_symbol = remaining_weight / len(request.symbols)
    capped_symbol_weight = min(per_symbol, request.user_context.max_single_asset_weight)
    allocations = [PortfolioAllocation(symbol=symbol, weight_pct=round(capped_symbol_weight, 2), portfolio_action=PortfolioAction.MAINTAIN_WEIGHT, rationale="Maintain exposure within max_single_asset_weight.") for symbol in request.symbols]
    allocated = sum(allocation.weight_pct for allocation in allocations)
    if request.user_context.allow_cash_position:
        allocations.append(PortfolioAllocation(symbol="CASH", weight_pct=round(100.0 - allocated, 2), portfolio_action=PortfolioAction.CASH_BUFFER, rationale="Cash buffer absorbs concentration and minimum cash constraints."))
    return ManagerSynthesisOutput(summary="Portfolio draft keeps concentration within user constraints.", time_horizon=request.user_context.investment_horizon, proposed_portfolio_action=PortfolioAction.MAINTAIN_WEIGHT, portfolio_allocation=allocations, portfolio_summary=portfolio_summary(agent_outputs), risk_warnings=agent_outputs.risk_agent.risk_factors, limitations=agent_limitations(agent_outputs), data_citations=tool_citations(tool_results))


def _derive_recommendation(agent_outputs: AgentOutputBundle) -> Recommendation:
    market_stance = _dominant_market_stance(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    valuation_label = _valuation_label(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label
    if risk_label == RiskLabel.CRITICAL:
        return Recommendation.SELL if market_stance == SignalStance.BEARISH else Recommendation.HOLD
    if market_stance == SignalStance.BULLISH and valuation_label != ValuationLabel.OVERVALUED:
        return Recommendation.BUY
    if market_stance == SignalStance.BEARISH or sentiment_label == SentimentLabel.BEARISH:
        return Recommendation.SELL
    if valuation_label == ValuationLabel.OVERVALUED:
        return Recommendation.HOLD
    return Recommendation.HOLD


def _dominant_market_stance(agent_outputs: AgentOutputBundle) -> SignalStance | None:
    if not agent_outputs.market_data_agent.market_signals:
        return None
    signal = max(agent_outputs.market_data_agent.market_signals, key=lambda item: item.confidence)
    return signal.stance


def _sentiment_label(agent_outputs: AgentOutputBundle) -> SentimentLabel | None:
    return None if agent_outputs.sentiment_agent is None else agent_outputs.sentiment_agent.sentiment_label


def _valuation_label(agent_outputs: AgentOutputBundle) -> ValuationLabel | None:
    return None if agent_outputs.valuation_agent is None else agent_outputs.valuation_agent.valuation_label


def _sentiment_stance(label: SentimentLabel) -> SignalStance:
    if label == SentimentLabel.BULLISH:
        return SignalStance.BULLISH
    if label == SentimentLabel.BEARISH:
        return SignalStance.BEARISH
    if label == SentimentLabel.MIXED:
        return SignalStance.MIXED
    return SignalStance.NEUTRAL


def _valuation_stance(label: ValuationLabel) -> SignalStance:
    if label == ValuationLabel.UNDERVALUED:
        return SignalStance.BULLISH
    if label == ValuationLabel.OVERVALUED:
        return SignalStance.BEARISH
    return SignalStance.NEUTRAL


def _risk_stance(label: RiskLabel) -> SignalStance:
    if label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        return SignalStance.BEARISH
    if label == RiskLabel.LOW:
        return SignalStance.BULLISH
    return SignalStance.NEUTRAL


def _supporting_signals(agent_outputs: AgentOutputBundle) -> list[str]:
    signals: list[str] = []
    for market_signal in agent_outputs.market_data_agent.market_signals:
        signals.extend(market_signal.drivers)
    if agent_outputs.sentiment_agent is not None:
        signals.extend(agent_outputs.sentiment_agent.top_drivers)
    if agent_outputs.valuation_agent is not None:
        signals.extend(agent_outputs.valuation_agent.valuation_drivers)
    return unique(signals)


def _draft_conflicts(agent_outputs: AgentOutputBundle) -> list[str]:
    conflicts: list[str] = []
    market_stance = _dominant_market_stance(agent_outputs)
    risk_label = agent_outputs.risk_agent.risk_label
    valuation_label = _valuation_label(agent_outputs)
    sentiment_label = _sentiment_label(agent_outputs)
    if market_stance == SignalStance.BULLISH and risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        conflicts.append("Bullish market signal conflicts with high risk.")
    if valuation_label == ValuationLabel.OVERVALUED and risk_label in {RiskLabel.HIGH, RiskLabel.CRITICAL}:
        conflicts.append("Overvaluation compounds high-risk exposure.")
    if market_stance == SignalStance.BULLISH and sentiment_label == SentimentLabel.BEARISH:
        conflicts.append("Bullish technical signal conflicts with bearish sentiment.")
    return conflicts
