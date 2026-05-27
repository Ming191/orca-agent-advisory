from datetime import datetime

import pytest

from app.schemas.enums import DecisionMode, ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.infrastructure.bigdata.bigdata_ml_provider import BigdataMlToolResultProvider


def _request(
    symbols: list[str], mode: DecisionMode = DecisionMode.SINGLE_SYMBOL_ADVISORY
) -> AdvisoryDecisionRequest:
    timestamp = datetime.fromisoformat("2026-05-27 23:30:00")
    return AdvisoryDecisionRequest(
        request_id="req_bigdata_ml_test",
        timestamp=timestamp,
        as_of_timestamp=timestamp,
        user_query="test bigdata ml provider",
        decision_mode=mode,
        symbols=symbols,
    )


def _adbe_row(close: float = 442.25) -> dict[str, object]:
    return {
        "Symbol": "ADBE",
        "Datetime": "2026-05-27 00:00:00",
        "model_version": "xgb_v1",
        "pred_a": 0.541499137878418,
        "risk_prob": 0.0377756766974926,
        "final_score": 0.51,
        "feature_version": "price_v1_notebook_ac",
        "prediction_process_date": "2026-05-27 23:12:00",
        "source_feature_process_date": "2026-05-27 22:59:00",
        "Close": close,
        "r1": 0.0125,
        "RVOL20": 1.34,
        "RSI14": 58.2,
        "MACD_hist": 0.17,
        "BB_pctB": 0.84,
        "BB_width": 0.11,
        "EMA20_50_spread": 2.1,
        "EMA20_slope": 0.4,
        "ROC10": 1.9,
        "ADX14": 24.0,
    }


def test_bigdata_ml_provider_returns_adbe_context() -> None:
    request = _request(["ADBE"])
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.SUCCESS
    prediction = bundle.ml_predictions.data["ADBE"]
    assert prediction.predicted_direction == "UP"
    assert prediction.probability_up == pytest.approx(0.541499137878418)
    assert prediction.probability_down == pytest.approx(0.0377756766974926)
    assert prediction.model_version == "xgb_v1"
    assert prediction.feature_window == "price_v1_notebook_ac"

    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.SUCCESS
    market_feature = bundle.market_features.data["ADBE"]
    assert market_feature.latest_price == pytest.approx(442.25)
    assert market_feature.price_change_pct_1d == pytest.approx(0.0125)
    assert market_feature.volume_ratio_20d == pytest.approx(1.34)
    assert market_feature.trend_direction == "UP"
    assert market_feature.technical_indicators.rsi_14 == pytest.approx(58.2)
    assert market_feature.technical_indicators.macd_signal == "BULLISH"
    assert market_feature.technical_indicators.bollinger_position == "UPPER"
    bundle.validate_required_for(request)


def test_bigdata_ml_provider_missing_symbol_returns_unavailable() -> None:
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(
        _request(["ZZZZ"])
    )

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.UNAVAILABLE
    assert bundle.ml_predictions.error_message
    assert bundle.ml_predictions.data == {}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.UNAVAILABLE
    assert bundle.market_features.error_message
    assert bundle.market_features.data == {}


def test_bigdata_ml_provider_partial_two_symbols() -> None:
    request = _request(["ADBE", "MSFT"], DecisionMode.PORTFOLIO_RECOMMENDATION)
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row()]).get_tool_results(request)

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.PARTIAL
    assert set(bundle.ml_predictions.data) == {"ADBE"}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.PARTIAL
    assert set(bundle.market_features.data) == {"ADBE"}


def test_bigdata_ml_provider_invalid_close_skipped() -> None:
    bundle = BigdataMlToolResultProvider(row_loader=lambda _: [_adbe_row(close=0.0)]).get_tool_results(
        _request(["ADBE"])
    )

    assert bundle.ml_predictions is not None
    assert bundle.ml_predictions.status == ToolStatus.UNAVAILABLE
    assert bundle.ml_predictions.data == {}
    assert bundle.market_features is not None
    assert bundle.market_features.status == ToolStatus.UNAVAILABLE
    assert bundle.market_features.data == {}
