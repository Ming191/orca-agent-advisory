import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from app.application.mappers.bigdata_ml_mapper import (
    is_valid_context_row,
    market_feature_from_row,
    prediction_from_row,
    row_sort_key,
)
from app.schemas.enums import ToolStatus
from app.schemas.request import AdvisoryDecisionRequest
from app.schemas.tool_results import (
    Freshness,
    MarketFeature,
    MarketFeatureToolResult,
    MlPrediction,
    MlPredictionToolResult,
    ToolResultBundle,
)


RowLoader = Callable[[AdvisoryDecisionRequest], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class BigdataMlTableConfig:
    prediction_table: str = field(
        default_factory=lambda: os.getenv("ORCA_ML_PREDICTION_TABLE", "ml_ready.stock_predictions")
    )
    feature_table: str = field(
        default_factory=lambda: os.getenv("ORCA_ML_FEATURE_TABLE", "ml_ready.stock_price_features")
    )
    curated_price_table: str = field(
        default_factory=lambda: os.getenv("ORCA_CURATED_PRICE_TABLE", "curated.us_stock_eod_prices")
    )
    catalog: str = field(default_factory=lambda: os.getenv("ORCA_ICEBERG_CATALOG", "nessie"))
    max_age_seconds: int = field(
        default_factory=lambda: int(os.getenv("ORCA_CONTEXT_MAX_AGE_SECONDS", "86400"))
    )
    spark_app_name: str | None = None

    def table_ref(self, table_name: str) -> str:
        if not self.catalog or table_name.startswith(f"{self.catalog}."):
            return table_name
        return f"{self.catalog}.{table_name}"


class BigdataMlToolResultProvider:
    """ToolResultProvider-compatible adapter for joined bigdata/ML rows."""

    def __init__(
        self,
        row_loader: RowLoader | None = None,
        table_config: BigdataMlTableConfig | None = None,
    ) -> None:
        self.row_loader = row_loader
        self.table_config = table_config or BigdataMlTableConfig()

    def get_tool_results(self, request: AdvisoryDecisionRequest) -> ToolResultBundle:
        rows = list((self.row_loader or self._load_live_rows)(request))
        rows_by_symbol = _valid_latest_rows_by_symbol(rows, set(request.symbols))
        found_symbols = [symbol for symbol in request.symbols if symbol in rows_by_symbol]
        missing_symbols = [symbol for symbol in request.symbols if symbol not in rows_by_symbol]

        if not found_symbols:
            status = ToolStatus.UNAVAILABLE
            error_message = "no valid bigdata ML context found for requested symbols"
            as_of_timestamp = request.as_of_timestamp
        elif missing_symbols:
            status = ToolStatus.PARTIAL
            error_message = "missing valid bigdata ML context for symbols: " + ", ".join(missing_symbols)
            as_of_timestamp = _freshest_timestamp(rows_by_symbol.values(), request.as_of_timestamp)
        else:
            status = ToolStatus.SUCCESS
            error_message = None
            as_of_timestamp = _freshest_timestamp(rows_by_symbol.values(), request.as_of_timestamp)

        freshness = Freshness(
            is_stale=False,
            last_updated_at=as_of_timestamp,
            max_age_seconds=self.table_config.max_age_seconds,
        )

        ml_data: dict[str, MlPrediction] = {}
        market_data: dict[str, MarketFeature] = {}
        source_refs: list[str] = []

        for symbol in found_symbols:
            row = rows_by_symbol[symbol]
            prediction = prediction_from_row(row)
            ml_data[symbol] = prediction
            market_data[symbol] = market_feature_from_row(row, prediction.predicted_direction)
            date_ref = str(row.get("Datetime") or row.get("prediction_process_date") or "")
            source_refs.extend(
                [
                    f"{self.table_config.prediction_table}:{symbol}:{date_ref}",
                    f"{self.table_config.feature_table}:{symbol}:{date_ref}",
                    f"{self.table_config.curated_price_table}:{symbol}:{date_ref}",
                ]
            )

        return ToolResultBundle(
            market_features=MarketFeatureToolResult(
                tool="MarketFeatureTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=source_refs,
                error_message=error_message,
                data=market_data,
            ),
            ml_predictions=MlPredictionToolResult(
                tool="MlPredictionTool",
                status=status,
                request_id=request.request_id,
                as_of_timestamp=as_of_timestamp,
                freshness=freshness,
                source_refs=source_refs,
                error_message=error_message,
                data=ml_data,
            ),
        )

    def _load_live_rows(self, request: AdvisoryDecisionRequest) -> Iterable[Mapping[str, Any]]:
        from pyspark.sql import SparkSession, functions as F  # type: ignore[import-not-found]

        app_name = self.table_config.spark_app_name or "orca-bigdata-ml-provider"
        spark = SparkSession.builder.appName(app_name).getOrCreate()
        symbols = [symbol.upper() for symbol in request.symbols]
        as_of_date = request.metadata.get("as_of_date")

        p = spark.table(self.table_config.table_ref(self.table_config.prediction_table)).alias("p")
        f = spark.table(self.table_config.table_ref(self.table_config.feature_table)).alias("f")
        c = spark.table(self.table_config.table_ref(self.table_config.curated_price_table)).alias("c")

        p = p.where(F.col("Symbol").isin(symbols))
        if as_of_date:
            p = p.where(F.to_date(F.col("Datetime")) <= F.lit(str(as_of_date)))
        max_datetime = p.agg(F.max("Datetime").alias("max_datetime")).collect()[0]["max_datetime"]
        if max_datetime is None:
            return []

        p = p.where(F.col("Datetime") == F.lit(max_datetime))
        joined = (
            p.join(f, ["Symbol", "Datetime"], "left")
            .join(c, ["Symbol", "Datetime"], "left")
            .select(
                F.col("Symbol"),
                F.col("Datetime"),
                F.col("p.model_version"),
                F.col("p.pred_a"),
                F.col("p.risk_prob"),
                F.col("p.final_score"),
                F.col("p.feature_version"),
                F.col("p.process_date").alias("prediction_process_date"),
                F.col("p.source_feature_process_date"),
                F.col("c.Close"),
                F.col("f.r1"),
                F.col("f.RVOL20"),
                F.col("f.RSI14"),
                F.col("f.MACD_hist"),
                F.col("f.BB_pctB"),
                F.col("f.BB_width"),
                F.col("f.EMA20_50_spread"),
                F.col("f.EMA20_slope"),
                F.col("f.ROC10"),
                F.col("f.ADX14"),
            )
        )
        return [row.asDict(recursive=True) for row in joined.collect()]


def _valid_latest_rows_by_symbol(
    rows: Iterable[Mapping[str, Any]], requested_symbols: set[str]
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("Symbol") or "").strip().upper()
        if symbol not in requested_symbols or not is_valid_context_row(row):
            continue
        current = latest.get(symbol)
        if current is None or row_sort_key(row) > row_sort_key(current):
            latest[symbol] = row
    return latest


def _freshest_timestamp(rows: Iterable[Mapping[str, Any]], fallback: datetime) -> datetime:
    timestamps = [max(row_sort_key(row)) for row in rows]
    timestamps = [timestamp for timestamp in timestamps if timestamp != datetime.min]
    return max(timestamps) if timestamps else fallback
