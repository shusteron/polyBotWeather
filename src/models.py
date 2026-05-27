from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator


class MarketData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str
    location: Optional[str] = None
    event_type: Optional[str] = None
    threshold: Optional[float] = None
    threshold_unit: Optional[str] = None
    threshold_direction: str = "exact"  # "exact", "above", "below"
    resolution_date: Optional[str] = None
    yes_price: float = 0.5
    no_price: float = 0.5
    spread: float = 0.0
    liquidity: float = 0.0
    volume: float = 0.0
    resolution_source: Optional[str] = None
    resolution_station: Optional[str] = None

    @field_validator("yes_price", "no_price", mode="before")
    @classmethod
    def clamp_price(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.001, min(0.999, f))


class WeatherForecast(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    location: str
    lat: float
    lon: float
    target_date: str
    model_name: str
    ensemble_members: list[float] = []
    mean: float = 0.0
    std: float = 0.0
    spread: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0


class EnsembleAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    forecasts: list[WeatherForecast] = []
    weighted_mean: float = 0.0
    weighted_std: float = 0.0
    provider_agreement: float = 0.0
    confidence_range: float = 0.0
    all_members: list[float] = []


class ForecastStabilityRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    location: str
    target_date: str
    runs: list[dict] = []
    stability_score: float = 0.0
    direction_consistent: bool = False


class ConfidenceScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total: float = 0.0
    ensemble_spread_score: float = 0.0
    stability_score: float = 0.0
    calibration_score: float = 0.0
    threshold_distance_score: float = 0.0
    liquidity_score: float = 0.0
    horizon_score: float = 0.0
    breakdown: dict = {}


class TradeSignal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    market_id: str
    action: str  # "BUY_YES" | "BUY_NO" | "NO_TRADE"
    model_probability: float
    market_probability: float
    edge: float
    confidence_score: ConfidenceScore
    rejection_reasons: list[str] = []
    kelly_size: float = 0.0
    recommended_size: float = 0.0


class Trade(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: TradeSignal
    market: MarketData
    timestamp: str
    paper_fill_price: float
    paper_size: float
    status: str = "OPEN"   # "OPEN" | "CLOSED" | "CANCELLED"
    outcome: Optional[bool] = None
    pnl: Optional[float] = None


class CalibrationRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    market_id: str
    predicted_probability: float
    market_probability: float
    edge: float
    confidence: float
    outcome: Optional[bool] = None
    resolution_date: Optional[str] = None
    pnl: Optional[float] = None
    location: Optional[str] = None          # city name — for per-city reliability tracking
    direction_type: Optional[str] = None    # "same" or "contrarian" — for zone analysis
