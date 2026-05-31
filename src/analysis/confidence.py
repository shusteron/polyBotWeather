from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from ..models import (
    ConfidenceScore,
    EnsembleAnalysis,
    ForecastStabilityRecord,
    MarketData,
)


class ConfidenceScorer:
    """
    Unified 0-100 confidence scoring system.

    Components and weights:
      ensemble_spread   0.25  — how much do the 250+ members disagree?
      stability         0.20  — has the forecast been stable across past scans?
      calibration       0.15  — overall model Brier score
      liquidity         0.15  — market depth (calibrated to weather market scale)
      horizon           0.10  — days until resolution
      zone_reliability  0.10  — how accurate is the model at THIS probability level?
      city_reliability  0.05  — how accurate is the model for THIS city?

    Note: threshold_distance was removed. It gave 0 points for near-threshold
    markets — exactly where edge exists. Edge is now handled exclusively by
    the trade filter's min_edge thresholds, not the confidence score.
    """

    WEIGHTS = {
        "ensemble_spread":  0.25,
        "stability":        0.20,
        "calibration":      0.15,
        "liquidity":        0.15,
        "horizon":          0.10,
        "zone_reliability": 0.10,
        "city_reliability": 0.05,
    }

    def score(
        self,
        market: MarketData,
        ensemble: EnsembleAnalysis,
        stability: ForecastStabilityRecord,
        calibration_score: float,
        edge: float,
        zone_reliability: float = 50.0,
        city_reliability: float = 50.0,
    ) -> ConfidenceScore:
        ensemble_spread_score = self._ensemble_spread_score(ensemble)
        stab_score            = stability.stability_score * 100.0
        calib_score           = float(calibration_score)
        liquidity_score       = self._liquidity_score(market)
        horizon_score         = self._horizon_score(market)
        zone_rel_score        = float(zone_reliability)
        city_rel_score        = float(city_reliability)

        total = (
            ensemble_spread_score * self.WEIGHTS["ensemble_spread"]
            + stab_score          * self.WEIGHTS["stability"]
            + calib_score         * self.WEIGHTS["calibration"]
            + liquidity_score     * self.WEIGHTS["liquidity"]
            + horizon_score       * self.WEIGHTS["horizon"]
            + zone_rel_score      * self.WEIGHTS["zone_reliability"]
            + city_rel_score      * self.WEIGHTS["city_reliability"]
        )

        breakdown = {
            "ensemble_spread":  round(ensemble_spread_score, 2),
            "stability":        round(stab_score, 2),
            "calibration":      round(calib_score, 2),
            "liquidity":        round(liquidity_score, 2),
            "horizon":          round(horizon_score, 2),
            "zone_reliability": round(zone_rel_score, 2),
            "city_reliability": round(city_rel_score, 2),
            "weights":          self.WEIGHTS,
        }

        return ConfidenceScore(
            total=round(total, 2),
            ensemble_spread_score=round(ensemble_spread_score, 2),
            stability_score=round(stab_score, 2),
            calibration_score=round(calib_score, 2),
            threshold_distance_score=0.0,
            liquidity_score=round(liquidity_score, 2),
            horizon_score=round(horizon_score, 2),
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------ #
    #  Sub-scorers                                                         #
    # ------------------------------------------------------------------ #

    def _ensemble_spread_score(self, ensemble: EnsembleAnalysis) -> float:
        """0 std → 100.  std ≥ 5°C → 0."""
        std = ensemble.weighted_std
        return max(0.0, float(100.0 * (1.0 - min(std / 5.0, 1.0))))

    def _liquidity_score(self, market: MarketData) -> float:
        """< $100 → 0.  ≥ $5k → 100.  Calibrated for weather prediction markets."""
        if market.liquidity < 100:
            return 0.0
        return max(0.0, float(100.0 * min((market.liquidity - 100) / (5_000 - 100), 1.0)))

    def _horizon_score(self, market: MarketData) -> float:
        """≤ 3 days → 100 (short-range reliable).  14+ days → 40."""
        if not market.resolution_date:
            return 60.0
        try:
            resolution = datetime.fromisoformat(
                market.resolution_date.replace("Z", "+00:00")
            )
            now  = datetime.now(resolution.tzinfo)
            days = (resolution - now).total_seconds() / 86_400.0
        except Exception:
            return 60.0

        if days <= 0:
            return 0.0
        elif days <= 3:
            return 100.0
        elif days >= 14:
            return 40.0
        slope = (40.0 - 100.0) / (14.0 - 3.0)
        return float(100.0 + slope * (days - 3.0))
