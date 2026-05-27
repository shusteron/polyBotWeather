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
      ensemble_spread   0.20  — how much do the 150+ members disagree?
      stability         0.15  — has the forecast been stable across past scans?
      calibration       0.10  — overall model Brier score
      threshold_dist    0.20  — how far is the forecast from the betting threshold?
      liquidity         0.10  — market depth
      horizon           0.05  — days until resolution
      zone_reliability  0.15  — how accurate is the model at THIS probability level?
      city_reliability  0.05  — how accurate is the model for THIS city?
    """

    WEIGHTS = {
        "ensemble_spread":  0.20,
        "stability":        0.15,
        "calibration":      0.10,
        "threshold_distance": 0.20,
        "liquidity":        0.10,
        "horizon":          0.05,
        "zone_reliability": 0.15,
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
        ensemble_spread_score    = self._ensemble_spread_score(ensemble)
        stab_score               = stability.stability_score * 100.0
        calib_score              = float(calibration_score)
        threshold_distance_score = self._threshold_distance_score(market, ensemble)
        liquidity_score          = self._liquidity_score(market)
        horizon_score            = self._horizon_score(market)
        zone_rel_score           = float(zone_reliability)
        city_rel_score           = float(city_reliability)

        total = (
            ensemble_spread_score    * self.WEIGHTS["ensemble_spread"]
            + stab_score             * self.WEIGHTS["stability"]
            + calib_score            * self.WEIGHTS["calibration"]
            + threshold_distance_score * self.WEIGHTS["threshold_distance"]
            + liquidity_score        * self.WEIGHTS["liquidity"]
            + horizon_score          * self.WEIGHTS["horizon"]
            + zone_rel_score         * self.WEIGHTS["zone_reliability"]
            + city_rel_score         * self.WEIGHTS["city_reliability"]
        )

        breakdown = {
            "ensemble_spread":    round(ensemble_spread_score, 2),
            "stability":          round(stab_score, 2),
            "calibration":        round(calib_score, 2),
            "threshold_distance": round(threshold_distance_score, 2),
            "liquidity":          round(liquidity_score, 2),
            "horizon":            round(horizon_score, 2),
            "zone_reliability":   round(zone_rel_score, 2),
            "city_reliability":   round(city_rel_score, 2),
            "weights":            self.WEIGHTS,
        }

        return ConfidenceScore(
            total=round(total, 2),
            ensemble_spread_score=round(ensemble_spread_score, 2),
            stability_score=round(stab_score, 2),
            calibration_score=round(calib_score, 2),
            threshold_distance_score=round(threshold_distance_score, 2),
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

    def _threshold_distance_score(
        self, market: MarketData, ensemble: EnsembleAnalysis
    ) -> float:
        """
        < 1°C from threshold → 0 (too close to call).
        ≥ 5°C from threshold → 100.
        """
        if market.threshold is None:
            return 50.0
        distance = abs(ensemble.weighted_mean - market.threshold)
        if distance < 1.0:
            return 0.0
        elif distance >= 5.0:
            return 100.0
        return float((distance - 1.0) / 4.0 * 100.0)

    def _liquidity_score(self, market: MarketData) -> float:
        """< $1k → 0.  ≥ $50k → 100."""
        return max(0.0, float(100.0 * min(market.liquidity / 50_000.0, 1.0)))

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
