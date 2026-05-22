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
    Combines ensemble spread, forecast stability, calibration history,
    threshold distance, liquidity, and forecast horizon.
    """

    WEIGHTS = {
        "ensemble_spread": 0.25,
        "stability": 0.20,
        "calibration": 0.20,
        "threshold_distance": 0.20,
        "liquidity": 0.10,
        "horizon": 0.05,
    }

    def score(
        self,
        market: MarketData,
        ensemble: EnsembleAnalysis,
        stability: ForecastStabilityRecord,
        calibration_score: float,
        edge: float,
    ) -> ConfidenceScore:
        """
        Compute all sub-scores and weighted total confidence.
        """
        ensemble_spread_score = self._ensemble_spread_score(ensemble)
        stab_score = stability.stability_score * 100.0
        calib_score = float(calibration_score)
        threshold_distance_score = self._threshold_distance_score(market, ensemble)
        liquidity_score = self._liquidity_score(market)
        horizon_score = self._horizon_score(market)

        total = (
            ensemble_spread_score * self.WEIGHTS["ensemble_spread"]
            + stab_score * self.WEIGHTS["stability"]
            + calib_score * self.WEIGHTS["calibration"]
            + threshold_distance_score * self.WEIGHTS["threshold_distance"]
            + liquidity_score * self.WEIGHTS["liquidity"]
            + horizon_score * self.WEIGHTS["horizon"]
        )

        breakdown = {
            "ensemble_spread": round(ensemble_spread_score, 2),
            "stability": round(stab_score, 2),
            "calibration": round(calib_score, 2),
            "threshold_distance": round(threshold_distance_score, 2),
            "liquidity": round(liquidity_score, 2),
            "horizon": round(horizon_score, 2),
            "weights": self.WEIGHTS,
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

    def _ensemble_spread_score(self, ensemble: EnsembleAnalysis) -> float:
        """
        Score based on weighted std of the ensemble.
        0 std → 100. std >= 5°C → 0.
        """
        std = ensemble.weighted_std
        score = 100.0 * (1.0 - min(std / 5.0, 1.0))
        return max(0.0, float(score))

    def _threshold_distance_score(
        self, market: MarketData, ensemble: EnsembleAnalysis
    ) -> float:
        """
        Score based on distance between forecast mean and market threshold.
        < 1°C from threshold → 0 (too risky / close call).
        >= 5°C from threshold → 100.
        """
        if market.threshold is None:
            return 50.0  # neutral if no threshold defined

        distance = abs(ensemble.weighted_mean - market.threshold)

        if distance < 1.0:
            return 0.0
        elif distance >= 5.0:
            return 100.0
        else:
            # Linear interpolation between 1 and 5 degrees
            return float((distance - 1.0) / 4.0 * 100.0)

    def _liquidity_score(self, market: MarketData) -> float:
        """
        Score based on market liquidity.
        < $1k → 0. >= $50k → 100. Linear in between.
        """
        liq = market.liquidity
        score = 100.0 * min(liq / 50_000.0, 1.0)
        return max(0.0, float(score))

    def _horizon_score(self, market: MarketData) -> float:
        """
        Score based on time until resolution.
        <= 3 days → 100 (short-range is reliable).
        14+ days → 40 (long-range is uncertain).
        """
        if not market.resolution_date:
            return 60.0  # moderate default

        try:
            resolution = datetime.fromisoformat(
                market.resolution_date.replace("Z", "+00:00")
            )
            now = datetime.now(resolution.tzinfo)
            days = (resolution - now).total_seconds() / 86_400.0
        except Exception:
            return 60.0

        if days <= 0:
            return 0.0
        elif days <= 3:
            return 100.0
        elif days >= 14:
            return 40.0
        else:
            # Linear from 100 at 3 days to 40 at 14 days
            slope = (40.0 - 100.0) / (14.0 - 3.0)
            return float(100.0 + slope * (days - 3.0))
