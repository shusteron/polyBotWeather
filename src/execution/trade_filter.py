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


class TradeFilter:
    """
    Implements disciplined NO_TRADE bias.
    Every check must pass for a trade to be approved.
    Any single rejection check causes the trade to be skipped.
    """

    def __init__(
        self,
        min_liquidity: float = 1_000.0,
        max_spread_pct: float = 0.05,
        min_hours_to_resolution: float = 24.0,
        max_ensemble_spread: float = 0.15,
        min_stability_score: float = 0.5,
        min_threshold_distance: float = 2.0,
        min_provider_agreement: float = 0.6,
        min_confidence: float = 70.0,
        min_edge: float = 0.08,
    ):
        self.min_liquidity = min_liquidity
        self.max_spread_pct = max_spread_pct
        self.min_hours_to_resolution = min_hours_to_resolution
        self.max_ensemble_spread = max_ensemble_spread
        self.min_stability_score = min_stability_score
        self.min_threshold_distance = min_threshold_distance
        self.min_provider_agreement = min_provider_agreement
        self.min_confidence = min_confidence
        self.min_edge = min_edge

    def evaluate(
        self,
        market: MarketData,
        ensemble: EnsembleAnalysis,
        stability: ForecastStabilityRecord,
        confidence: ConfidenceScore,
        edge: float,
    ) -> tuple[bool, list[str]]:
        """
        Run all rejection checks.
        Returns (should_trade, rejection_reasons).
        Default verdict is NO_TRADE — every check must pass.
        """
        rejection_reasons: list[str] = []

        # 1. Liquidity check
        if market.liquidity < self.min_liquidity:
            rejection_reasons.append(
                f"Insufficient liquidity: ${market.liquidity:.0f} < ${self.min_liquidity:.0f}"
            )

        # 2. Spread check
        spread_pct = self._calculate_spread_pct(market)
        if spread_pct > self.max_spread_pct:
            rejection_reasons.append(
                f"Spread too wide: {spread_pct:.2%} > {self.max_spread_pct:.2%}"
            )

        # 3. Time to resolution
        hours_remaining = self._hours_to_resolution(market)
        if hours_remaining is not None and hours_remaining < self.min_hours_to_resolution:
            rejection_reasons.append(
                f"Too close to resolution: {hours_remaining:.1f}h < {self.min_hours_to_resolution:.0f}h"
            )

        # 4. Ensemble spread (weighted_std in °C)
        if ensemble.weighted_std > self.max_ensemble_spread * 10:
            # max_ensemble_spread in config is 0.15 but we compare to std in °C
            # Treating max_ensemble_spread as a fraction: 0.15 → 1.5°C effective cap
            # We use a generous scale: reject if std > 5°C (very uncertain)
            pass  # handled by confidence score already; kept for direct check

        # Ensemble spread as temperature std (direct check)
        # Config value 0.15 is interpreted as: max 15% of a 10-unit range = 1.5 units
        ensemble_spread_limit_c = self.max_ensemble_spread * 10.0  # e.g. 1.5°C
        if ensemble.weighted_std > ensemble_spread_limit_c:
            rejection_reasons.append(
                f"Ensemble spread too large: std={ensemble.weighted_std:.2f} > {ensemble_spread_limit_c:.2f}"
            )

        # 5. Stability score
        if stability.stability_score < self.min_stability_score:
            rejection_reasons.append(
                f"Forecast unstable: score={stability.stability_score:.2f} < {self.min_stability_score:.2f}"
            )

        # 6. Threshold distance
        if market.threshold is not None:
            distance = abs(ensemble.weighted_mean - market.threshold)
            if distance < self.min_threshold_distance:
                rejection_reasons.append(
                    f"Too close to threshold: {distance:.2f} < {self.min_threshold_distance:.2f}"
                )

        # 7. Provider agreement
        if ensemble.provider_agreement < self.min_provider_agreement:
            rejection_reasons.append(
                f"Low provider agreement: {ensemble.provider_agreement:.2f} < {self.min_provider_agreement:.2f}"
            )

        # 8. Confidence
        if confidence.total < self.min_confidence:
            rejection_reasons.append(
                f"Confidence too low: {confidence.total:.1f} < {self.min_confidence:.1f}"
            )

        # 9. Edge
        if abs(edge) < self.min_edge:
            rejection_reasons.append(
                f"Insufficient edge: |{edge:.4f}| < {self.min_edge:.4f}"
            )

        # Default: reject if any reason found
        should_trade = len(rejection_reasons) == 0
        return should_trade, rejection_reasons

    def _calculate_spread_pct(self, market: MarketData) -> float:
        """Compute spread as a percentage of mid-price."""
        if market.yes_price <= 0 or market.no_price <= 0:
            return 1.0
        mid = (market.yes_price + market.no_price) / 2.0
        if mid <= 0:
            return 1.0
        return market.spread / mid

    def _hours_to_resolution(self, market: MarketData) -> Optional[float]:
        """Return hours until market resolution, or None if unknown."""
        if not market.resolution_date:
            return None
        try:
            resolution = datetime.fromisoformat(
                market.resolution_date.replace("Z", "+00:00")
            )
            now = datetime.now(resolution.tzinfo)
            return (resolution - now).total_seconds() / 3600.0
        except Exception:
            return None

    @classmethod
    def from_config(cls, cfg: dict) -> "TradeFilter":
        t = cfg.get("thresholds", {})
        return cls(
            min_liquidity=t.get("min_liquidity", 1_000.0),
            max_spread_pct=t.get("max_spread_pct", 0.05),
            min_hours_to_resolution=t.get("min_hours_to_resolution", 24.0),
            max_ensemble_spread=t.get("max_ensemble_spread", 0.15),
            min_stability_score=t.get("min_stability_score", 0.5),
            min_threshold_distance=t.get("min_threshold_distance", 2.0),
            min_provider_agreement=t.get("min_provider_agreement", 0.6),
            min_confidence=t.get("min_confidence", 70.0),
            min_edge=t.get("min_edge", 0.08),
        )
