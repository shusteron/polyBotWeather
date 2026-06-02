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
    Disciplined NO_TRADE bias — every check must pass.

    Key improvements over v1:
      • Minimum entry price — rejects sub-cent lottery tickets
      • Dynamic edge requirement — when model and market lean the SAME direction
        (e.g. both say YES is likely, but disagree on how likely) a much larger
        gap is required, because weather models are not precise enough to
        reliably distinguish 86% from 99.7%.  When the model is genuinely
        contrarian to the market the bar is lower.
      • Zone reliability gate — once enough calibration history exists, skip
        probability zones where the model has proven inaccurate.
    """

    def __init__(
        self,
        # Market quality
        min_liquidity: float = 1_000.0,
        max_spread_pct: float = 0.05,
        min_hours_to_resolution: float = 24.0,
        # Forecast quality
        max_ensemble_spread: float = 0.15,
        min_stability_score: float = 0.5,
        min_threshold_distance: float = 2.0,
        min_provider_agreement: float = 0.6,
        # Overall confidence
        min_confidence: float = 72.0,
        # --- New in v2 ---
        # Price floor — never buy a side priced below this
        min_market_price: float = 0.005,
        # Edge requirements split by direction alignment
        min_edge_same_direction: float = 0.30,  # both model+market lean same way
        min_edge_contrarian: float = 0.15,       # model and market disagree on direction
        # Zone reliability (0-100).  Only enforced once enough samples exist.
        min_zone_reliability: float = 40.0,
        # Minimum model probability for the side being purchased.
        # Prevents buying a side the model itself considers unlikely (e.g. model=68% YES → don't buy NO).
        min_purchased_side_prob: float = 0.35,
    ):
        self.min_liquidity             = min_liquidity
        self.max_spread_pct            = max_spread_pct
        self.min_hours_to_resolution   = min_hours_to_resolution
        self.max_ensemble_spread       = max_ensemble_spread
        self.min_stability_score       = min_stability_score
        self.min_threshold_distance    = min_threshold_distance
        self.min_provider_agreement    = min_provider_agreement
        self.min_confidence            = min_confidence
        self.min_market_price          = min_market_price
        self.min_edge_same_direction   = min_edge_same_direction
        self.min_edge_contrarian       = min_edge_contrarian
        self.min_zone_reliability      = min_zone_reliability
        self.min_purchased_side_prob   = min_purchased_side_prob

    def evaluate(
        self,
        market: MarketData,
        ensemble: EnsembleAnalysis,
        stability: ForecastStabilityRecord,
        confidence: ConfidenceScore,
        edge: float,
        model_prob: float = 0.5,
        zone_reliability: float = 50.0,
        zone_sample_count: int = 0,
        min_zone_samples: int = 20,
    ) -> tuple[bool, list[str]]:
        """
        Run all rejection checks.
        Returns (should_trade, rejection_reasons).
        """
        rejection_reasons: list[str] = []

        # 1. Liquidity
        if market.liquidity < self.min_liquidity:
            rejection_reasons.append(
                f"Insufficient liquidity: ${market.liquidity:.0f} < ${self.min_liquidity:.0f}"
            )

        # 2. Spread
        spread_pct = self._spread_pct(market)
        if spread_pct > self.max_spread_pct:
            rejection_reasons.append(
                f"Spread too wide: {spread_pct:.2%} > {self.max_spread_pct:.2%}"
            )

        # 3. Time to resolution
        hours = self._hours_to_resolution(market)
        if hours is not None and hours < self.min_hours_to_resolution:
            rejection_reasons.append(
                f"Too close to resolution: {hours:.1f}h < {self.min_hours_to_resolution:.0f}h"
            )

        # 4. Ensemble spread
        spread_limit_c = self.max_ensemble_spread * 10.0
        if ensemble.weighted_std > spread_limit_c:
            rejection_reasons.append(
                f"Ensemble spread too large: std={ensemble.weighted_std:.2f}°C > {spread_limit_c:.2f}°C"
            )

        # 5. Forecast stability
        if stability.stability_score < self.min_stability_score:
            rejection_reasons.append(
                f"Forecast unstable: {stability.stability_score:.2f} < {self.min_stability_score:.2f}"
            )

        # 6. Threshold distance
        if market.threshold is not None:
            distance = abs(ensemble.weighted_mean - market.threshold)
            if distance < self.min_threshold_distance:
                rejection_reasons.append(
                    f"Too close to threshold: {distance:.2f}°C < {self.min_threshold_distance:.2f}°C"
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

        # 9. Minimum entry price (NO LOTTERY TICKETS)
        #    The side we are buying must be priced above min_market_price.
        entry_price = market.yes_price if edge > 0 else market.no_price
        if entry_price < self.min_market_price:
            rejection_reasons.append(
                f"Entry price too low: {entry_price:.4f} < {self.min_market_price:.4f} "
                f"(sub-cent bets have too much variance)"
            )

        # 10. Dynamic edge — requirement depends on whether model and market
        #     lean the SAME direction or OPPOSITE directions.
        #
        #     Same direction example: model=86% YES, market=99.7% YES
        #     → both say YES is likely; the model is just less confident.
        #     Weather models are NOT precise enough at the 85-99% range to
        #     reliably distinguish these.  Require a large gap (default 30%).
        #
        #     Contrarian example: model=39% YES, market=1.6% YES
        #     → market says nearly impossible; model says real chance.
        #     This is a strong structural disagreement. Lower bar (default 15%).
        model_leans_yes  = model_prob > 0.5
        market_leans_yes = market.yes_price > 0.5
        same_direction   = (model_leans_yes == market_leans_yes)

        required_edge = (
            self.min_edge_same_direction if same_direction
            else self.min_edge_contrarian
        )
        direction_label = "same-direction" if same_direction else "contrarian"

        if abs(edge) < required_edge:
            rejection_reasons.append(
                f"Insufficient {direction_label} edge: "
                f"|{edge:.4f}| < {required_edge:.4f}"
            )

        # 11. Zone reliability — once we have enough history, skip zones
        #     where the model has consistently been wrong.
        if zone_sample_count >= min_zone_samples:
            if zone_reliability < self.min_zone_reliability:
                rejection_reasons.append(
                    f"Model unreliable in this probability zone: "
                    f"{zone_reliability:.1f} < {self.min_zone_reliability:.1f} "
                    f"(n={zone_sample_count})"
                )

        # 12. Model must believe in the side being purchased.
        #     If edge > 0 we buy YES; if edge < 0 we buy NO.
        #     Blocks cases like: model=68% YES → edge negative → buy NO at 32% model prob.
        purchased_side_prob = model_prob if edge > 0 else (1.0 - model_prob)
        if purchased_side_prob < self.min_purchased_side_prob:
            side = "YES" if edge > 0 else "NO"
            rejection_reasons.append(
                f"Model probability for {side} side too low: "
                f"{purchased_side_prob:.3f} < {self.min_purchased_side_prob:.3f}"
            )

        should_trade = len(rejection_reasons) == 0
        return should_trade, rejection_reasons

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _spread_pct(self, market: MarketData) -> float:
        if market.yes_price <= 0 or market.no_price <= 0:
            return 1.0
        mid = (market.yes_price + market.no_price) / 2.0
        return market.spread / mid if mid > 0 else 1.0

    def _hours_to_resolution(self, market: MarketData) -> Optional[float]:
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
            min_liquidity            = t.get("min_liquidity",             1_000.0),
            max_spread_pct           = t.get("max_spread_pct",            0.05),
            min_hours_to_resolution  = t.get("min_hours_to_resolution",   24.0),
            max_ensemble_spread      = t.get("max_ensemble_spread",       0.15),
            min_stability_score      = t.get("min_stability_score",       0.5),
            min_threshold_distance   = t.get("min_threshold_distance",    2.0),
            min_provider_agreement   = t.get("min_provider_agreement",    0.6),
            min_confidence           = t.get("min_confidence",            72.0),
            min_market_price         = t.get("min_market_price",          0.005),
            min_edge_same_direction  = t.get("min_edge_same_direction",   0.30),
            min_edge_contrarian      = t.get("min_edge_contrarian",       0.15),
            min_zone_reliability     = t.get("min_zone_reliability",      40.0),
            min_purchased_side_prob  = t.get("min_purchased_side_prob",   0.35),
        )
