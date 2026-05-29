from __future__ import annotations

import numpy as np
from loguru import logger

from ..models import EnsembleAnalysis, WeatherForecast

DEFAULT_WEIGHTS = {
    "ecmwf_ifs025": 0.40,   # best global NWP model
    "gfs025":        0.25,   # NOAA operational
    "icon_seamless": 0.20,   # DWD, excellent over Europe
    "gem_global":    0.10,   # Canadian, good mid-latitudes
    "bom_access_global_ensemble": 0.05,  # Australian BOM
    "deterministic": 0.05,
}


class EnsembleAnalyzer:
    def __init__(self, model_weights: dict[str, float] | None = None):
        self.model_weights = model_weights or DEFAULT_WEIGHTS

    def analyze(
        self, forecasts: list[WeatherForecast], weights: dict[str, float] | None = None
    ) -> EnsembleAnalysis:
        """
        Aggregate multiple model forecasts into an EnsembleAnalysis.
        Applies configurable model weights.
        """
        if not forecasts:
            logger.warning("No forecasts provided to EnsembleAnalyzer.analyze")
            return EnsembleAnalysis()

        weights = weights or self.model_weights

        all_members: list[float] = []
        weighted_means: list[float] = []
        weighted_stds: list[float] = []
        model_means: list[float] = []
        total_weight = 0.0

        for forecast in forecasts:
            w = weights.get(forecast.model_name, 1.0 / len(forecasts))
            if forecast.ensemble_members:
                all_members.extend(forecast.ensemble_members)
                weighted_means.append(forecast.mean * w)
                weighted_stds.append(forecast.std * w)
                model_means.append(forecast.mean)
                total_weight += w

        if total_weight == 0.0 or not all_members:
            logger.warning("Zero total weight or no members in ensemble analysis")
            return EnsembleAnalysis(forecasts=forecasts)

        # Normalize
        weighted_mean = sum(weighted_means) / total_weight
        weighted_std = sum(weighted_stds) / total_weight

        # Provider agreement: 1 - CV across model means
        if len(model_means) > 1:
            mean_of_means = np.mean(model_means)
            std_of_means = np.std(model_means)
            if mean_of_means != 0:
                cv = std_of_means / abs(mean_of_means)
                provider_agreement = float(max(0.0, 1.0 - cv))
            else:
                provider_agreement = 1.0 if std_of_means == 0 else 0.0
        else:
            provider_agreement = 1.0

        all_arr = np.array(all_members)
        confidence_range = float(np.percentile(all_arr, 90) - np.percentile(all_arr, 10))

        return EnsembleAnalysis(
            forecasts=forecasts,
            weighted_mean=float(weighted_mean),
            weighted_std=float(weighted_std),
            provider_agreement=float(provider_agreement),
            confidence_range=float(confidence_range),
            all_members=all_members,
        )

    @staticmethod
    def _laplace_smooth(hits: int, n: int) -> float:
        """
        Laplace (add-one) smoothing — prevents returning exactly 0.0 or 1.0.

        Formula: (hits + 1) / (n + 2)

        Examples with 150 members:
          All agree YES  → 151/152 = 0.993  (not 1.0)
          None agree     →   1/152 = 0.0066 (not 0.0)
          Half agree     →  76/152 = 0.500  (unchanged)
        """
        return (hits + 1) / (n + 2)

    def calculate_probability_above_threshold(
        self, analysis: EnsembleAnalysis, threshold: float
    ) -> float:
        """
        Estimate P(value > threshold) using Laplace-smoothed ensemble fraction.
        Never returns exactly 0.0 or 1.0 — acknowledges model uncertainty.
        """
        if not analysis.all_members:
            logger.warning("No ensemble members to calculate probability above threshold")
            return 0.5
        members = np.array(analysis.all_members)
        hits = int(np.sum(members > threshold))
        n = len(members)
        raw = hits / n
        prob = self._laplace_smooth(hits, n)
        if raw in (0.0, 1.0):
            logger.warning(
                f"All {n} ensemble members {'above' if raw == 1.0 else 'below'} "
                f"threshold {threshold:.1f} — Laplace-smoothed to {prob:.4f}"
            )
        return prob

    def calculate_probability_below_threshold(
        self, analysis: EnsembleAnalysis, threshold: float
    ) -> float:
        """
        Estimate P(value < threshold) using Laplace-smoothed ensemble fraction.
        Never returns exactly 0.0 or 1.0.
        """
        if not analysis.all_members:
            return 0.5
        members = np.array(analysis.all_members)
        hits = int(np.sum(members < threshold))
        n = len(members)
        raw = hits / n
        prob = self._laplace_smooth(hits, n)
        if raw in (0.0, 1.0):
            logger.warning(
                f"All {n} ensemble members {'below' if raw == 1.0 else 'above'} "
                f"threshold {threshold:.1f} — Laplace-smoothed to {prob:.4f}"
            )
        return prob

    def calculate_probability_in_range(
        self, analysis: EnsembleAnalysis, low: float, high: float
    ) -> float:
        """
        Estimate P(low <= value <= high) using Laplace-smoothed ensemble fraction.
        Never returns exactly 0.0 or 1.0.
        """
        if not analysis.all_members:
            return 0.5
        members = np.array(analysis.all_members)
        hits = int(np.sum((members >= low) & (members <= high)))
        n = len(members)
        return self._laplace_smooth(hits, n)

    def get_model_breakdown(self, analysis: EnsembleAnalysis) -> dict[str, dict]:
        """Return a summary dict of each model's contribution."""
        breakdown = {}
        for f in analysis.forecasts:
            breakdown[f.model_name] = {
                "mean": f.mean,
                "std": f.std,
                "spread": f.spread,
                "n_members": len(f.ensemble_members),
                "p10": f.p10,
                "p50": f.p50,
                "p90": f.p90,
            }
        return breakdown
