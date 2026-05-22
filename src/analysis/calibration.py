from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from loguru import logger

from ..models import CalibrationRecord

CALIBRATION_FILE = os.path.join("data", "calibration.json")


class CalibrationEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.calibration_file = os.path.join(data_dir, "calibration.json")
        self._records: list[dict] = []
        self._load_records()

    def record_trade(self, record: CalibrationRecord) -> None:
        """Persist a new calibration record to the JSON store."""
        self._records.append(record.model_dump())
        self._save_records()

    def update_outcome(self, market_id: str, outcome: bool, pnl: float) -> None:
        """Update the outcome and PnL for an existing calibration record."""
        for r in self._records:
            if r.get("market_id") == market_id and r.get("outcome") is None:
                r["outcome"] = outcome
                r["pnl"] = pnl
                break
        self._save_records()

    def get_calibration_score(
        self,
        region: Optional[str] = None,
        horizon_days: Optional[int] = None,
        season: Optional[str] = None,
    ) -> float:
        """
        Return a calibration score (0-100).
        Filters by region, horizon, and/or season if provided.
        Perfect calibration → 50% predicted = 50% actual hit rate.
        """
        records = self._get_resolved_records()
        if not records:
            return 50.0  # neutral when no history

        brier = self.get_brier_score(records)
        # Brier score range: 0 (perfect) to 1 (worst). Convert to 0-100 score.
        # A random predictor scores ~0.25. Score 100 = Brier 0, Score 0 = Brier 0.5+
        score = float(max(0.0, 100.0 * (1.0 - brier / 0.25)))
        return min(100.0, score)

    def get_brier_score(self, records: list[dict]) -> float:
        """
        Compute the Brier score for a list of resolved calibration records.
        Brier = mean((predicted_prob - outcome)^2)
        """
        resolved = [r for r in records if r.get("outcome") is not None]
        if not resolved:
            return 0.25  # return naive score when no data

        errors = []
        for r in resolved:
            p = float(r.get("predicted_probability", 0.5))
            o = float(r["outcome"])
            errors.append((p - o) ** 2)
        return float(np.mean(errors))

    def get_calibration_curve(self, records: list[dict]) -> list[dict]:
        """
        Bin predictions into deciles and compute actual hit rates per bin.
        Returns a list of dicts: {bin_low, bin_high, predicted_mid, actual_rate, count}
        """
        resolved = [r for r in records if r.get("outcome") is not None]
        bins = []
        bin_edges = list(range(0, 110, 10))

        for i in range(len(bin_edges) - 1):
            low = bin_edges[i] / 100.0
            high = bin_edges[i + 1] / 100.0
            bucket = [
                r for r in resolved
                if low <= float(r.get("predicted_probability", 0)) < high
            ]
            actual_rate = float(np.mean([r["outcome"] for r in bucket])) if bucket else 0.0
            bins.append({
                "bin_low": low,
                "bin_high": high,
                "predicted_mid": (low + high) / 2.0,
                "actual_rate": actual_rate,
                "count": len(bucket),
            })
        return bins

    def get_region_reliability(self, region: str) -> float:
        """
        Return a calibration score (0-100) for trades in a specific region.
        Currently uses market title matching as a proxy for region.
        """
        region_records = [
            r for r in self._records
            if region.lower() in str(r.get("market_id", "")).lower()
        ]
        if not region_records:
            return 50.0
        resolved = [r for r in region_records if r.get("outcome") is not None]
        if not resolved:
            return 50.0
        brier = self.get_brier_score(resolved)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - brier / 0.25))))

    def _get_resolved_records(self) -> list[dict]:
        return [r for r in self._records if r.get("outcome") is not None]

    def _load_records(self) -> None:
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r") as f:
                    self._records = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load calibration records: {exc}")
                self._records = []
        else:
            self._records = []

    def _save_records(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            with open(self.calibration_file, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
        except Exception as exc:
            logger.error(f"Failed to save calibration records: {exc}")
