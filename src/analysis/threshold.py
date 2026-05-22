from __future__ import annotations


class ThresholdDistanceEngine:
    """
    Analyze how close a forecast value is to a market threshold.
    Close calls (low distance) are risky and should be penalized.
    """

    def calculate_distance(self, forecast_value: float, threshold: float) -> float:
        """
        Return the absolute distance between forecast and threshold.
        Units match the forecast/threshold (e.g. degrees C or F).
        """
        return abs(forecast_value - threshold)

    def get_distance_score(self, distance: float) -> float:
        """
        Convert a distance value to a 0-100 score.
        0 distance → 0 (maximum uncertainty at threshold).
        1 unit → 25.
        5+ units → 100 (confident, well away from threshold).
        """
        if distance <= 0.0:
            return 0.0
        elif distance >= 5.0:
            return 100.0
        else:
            # Non-linear: sqrt gives extra reward for moving away quickly
            return float(min(100.0, (distance / 5.0) ** 0.5 * 100.0))

    def is_close_call(self, distance: float, min_distance: float = 2.0) -> bool:
        """
        Return True if the distance is below the minimum acceptable distance.
        A close call means the forecast is too near the threshold for confidence.
        """
        return distance < min_distance

    def get_direction_label(self, forecast_value: float, threshold: float) -> str:
        """Return 'ABOVE' or 'BELOW' based on forecast vs threshold."""
        return "ABOVE" if forecast_value > threshold else "BELOW"

    def calculate_probability_distance(
        self, probability: float, neutral: float = 0.5
    ) -> float:
        """
        Return how far a probability is from the neutral 50% mark.
        Used as a secondary signal for confidence.
        """
        return abs(probability - neutral)
