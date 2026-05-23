from __future__ import annotations

import time
from datetime import date
from typing import Optional

import numpy as np
import requests
from loguru import logger

from ..models import WeatherForecast

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ENSEMBLE_MODELS = {
    "ecmwf_ifs025": "ECMWF IFS 0.25° (50 members)",
    "gfs025":        "GFS 0.25° (30 members)",
    "icon_seamless": "ICON Seamless (39 members)",
    "gem_global":    "GEM Global (20 members)",
    "bom_access_global_ensemble": "BOM ACCESS (17 members)",
}

VARIABLE_MAP = {
    "temperature": "temperature_2m",
    "temperature_max": "temperature_2m_max",
    "temperature_min": "temperature_2m_min",
    "precipitation": "precipitation_sum",
    "wind": "windspeed_10m_max",
}


class OpenMeteoClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WeatherPredictionBot/1.0"})

    def get_ensemble_forecast(
        self,
        lat: float,
        lon: float,
        target_date: str,
        variable: str = "temperature_2m",
        location: str = "unknown",
        measurement: str = "max",  # "max", "min", or "mean"
    ) -> list[WeatherForecast]:
        """
        Fetch ensemble forecasts for all configured models.
        measurement: how to aggregate hourly → daily ("max" for highest temp, "min" for lowest)
        """
        results: list[WeatherForecast] = []
        for model_key in ENSEMBLE_MODELS:
            forecast = self._fetch_model_ensemble(
                lat, lon, target_date, model_key, variable, location, measurement
            )
            if forecast:
                results.append(forecast)
            time.sleep(0.5)  # respect Open-Meteo rate limits
        return results

    def _fetch_model_ensemble(
        self,
        lat: float,
        lon: float,
        target_date: str,
        model: str,
        variable: str,
        location: str,
        measurement: str = "max",
    ) -> Optional[WeatherForecast]:
        try:
            # Determine the date range
            tdate = date.fromisoformat(target_date)
            start = tdate.strftime("%Y-%m-%d")
            end = tdate.strftime("%Y-%m-%d")

            # Build ensemble member variable list
            # Open-Meteo ensemble uses member0..memberN suffix
            hourly_variable = variable

            params = {
                "latitude": lat,
                "longitude": lon,
                "models": model,
                "hourly": hourly_variable,
                "start_date": start,
                "end_date": end,
                "timezone": "UTC",
            }

            resp = self.session.get(ENSEMBLE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            members = self._parse_ensemble_members(data, model, target_date, variable, measurement)
            if not members:
                logger.warning(f"No ensemble members parsed for model {model}")
                return None

            arr = np.array(members)
            return WeatherForecast(
                location=location,
                lat=lat,
                lon=lon,
                target_date=target_date,
                model_name=model,
                ensemble_members=members,
                mean=float(np.mean(arr)),
                std=float(np.std(arr)),
                spread=float(np.percentile(arr, 90) - np.percentile(arr, 10)),
                p10=float(np.percentile(arr, 10)),
                p25=float(np.percentile(arr, 25)),
                p50=float(np.percentile(arr, 50)),
                p75=float(np.percentile(arr, 75)),
                p90=float(np.percentile(arr, 90)),
            )

        except Exception as exc:
            logger.warning(f"Failed to fetch ensemble for model {model}: {exc}")
            return None

    def _parse_ensemble_members(
        self,
        data: dict,
        model_name: str,
        target_date: str,
        variable: str,
        measurement: str = "max",
    ) -> list[float]:
        """
        Extract one value per ensemble member for the target day.
        measurement controls how hourly → daily aggregation works:
          "max"  → daily maximum (for Highest temperature markets)
          "min"  → daily minimum (for Lowest temperature markets)
          "mean" → daily mean
          "sum"  → daily sum (for precipitation)
        """
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        members: list[float] = []

        member_keys = [k for k in hourly if k.startswith(variable) and "member" in k]

        def agg(vals: list[float]) -> float:
            if measurement == "max":
                return float(np.max(vals))
            elif measurement == "min":
                return float(np.min(vals))
            elif measurement == "sum":
                return float(np.sum(vals))
            else:
                return float(np.mean(vals))

        if not member_keys:
            values = hourly.get(variable, [])
            daily_vals = self._filter_daily(values, times, target_date)
            if daily_vals:
                members = [agg(daily_vals)]
        else:
            for key in member_keys:
                values = hourly.get(key, [])
                daily_vals = self._filter_daily(values, times, target_date)
                if daily_vals:
                    members.append(agg(daily_vals))

        return members

    def _filter_daily(self, values: list, times: list, target_date: str) -> list[float]:
        """Return all non-null hourly floats for the target date."""
        result = []
        for t, v in zip(times, values):
            if t.startswith(target_date) and v is not None:
                try:
                    result.append(float(v))
                except (TypeError, ValueError):
                    pass
        return result

    def get_deterministic_forecast(
        self, lat: float, lon: float, target_date: str, location: str = "unknown"
    ) -> Optional[WeatherForecast]:
        """Standard deterministic forecast for comparison."""
        try:
            tdate = date.fromisoformat(target_date)
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "start_date": tdate.strftime("%Y-%m-%d"),
                "end_date": tdate.strftime("%Y-%m-%d"),
                "timezone": "UTC",
            }
            resp = self.session.get(FORECAST_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_max", [])
            if not temps:
                return None

            val = float(temps[0]) if temps[0] is not None else 0.0
            return WeatherForecast(
                location=location,
                lat=lat,
                lon=lon,
                target_date=target_date,
                model_name="deterministic",
                ensemble_members=[val],
                mean=val,
                std=0.0,
                spread=0.0,
                p10=val,
                p25=val,
                p50=val,
                p75=val,
                p90=val,
            )
        except Exception as exc:
            logger.warning(f"Failed to fetch deterministic forecast: {exc}")
            return None
