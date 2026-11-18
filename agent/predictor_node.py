"""
Agent tool node wrapping the LightGBM delay predictor and empirical trip distribution models.
Provides natural-language entity extraction and trip reliability forecasting.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from models.delay_predictor import delay_predictor
from models.trip_stats import TripReliabilityStats, calculate_empirical_stats

logger = logging.getLogger("septa_ops.agent.predictor_node")

STOP_ALIAS_MAP = {
    "temple": "STOP_TEMPLE",
    "temple university": "STOP_TEMPLE",
    "temple u": "STOP_TEMPLE",
    "30th": "STOP_30TH",
    "30th st": "STOP_30TH",
    "30th street": "STOP_30TH",
    "30th street station": "STOP_30TH",
    "jefferson": "STOP_JEFF",
    "jefferson station": "STOP_JEFF",
    "market east": "STOP_JEFF",
    "suburban": "STOP_SUB",
    "suburban station": "STOP_SUB",
    "phl": "STOP_PHL_A",
    "airport": "STOP_PHL_A",
    "phl airport": "STOP_PHL_A",
    "terminal a": "STOP_PHL_A",
    "terminal b": "STOP_PHL_B",
    "terminal c": "STOP_PHL_CD",
    "terminal d": "STOP_PHL_CD",
    "terminal e": "STOP_PHL_EF",
    "terminal f": "STOP_PHL_EF",
    "city hall": "STOP_CITY_HALL",
}


def normalize_stop_identifier(name: str) -> str:
    """Normalizes natural-language station names to canonical GTFS stop IDs."""
    cleaned = name.lower().strip()
    for alias, stop_id in STOP_ALIAS_MAP.items():
        if alias in cleaned:
            return stop_id
    if name.upper().startswith("STOP_"):
        return name.upper()
    return "STOP_TEMPLE" if "temple" in cleaned else "STOP_PHL_A"


@dataclass
class TripPredictionResponse:
    origin_stop_id: str
    origin_name: str
    destination_stop_id: str
    destination_name: str
    route_name: str
    scheduled_duration_min: int
    predicted_delay_min: float
    delay_lower_bound_min: float
    delay_upper_bound_min: float
    p50_delay_min: float
    p90_delay_min: float
    on_time_probability: float
    crowding_index: float
    crowding_level: str
    recommended_departure_buffer_min: int
    narrative_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TripPredictorNode:
    """Agent tool wrapper for commuter reliability analytics."""

    def __init__(self):
        self.predictor = delay_predictor

    def predict_trip(
        self,
        origin: str = "STOP_TEMPLE",
        destination: str = "STOP_PHL_A",
        departure_hour: int = 8,
        is_weekend: bool = False,
        precipitation_inches: float = 0.0,
        temp_f: float = 68.0,
        wind_speed_mph: float = 8.0,
        has_active_detour: bool = False,
    ) -> TripPredictionResponse:
        """
        Computes empirical percentiles and LightGBM regression forecast for an OD pair.
        """
        origin_id = normalize_stop_identifier(origin)
        dest_id = normalize_stop_identifier(destination)

        # 1. Empirical distribution metrics
        weather_factor = 1.0 + (1.8 * precipitation_inches) + (0.01 * max(0, wind_speed_mph - 15.0))
        if has_active_detour:
            weather_factor += 0.4

        emp_stats: TripReliabilityStats = calculate_empirical_stats(
            origin_id=origin_id,
            destination_id=dest_id,
            departure_hour=departure_hour,
            is_weekend=is_weekend,
            weather_impact_factor=weather_factor,
        )

        # 2. LightGBM delay prediction
        ml_inputs = {
            "origin_stop_id": origin_id,
            "destination_stop_id": dest_id,
            "departure_hour": departure_hour,
            "is_weekend": int(is_weekend),
            "temp_f": temp_f,
            "precipitation_inches": precipitation_inches,
            "wind_speed_mph": wind_speed_mph,
            "has_active_detour": int(has_active_detour),
            "historical_route_otp": emp_stats.on_time_probability,
        }
        ml_res = self.predictor.predict(ml_inputs)
        pred_delay = ml_res["predicted_delay_min"]

        # 3. Formulate commuter narrative
        narrative = (
            f"Trip from {emp_stats.origin_stop_name} to {emp_stats.destination_stop_name} "
            f"via {emp_stats.route_name}: Scheduled travel time is {emp_stats.scheduled_duration_min} mins. "
            f"Under current conditions, expected delay is {pred_delay:.1f} mins (p50: {emp_stats.p50_delay_min} min, "
            f"p90: {emp_stats.p90_delay_min} min). On-time performance probability is {int(emp_stats.on_time_probability * 100)}%. "
            f"Vehicle crowding is projected to be {emp_stats.crowding_level}. "
            f"Recommended safety buffer: Depart {emp_stats.recommended_buffer_min} minutes earlier than standard schedule."
        )

        return TripPredictionResponse(
            origin_stop_id=origin_id,
            origin_name=emp_stats.origin_stop_name,
            destination_stop_id=dest_id,
            destination_name=emp_stats.destination_stop_name,
            route_name=emp_stats.route_name,
            scheduled_duration_min=emp_stats.scheduled_duration_min,
            predicted_delay_min=pred_delay,
            delay_lower_bound_min=ml_res["lower_bound_min"],
            delay_upper_bound_min=ml_res["upper_bound_min"],
            p50_delay_min=emp_stats.p50_delay_min,
            p90_delay_min=emp_stats.p90_delay_min,
            on_time_probability=emp_stats.on_time_probability,
            crowding_index=emp_stats.crowding_index,
            crowding_level=emp_stats.crowding_level,
            recommended_departure_buffer_min=emp_stats.recommended_buffer_min,
            narrative_summary=narrative,
        )


# Singleton instance
predictor_node = TripPredictorNode()


if __name__ == "__main__":
    node = TripPredictorNode()
    resp = node.predict_trip(origin="Temple University", destination="PHL Airport", departure_hour=8)
    print("Trip Predictor Response:\n", resp.narrative_summary)
