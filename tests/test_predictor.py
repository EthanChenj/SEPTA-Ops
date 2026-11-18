"""
Unit tests for the trip stats empirical distributions and LightGBM delay predictor.
"""

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None
from models.trip_stats import (
    TripReliabilityStats,
    calculate_empirical_stats,
    categorize_crowding,
)
from models.delay_predictor import DelayPredictor, delay_predictor
from agent.predictor_node import predictor_node, normalize_stop_identifier


def test_empirical_stats_temple_to_phl():
    stats = calculate_empirical_stats("STOP_TEMPLE", "STOP_PHL_A", departure_hour=8, is_weekend=False)
    assert isinstance(stats, TripReliabilityStats)
    assert stats.scheduled_duration_min == 38
    assert stats.p50_delay_min > 0.0
    assert stats.p90_delay_min > stats.p50_delay_min
    assert stats.on_time_probability > 0.70
    assert stats.recommended_buffer_min >= 15


def test_empirical_stats_weather_impact():
    clear_stats = calculate_empirical_stats("STOP_TEMPLE", "STOP_PHL_A", departure_hour=8, weather_impact_factor=1.0)
    rain_stats = calculate_empirical_stats("STOP_TEMPLE", "STOP_PHL_A", departure_hour=8, weather_impact_factor=1.5)

    assert rain_stats.p90_delay_min > clear_stats.p90_delay_min
    assert rain_stats.on_time_probability < clear_stats.on_time_probability


def test_categorize_crowding():
    assert "Low" in categorize_crowding(0.25)
    assert "Moderate" in categorize_crowding(0.55)
    assert "High" in categorize_crowding(0.85)
    assert "Standing Room" in categorize_crowding(1.10)


def test_delay_predictor_feature_building():
    predictor = DelayPredictor()
    df_feat = predictor.build_features({
        "origin_stop_id": "STOP_TEMPLE",
        "destination_stop_id": "STOP_PHL_A",
        "departure_hour": 8,
        "is_weekend": False,
        "temp_f": 72.0,
        "precipitation_inches": 0.2,
    })
    if hasattr(df_feat, "iloc"):
        assert len(df_feat) == 1
        assert df_feat["is_peak_hour"].iloc[0] == 1
        assert df_feat["is_airport_line"].iloc[0] == 1
    else:
        assert df_feat["is_peak_hour"] == 1
        assert df_feat["is_airport_line"] == 1


def test_delay_predictor_train_and_predict():
    predictor = DelayPredictor()
    metrics = predictor.train(n_samples=500)
    assert metrics["train_mae"] >= 0.0

    res = predictor.predict({
        "departure_hour": 17,
        "is_weekend": False,
        "precipitation_inches": 0.3,
        "is_airport_line": 1,
    })
    assert "predicted_delay_min" in res
    assert res["predicted_delay_min"] > 0.0
    assert res["upper_bound_min"] >= res["predicted_delay_min"]
    assert "advisory" in res


def test_stop_normalization():
    assert normalize_stop_identifier("temple") == "STOP_TEMPLE"
    assert normalize_stop_identifier("30th street") == "STOP_30TH"
    assert normalize_stop_identifier("Airport") == "STOP_PHL_A"
    assert normalize_stop_identifier("Terminal B") == "STOP_PHL_B"


def test_predictor_node_tool():
    resp = predictor_node.predict_trip(
        origin="Temple University",
        destination="PHL Airport",
        departure_hour=8,
    )
    assert resp.scheduled_duration_min > 0
    assert resp.predicted_delay_min > 0
    assert "Temple" in resp.origin_name
    assert "Airport" in resp.destination_name
    assert len(resp.narrative_summary) > 20
