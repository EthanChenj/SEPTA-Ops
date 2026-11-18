"""
Unit tests for Pandera validation schemas, ETL structured generation, and PDF chunking.
"""

from datetime import date, datetime
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False
    pd = None
try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from etl.schemas import validate_dataframe
from etl.structured_ingest import (
    BASELINE_ROUTES,
    BASELINE_STOPS,
    generate_date_dimension,
    generate_detours_records,
    generate_otp_records,
    generate_ridership_records,
    generate_weather_records,
)
from etl.pdf_ingest import (
    BASELINE_BUDGET_CHUNKS,
    chunk_document_text,
    generate_text_embedding,
)


def test_baseline_routes_and_stops_validation():
    if HAS_PANDAS and pd is not None:
        df_routes = pd.DataFrame(BASELINE_ROUTES)
        df_stops = pd.DataFrame(BASELINE_STOPS)

        val_routes = validate_dataframe(df_routes, "route")
        val_stops = validate_dataframe(df_stops, "stop")

        assert len(val_routes) == len(BASELINE_ROUTES)
        assert len(val_stops) == len(BASELINE_STOPS)
        assert "AIR" in val_routes["route_id"].values
        assert "STOP_TEMPLE" in val_stops["stop_id"].values
    else:
        assert len(BASELINE_ROUTES) >= 5
        assert len(BASELINE_STOPS) >= 5


def test_date_dimension_generation():
    df_dates = generate_date_dimension(date(2026, 7, 1), num_days=10)
    assert len(df_dates) == 10
    if HAS_PANDAS and pd is not None and hasattr(df_dates, "columns"):
        assert "fiscal_year" in df_dates.columns
        assert "is_weekend" in df_dates.columns
        assert df_dates["fiscal_year"].iloc[0] == 2027
    else:
        assert "fiscal_year" in df_dates[0]
        assert df_dates[0]["fiscal_year"] == 2027


def test_otp_records_generation_and_validation():
    routes = ["AIR", "BSL"]
    dates = [date(2026, 11, 1), date(2026, 11, 2)]
    df_otp = generate_otp_records(routes, dates)

    assert len(df_otp) == 4
    if HAS_PANDAS and pd is not None and hasattr(df_otp, "columns"):
        val_otp = validate_dataframe(df_otp, "otp")
        assert val_otp["otp_percentage"].min() >= 0.0
        assert val_otp["otp_percentage"].max() <= 100.0
    else:
        assert all(0.0 <= r["otp_percentage"] <= 100.0 for r in df_otp)


def test_ridership_generation():
    routes = ["AIR"]
    dates = [date(2026, 11, 1)]
    df_ridership = generate_ridership_records(routes, dates)

    assert len(df_ridership) > 0
    if HAS_PANDAS and pd is not None and hasattr(df_ridership, "columns"):
        assert "load_factor" in df_ridership.columns
        assert df_ridership["boardings"].min() >= 0
    else:
        assert "load_factor" in df_ridership[0]
        assert df_ridership[0]["boardings"] >= 0


def test_weather_generation():
    dates = [date(2026, 11, 1)]
    df_weather = generate_weather_records(dates)

    assert len(df_weather) == 24
    if HAS_PANDAS and pd is not None and hasattr(df_weather, "columns"):
        assert "temp_f" in df_weather.columns
        assert "precipitation_inches" in df_weather.columns
    else:
        assert "temp_f" in df_weather[0]
        assert "precipitation_inches" in df_weather[0]


def test_detours_generation():
    df_detours = generate_detours_records()
    assert len(df_detours) >= 3
    if HAS_PANDAS and pd is not None and hasattr(df_detours, "between"):
        assert all(df_detours["impact_score"].between(1, 5))
    else:
        assert all(1 <= r["impact_score"] <= 5 for r in df_detours)


def test_text_embedding_generation():
    text_sample = "SEPTA Trolley Modernization ADA Accessible Light Rail"
    emb = generate_text_embedding(text_sample, dim=1536)

    assert isinstance(emb, list)
    assert len(emb) == 1536
    # Vector should have non-zero elements
    assert sum(abs(x) for x in emb[:10]) > 0.0


def test_chunk_document_text():
    sample_doc = (
        "SECTION 1: OVERVIEW\n"
        "This is the opening text for the operating budget overview.\n"
        "The deficit is significant and requires strategic planning.\n"
        "SECTION 2: MODERNIZATION\n"
        "Fleet overhauls are underway across regional rail corridors.\n"
    )
    chunks = chunk_document_text(sample_doc, document_title="Test Doc", chunk_size=80, overlap=20)
    assert len(chunks) >= 1
    assert "document_title" in chunks[0]
    assert "content" in chunks[0]
