"""
Data validation contracts and Pandera schemas for transit operations ETL.
Enforces type constraints, nullability, and domain boundaries.
"""

from __future__ import annotations

try:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
    HAS_PANDERA = True
except ImportError:  # pragma: no cover
    HAS_PANDERA = False
    pa = None
    Column = None
    Check = None
    DataFrameSchema = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False
    pd = None


if HAS_PANDERA:
    # --------------------------------------------------------------------------
    # GTFS & Route Dimension Schema
    # --------------------------------------------------------------------------
    RouteSchema = DataFrameSchema(
        columns={
            "route_id": Column(str, nullable=False, unique=True),
            "route_short_name": Column(str, nullable=False),
            "route_long_name": Column(str, nullable=False),
            "route_type": Column(int, Check.isin([0, 1, 2, 3])),
            "mode_category": Column(str, Check.isin(["Regional Rail", "Subway/Elevated", "Trolley", "Bus"])),
            "description": Column(str, nullable=True),
        },
        strict=True,
        coerce=True,
    )

    # --------------------------------------------------------------------------
    # GTFS & Stop Dimension Schema
    # --------------------------------------------------------------------------
    StopSchema = DataFrameSchema(
        columns={
            "stop_id": Column(str, nullable=False, unique=True),
            "stop_name": Column(str, nullable=False),
            "stop_lat": Column(float, Check.in_range(39.0, 41.0), nullable=True),
            "stop_lon": Column(float, Check.in_range(-76.0, -74.0), nullable=True),
            "wheelchair_boarding": Column(int, Check.isin([0, 1, 2]), nullable=True),
            "zone_id": Column(str, nullable=True),
            "is_major_hub": Column(bool, nullable=False),
        },
        strict=True,
        coerce=True,
    )

    # --------------------------------------------------------------------------
    # On-Time Performance (OTP) Fact Schema
    # --------------------------------------------------------------------------
    OTPSchema = DataFrameSchema(
        columns={
            "date": Column(pa.DateTime, nullable=False),
            "route_id": Column(str, nullable=False),
            "scheduled_trips": Column(int, Check.greater_than_or_equal_to(0)),
            "on_time_trips": Column(int, Check.greater_than_or_equal_to(0)),
            "delayed_trips": Column(int, Check.greater_than_or_equal_to(0)),
            "cancelled_trips": Column(int, Check.greater_than_or_equal_to(0)),
            "otp_percentage": Column(float, Check.in_range(0.0, 100.0)),
            "average_delay_seconds": Column(float, Check.greater_than_or_equal_to(0.0)),
        },
        strict=True,
        coerce=True,
    )

    # --------------------------------------------------------------------------
    # Ridership & Crowding Fact Schema
    # --------------------------------------------------------------------------
    RidershipSchema = DataFrameSchema(
        columns={
            "date": Column(pa.DateTime, nullable=False),
            "route_id": Column(str, nullable=False),
            "stop_id": Column(str, nullable=True),
            "hour_of_day": Column(int, Check.in_range(0, 23)),
            "boardings": Column(int, Check.greater_than_or_equal_to(0)),
            "alightings": Column(int, Check.greater_than_or_equal_to(0)),
            "unlinked_passenger_trips": Column(int, Check.greater_than_or_equal_to(0)),
            "load_factor": Column(float, Check.greater_than_or_equal_to(0.0)),
        },
        strict=True,
        coerce=True,
    )

    # --------------------------------------------------------------------------
    # Detours & Disruptions Schema
    # --------------------------------------------------------------------------
    DetourSchema = DataFrameSchema(
        columns={
            "route_id": Column(str, nullable=False),
            "start_datetime": Column(pa.DateTime, nullable=False),
            "end_datetime": Column(pa.DateTime, nullable=True),
            "reason": Column(str, nullable=False),
            "detour_type": Column(str, Check.isin(["Construction", "Emergency", "Event", "Maintenance", "Weather"])),
            "impact_score": Column(int, Check.in_range(1, 5)),
        },
        strict=True,
        coerce=True,
    )

    # --------------------------------------------------------------------------
    # Regional Weather Schema
    # --------------------------------------------------------------------------
    WeatherSchema = DataFrameSchema(
        columns={
            "date": Column(pa.DateTime, nullable=False),
            "hour_of_day": Column(int, Check.in_range(0, 23)),
            "temp_f": Column(float, Check.in_range(-20.0, 120.0)),
            "precipitation_inches": Column(float, Check.greater_than_or_equal_to(0.0)),
            "snowfall_inches": Column(float, Check.greater_than_or_equal_to(0.0)),
            "wind_speed_mph": Column(float, Check.greater_than_or_equal_to(0.0)),
            "weather_condition": Column(str, nullable=False),
        },
        strict=True,
        coerce=True,
    )
else:
    # Graceful pure-python fallback schema stubs
    RouteSchema = None
    StopSchema = None
    OTPSchema = None
    RidershipSchema = None
    DetourSchema = None
    WeatherSchema = None


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Validates a DataFrame against the specified Pandera schema contract.
    If Pandera is unavailable, performs basic column and type validation.
    """
    schemas = {
        "route": RouteSchema,
        "stop": StopSchema,
        "otp": OTPSchema,
        "ridership": RidershipSchema,
        "detour": DetourSchema,
        "weather": WeatherSchema,
    }
    target_schema = schemas.get(schema_name.lower())
    if HAS_PANDERA and target_schema is not None:
        return target_schema.validate(df)
    
    # Fallback basic validation
    if schema_name == "otp":
        required = ["date", "route_id", "scheduled_trips", "on_time_trips", "otp_percentage"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in OTP DataFrame")
    elif schema_name == "ridership":
        required = ["date", "route_id", "hour_of_day", "boardings", "load_factor"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in Ridership DataFrame")
    return df
