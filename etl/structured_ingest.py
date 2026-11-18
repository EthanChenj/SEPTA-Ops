"""
Structured transit data ingestion pipeline for SEPTA-Ops.
Handles GTFS routes, stops, date dimension, OTP, ridership, detours, and weather.
"""

from __future__ import annotations

import logging
import os
import math
import random
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False
    pd = None

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    HAS_SQLALCHEMY = False
    create_engine = None
    text = None

from etl.schemas import validate_dataframe

logger = logging.getLogger("septa_ops.etl.structured_ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://septa_admin:septa_secure_password@localhost:5432/septa_ops"
)


# ------------------------------------------------------------------------------
# Baseline Reference Datasets
# ------------------------------------------------------------------------------

BASELINE_ROUTES = [
    {
        "route_id": "AIR",
        "route_short_name": "Airport",
        "route_long_name": "Airport Regional Rail Line",
        "route_type": 2,
        "mode_category": "Regional Rail",
        "description": "Connects Temple University and Center City Philadelphia directly to PHL Airport Terminals.",
    },
    {
        "route_id": "PAO",
        "route_short_name": "Paoli/Thorndale",
        "route_long_name": "Paoli/Thorndale Regional Rail Line",
        "route_type": 2,
        "mode_category": "Regional Rail",
        "description": "High-volume commuter line connecting Center City to Main Line western suburbs.",
    },
    {
        "route_id": "TRE",
        "route_short_name": "Trenton",
        "route_long_name": "Trenton Regional Rail Line",
        "route_type": 2,
        "mode_category": "Regional Rail",
        "description": "Northeast corridor connecting Center City to Trenton Transit Center (NJ Transit transfer).",
    },
    {
        "route_id": "LAN",
        "route_short_name": "Lansdale/Doylestown",
        "route_long_name": "Lansdale/Doylestown Regional Rail Line",
        "route_type": 2,
        "mode_category": "Regional Rail",
        "description": "North suburban corridor connecting Center City to Montgomery and Bucks counties.",
    },
    {
        "route_id": "BSL",
        "route_short_name": "BSL",
        "route_long_name": "Broad Street Line Subway",
        "route_type": 1,
        "mode_category": "Subway/Elevated",
        "description": "North-South heavy rail spine between Fern Rock and NRG Stadium via City Hall.",
    },
    {
        "route_id": "MFL",
        "route_short_name": "MFL",
        "route_long_name": "Market-Frankford Line",
        "route_type": 1,
        "mode_category": "Subway/Elevated",
        "description": "East-West heavy rail spine connecting Frankford Transportation Center to 69th Street.",
    },
    {
        "route_id": "T10",
        "route_short_name": "Route 10",
        "route_long_name": "Lancaster Avenue Subway-Surface Trolley",
        "route_type": 0,
        "mode_category": "Trolley",
        "description": "West Philadelphia street-running trolley operating through Center City trolley tunnel.",
    },
    {
        "route_id": "BUS_23",
        "route_short_name": "Route 23",
        "route_long_name": "Chestnut Hill to Center City Bus",
        "route_type": 3,
        "mode_category": "Bus",
        "description": "High-ridership north-south bus route along Germantown and 11th/12th Streets.",
    },
    {
        "route_id": "BUS_47",
        "route_short_name": "Route 47",
        "route_long_name": "Whitman Plaza to 5th & Godfrey Bus",
        "route_type": 3,
        "mode_category": "Bus",
        "description": "Dense urban corridor bus line along 7th/8th Streets through Center City.",
    },
]

BASELINE_STOPS = [
    {
        "stop_id": "STOP_TEMPLE",
        "stop_name": "Temple University Station",
        "stop_lat": 39.9814,
        "stop_lon": -75.1495,
        "wheelchair_boarding": 1,
        "zone_id": "Zone 1",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_JEFF",
        "stop_name": "Jefferson Station (Center City)",
        "stop_lat": 39.9525,
        "stop_lon": -75.1582,
        "wheelchair_boarding": 1,
        "zone_id": "CC",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_SUB",
        "stop_name": "Suburban Station (Penn Center)",
        "stop_lat": 39.9540,
        "stop_lon": -75.1675,
        "wheelchair_boarding": 1,
        "zone_id": "CC",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_30TH",
        "stop_name": "William H. Gray III 30th Street Station",
        "stop_lat": 39.9558,
        "stop_lon": -75.1820,
        "wheelchair_boarding": 1,
        "zone_id": "CC",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_PHL_A",
        "stop_name": "PHL Airport Terminal A",
        "stop_lat": 39.8785,
        "stop_lon": -75.2405,
        "wheelchair_boarding": 1,
        "zone_id": "Airport",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_PHL_B",
        "stop_name": "PHL Airport Terminal B",
        "stop_lat": 39.8770,
        "stop_lon": -75.2415,
        "wheelchair_boarding": 1,
        "zone_id": "Airport",
        "is_major_hub": False,
    },
    {
        "stop_id": "STOP_PHL_CD",
        "stop_name": "PHL Airport Terminals C & D",
        "stop_lat": 39.8760,
        "stop_lon": -75.2420,
        "wheelchair_boarding": 1,
        "zone_id": "Airport",
        "is_major_hub": False,
    },
    {
        "stop_id": "STOP_PHL_EF",
        "stop_name": "PHL Airport Terminals E & F",
        "stop_lat": 39.8745,
        "stop_lon": -75.2435,
        "wheelchair_boarding": 1,
        "zone_id": "Airport",
        "is_major_hub": True,
    },
    {
        "stop_id": "STOP_CITY_HALL",
        "stop_name": "City Hall Station (BSL / MFL)",
        "stop_lat": 39.9526,
        "stop_lon": -75.1635,
        "wheelchair_boarding": 1,
        "zone_id": "Transit",
        "is_major_hub": True,
    },
]


def generate_date_dimension(start_date: date, num_days: int = 90) -> Any:
    """Generates calendar and fiscal date records."""
    records = []
    for i in range(num_days):
        cur = start_date + timedelta(days=i)
        is_weekend = cur.weekday() >= 5
        # SEPTA fiscal year begins July 1
        fy = cur.year + 1 if cur.month >= 7 else cur.year
        # Fiscal quarter
        fq = ((cur.month - 7) % 12) // 3 + 1
        records.append({
            "date_key": cur,
            "day_of_week": cur.weekday() + 1,
            "day_name": cur.strftime("%A"),
            "is_weekend": is_weekend,
            "is_holiday": False,
            "fiscal_year": fy,
            "fiscal_quarter": fq,
        })
    return pd.DataFrame(records) if (HAS_PANDAS and pd is not None) else records


def generate_otp_records(routes: List[str], dates: List[date]) -> Any:
    """Generates realistic On-Time Performance (OTP) fact records."""
    records = []
    random.seed(42)
    if HAS_NUMPY and np is not None:
        np.random.seed(42)

    base_otp_map = {
        "AIR": 0.915,
        "PAO": 0.880,
        "TRE": 0.875,
        "LAN": 0.890,
        "BSL": 0.910,
        "MFL": 0.865,
        "T10": 0.820,
        "BUS_23": 0.795,
        "BUS_47": 0.780,
    }

    base_trips_map = {
        "AIR": 76,
        "PAO": 88,
        "TRE": 64,
        "LAN": 72,
        "BSL": 240,
        "MFL": 280,
        "T10": 160,
        "BUS_23": 190,
        "BUS_47": 180,
    }

    for d in dates:
        is_weekend = d.weekday() >= 5
        for route in routes:
            base_otp = base_otp_map.get(route, 0.85)
            scheduled = base_trips_map.get(route, 100)
            if is_weekend:
                scheduled = int(scheduled * 0.65)
                base_otp += 0.02  # Weekend traffic is generally smoother

            # Noise / weather fluctuation
            noise = float(np.random.normal(0, 0.035)) if (HAS_NUMPY and np is not None) else random.gauss(0, 0.035)
            actual_otp = max(0.60, min(0.99, base_otp + noise))
            
            on_time = int(scheduled * actual_otp)
            cancelled = max(0, int(random.expovariate(1.0 / 1.2))) if not (HAS_NUMPY and np is not None) else max(0, int(np.random.exponential(1.2)))
            delayed = max(0, scheduled - on_time - cancelled)
            avg_delay = float(random.uniform(240, 720)) if delayed > 0 else 0.0

            records.append({
                "date": pd.to_datetime(d) if (HAS_PANDAS and pd is not None) else d,
                "route_id": route,
                "scheduled_trips": scheduled,
                "on_time_trips": on_time,
                "delayed_trips": delayed,
                "cancelled_trips": cancelled,
                "otp_percentage": round(actual_otp * 100, 2),
                "average_delay_seconds": round(avg_delay, 2),
            })
    return pd.DataFrame(records) if (HAS_PANDAS and pd is not None) else records


def generate_ridership_records(routes: List[str], dates: List[date]) -> Any:
    """Generates hourly boardings and load factors."""
    records = []
    random.seed(42)
    # Peak hours: 7-9 AM, 4-6 PM
    for d in dates[-14:]:  # 14 days of granular hourly data
        is_weekend = d.weekday() >= 5
        for route in routes:
            for hour in range(6, 23):
                is_peak = (7 <= hour <= 9) or (16 <= hour <= 18)
                if is_weekend:
                    base_boardings = 120 if 11 <= hour <= 17 else 50
                    load_factor = random.uniform(0.3, 0.65)
                else:
                    if is_peak:
                        base_boardings = 450 + random.randint(0, 150)
                        load_factor = random.uniform(0.75, 1.15)
                    else:
                        base_boardings = 160 + random.randint(0, 80)
                        load_factor = random.uniform(0.35, 0.65)

                alightings = int(base_boardings * random.uniform(0.9, 1.1))
                records.append({
                    "date": pd.to_datetime(d) if (HAS_PANDAS and pd is not None) else d,
                    "route_id": route,
                    "stop_id": "STOP_TEMPLE" if route == "AIR" else None,
                    "hour_of_day": hour,
                    "boardings": base_boardings,
                    "alightings": alightings,
                    "unlinked_passenger_trips": base_boardings,
                    "load_factor": round(load_factor, 2),
                })
    return pd.DataFrame(records) if (HAS_PANDAS and pd is not None) else records


def generate_weather_records(dates: List[date]) -> Any:
    """Generates hourly weather observations."""
    records = []
    random.seed(42)
    for d in dates[-30:]:
        for hour in range(0, 24):
            temp = 68.0 + 10.0 * math.sin((hour - 8) / 24.0 * 2 * math.pi) + random.uniform(-4, 4)
            precip = 0.0
            condition = "Clear"
            if random.random() < 0.15:
                precip = round(random.uniform(0.05, 0.45), 2)
                condition = "Rain" if precip < 0.25 else "Heavy Rain"

            records.append({
                "date": pd.to_datetime(d) if (HAS_PANDAS and pd is not None) else d,
                "hour_of_day": hour,
                "temp_f": round(temp, 1),
                "precipitation_inches": precip,
                "snowfall_inches": 0.0,
                "wind_speed_mph": round(random.uniform(4.0, 18.0), 1),
                "weather_condition": condition,
            })
    return pd.DataFrame(records) if (HAS_PANDAS and pd is not None) else records


def generate_detours_records() -> Any:
    """Generates sample detours and service disruptions."""
    now = datetime.now()
    records = [
        {
            "route_id": "T10",
            "start_datetime": now - timedelta(days=5),
            "end_datetime": now + timedelta(days=2),
            "reason": "Center City Trolley Tunnel Track & Catenary Maintenance Blitz",
            "detour_type": "Maintenance",
            "impact_score": 4,
        },
        {
            "route_id": "PAO",
            "start_datetime": now - timedelta(days=1),
            "end_datetime": now + timedelta(days=1),
            "reason": "Overhead wire inspection between Bryn Mawr and Paoli",
            "detour_type": "Maintenance",
            "impact_score": 3,
        },
        {
            "route_id": "BUS_23",
            "start_datetime": now - timedelta(days=3),
            "end_datetime": now + timedelta(days=4),
            "reason": "Germantown Avenue water main renewal detour via Wayne Ave",
            "detour_type": "Construction",
            "impact_score": 2,
        },
        {
            "route_id": "AIR",
            "start_datetime": now - timedelta(days=10),
            "end_datetime": now - timedelta(days=8),
            "reason": "Signal upgrade near Eastwick Station",
            "detour_type": "Maintenance",
            "impact_score": 2,
        },
    ]
    return pd.DataFrame(records) if (HAS_PANDAS and pd is not None) else records


class StructuredIngestionPipeline:
    """Ingestion pipeline coordinator for PostgreSQL transit tables."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or DATABASE_URL
        self.engine = None
        if HAS_SQLALCHEMY:
            try:
                self.engine = create_engine(self.db_url, connect_args={"connect_timeout": 3})
            except Exception as e:
                logger.warning(f"Could not connect to database at {self.db_url}: {e}")

    def run_ingestion(self) -> Dict[str, int]:
        """Runs the complete ingestion pipeline and returns ingested row counts."""
        logger.info("Starting SEPTA structured transit ingestion pipeline...")
        counts: Dict[str, int] = {}

        # 1. Routes & Stops
        df_routes = pd.DataFrame(BASELINE_ROUTES)
        df_stops = pd.DataFrame(BASELINE_STOPS)
        validate_dataframe(df_routes, "route")
        validate_dataframe(df_stops, "stop")

        # 2. Date Dimension
        start_date = date.today() - timedelta(days=89)
        df_dates = generate_date_dimension(start_date, num_days=90)

        # 3. Facts
        route_ids = [r["route_id"] for r in BASELINE_ROUTES]
        calendar_dates = df_dates["date_key"].tolist()

        df_otp = generate_otp_records(route_ids, calendar_dates)
        validate_dataframe(df_otp, "otp")

        df_ridership = generate_ridership_records(route_ids, calendar_dates)
        validate_dataframe(df_ridership, "ridership")

        df_weather = generate_weather_records(calendar_dates)
        validate_dataframe(df_weather, "weather")

        df_detours = generate_detours_records()
        validate_dataframe(df_detours, "detour")

        counts["dim_routes"] = len(df_routes)
        counts["dim_stops"] = len(df_stops)
        counts["dim_date"] = len(df_dates)
        counts["fact_otp"] = len(df_otp)
        counts["fact_ridership"] = len(df_ridership)
        counts["fact_weather"] = len(df_weather)
        counts["fact_detours"] = len(df_detours)

        # Write to Database if available
        if self.engine:
            try:
                with self.engine.begin() as conn:
                    logger.info("Persisting reference dimensions to database...")
                    df_routes.to_sql("dim_routes", conn, if_exists="append", index=False)
                    df_stops.to_sql("dim_stops", conn, if_exists="append", index=False)
                    df_dates.to_sql("dim_date", conn, if_exists="append", index=False)

                    logger.info("Persisting operational facts to database...")
                    df_otp.to_sql("fact_otp", conn, if_exists="append", index=False)
                    df_ridership.to_sql("fact_ridership", conn, if_exists="append", index=False)
                    df_weather.to_sql("fact_weather", conn, if_exists="append", index=False)
                    df_detours.to_sql("fact_detours", conn, if_exists="append", index=False)
                logger.info("Database ingestion successfully committed.")
            except Exception as e:
                logger.warning(f"Database write skipped or failed (DB may be offline): {e}")

        logger.info(f"Ingestion completed. Summary: {counts}")
        return counts


if __name__ == "__main__":
    pipeline = StructuredIngestionPipeline()
    summary = pipeline.run_ingestion()
    print("Ingestion Summary:", summary)
