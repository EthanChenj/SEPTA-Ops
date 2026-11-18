-- ==============================================================================
-- SEPTA Operations & Intelligence Star Schema & Vector Store
-- Migration 001: Initial Schema Definition
-- ==============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- DIMENSION TABLES
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_routes (
    route_id VARCHAR(32) PRIMARY KEY,
    route_short_name VARCHAR(64) NOT NULL,
    route_long_name VARCHAR(128) NOT NULL,
    route_type INT NOT NULL,
    mode_category VARCHAR(32) NOT NULL,
    description TEXT
);

COMMENT ON TABLE dim_routes IS 'Catalog of SEPTA transit lines, including Regional Rail, Subways, Trolleys, and Bus routes.';
COMMENT ON COLUMN dim_routes.route_id IS 'Unique identifier for the route (e.g., AIR for Airport Line, BSL for Broad Street Line, MFL for Market-Frankford Line, BUS_23).';
COMMENT ON COLUMN dim_routes.route_short_name IS 'Short display code or route number.';
COMMENT ON COLUMN dim_routes.route_long_name IS 'Full route name or terminal pair.';
COMMENT ON COLUMN dim_routes.route_type IS 'GTFS standard route type: 0=Tram/Trolley, 1=Subway/Metro, 2=Regional Rail, 3=Bus.';
COMMENT ON COLUMN dim_routes.mode_category IS 'Human-readable operating category: Regional Rail, Subway/Elevated, Trolley, Bus.';
COMMENT ON COLUMN dim_routes.description IS 'Operational summary and corridor serviced.';

CREATE TABLE IF NOT EXISTS dim_stops (
    stop_id VARCHAR(32) PRIMARY KEY,
    stop_name VARCHAR(128) NOT NULL,
    stop_lat NUMERIC(9,6),
    stop_lon NUMERIC(9,6),
    wheelchair_boarding SMALLINT DEFAULT 1,
    zone_id VARCHAR(16),
    is_major_hub BOOLEAN DEFAULT FALSE
);

COMMENT ON TABLE dim_stops IS 'Stations, stops, and terminals across the SEPTA service area.';
COMMENT ON COLUMN dim_stops.stop_id IS 'Unique GTFS stop identifier (e.g., STOP_TEMPLE, STOP_PHL_A, STOP_30TH).';
COMMENT ON COLUMN dim_stops.stop_name IS 'Full station or terminal name (e.g., Temple University, Philadelphia International Airport Terminal A).';
COMMENT ON COLUMN dim_stops.is_major_hub IS 'Boolean flag indicating a major transfer terminal or rail hub.';

CREATE TABLE IF NOT EXISTS dim_date (
    date_key DATE PRIMARY KEY,
    day_of_week SMALLINT NOT NULL,
    day_name VARCHAR(16) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    fiscal_year SMALLINT NOT NULL,
    fiscal_quarter SMALLINT NOT NULL
);

COMMENT ON TABLE dim_date IS 'Date dimension supporting calendar and fiscal year analytics (SEPTA FY begins July 1).';
COMMENT ON COLUMN dim_date.date_key IS 'Calendar date in YYYY-MM-DD format.';
COMMENT ON COLUMN dim_date.fiscal_year IS 'SEPTA fiscal year (e.g., FY2026 runs July 1, 2025 to June 30, 2026).';

-- -----------------------------------------------------------------------------
-- FACT TABLES
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_otp (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL REFERENCES dim_date(date_key),
    route_id VARCHAR(32) NOT NULL REFERENCES dim_routes(route_id),
    scheduled_trips INT NOT NULL,
    on_time_trips INT NOT NULL,
    delayed_trips INT NOT NULL,
    cancelled_trips INT NOT NULL,
    otp_percentage NUMERIC(5,2) NOT NULL,
    average_delay_seconds NUMERIC(8,2) NOT NULL
);

COMMENT ON TABLE fact_otp IS 'Daily aggregated On-Time Performance (OTP) by transit route.';
COMMENT ON COLUMN fact_otp.otp_percentage IS 'Percentage of scheduled trips arriving within on-time threshold (<= 5:59 mins for rail, <= 4:59 mins for bus). Value between 0.00 and 100.00.';
COMMENT ON COLUMN fact_otp.average_delay_seconds IS 'Average delay duration in seconds for delayed trips on this route-date.';

CREATE TABLE IF NOT EXISTS fact_ridership (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL REFERENCES dim_date(date_key),
    route_id VARCHAR(32) NOT NULL REFERENCES dim_routes(route_id),
    stop_id VARCHAR(32) REFERENCES dim_stops(stop_id),
    hour_of_day SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    boardings INT NOT NULL,
    alightings INT NOT NULL,
    unlinked_passenger_trips INT NOT NULL,
    load_factor NUMERIC(4,2) NOT NULL
);

COMMENT ON TABLE fact_ridership IS 'Hourly passenger boardings, alightings, and vehicle crowding load factor.';
COMMENT ON COLUMN fact_ridership.load_factor IS 'Ratio of passenger volume to seated capacity (1.00 = 100% seated capacity, >1.00 = standing room only).';
COMMENT ON COLUMN fact_ridership.unlinked_passenger_trips IS 'Total passenger boardings recorded across the interval.';

CREATE TABLE IF NOT EXISTS fact_detours (
    id BIGSERIAL PRIMARY KEY,
    route_id VARCHAR(32) NOT NULL REFERENCES dim_routes(route_id),
    start_datetime TIMESTAMPTZ NOT NULL,
    end_datetime TIMESTAMPTZ,
    reason TEXT NOT NULL,
    detour_type VARCHAR(64) NOT NULL,
    impact_score INT CHECK (impact_score BETWEEN 1 AND 5)
);

COMMENT ON TABLE fact_detours IS 'Historical and active route disruptions, detours, and track work.';
COMMENT ON COLUMN fact_detours.detour_type IS 'Category of disruption: Construction, Emergency, Event, Maintenance, Weather.';
COMMENT ON COLUMN fact_detours.impact_score IS 'Severity rating from 1 (minor reroute, <5 min delay) to 5 (full service suspension).';

CREATE TABLE IF NOT EXISTS fact_weather (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL REFERENCES dim_date(date_key),
    hour_of_day SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    temp_f NUMERIC(4,1) NOT NULL,
    precipitation_inches NUMERIC(4,2) NOT NULL DEFAULT 0.00,
    snowfall_inches NUMERIC(4,2) NOT NULL DEFAULT 0.00,
    wind_speed_mph NUMERIC(4,1) NOT NULL,
    weather_condition VARCHAR(64) NOT NULL
);

COMMENT ON TABLE fact_weather IS 'Hourly Philadelphia regional weather observations correlated with transit delays.';
COMMENT ON COLUMN fact_weather.precipitation_inches IS 'Total liquid precipitation accumulation in inches during the hour.';
COMMENT ON COLUMN fact_weather.snowfall_inches IS 'Snow accumulation in inches during the hour.';

-- -----------------------------------------------------------------------------
-- HYBRID RAG VECTOR STORE (pgvector + tsvector)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS budget_document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_title VARCHAR(255) NOT NULL,
    fiscal_year VARCHAR(16) NOT NULL,
    page_number INT NOT NULL,
    section_header TEXT,
    content TEXT NOT NULL,
    embedding vector(1536),
    tsv_content tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(section_header, '') || ' ' || coalesce(content, ''))
    ) STORED
);

COMMENT ON TABLE budget_document_chunks IS 'Chunked and vectorized SEPTA Operating Budget, Capital Program, and strategic policy PDFs for Hybrid RAG.';
COMMENT ON COLUMN budget_document_chunks.embedding IS 'Dense vector embeddings (1536-dim) generated from document chunk text.';
COMMENT ON COLUMN budget_document_chunks.tsv_content IS 'Generated Postgres full-text search tsvector over section header and chunk body for sparse lexical retrieval.';

-- -----------------------------------------------------------------------------
-- INDEXES
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_fact_otp_route_date ON fact_otp(route_id, date);
CREATE INDEX IF NOT EXISTS idx_fact_ridership_route_date ON fact_ridership(route_id, date, hour_of_day);
CREATE INDEX IF NOT EXISTS idx_fact_detours_route ON fact_detours(route_id, start_datetime);
CREATE INDEX IF NOT EXISTS idx_fact_weather_date ON fact_weather(date, hour_of_day);

-- HNSW Vector Index for dense cosine distance search (<=>)
CREATE INDEX IF NOT EXISTS idx_budget_chunks_hnsw 
    ON budget_document_chunks 
    USING hnsw (embedding vector_cosine_ops);

-- GIN Index for sparse keyword search (@@)
CREATE INDEX IF NOT EXISTS idx_budget_chunks_tsv 
    ON budget_document_chunks 
    USING gin (tsv_content);

CREATE INDEX IF NOT EXISTS idx_budget_chunks_doc_page 
    ON budget_document_chunks (document_title, page_number);
