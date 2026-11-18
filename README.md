# SEPTA-Ops: Transit Intelligence, AST-Guarded Text-to-SQL & Commuter Predictor

[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-blue.svg)](https://github.com/pgvector/pgvector)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end transit data engineering, predictive modeling, and multi-hop AI intelligence platform for **SEPTA** (Southeastern Pennsylvania Transportation Authority).

---

## 🚆 System Architecture & Capabilities

```text
                                  ┌─────────────────────────────┐
                                  │      Streamlit Web UI       │
                                  │ (app/tab_copilot & predictor)│
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   LangGraph State Machine   │
                                  │      (agent/graph.py)       │
                                  └──────────────┬──────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
       ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
       │   SQL Guard & Node   │      │   Hybrid RAG Node    │      │  Trip Predictor Node │
       │ (sqlglot AST + LIMIT)│      │  (pgvector + tsvector│      │  (Empirical p50/p90  │
       │  least-privilege role│      │      via RRF)        │      │     + LightGBM)      │
       └──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
                  │                             │                             │
                  ▼                             ▼                             ▼
       ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
       │ PostgreSQL 16 DB     │      │ Budget & Policy PDFs │      │ Serialized Model     │
       │ (Star Schema Tables) │      │ (Vector & TSV Chunks)│      │ (models/artifacts/)  │
       └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

The system provides two dedicated operational workflows:

1. **Operations & Policy Copilot:**
   - **AST-Guarded Text-to-SQL:** Dynamic schema context inspection with `sqlglot` AST safety validation, table whitelisting, prohibited function filtering, automatic `LIMIT 100` injection, and self-correcting query execution under a least-privilege `readonly_copilot` role.
   - **Hybrid RAG with Reciprocal Rank Fusion (RRF):** Dense 1536-dim vector embeddings combined with PostgreSQL sparse full-text search (`tsvector` / `ts_rank_cd`) over the SEPTA FY26–27 Operating Budget and Capital Program.
   - **Interactive Plotly Visualizations:** Automated conversion of tabular SQL metrics into interactive line and bar charts.

2. **Commuter & Airport Trip Reliability Predictor:**
   - **Empirical Tail Distributions:** Computes median delay ($p_{50}$), worst-case delay ($p_{90}$, $p_{95}$), and passenger load crowding index across transit corridors (e.g., Temple University Station to PHL Airport).
   - **LightGBM Delay Forecasting:** Predicts expected trip delays conditioned on time-of-day, peak hours, regional weather (rain, wind), and corridor maintenance detours.
   - **Flight Buffer Advisory:** Recommends exact departure buffer times to ensure 95%+ probability of on-time airport arrival.

---

## 📁 Repository Structure

```text
├── docker-compose.yml          # PostgreSQL 16 + pgvector container
├── pyproject.toml              # Project dependencies and pytest config
├── .env.example                # Sample environment configuration
├── AGENTS.md                   # Architectural contracts and module guidelines
├── migrations/
│   ├── 001_init_schema.sql     # Star schema, pgvector tables, column comments
│   └── 002_readonly_role.sql   # Least-privilege readonly_copilot DB role
├── etl/
│   ├── schemas.py              # Pandera data validation contracts
│   ├── structured_ingest.py    # GTFS, ridership, OTP, detours, weather ingestion
│   └── pdf_ingest.py           # Docling/PyMuPDF parser + chunking + embedding upsert
├── models/
│   ├── trip_stats.py           # Empirical p50/p90 delay & crowding aggregations
│   ├── delay_predictor.py      # LightGBM training, feature engineering & inference
│   └── artifacts/              # Serialized LightGBM model files (.joblib)
├── agent/
│   ├── sql_guard.py            # sqlglot AST validator + LIMIT injection
│   ├── sql_node.py             # Schema context + self-correcting SQL execution
│   ├── rag_node.py             # Hybrid pgvector + tsvector RRF retrieval
│   ├── predictor_node.py       # Agent tool wrapper around models/delay_predictor.py
│   ├── chart_node.py           # Plotly JSON spec generator from query/model outputs
│   └── graph.py                # LangGraph state machine & intent router
├── app/
│   ├── main.py                 # Streamlit entrypoint & session state
│   ├── tab_copilot.py          # Tab 1: Operations Copilot + SQL/PDF Audit Trail
│   └── tab_predictor.py        # Tab 2: Commuter & PHL Airport Reliability Predictor
├── evals/
│   ├── golden_set.json         # 50 curated test queries (SQL, RAG, Hybrid, Adversarial)
│   └── run_eval.py             # Automated accuracy, guardrail & latency benchmark
└── tests/
    ├── test_db_and_etl.py      # Pandera schema and ETL generation tests
    ├── test_sql_guard.py       # sqlglot AST guardrail unit tests
    └── test_predictor.py       # Empirical distribution and LightGBM tests
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional, for live PostgreSQL + pgvector)

### 2. Environment Setup
```bash
# Clone and enter workspace
cd SEPTA-Ops

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env
```

### 3. Start Database (Optional)
```bash
docker compose up -d
```
The container automatically runs `migrations/001_init_schema.sql` and `migrations/002_readonly_role.sql` on startup.

### 4. Run Ingestion Pipelines
```bash
# Populate baseline GTFS, OTP, ridership, weather, and detours
python -m etl.structured_ingest

# Chunk and vectorize SEPTA budget documents
python -m etl.pdf_ingest
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Automated Testing & Evaluation

### Run Test Suite
```bash
pytest -v
```

### Run Golden Set Evaluation Benchmark
Evaluates the 50-query golden set across SQL, RAG, Hybrid, and Adversarial categories:
```bash
python evals/run_eval.py
```

Benchmark output:
- **Accuracy & Intent Routing Pass Rate**
- **AST Guardrail Catch Rate** (100% intercept on SQL injections & mutations)
- **Latency Percentiles** ($p_{50}, p_{90}, p_{99}$ in ms)
- Detailed results saved to `evals/eval_report.json`.

---

## 🛡️ Security & AST Guardrails

All database interactions pass through `agent/sql_guard.py`:
- **Statement Type Checking:** Only `SELECT` statements are permitted. `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and `TRUNCATE` are blocked.
- **Table Whitelisting:** Enforces access only to `dim_routes`, `dim_stops`, `dim_date`, `fact_otp`, `fact_ridership`, `fact_detours`, `fact_weather`, and `budget_document_chunks`.
- **Command Stacking Prevention:** Multiple statements separated by semicolons are intercepted.
- **Automatic LIMIT Clamping:** Prevents denial-of-service memory exhaustion by ensuring every query has a `LIMIT` (default: 100, clamped max: 500).
- **Least-Privilege Connection:** Executed via `readonly_copilot` role with transaction read-only and statement timeout limits.
