"""
Unit tests for the sqlglot AST SQL Validator and Guardrail engine.
Verifies read-only safety, table whitelisting, and LIMIT injection.
"""

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None
from agent.sql_guard import (
    ALLOWED_TABLES,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SQLGuardResult,
    validate_and_guard_sql,
)


def test_valid_select_query():
    sql = "SELECT route_id, route_long_name FROM dim_routes"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is True
    assert "LIMIT 100" in res.sanitized_sql.upper()
    assert res.injected_limit is True


def test_valid_join_query():
    sql = """
        SELECT r.route_short_name, avg(f.otp_percentage) as avg_otp
        FROM fact_otp f
        JOIN dim_routes r ON f.route_id = r.route_id
        GROUP BY r.route_short_name
        LIMIT 25
    """
    res = validate_and_guard_sql(sql)
    assert res.is_valid is True
    assert "LIMIT 25" in res.sanitized_sql.upper()
    assert res.injected_limit is False


def test_valid_cte_query():
    sql = """
        WITH peak_otp AS (
            SELECT route_id, otp_percentage FROM fact_otp WHERE otp_percentage > 90.0
        )
        SELECT * FROM peak_otp
    """
    res = validate_and_guard_sql(sql)
    assert res.is_valid is True
    assert "LIMIT 100" in res.sanitized_sql.upper()


def test_clamp_excessive_limit():
    sql = "SELECT * FROM fact_ridership LIMIT 2000"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is True
    assert f"LIMIT {MAX_LIMIT}" in res.sanitized_sql.upper()
    assert res.injected_limit is True


def test_block_drop_table():
    sql = "DROP TABLE dim_routes"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "prohibited" in res.error_message.lower() or "disallowed" in res.error_message.lower()


def test_block_delete():
    sql = "DELETE FROM fact_otp WHERE id > 0"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "prohibited" in res.error_message.lower() or "disallowed" in res.error_message.lower()


def test_block_insert():
    sql = "INSERT INTO dim_routes (route_id) VALUES ('HACK')"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "prohibited" in res.error_message.lower() or "disallowed" in res.error_message.lower()


def test_block_multiple_stacked_statements():
    sql = "SELECT * FROM dim_routes; DROP TABLE dim_routes;"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "multiple" in res.error_message.lower() or "chaining" in res.error_message.lower() or "prohibited" in res.error_message.lower()


def test_block_unauthorized_table():
    sql = "SELECT * FROM secret_user_credentials"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "unauthorized table" in res.error_message.lower() or "denied" in res.error_message.lower()


def test_block_dangerous_function():
    sql = "SELECT pg_sleep(10)"
    res = validate_and_guard_sql(sql)
    assert res.is_valid is False
    assert "dangerous function" in res.error_message.lower() or "blocked" in res.error_message.lower()


def test_empty_sql():
    res = validate_and_guard_sql("   ")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()
