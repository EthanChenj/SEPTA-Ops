"""
SQL AST Validator and Guardrail Node for Operations Copilot.
Uses sqlglot to parse, inspect, validate, and inject LIMIT clauses into candidate SQL queries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover
    HAS_SQLGLOT = False
    sqlglot = None
    exp = None

logger = logging.getLogger("septa_ops.agent.sql_guard")

ALLOWED_TABLES: Set[str] = {
    "dim_routes",
    "dim_stops",
    "dim_date",
    "fact_otp",
    "fact_ridership",
    "fact_detours",
    "fact_weather",
    "budget_document_chunks",
}

FORBIDDEN_FUNCTIONS: Set[str] = {
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_read_file",
    "pg_write_file",
    "dblink",
    "dblink_exec",
    "system",
    "query_to_xml",
    "lo_import",
    "lo_export",
}

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


@dataclass
class SQLGuardResult:
    is_valid: bool
    sanitized_sql: str
    error_message: Optional[str] = None
    injected_limit: bool = False
    tables_referenced: Optional[List[str]] = None


def validate_and_guard_sql(
    sql_text: str,
    allowed_tables: Optional[Set[str]] = None,
    default_limit: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
) -> SQLGuardResult:
    """
    Parses and verifies candidate SQL query using AST inspection.
    Ensures read-only safety, enforces table whitelisting, and injects LIMIT.
    """
    cleaned = sql_text.strip()
    # Strip markdown backticks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return SQLGuardResult(is_valid=False, sanitized_sql="", error_message="Empty SQL statement.")

    tables_allowed = allowed_tables or ALLOWED_TABLES

    if HAS_SQLGLOT:
        try:
            expressions = sqlglot.parse(cleaned, read="postgres")
        except Exception as e:
            return SQLGuardResult(is_valid=False, sanitized_sql=cleaned, error_message=f"SQL syntax error: {e}")

        # Block multiple stacked queries
        if len(expressions) > 1:
            return SQLGuardResult(
                is_valid=False,
                sanitized_sql=cleaned,
                error_message="Multiple SQL statements detected. Command chaining is strictly forbidden.",
            )

        parsed_ast = expressions[0]
        if parsed_ast is None:
            return SQLGuardResult(is_valid=False, sanitized_sql="", error_message="Could not parse SQL AST.")

        # Disallow non-SELECT operations (Insert, Update, Delete, Drop, Alter, Command, etc.)
        if not isinstance(parsed_ast, (exp.Select, exp.Union)):
            return SQLGuardResult(
                is_valid=False,
                sanitized_sql=cleaned,
                error_message=f"Disallowed statement type: '{type(parsed_ast).__name__}'. Only SELECT queries are permitted.",
            )

        # Inspect all AST nodes
        tables_found = set()
        for node in parsed_ast.walk():
            # Check for forbidden statement mutations
            if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Command, exp.Truncate)):
                return SQLGuardResult(
                    is_valid=False,
                    sanitized_sql=cleaned,
                    error_message=f"Mutation statement '{type(node).__name__}' is prohibited.",
                )

            # Check for forbidden function calls
            if isinstance(node, exp.Anonymous):
                func_name = node.name.lower()
                if func_name in FORBIDDEN_FUNCTIONS:
                    return SQLGuardResult(
                        is_valid=False,
                        sanitized_sql=cleaned,
                        error_message=f"Dangerous function '{func_name}' is blocked by security guardrails.",
                    )

            # Check table references
            if isinstance(node, exp.Table):
                tbl_name = node.name.lower()
                # Ignore CTE aliases if declared
                tables_found.add(tbl_name)

        # Validate referenced tables against whitelist (ignoring CTE names defined in query)
        cte_names = set()
        with_clause = parsed_ast.args.get("with")
        if with_clause:
            for cte in with_clause.expressions:
                cte_names.add(cte.alias_or_name.lower())

        invalid_tables = [t for t in tables_found if t not in tables_allowed and t not in cte_names]
        if invalid_tables:
            return SQLGuardResult(
                is_valid=False,
                sanitized_sql=cleaned,
                error_message=f"Access denied to unauthorized table(s): {', '.join(invalid_tables)}. Permitted tables: {', '.join(sorted(tables_allowed))}.",
            )

        # Inject or clamp LIMIT
        injected = False
        limit_expr = parsed_ast.args.get("limit")
        if limit_expr is None:
            parsed_ast = parsed_ast.limit(default_limit)
            injected = True
        else:
            try:
                curr_limit = int(limit_expr.expression.this)
                if curr_limit > max_limit or curr_limit <= 0:
                    parsed_ast.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
                    injected = True
            except Exception:
                parsed_ast.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
                injected = True

        sanitized_sql = parsed_ast.sql(dialect="postgres")
        return SQLGuardResult(
            is_valid=True,
            sanitized_sql=sanitized_sql,
            injected_limit=injected,
            tables_referenced=list(tables_found),
        )

    else:
        # Regex fallback validator when sqlglot is not loaded
        upper_sql = cleaned.upper()
        prohibited_keywords = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "EXEC ", ";"]
        for kw in prohibited_keywords:
            if kw in upper_sql:
                return SQLGuardResult(
                    is_valid=False,
                    sanitized_sql=cleaned,
                    error_message=f"Prohibited operation '{kw.strip()}' found in query.",
                )

        if not upper_sql.startswith("SELECT") and not upper_sql.startswith("WITH"):
            return SQLGuardResult(
                is_valid=False,
                sanitized_sql=cleaned,
                error_message="Query must begin with SELECT or WITH.",
            )

        # Check for forbidden functions
        for func in FORBIDDEN_FUNCTIONS:
            if func.upper() in upper_sql:
                return SQLGuardResult(
                    is_valid=False,
                    sanitized_sql=cleaned,
                    error_message=f"Dangerous function '{func}' is blocked by security guardrails.",
                )

        # Check for table references against whitelist
        from_matches = re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_0-9]+)", cleaned, flags=re.IGNORECASE)
        for tbl in from_matches:
            tbl_lower = tbl.lower()
            if tbl_lower not in tables_allowed:
                return SQLGuardResult(
                    is_valid=False,
                    sanitized_sql=cleaned,
                    error_message=f"Access denied to unauthorized table: '{tbl}'.",
                )

        # Basic LIMIT check
        injected = False
        limit_match = re.search(r"LIMIT\s+(\d+)", cleaned, flags=re.IGNORECASE)
        if limit_match:
            curr_val = int(limit_match.group(1))
            if curr_val > max_limit or curr_val <= 0:
                cleaned = re.sub(r"LIMIT\s+\d+", f"LIMIT {max_limit}", cleaned, flags=re.IGNORECASE)
                injected = True
        else:
            cleaned = f"{cleaned.rstrip(';')} LIMIT {default_limit}"
            injected = True

        return SQLGuardResult(is_valid=True, sanitized_sql=cleaned, injected_limit=injected)


if __name__ == "__main__":
    # Test valid query
    q1 = "SELECT route_id, avg(otp_percentage) FROM fact_otp GROUP BY route_id"
    res1 = validate_and_guard_sql(q1)
    print("Test 1 (Valid):", res1)

    # Test malicious query
    q2 = "SELECT * FROM dim_routes; DROP TABLE dim_routes;"
    res2 = validate_and_guard_sql(q2)
    print("Test 2 (Malicious Stacked):", res2)

    # Test unauthorized table
    q3 = "SELECT * FROM pg_user"
    res3 = validate_and_guard_sql(q3)
    print("Test 3 (Unauthorized table):", res3)
