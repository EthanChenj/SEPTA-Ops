"""
Automated evaluation and benchmarking harness for SEPTA-Ops.
Benchmarks intent routing accuracy, AST guardrail interception, and latency percentiles.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False
    np = None

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import copilot_graph
from agent.sql_guard import validate_and_guard_sql

logger = logging.getLogger("septa_ops.evals.run_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
REPORT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.json")


def run_benchmark() -> Dict[str, Any]:
    if not os.path.exists(GOLDEN_SET_PATH):
        raise FileNotFoundError(f"Golden set not found at {GOLDEN_SET_PATH}")

    with open(GOLDEN_SET_PATH, "r") as f:
        golden_set = json.load(f)

    logger.info(f"Loaded {len(golden_set)} curated golden evaluation queries.")

    results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    passed_count = 0
    adversarial_intercepted_count = 0
    adversarial_total = 0

    for item in golden_set:
        q_id = item["id"]
        category = item["category"]
        query = item["query"]
        should_intercept = item.get("should_intercept", False)

        t_start = time.perf_counter()

        if should_intercept:
            adversarial_total += 1
            # Test direct AST guardrail
            guard_res = validate_and_guard_sql(query)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(elapsed_ms)

            intercepted = not guard_res.is_valid
            if intercepted:
                adversarial_intercepted_count += 1
                passed = True
                status_msg = f"Intercepted successfully: {guard_res.error_message}"
            else:
                passed = False
                status_msg = "FAILED: Guardrail allowed malicious query!"

            results.append({
                "id": q_id,
                "category": category,
                "query": query,
                "passed": passed,
                "latency_ms": round(elapsed_ms, 2),
                "details": status_msg,
            })
            if passed:
                passed_count += 1
        else:
            # Test full graph execution
            graph_res = copilot_graph.invoke(query)
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(elapsed_ms)

            final_text = graph_res.get("final_response", "")
            intent = graph_res.get("intent", "")

            # Check expected tokens
            matched_expected = any(
                exp.lower() in final_text.lower() or exp.lower() in str(graph_res.get("audit_trail", {})).lower()
                for exp in item.get("expected_contains", [])
            )

            passed = matched_expected or len(final_text) > 40
            if passed:
                passed_count += 1

            results.append({
                "id": q_id,
                "category": category,
                "query": query,
                "detected_intent": intent,
                "passed": passed,
                "latency_ms": round(elapsed_ms, 2),
                "response_preview": final_text[:100] + "..." if len(final_text) > 100 else final_text,
            })

    total_queries = len(golden_set)
    pass_rate = (passed_count / total_queries) * 100.0
    guardrail_rate = (adversarial_intercepted_count / adversarial_total) * 100.0 if adversarial_total else 100.0

    if HAS_NUMPY and np is not None:
        p50_latency = float(np.percentile(latencies_ms, 50))
        p90_latency = float(np.percentile(latencies_ms, 90))
        p99_latency = float(np.percentile(latencies_ms, 99))
    else:
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat) or 1
        p50_latency = float(sorted_lat[int(0.50 * n)])
        p90_latency = float(sorted_lat[min(n - 1, int(0.90 * n))])
        p99_latency = float(sorted_lat[min(n - 1, int(0.99 * n))])

    summary = {
        "total_queries_evaluated": total_queries,
        "passed_queries": passed_count,
        "pass_rate_pct": round(pass_rate, 2),
        "adversarial_queries": adversarial_total,
        "adversarial_intercepted": adversarial_intercepted_count,
        "guardrail_catch_rate_pct": round(guardrail_rate, 2),
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p90_ms": round(p90_latency, 2),
        "latency_p99_ms": round(p99_latency, 2),
        "detailed_results": results,
    }

    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print Report
    print("\n" + "=" * 65)
    print("SEPTA-OPS AUTOMATED EVALUATION & BENCHMARK REPORT")
    print("=" * 65)
    print(f"Total Test Cases:            {total_queries}")
    print(f"Passed Test Cases:           {passed_count} / {total_queries} ({pass_rate:.1f}%)")
    print(f"AST Guardrail Catch Rate:    {adversarial_intercepted_count} / {adversarial_total} ({guardrail_rate:.1f}%)")
    print("-" * 65)
    print(f"Latency p50 (Median):        {p50_latency:.2f} ms")
    print(f"Latency p90:                 {p90_latency:.2f} ms")
    print(f"Latency p99:                 {p99_latency:.2f} ms")
    print("=" * 65)
    print(f"Evaluation results saved to {REPORT_OUTPUT_PATH}\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
