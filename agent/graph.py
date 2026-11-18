"""
LangGraph state machine and multi-hop intent router for SEPTA Operations Copilot.
Coordinates AST-guarded Text-to-SQL, Hybrid RAG with RRF, and trip prediction nodes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAS_LANGGRAPH = False
    StateGraph = None
    END = "__end__"

from agent.chart_node import chart_node
from agent.predictor_node import predictor_node
from agent.rag_node import rag_node
from agent.sql_node import sql_node

logger = logging.getLogger("septa_ops.agent.graph")


class CopilotState(TypedDict, total=False):
    user_query: str
    intent: str  # "sql", "rag", "hybrid", "predict_trip", "general"
    sql_query: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    rag_context: Optional[Dict[str, Any]]
    prediction_result: Optional[Dict[str, Any]]
    chart_spec: Optional[Dict[str, Any]]
    final_response: str
    error: Optional[str]
    audit_trail: Dict[str, Any]


def classify_intent(query: str) -> str:
    """Classifies user intent into SQL, RAG, Hybrid, Trip Prediction, or General."""
    q_lower = query.lower()

    # Predictive trip inquiries
    if any(k in q_lower for k in ["predict", "travel to", "how long will it take", "temple to", "airport trip", "departure buffer", "crowding level"]):
        return "predict_trip"

    # Hybrid multi-hop inquiries
    has_policy = any(k in q_lower for k in ["budget", "deficit", "modernization", "plan", "policy", "capital program", "funding"])
    has_data = any(k in q_lower for k in ["otp", "on-time", "ridership", "boardings", "average", "delayed", "weather", "performance", "compare"])

    if has_policy and has_data:
        return "hybrid"

    # Strategic policy / RAG documents
    if has_policy:
        return "rag"

    # Tabular / operational SQL
    if has_data or any(k in q_lower for k in ["route", "line", "highest", "lowest", "count", "trips", "show me"]):
        return "sql"

    return "general"


def router_node(state: CopilotState) -> CopilotState:
    """Determines query intent and sets up the execution plan."""
    query = state.get("user_query", "")
    intent = classify_intent(query)
    audit = state.get("audit_trail", {})
    audit["intent_classified"] = intent
    return {**state, "intent": intent, "audit_trail": audit}


def sql_worker_node(state: CopilotState) -> CopilotState:
    """Generates candidate SQL and executes through the AST-guarded sql_node."""
    query = state.get("user_query", "")
    q_lower = query.lower()
    audit = state.get("audit_trail", {})

    # Template-assisted Text-to-SQL logic for core transit queries
    candidate_sql = state.get("sql_query")
    if not candidate_sql:
        if "airport" in q_lower or "air" in q_lower:
            candidate_sql = (
                "SELECT date, scheduled_trips, on_time_trips, delayed_trips, otp_percentage "
                "FROM fact_otp WHERE route_id = 'AIR' ORDER BY date DESC LIMIT 10"
            )
        elif "ridership" in q_lower or "busiest" in q_lower or "boardings" in q_lower:
            candidate_sql = (
                "SELECT r.route_short_name, r.mode_category, sum(f.unlinked_passenger_trips) as total_ridership "
                "FROM fact_ridership f JOIN dim_routes r ON f.route_id = r.route_id "
                "GROUP BY r.route_short_name, r.mode_category ORDER BY total_ridership DESC LIMIT 10"
            )
        elif "detour" in q_lower or "disruption" in q_lower:
            candidate_sql = (
                "SELECT route_id, reason, detour_type, impact_score "
                "FROM fact_detours ORDER BY start_datetime DESC LIMIT 10"
            )
        else:
            candidate_sql = (
                "SELECT r.route_short_name, r.mode_category, round(avg(f.otp_percentage), 2) as avg_otp "
                "FROM fact_otp f JOIN dim_routes r ON f.route_id = r.route_id "
                "GROUP BY r.route_short_name, r.mode_category ORDER BY avg_otp DESC LIMIT 10"
            )

    exec_res = sql_node.execute_query(candidate_sql)
    audit["sql_query_executed"] = exec_res.sanitized_sql or candidate_sql
    audit["sql_success"] = exec_res.success
    audit["sql_row_count"] = exec_res.row_count
    audit["sql_error"] = exec_res.error

    chart = None
    if exec_res.success and exec_res.rows:
        chart = chart_node.generate_chart_spec(exec_res.rows)

    return {
        **state,
        "sql_query": candidate_sql,
        "sql_result": exec_res.to_dict(),
        "chart_spec": chart,
        "audit_trail": audit,
    }


def rag_worker_node(state: CopilotState) -> CopilotState:
    """Executes Hybrid Dense + Sparse RAG retrieval with RRF."""
    query = state.get("user_query", "")
    audit = state.get("audit_trail", {})

    retrieval_res = rag_node.retrieve(query, top_k=3)
    audit["rag_sources"] = retrieval_res.source_documents
    audit["rag_top_chunk_rrf"] = retrieval_res.chunks[0].rrf_score if retrieval_res.chunks else 0.0

    return {
        **state,
        "rag_context": retrieval_res.to_dict(),
        "audit_trail": audit,
    }


def predict_worker_node(state: CopilotState) -> CopilotState:
    """Executes empirical delay distribution calculation and LightGBM inference."""
    query = state.get("user_query", "")
    audit = state.get("audit_trail", {})

    # Entity extraction
    origin = "STOP_TEMPLE" if "temple" in query.lower() else "STOP_30TH"
    dest = "STOP_PHL_A"

    pred_res = predictor_node.predict_trip(origin=origin, destination=dest, departure_hour=8)
    audit["prediction_route"] = pred_res.route_name
    audit["p50_delay"] = pred_res.p50_delay_min
    audit["p90_delay"] = pred_res.p90_delay_min

    chart = chart_node.generate_delay_distribution_chart(
        p50=pred_res.p50_delay_min,
        p90=pred_res.p90_delay_min,
        p95=pred_res.p90_delay_min * 1.5,
        scheduled_min=pred_res.scheduled_duration_min,
        title=f"Delay Distribution: {pred_res.origin_name} to {pred_res.destination_name}",
    )

    return {
        **state,
        "prediction_result": pred_res.to_dict(),
        "chart_spec": chart,
        "audit_trail": audit,
    }


def synthesizer_node(state: CopilotState) -> CopilotState:
    """Compiles the final cohesive response based on accumulated node outputs."""
    intent = state.get("intent", "general")
    query = state.get("user_query", "")
    parts = []

    if intent == "predict_trip" and state.get("prediction_result"):
        pred = state["prediction_result"]
        parts.append(f"### 🚆 Commuter Trip Reliability: {pred['origin_name']} ➔ {pred['destination_name']}\n")
        parts.append(f"- **Scheduled Travel Time:** {pred['scheduled_duration_min']} minutes")
        parts.append(f"- **Expected Delay (LightGBM):** {pred['predicted_delay_min']} minutes")
        parts.append(f"- **Empirical Delay Percentiles:** $p_{{50}} = {pred['p50_delay_min']}$ min | $p_{{90}} = {pred['p90_delay_min']}$ min")
        parts.append(f"- **On-Time Arrival Probability:** {int(pred['on_time_probability'] * 100)}%")
        parts.append(f"- **Crowding Level:** {pred['crowding_level']}")
        parts.append(f"\n> **Advisory Recommendation:** Leave {pred['recommended_departure_buffer_min']} minutes earlier than scheduled to ensure high on-time confidence.")

    elif intent == "sql" and state.get("sql_result"):
        res = state["sql_result"]
        if res.get("success"):
            rows = res.get("rows", [])
            parts.append(f"### 📊 Operational Query Results ({res.get('row_count', 0)} rows)\n")
            if rows:
                # Markdown table preview
                cols = res.get("columns", [])
                header = "| " + " | ".join(cols) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                parts.append(header)
                parts.append(sep)
                for r in rows[:5]:
                    parts.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
                if len(rows) > 5:
                    parts.append(f"\n*...and {len(rows) - 5} more rows.*")
            else:
                parts.append("No matching records found for this criteria.")
        else:
            parts.append(f"⚠️ Query Execution Interrupted: {res.get('error')}")

    elif intent == "rag" and state.get("rag_context"):
        rag_data = state["rag_context"]
        chunks = rag_data.get("chunks", [])
        parts.append("### 📑 Policy & Budget Document Intelligence\n")
        for c in chunks:
            parts.append(f"**From {c['document_title']} (p. {c['page_number']} - {c['section_header']}):**")
            parts.append(f"> {c['content']}\n")

    elif intent == "hybrid":
        parts.append("### 🔄 Multi-Hop Intelligence: Operational Metrics & Strategic Policy Context\n")
        if state.get("sql_result") and state["sql_result"].get("success"):
            parts.append(f"**Operations Telemetry:** Query returned {state['sql_result'].get('row_count')} records.")
        if state.get("rag_context"):
            parts.append("\n**Strategic Budget/Policy Insights:**")
            for c in state["rag_context"].get("chunks", [])[:2]:
                parts.append(f"- *{c['section_header']} ({c['document_title']})*: {c['content']}")

    else:
        parts.append(f"SEPTA-Ops Copilot processed query: '{query}'. Please select an operational metric or trip route.")

    return {**state, "final_response": "\n".join(parts)}


class CopilotGraph:
    """Graph orchestrator supporting both LangGraph and pure-python execution."""

    def __init__(self):
        self._compiled_graph = None
        if HAS_LANGGRAPH:
            self._build_langgraph()

    def _build_langgraph(self) -> None:
        builder = StateGraph(CopilotState)
        builder.add_node("router", router_node)
        builder.add_node("sql_worker", sql_worker_node)
        builder.add_node("rag_worker", rag_worker_node)
        builder.add_node("predict_worker", predict_worker_node)
        builder.add_node("synthesizer", synthesizer_node)

        builder.set_entry_point("router")

        def route_decision(state: CopilotState) -> str:
            intent = state.get("intent", "general")
            if intent == "predict_trip":
                return "predict_worker"
            elif intent == "rag":
                return "rag_worker"
            elif intent == "sql":
                return "sql_worker"
            elif intent == "hybrid":
                return "sql_worker"
            return "synthesizer"

        builder.add_conditional_edges("router", route_decision, {
            "predict_worker": "predict_worker",
            "rag_worker": "rag_worker",
            "sql_worker": "sql_worker",
            "synthesizer": "synthesizer",
        })

        builder.add_edge("predict_worker", "synthesizer")
        builder.add_edge("sql_worker", "synthesizer")
        builder.add_edge("rag_worker", "synthesizer")
        builder.add_edge("synthesizer", END)

        self._compiled_graph = builder.compile()

    def invoke(self, query: str) -> CopilotState:
        """Executes the state graph over the user query."""
        initial_state: CopilotState = {
            "user_query": query,
            "audit_trail": {},
        }

        if self._compiled_graph is not None:
            return self._compiled_graph.invoke(initial_state)

        # Pure-python fallback state machine runner
        state = router_node(initial_state)
        intent = state.get("intent")
        if intent == "predict_trip":
            state = predict_worker_node(state)
        elif intent == "rag":
            state = rag_worker_node(state)
        elif intent == "sql":
            state = sql_worker_node(state)
        elif intent == "hybrid":
            state = sql_worker_node(state)
            state = rag_worker_node(state)

        state = synthesizer_node(state)
        return state


# Singleton instance
copilot_graph = CopilotGraph()


if __name__ == "__main__":
    graph = CopilotGraph()
    res = graph.invoke("What is the on-time performance of the Airport line?")
    print("Graph execution response:\n", res["final_response"])
