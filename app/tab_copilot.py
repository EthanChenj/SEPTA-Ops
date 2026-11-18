"""
Tab 1: Operations & Policy Copilot with SQL/PDF Audit Trail and Plotly chart rendering.
"""

from __future__ import annotations

import streamlit as st
from agent.graph import copilot_graph


def render_copilot_tab() -> None:
    st.subheader("Operations & Policy Intelligence Copilot")
    st.write(
        "Ask natural-language questions across SEPTA's operational database (OTP, ridership, detours, weather) "
        "and strategic policy documents (FY26-27 Operating Budget & Capital Program)."
    )

    # Suggestion Pills
    st.markdown("**Suggested Inquiries:**")
    col1, col2, col3, col4 = st.columns(4)
    sample_queries = [
        "What is the on-time performance of the Airport Line?",
        "What is the projected deficit in the FY26 budget?",
        "Rank routes by ridership volume",
        "Explain the Trolley Modernization capital plan",
    ]
    prompt_to_submit = None

    if col1.button("✈️ Airport Line OTP", use_container_width=True):
        prompt_to_submit = sample_queries[0]
    if col2.button("💰 FY26 Budget Deficit", use_container_width=True):
        prompt_to_submit = sample_queries[1]
    if col3.button("👥 Top Ridership Lines", use_container_width=True):
        prompt_to_submit = sample_queries[2]
    if col4.button("🚊 Trolley Modernization", use_container_width=True):
        prompt_to_submit = sample_queries[3]

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_spec"):
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure(msg["chart_spec"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Chart render notice: {e}")

            # Audit Trail Expander
            audit = msg.get("audit")
            if audit:
                with st.expander("🔍 SQL & Document Audit Trail", expanded=False):
                    if audit.get("sql_query_executed"):
                        st.markdown("**Sanitized SQL Query (AST Guarded):**")
                        st.code(audit["sql_query_executed"], language="sql")
                        st.caption(
                            f"Rows returned: {audit.get('sql_row_count', 0)} | "
                            f"Execution Success: {audit.get('sql_success', False)}"
                        )
                    if audit.get("rag_sources"):
                        st.markdown("**Retrieved Policy Documents (Hybrid RRF):**")
                        for src in audit["rag_sources"]:
                            st.markdown(f"- 📄 `{src}`")
                        st.caption(f"Top Chunk RRF Score: {audit.get('rag_top_chunk_rrf', 0.0)}")

    # Handle Input from Chat or Buttons
    user_input = st.chat_input("Ask a question about SEPTA operations or budget...") or prompt_to_submit

    if user_input:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process with LangGraph State Machine
        with st.spinner("Analyzing SEPTA data & strategic documents..."):
            result = copilot_graph.invoke(user_input)
            response_text = result.get("final_response", "No response generated.")
            chart_spec = result.get("chart_spec")
            audit_trail = result.get("audit_trail", {})

        # Append Assistant Response
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "chart_spec": chart_spec,
            "audit": audit_trail,
        }
        st.session_state.messages.append(assistant_msg)
        st.rerun()
