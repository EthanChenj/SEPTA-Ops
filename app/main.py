"""
Streamlit main application entrypoint for SEPTA-Ops.
Sets up session state, custom styling, and multi-tab operational dashboard.
"""

from __future__ import annotations

import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="SEPTA-Ops Intelligence Platform",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom branding CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00539b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #f1f5f9;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
    }
    .audit-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initializes Streamlit session state stores."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to **SEPTA-Ops Operations & Policy Copilot**! "
                    "Ask me about transit on-time performance (OTP), ridership patterns, "
                    "service disruptions, or the FY26–27 operating budget and capital plan."
                ),
            }
        ]
    if "audit_history" not in st.session_state:
        st.session_state.audit_history = []
    if "current_chart" not in st.session_state:
        st.session_state.current_chart = None


def main() -> None:
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/SEPTA.svg/320px-SEPTA.svg.png", width=140)
        st.title("SEPTA-Ops")
        st.caption("Transit Data Engineering & Multi-Hop AI")
        st.divider()

        st.subheader("System Health")
        st.success("● Database: PostgreSQL 16 + pgvector", icon="🗄️")
        st.info("● AST Guardrails: Active (sqlglot)", icon="🛡️")
        st.success("● LightGBM Predictor: Ready", icon="🤖")
        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.audit_history = []
            st.session_state.current_chart = None
            st.rerun()

        st.markdown("---")
        st.caption("SEPTA-Ops Copilot • Architecture: AST-Guarded SQL + Hybrid RRF + LightGBM")

    # Header
    st.markdown('<div class="main-header">SEPTA Operations & Transit Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-hop Text-to-SQL analytics, budget document intelligence, '
        'and origin–destination commuter delay prediction.</div>',
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab1, tab2 = st.tabs([
        "🔍 Tab 1: Operations & Policy Copilot",
        "✈️ Tab 2: Commuter & PHL Airport Predictor",
    ])

    with tab1:
        from app.tab_copilot import render_copilot_tab
        render_copilot_tab()

    with tab2:
        from app.tab_predictor import render_predictor_tab
        render_predictor_tab()


if __name__ == "__main__":
    main()
