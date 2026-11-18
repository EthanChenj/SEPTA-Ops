"""
Tab 2: Commuter & PHL Airport Reliability Predictor.
Calculates empirical p50/p90 delays, vehicle crowding, and LightGBM departure buffers.
"""

from __future__ import annotations

import streamlit as st
from agent.chart_node import chart_node
from agent.predictor_node import predictor_node

STATION_OPTIONS = {
    "STOP_TEMPLE": "Temple University Station (Regional Rail)",
    "STOP_30TH": "William H. Gray III 30th Street Station",
    "STOP_JEFF": "Jefferson Station (Center City)",
    "STOP_SUB": "Suburban Station (Penn Center)",
    "STOP_CITY_HALL": "City Hall Station (Broad St / Market St)",
}

AIRPORT_DESTINATIONS = {
    "STOP_PHL_A": "PHL Airport Terminal A (International)",
    "STOP_PHL_B": "PHL Airport Terminal B",
    "STOP_PHL_CD": "PHL Airport Terminals C & D",
    "STOP_PHL_EF": "PHL Airport Terminals E & F",
}


def render_predictor_tab() -> None:
    st.subheader("Commuter & PHL Airport Reliability Predictor")
    st.write(
        "Forecast origin–destination transit delay distributions ($p_{50}$, $p_{90}$), "
        "crowding index, and recommended departure buffers using empirical telemetry and LightGBM."
    )

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            origin_id = st.selectbox(
                "Origin Station",
                options=list(STATION_OPTIONS.keys()),
                format_func=lambda x: STATION_OPTIONS[x],
                index=0,  # Default: Temple University
            )
        with c2:
            dest_id = st.selectbox(
                "Destination Terminal / Station",
                options=list(AIRPORT_DESTINATIONS.keys()),
                format_func=lambda x: AIRPORT_DESTINATIONS[x],
                index=0,  # Default: PHL Airport Terminal A
            )

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        departure_hour = st.slider("Planned Departure Hour", min_value=5, max_value=23, value=8, format="%d:00")
    with col_t2:
        day_type = st.radio("Trip Day", ["Weekday (Commuter Peak)", "Weekend"], horizontal=True)
        is_weekend = "Weekend" in day_type
    with col_t3:
        weather_cond = st.selectbox("Observed / Forecast Weather", ["Clear / Dry", "Light Rain", "Heavy Rain", "Snow"])

    precip_map = {"Clear / Dry": 0.0, "Light Rain": 0.15, "Heavy Rain": 0.45, "Snow": 0.30}
    precip_inches = precip_map[weather_cond]

    detour_active = st.checkbox("Simulate active track maintenance / corridor detour")

    st.write("")
    if st.button("🚀 Calculate Trip Reliability & Buffer", type="primary", use_container_width=True):
        with st.spinner("Executing empirical percentile modeling & LightGBM inference..."):
            pred = predictor_node.predict_trip(
                origin=origin_id,
                destination=dest_id,
                departure_hour=departure_hour,
                is_weekend=is_weekend,
                precipitation_inches=precip_inches,
                has_active_detour=detour_active,
            )

        st.divider()

        # KPI Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Scheduled Duration", f"{pred.scheduled_duration_min} min")
        m2.metric("Predicted Delay", f"{pred.predicted_delay_min:.1f} min", delta=f"{pred.predicted_delay_min - pred.p50_delay_min:+.1f} vs med")
        m3.metric("p50 (Median Delay)", f"{pred.p50_delay_min} min")
        m4.metric("p90 (Worst 10%)", f"{pred.p90_delay_min} min")
        m5.metric("On-Time Probability", f"{int(pred.on_time_probability * 100)}%")

        # Advisory Callout Box
        st.info(f"💡 **Trip Reliability Advisory:** {pred.narrative_summary}")

        # Distribution Chart
        chart_spec = chart_node.generate_delay_distribution_chart(
            p50=pred.p50_delay_min,
            p90=pred.p90_delay_min,
            p95=pred.p90_delay_min * 1.4,
            scheduled_min=pred.scheduled_duration_min,
            title=f"Delay Distribution & Thresholds: {pred.origin_name} to {pred.destination_name}",
        )

        try:
            import plotly.graph_objects as go
            fig = go.Figure(chart_spec)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Chart visual: {e}")

        # Crowding and Buffer Details
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("### 👥 Passenger Crowding Index")
            st.write(f"Estimated Load Factor: **{pred.crowding_index:.2f}** ({pred.crowding_level})")
            st.progress(min(1.0, pred.crowding_index / 1.2))
        with col_b2:
            st.markdown("### ✈️ Flight Connection Departure Buffer")
            st.success(f"Recommended Safety Buffer: **Leave {pred.recommended_departure_buffer_min} minutes early**")
            st.caption("Accounts for p90 tail delays, station platform transfers, and airport security check lines.")
