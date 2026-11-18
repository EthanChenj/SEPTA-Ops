"""
Plotly chart generator for SQL and predictor outputs.
Transforms tabular query results and predictive distributions into Plotly JSON specifications.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class ChartNode:
    """Detects tabular patterns and compiles Plotly visualization specifications."""

    @staticmethod
    def generate_chart_spec(rows: List[Dict[str, Any]], title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes column types and row values to generate an appropriate Plotly chart specification.
        Returns a dict suitable for st.plotly_chart or json serialization.
        """
        if not rows or len(rows) < 2:
            return None

        columns = list(rows[0].keys())
        if len(columns) < 2:
            return None

        # Classify columns as categorical, temporal, or numerical
        numeric_cols = []
        string_cols = []
        date_cols = []

        for col in columns:
            val = rows[0][col]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_cols.append(col)
            elif "date" in col.lower() or "time" in col.lower() or "day" in col.lower():
                date_cols.append(col)
            else:
                string_cols.append(col)

        if not numeric_cols:
            return None

        # Select X and Y axes
        x_col = (date_cols + string_cols)[0] if (date_cols + string_cols) else columns[0]
        y_col = numeric_cols[0]

        x_vals = [r.get(x_col) for r in rows]
        y_vals = [r.get(y_col) for r in rows]

        chart_title = title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"

        # Choose chart type: Line if temporal, Bar otherwise
        is_temporal = bool(date_cols and x_col in date_cols)
        chart_type = "scatter" if is_temporal else "bar"

        trace: Dict[str, Any] = {
            "x": x_vals,
            "y": y_vals,
            "type": chart_type,
            "marker": {
                "color": "#00539b",  # SEPTA Blue
            },
        }

        if is_temporal:
            trace["mode"] = "lines+markers"
            trace["line"] = {"color": "#00539b", "width": 2.5}

        spec = {
            "data": [trace],
            "layout": {
                "title": {"text": chart_title, "font": {"size": 16, "color": "#1e293b"}},
                "xaxis": {
                    "title": x_col.replace("_", " ").title(),
                    "showgrid": True,
                    "gridcolor": "#f1f5f9",
                },
                "yaxis": {
                    "title": y_col.replace("_", " ").title(),
                    "showgrid": True,
                    "gridcolor": "#f1f5f9",
                },
                "template": "plotly_white",
                "margin": {"l": 50, "r": 30, "t": 50, "b": 50},
                "height": 380,
            },
        }
        return spec

    @staticmethod
    def generate_delay_distribution_chart(
        p50: float, p90: float, p95: float, scheduled_min: int, title: str = "Trip Delay Distribution"
    ) -> Dict[str, Any]:
        """Generates a gauge / horizontal bar distribution chart for trip predictions."""
        categories = ["Scheduled Travel Time", "p50 (Median Delay)", "p90 (Worst 10%)", "p95 (Severe Disruption)"]
        values = [scheduled_min, p50, p90, p95]
        colors = ["#00539b", "#10b981", "#f59e0b", "#ef4444"]

        spec = {
            "data": [
                {
                    "x": values,
                    "y": categories,
                    "type": "bar",
                    "orientation": "h",
                    "marker": {"color": colors},
                    "text": [f"{v:.1f} min" if isinstance(v, float) else f"{v} min" for v in values],
                    "textposition": "auto",
                }
            ],
            "layout": {
                "title": {"text": title, "font": {"size": 16, "color": "#1e293b"}},
                "xaxis": {"title": "Duration / Delay (Minutes)", "showgrid": True, "gridcolor": "#f1f5f9"},
                "yaxis": {"autorange": "reversed"},
                "template": "plotly_white",
                "margin": {"l": 160, "r": 30, "t": 50, "b": 50},
                "height": 320,
            },
        }
        return spec


# Singleton instance
chart_node = ChartNode()
