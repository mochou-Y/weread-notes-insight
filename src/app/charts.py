"""Plotly chart helpers with a shared visual style."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.app.theme import COLORS, PALETTE, STATUS_COLORS


def apply_chart_theme(fig, height: int | None = None):
    """Apply common chart layout defaults."""
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["text"], "family": "Arial, sans-serif"},
        "margin": {"l": 40, "r": 30, "t": 40, "b": 40},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
        "hoverlabel": {
            "bgcolor": COLORS["surface"],
            "bordercolor": COLORS["border"],
            "font": {"color": COLORS["text"]},
        },
        "colorway": PALETTE,
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(107, 114, 128, 0.14)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def horizontal_bar(df: pd.DataFrame, x: str, y: str, color: str, x_label: str, height: int = 450):
    """Horizontal bar chart with readable fixed-color bars."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        labels={x: x_label, y: ""},
        text=x,
    )
    fig.update_traces(
        marker_color=color,
        texttemplate="%{text:.1%}" if df[x].max() <= 1 else "%{text}",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    return apply_chart_theme(fig, height=height)


def status_color_map() -> dict[str, str]:
    """Colors for temporal theme status buckets."""
    return STATUS_COLORS.copy()


def qualitative_palette() -> list[str]:
    """Main qualitative color palette."""
    return PALETTE.copy()
