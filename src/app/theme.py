"""Shared visual theme for the Streamlit app."""

from __future__ import annotations

import streamlit as st


COLORS = {
    "bg": "#F8F6F0",
    "surface": "#FFFFFF",
    "surface_muted": "#F1EFE7",
    "text": "#1F2933",
    "text_muted": "#6B7280",
    "border": "#E5E0D6",
    "primary": "#315C72",
    "primary_soft": "#DDEBF0",
    "secondary": "#5F8D4E",
    "secondary_soft": "#E8EEDC",
    "accent": "#C47A3A",
    "accent_soft": "#F4E3D3",
    "purple": "#7C6A9A",
    "danger": "#B85C5C",
    "neutral": "#9CA3AF",
}

STATUS_COLORS = {
    "稳定核": COLORS["primary"],
    "新兴": COLORS["secondary"],
    "淡出": COLORS["purple"],
    "阶段性": COLORS["accent"],
    "普通": COLORS["neutral"],
}

PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["purple"],
    "#4F8A8B",
    "#9A7B4F",
    "#6D7A99",
    "#A65F46",
]


def apply_theme() -> None:
    """Apply a lightweight product theme on top of Streamlit defaults."""
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {COLORS['bg']};
            --surface: {COLORS['surface']};
            --surface-muted: {COLORS['surface_muted']};
            --text: {COLORS['text']};
            --text-muted: {COLORS['text_muted']};
            --border: {COLORS['border']};
            --primary: {COLORS['primary']};
            --primary-soft: {COLORS['primary_soft']};
            --accent: {COLORS['accent']};
        }}

        .stApp {{
            background: radial-gradient(circle at top left, #FDFBF5 0, var(--app-bg) 34%, #F3F0E8 100%);
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F4F1E8 100%);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 12px;
            padding: 0.3rem 0.55rem;
        }}

        h1, h2, h3 {{
            color: var(--text);
            letter-spacing: -0.02em;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem;
            box-shadow: 0 10px 30px rgba(49, 92, 114, 0.06);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--text-muted);
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--primary);
        }}

        div[data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 10px 26px rgba(31, 41, 51, 0.04);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        .app-hero, .insight-card, .quote-card {{
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(49, 92, 114, 0.08);
        }}

        .app-hero {{
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.3rem;
        }}

        .app-eyebrow {{
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }}

        .app-hero h1 {{
            margin: 0;
            font-size: 2.05rem;
        }}

        .app-hero p, .section-caption, .quote-meta {{
            color: var(--text-muted);
        }}

        .section-title {{
            margin: 1.3rem 0 0.25rem;
            font-size: 1.25rem;
            font-weight: 750;
        }}

        .insight-card {{
            padding: 1rem 1.1rem;
            margin: 0.45rem 0 0.85rem;
            border-left: 5px solid var(--primary);
        }}

        .insight-card.accent {{ border-left-color: {COLORS['accent']}; }}
        .insight-card.secondary {{ border-left-color: {COLORS['secondary']}; }}
        .insight-card.purple {{ border-left-color: {COLORS['purple']}; }}

        .insight-card-title {{
            color: var(--text);
            font-weight: 750;
            margin-bottom: 0.35rem;
        }}

        .quote-card {{
            padding: 0.85rem 1rem;
            margin: 0.45rem 0;
        }}

        .quote-content {{
            color: var(--text);
            font-weight: 650;
            line-height: 1.65;
        }}

        .tag-pill {{
            display: inline-block;
            background: var(--primary-soft);
            color: var(--primary);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            margin: 0.12rem 0.18rem 0.12rem 0;
            font-size: 0.82rem;
            font-weight: 650;
        }}

        button[kind="primary"], .stButton > button {{
            border-radius: 999px;
            border: 1px solid var(--border);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def tone_color(tone: str) -> str:
    """Return a theme color by semantic tone."""
    return COLORS.get(tone, COLORS["primary"])
