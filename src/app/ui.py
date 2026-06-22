"""Small reusable UI helpers for Streamlit pages."""

from __future__ import annotations

import html

import streamlit as st


def page_header(title: str, subtitle: str = "", eyebrow: str = "阅读画像") -> None:
    """Render a consistent page title block."""
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    """Render a consistent section heading."""
    caption_html = f"<div class='section-caption'>{html.escape(caption)}</div>" if caption else ""
    st.markdown(
        f"<div class='section-title'>{html.escape(title)}</div>{caption_html}",
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str, tone: str = "primary") -> None:
    """Render a short narrative insight card."""
    st.markdown(
        f"""
        <div class="insight-card {html.escape(tone)}">
            <div class="insight-card-title">{html.escape(title)}</div>
            <div>{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quote_card(content: str, meta: str = "") -> None:
    """Render a note excerpt as evidence."""
    meta_html = f"<div class='quote-meta'>{html.escape(meta)}</div>" if meta else ""
    st.markdown(
        f"""
        <div class="quote-card">
            <div class="quote-content">{html.escape(content)}</div>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tag_list(tags: list[str]) -> None:
    """Render tags as theme pills."""
    if not tags:
        return
    pills = "".join(f"<span class='tag-pill'>{html.escape(str(tag))}</span>" for tag in tags)
    st.markdown(pills, unsafe_allow_html=True)
