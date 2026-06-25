# Implementation Plan

## Goal
Add a new Streamlit page for a report-style psychological reading profile while preserving the old `🔍 噪声洞察` page.

## Steps
1. Add small rendering helpers in `src/app/ui.py` for report/lens cards if needed.
2. Add helper functions in `src/app/main.py` to derive compact report lenses from existing `analysis`, `temporal_analysis`, `notes`, `themes`, and `book_map`.
3. Add `view_psychological_report(...)` as a new page renderer.
4. Add a new sidebar page item and route it without changing the existing `🔍 噪声洞察` route.
5. Verify Python syntax with `python -m compileall src/app`.

## Acceptance Criteria
- Existing `🔍 噪声洞察` remains routed to `view_noise_analysis`.
- New page appears as a separate navigation item.
- New page gives useful report-style judgments even if LLM analysis is missing, with a graceful setup message.
- Evidence remains optional and compact, not the primary display.
