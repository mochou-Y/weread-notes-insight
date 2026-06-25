# 新增心理画像报告页

## Current Snapshot
- Stage: Act
- Goal: Add a new Streamlit page/address for a multi-lens psychological reading profile report.
- Keep unchanged: Existing `认知画像洞察` page behavior and existing source analysis data files.
- Next: Implement the new page, wire it into navigation, and verify syntax/imports.

## Scope
- Add a separate page entry in the Streamlit sidebar.
- Reuse existing `noise_cross_cognitive.json`, temporal analysis, notes, themes, and book metadata.
- Keep the original `view_noise_analysis` route available and unchanged.

## Non-Scope
- No LLM prompt/data pipeline changes in this pass.
- No removal of existing expanders from old pages.
- No redesign of all pages.
