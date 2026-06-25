# Findings

## 需求相关现状证据
- 当前行为: `src/app/main.py` has Streamlit pages for overview, theme list, note evidence, noise/profile insights, and temporal evolution. The current profile page is `view_noise_analysis`, rendered through sidebar item `🔍 噪声洞察`.
- 相关文件/目录: `src/app/main.py` contains page routing and all current page renderers; `src/app/ui.py` contains reusable HTML card helpers; `src/app/theme.py` contains shared CSS/colors; `src/app/charts.py` contains Plotly helpers.
- 数据流/调用链: `main()` loads notes/themes/labels/UMAP through `load_data()`, loads profile analysis from `log/insights_output/noise_cross_cognitive.json`, and loads temporal analysis from `log/insights_output/temporal_evolution.json`.
- 相似实现/现有模式: Existing pages use `page_header`, `section`, `insight_card`, `quote_card`, `tag_list`, `st.columns`, Plotly charts, `st.selectbox`, and `st.expander` for evidence details.
- 验证方式: Python syntax compilation for `src/app/*.py`; optional Streamlit smoke run if needed.
- 当前工作区状态/历史债: No project README/AGENTS/plans existed before this thread. `src/app/main.py` is a large file with many page concerns in one module, but this task should avoid broad refactoring.
- 风险与约束: New page should reuse existing analysis fields defensively because JSON completeness can vary. The old `认知画像洞察` page must remain unchanged.
- 未确认问题: None blocking; user selected report direction, psychological-analysis tone, all lenses, and medium judgment strength.
- AI-readable 缺口: No README or test command documentation; this plan records the current validation path for this task.

## 根因/差距分析
- 观察到的事实: Several current experiences rely on `st.expander` lists, especially theme list, cross-domain books, and micro themes.
- 假设: The perceived weakness comes from placing evidence/list structure before interpretation.
- 支撑/反证: Existing profile page already has some narrative cards and charts, but cross-book/micro-theme sections still expose list-first structures.
- 结论: A new page should make a report-first path: medium-strength judgment, explanation, compact evidence, then optional expandable evidence.
