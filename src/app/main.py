"""Streamlit 应用入口"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api.weread import DataLoader
from src.data.models import Theme

NOISE_ANALYSIS_PATH = project_root / "log" / "insights_output" / "noise_cross_cognitive.json"
TEMPORAL_ANALYSIS_PATH = project_root / "log" / "insights_output" / "temporal_evolution.json"


def _noise_analysis_mtime() -> float:
    """分析结果文件的修改时间，用于缓存失效"""
    if NOISE_ANALYSIS_PATH.exists():
        return NOISE_ANALYSIS_PATH.stat().st_mtime
    return 0.0


@st.cache_data(show_spinner=False)
def load_noise_analysis(mtime: float):
    """加载噪声深度分析结果（文件更新后自动刷新）"""
    if not NOISE_ANALYSIS_PATH.exists():
        return None
    with open(NOISE_ANALYSIS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _temporal_analysis_mtime() -> float:
    if TEMPORAL_ANALYSIS_PATH.exists():
        return TEMPORAL_ANALYSIS_PATH.stat().st_mtime
    return 0.0


@st.cache_data(show_spinner=False)
def load_temporal_analysis(mtime: float):
    """加载时间维度分析结果"""
    if not TEMPORAL_ANALYSIS_PATH.exists():
        return None
    with open(TEMPORAL_ANALYSIS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_data():
    """加载数据（缓存）"""
    loader = DataLoader()

    # 加载笔记
    all_notes = loader.load_all_notes()

    # 过滤笔记：去除书签、空内容、包含[插图]的笔记
    notes = [
        n for n in all_notes
        if n.type != "bookmark"
        and n.content.strip()  # 过滤空内容
        and "[插图]" not in n.content  # 过滤内容包含[插图]
        and "[插图]" not in (n.context or "")  # 过滤context包含[插图]
    ]

    # 加载书籍
    books = loader.load_notebook()
    book_map = {b.book_id: b for b in books}

    # 加载聚类结果
    themes_path = loader.processed_dir / "themes.json"
    with open(themes_path, encoding="utf-8") as f:
        themes_data = json.load(f)
    themes = [Theme(**t) for t in themes_data["themes"]]

    # 加载 labels
    labels_path = loader.processed_dir / "labels.npy"
    labels = np.load(labels_path)

    # 加载 UMAP 坐标
    coords_path = loader.processed_dir / "umap_coords.npy"
    if coords_path.exists():
        coords_2d = np.load(coords_path)
    else:
        coords_2d = None

    return notes, book_map, themes, labels, coords_2d


def view_overview(notes, themes, labels, coords_2d):
    """概览散点图视图"""
    st.header("📊 聚类概览")

    if coords_2d is None:
        st.warning("未找到 UMAP 坐标数据，请重新运行 `cluster` 命令")
        return

    # 构建数据框
    theme_map = {t.id: t.label for t in themes}
    theme_map[-1] = "噪声"

    df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "主题": [theme_map.get(f"theme_{l}", "噪声") for l in labels],
        "内容": [n.content[:100] + "..." if len(n.content) > 100 else n.content for n in notes],
        "书籍": [n.book_title for n in notes],
    })

    # 绘制散点图
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="主题",
        hover_data=["内容", "书籍"],
        title=f"笔记聚类分布 ({len(themes)} 个主题, {len(notes)} 条笔记)",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=True,
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总笔记数", len(notes))
    with col2:
        st.metric("主题数", len(themes))
    with col3:
        noise_count = (labels == -1).sum()
        st.metric("噪声点", noise_count)


def view_themes(themes, notes, book_map, labels):
    """主题列表视图"""
    st.header("📚 主题列表")

    # 筛选器
    col1, col2 = st.columns([1, 3])
    with col1:
        min_size = st.number_input("最小笔记数", min_value=1, value=3)
        search = st.text_input("搜索主题", placeholder="输入关键词...")

    # 过滤主题
    filtered_themes = [t for t in themes if len(t.note_ids) >= min_size]
    if search:
        filtered_themes = [t for t in filtered_themes if search in t.label]

    st.write(f"共 {len(filtered_themes)} 个主题")

    # 主题卡片
    note_map = {n.id: n for n in notes}
    for theme in filtered_themes:
        with st.expander(f"**{theme.label}** ({len(theme.note_ids)} 条笔记)"):
            # 显示该主题下的笔记摘要
            sample_notes = [note_map[nid] for nid in theme.note_ids[:5] if nid in note_map]
            for note in sample_notes:
                st.markdown(f"- *{note.content[:80]}{'...' if len(note.content) > 80 else ''}* — 《{note.book_title}》")


def view_notes(notes, themes, book_map, labels):
    """笔记详情视图"""
    st.header("📝 笔记详情")

    # 筛选模式选择
    filter_mode = st.radio("筛选模式", ["按主题筛选", "按书籍筛选"], horizontal=True)

    if filter_mode == "按主题筛选":
        # 模式1：先选主题，再选书籍
        theme_labels = ["全部"] + [t.label for t in themes]
        selected_theme = st.selectbox("选择主题", theme_labels, key="theme_select")

        if selected_theme == "全部":
            filtered_notes = notes
        else:
            theme = next((t for t in themes if t.label == selected_theme), None)
            if theme:
                note_ids = set(theme.note_ids)
                filtered_notes = [n for n in notes if n.id in note_ids]
            else:
                filtered_notes = []

        # 书籍筛选（基于当前主题下的笔记）
        books_in_theme = sorted(set(n.book_title for n in filtered_notes))
        selected_book = st.selectbox("筛选书籍", ["全部"] + books_in_theme, key="book_select")

        if selected_book != "全部":
            filtered_notes = [n for n in filtered_notes if n.book_title == selected_book]

    else:
        # 模式2：先选书籍，再选主题
        all_books = sorted(set(n.book_title for n in notes))
        selected_book = st.selectbox("选择书籍", ["全部"] + all_books, key="book_select2")

        if selected_book == "全部":
            filtered_notes = notes
        else:
            filtered_notes = [n for n in notes if n.book_title == selected_book]

        # 主题筛选（基于当前书籍下的笔记）
        note_ids_set = set(n.id for n in filtered_notes)
        themes_with_book = [t for t in themes if any(nid in note_ids_set for nid in t.note_ids)]
        theme_labels = ["全部"] + [f"{t.label} ({sum(1 for nid in t.note_ids if nid in note_ids_set)}条)" for t in themes_with_book]
        selected_theme = st.selectbox("筛选主题", theme_labels, key="theme_select2")

        if selected_theme != "全部":
            # 提取主题名称（去除数量后缀）
            theme_name = selected_theme.split(" (")[0]
            theme = next((t for t in themes if t.label == theme_name), None)
            if theme:
                note_ids = set(theme.note_ids)
                filtered_notes = [n for n in filtered_notes if n.id in note_ids]
            else:
                filtered_notes = []

    st.write(f"共 {len(filtered_notes)} 条笔记")

    # 笔记列表
    for note in filtered_notes[:50]:  # 限制显示数量
        with st.container():
            st.markdown(f"**{note.content[:200]}{'...' if len(note.content) > 200 else ''}**")
            st.caption(f"📚 《{note.book_title}》 | 📖 {note.chapter} | 🕐 {note.create_time.strftime('%Y-%m-%d')}")
            st.divider()


def _horizontal_bar(df, x, y, title_color: str, x_label: str, height: int = 450):
    """横向柱状图，使用固定颜色避免低值过浅不可见"""
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation="h",
        labels={x: x_label, y: ""},
        text=x,
    )
    fig.update_traces(
        marker_color=title_color,
        texttemplate="%{text:.1%}" if df[x].max() <= 1 else "%{text}",
        textposition="outside",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=height,
        showlegend=False,
    )
    return fig


def _compute_cross_domain_books(notes, themes, bridge_notes: list[dict] | None = None):
    """识别跨领域书籍（前端 fallback，逻辑与 NoiseAnalyzer 一致）"""
    bridge_notes = bridge_notes or []
    note_to_theme: dict[str, str] = {}
    for theme in themes:
        for nid in theme.note_ids:
            note_to_theme[nid] = theme.label

    note_map = {n.id: n for n in notes}
    books: dict[str, dict] = {}

    def ensure_book(note):
        if note.book_id not in books:
            books[note.book_id] = {
                "book_id": note.book_id,
                "title": note.book_title,
                "author": note.book_author,
                "themes": set(),
                "bridge_note_count": 0,
                "bridge_pairs": set(),
                "note_count": 0,
                "sample_bridge_notes": [],
            }
        return books[note.book_id]

    for note in notes:
        b = ensure_book(note)
        b["note_count"] += 1
        if note.id in note_to_theme:
            b["themes"].add(note_to_theme[note.id])

    for bn in bridge_notes:
        note = note_map.get(bn["note_id"])
        if note is None:
            continue
        b = ensure_book(note)
        b["bridge_note_count"] += 1
        pair = tuple(sorted(bn["themes"]))
        b["bridge_pairs"].add(pair)
        for t in bn["themes"]:
            b["themes"].add(t)
        if len(b["sample_bridge_notes"]) < 3:
            b["sample_bridge_notes"].append(bn["content"])

    result = []
    for b in books.values():
        theme_count = len(b["themes"])
        if theme_count < 2 and b["bridge_note_count"] < 2:
            continue
        bridge_pair_count = len(b["bridge_pairs"])
        score = theme_count * 2 + b["bridge_note_count"] * 3 + bridge_pair_count
        result.append({
            "book_id": b["book_id"],
            "title": b["title"],
            "author": b["author"],
            "theme_count": theme_count,
            "themes": sorted(b["themes"]),
            "bridge_note_count": b["bridge_note_count"],
            "bridge_pairs": [list(p) for p in sorted(b["bridge_pairs"])],
            "note_count": b["note_count"],
            "cross_score": score,
            "sample_bridge_notes": b["sample_bridge_notes"],
        })

    result.sort(key=lambda x: -x["cross_score"])
    return result


def _load_bridge_notes():
    """加载桥接笔记明细缓存"""
    loader = DataLoader()
    path = loader.processed_dir / "noise_bridge_notes.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_bridge_graph(bridges: list[dict], min_count: int, max_nodes: int):
    """从桥接数据构建网络图所需的节点与边"""
    filtered = [b for b in bridges if b["count"] >= min_count and b["themes"][0] != b["themes"][1]]
    if not filtered:
        return [], [], {}

    node_weights: dict[str, int] = {}
    for b in filtered:
        t0, t1 = b["themes"]
        node_weights[t0] = node_weights.get(t0, 0) + b["count"]
        node_weights[t1] = node_weights.get(t1, 0) + b["count"]

    top_nodes = [n for n, _ in sorted(node_weights.items(), key=lambda x: -x[1])[:max_nodes]]
    top_set = set(top_nodes)

    edges = [
        {
            "source": b["themes"][0],
            "target": b["themes"][1],
            "count": b["count"],
            "insight": b.get("insight", ""),
            "sample_notes": b.get("sample_notes", []),
        }
        for b in filtered
        if b["themes"][0] in top_set and b["themes"][1] in top_set
    ]
    return top_nodes, edges, node_weights


def _bridge_network_figure(nodes: list[str], edges: list[dict], node_weights: dict[str, int]):
    """构建主题关联网络图（圆形布局，边宽=桥接强度，节点大小=活跃度）"""
    n = len(nodes)
    if n == 0:
        return None

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    max_count = max(e["count"] for e in edges) if edges else 1
    max_weight = max(node_weights.get(node, 0) for node in nodes) or 1

    fig = go.Figure()

    for e in edges:
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        ratio = e["count"] / max_count
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=1.5 + 6 * ratio, color=f"rgba(22, 163, 74, {0.35 + 0.55 * ratio})"),
            hoverinfo="text",
            hovertext=f"{e['source']} ↔ {e['target']}<br>{e['count']} 条桥接笔记",
            showlegend=False,
        ))

    node_sizes = [18 + 32 * node_weights.get(node, 0) / max_weight for node in nodes]
    fig.add_trace(go.Scatter(
        x=[pos[node][0] for node in nodes],
        y=[pos[node][1] for node in nodes],
        mode="markers+text",
        marker=dict(size=node_sizes, color="#2563EB", line=dict(width=2, color="white")),
        text=nodes,
        textposition="top center",
        textfont=dict(size=11),
        hovertext=[f"{node}<br>桥接活跃度: {node_weights.get(node, 0)}" for node in nodes],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig


def _render_master_theme(profile: dict):
    """渲染阅读母题区块"""
    master = profile.get("master_theme", {})
    if master.get("title"):
        st.subheader("🎯 阅读母题")
        st.caption("跨书籍、跨主题反复出现的核心生命议题，所有分析线索的汇入口")
        src = master.get("source", "llm")
        if src == "fallback":
            st.warning("LLM 暂不可用，以下为基于桥接数据的规则归纳。请检查 API 连接后重新运行 profile。")
        elif src == "cached" and master.get("llm_error"):
            st.info("本次 LLM 调用失败，展示的是上次成功生成的母题。")
        st.markdown(f"## {master['title']}")
        if master.get("statement"):
            st.markdown(f"**{master['statement']}**")
        if master.get("narrative"):
            st.markdown(master["narrative"])

        clue_col, echo_col = st.columns(2)
        with clue_col:
            if master.get("converging_clues"):
                st.markdown("**汇聚线索**")
                for clue in master["converging_clues"]:
                    st.markdown(f"- {clue}")
        with echo_col:
            if master.get("manifestations"):
                st.markdown("**书中回响**")
                for item in master["manifestations"]:
                    st.markdown(f"- {item}")
    elif master.get("llm_error") or master.get("error"):
        st.subheader("🎯 阅读母题")
        st.error(
            f"母题提炼失败：{master.get('llm_error') or master.get('error')}。"
            "请检查 OPENAI_API_KEY / OPENAI_BASE_URL 后重新运行 profile。"
        )


def view_noise_analysis(analysis, notes, themes, book_map):
    """噪声深度分析视图"""
    st.header("🔍 噪声深度分析")

    if analysis is None:
        st.info("尚未运行噪声分析。请先执行：`python -m src.main analyze`")
        return

    stats = analysis["noise_stats"]
    profile = analysis["user_profile"]
    generated_at = analysis.get("generated_at", "")

    # 概览指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("噪声笔记", stats["total"])
    with col2:
        st.metric("微主题", stats["sub_clusters"])
    with col3:
        st.metric("桥接模式", len(analysis.get("bridge_patterns", [])))
    if generated_at:
        st.caption(f"生成时间: {generated_at[:19].replace('T', ' ')}")

    # 认知风格
    cognitive = profile.get("cognitive_style", {})
    if cognitive.get("keywords") or cognitive.get("description"):
        st.subheader("🧠 认知风格")
        desc = cognitive.get("description", "")
        src = cognitive.get("source", "llm")
        if desc.startswith("分析失败"):
            st.error("认知风格分析失败，请检查 OPENAI_API_KEY / OPENAI_BASE_URL 网络连接后重新运行 profile。")
        elif src == "fallback":
            st.warning("LLM 暂不可用，以下为基于桥接数据的简要归纳。")
        elif src == "cached" and cognitive.get("llm_error"):
            st.info("本次 LLM 调用失败，展示的是上次成功生成的认知风格。")
        if cognitive.get("keywords"):
            st.markdown(" ".join(f"`{kw}`" for kw in cognitive["keywords"]))
        if desc and not desc.startswith("分析失败"):
            st.markdown(desc)

    # 图表区
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 知识域分布")
        domains = profile.get("knowledge_domains", [])[:15]
        if domains:
            df_domains = pd.DataFrame(domains)
            fig = _horizontal_bar(
                df_domains, x="weight", y="domain",
                title_color="#2563EB", x_label="占比",
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("🔗 深度领域")
        depth = profile.get("depth_indicators", [])[:12]
        if depth:
            df_depth = pd.DataFrame(depth)
            fig = _horizontal_bar(
                df_depth, x="bridge_count", y="domain",
                title_color="#EA580C", x_label="桥接次数",
            )
            st.plotly_chart(fig, use_container_width=True)

    # 主题关联网络（替代原交叉兴趣柱状图 + 桥接洞察列表）
    bridges = analysis.get("bridge_patterns", [])
    if bridges:
        st.subheader("🌉 主题关联网络")
        st.caption("节点 = 主题，节点越大表示桥接越活跃；连线越粗表示两个主题之间的桥接笔记越多")

        max_bridge_count = max(b["count"] for b in bridges)
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            min_count = st.slider(
                "最小桥接笔记数", min_value=1,
                max_value=min(20, max_bridge_count), value=4,
                key="bridge_min_count",
            )
        with ctrl_col2:
            max_nodes = st.slider("显示主题数", min_value=8, max_value=30, value=18, key="bridge_max_nodes")

        nodes, edges, node_weights = _prepare_bridge_graph(bridges, min_count, max_nodes)
        if not edges:
            st.warning("当前筛选条件下没有桥接关系，请降低最小桥接笔记数")
        else:
            fig = _bridge_network_figure(nodes, edges, node_weights)
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"显示 **{len(nodes)}** 个主题、**{len(edges)}** 条关联")

            edge_options = {
                f"{e['source']} ↔ {e['target']} ({e['count']} 条)": e
                for e in sorted(edges, key=lambda x: -x["count"])
            }
            selected_label = st.selectbox("查看桥接详情", list(edge_options.keys()), key="bridge_detail")
            selected = edge_options[selected_label]

            detail_col1, detail_col2 = st.columns([1, 1])
            with detail_col1:
                st.markdown(f"**{selected['source']}** ↔ **{selected['target']}**")
                st.metric("桥接笔记数", selected["count"])
                if selected.get("insight"):
                    st.info(selected["insight"])
            with detail_col2:
                st.markdown("**代表性笔记**")
                for note in selected.get("sample_notes", []):
                    st.markdown(f"- *{note}*")

    # 跨领域书籍
    cross_books = (analysis or {}).get("cross_domain_books")
    if not cross_books and analysis is not None:
        cross_books = _compute_cross_domain_books(notes, themes, _load_bridge_notes())
    elif analysis is None:
        cross_books = []

    if cross_books:
        st.subheader("📖 跨领域书籍")
        st.caption(
            "笔记横跨多个思维主题，或含大量「桥接笔记」——"
            "这类书可能是多种认知领域的交汇点，值得重读"
        )

        min_themes = st.slider("最少涉及主题数", min_value=2, max_value=6, value=2, key="cross_book_min_themes")
        filtered_books = [b for b in cross_books if b["theme_count"] >= min_themes]
        st.write(f"共 **{len(filtered_books)}** 本跨领域书籍")

        if filtered_books:
            top_books = filtered_books[:15]
            df_books = pd.DataFrame([
                {"书名": b["title"], "交叉指数": b["cross_score"]}
                for b in top_books
            ])
            fig = _horizontal_bar(
                df_books, x="交叉指数", y="书名",
                title_color="#7C3AED", x_label="交叉指数",
                height=max(300, len(top_books) * 32),
            )
            st.plotly_chart(fig, use_container_width=True)

            for book in filtered_books[:25]:
                book_meta = book_map.get(book["book_id"])
                header = (
                    f"《{book['title']}》— {book['theme_count']} 个主题"
                    f"，{book['bridge_note_count']} 条桥接笔记"
                )
                with st.expander(header):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if book_meta and book_meta.cover:
                            st.image(book_meta.cover, width=100)
                        st.caption(f"作者: {book['author']}")
                        st.metric("笔记数", book["note_count"])
                        st.metric("交叉指数", book["cross_score"])
                    with cols[1]:
                        st.markdown("**涉及主题**")
                        st.markdown(" ".join(f"`{t}`" for t in book["themes"]))
                        if book.get("bridge_pairs"):
                            st.markdown("**桥接主题对**")
                            for pair in book["bridge_pairs"][:8]:
                                st.markdown(f"- {pair[0]} ↔ {pair[1]}")
                        if book.get("sample_bridge_notes"):
                            st.markdown("**桥接笔记摘录**")
                            for note in book["sample_bridge_notes"]:
                                st.markdown(f"- *{note}*")
    elif analysis is not None:
        st.subheader("📖 跨领域书籍")
        st.info("未识别到跨领域书籍。请重新运行 `python -m src.main analyze --mode profile` 生成完整数据。")

    _render_master_theme(profile)

    # 噪声微主题
    micro_themes = analysis.get("micro_themes", [])
    if micro_themes:
        st.subheader("🔬 噪声微主题")
        note_map = {n.id: n for n in notes}
        for mt in micro_themes:
            with st.expander(f"**{mt['label']}** ({mt['size']} 条)"):
                for content in mt.get("sample_notes", []):
                    st.markdown(f"- *{content}*")
                note_ids = mt.get("note_ids", [])
                if note_ids:
                    books = sorted({
                        note_map[nid].book_title
                        for nid in note_ids[:50]
                        if nid in note_map
                    })
                    if books:
                        st.caption(f"涉及书籍: {', '.join(books[:8])}{'...' if len(books) > 8 else ''}")


def view_temporal(analysis):
    """时间演变视图"""
    st.header("📈 时间演变")

    if analysis is None:
        st.info("尚未运行时间分析。请先执行：`python -m src.main analyze --mode temporal`")
        return

    periods = analysis.get("periods", [])
    generated_at = analysis.get("generated_at", "")
    granularity = analysis.get("granularity", "quarter")
    drift_score = analysis.get("drift_score", 0)
    stability = analysis.get("stability", {})
    intensity = analysis.get("intensity", [])
    theme_timeline = analysis.get("theme_timeline", [])
    period_details = analysis.get("period_details", [])
    significant_shifts = analysis.get("significant_shifts", [])
    period_options = {d["period"]: d for d in period_details}

    if periods:
        if st.session_state.get("temporal_focus_period") not in periods:
            st.session_state.temporal_focus_period = periods[-1]
        focus_idx = periods.index(st.session_state.temporal_focus_period)
    else:
        focus_idx = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("时段数", len(periods))
    with col2:
        st.metric("主题漂移度", f"{drift_score:.2f}")
        st.caption("0-1 之间，越接近 0 表示相邻时段主题越稳定；越接近 1 表示兴趣结构变化越明显。")
    with col3:
        gran_label = {"quarter": "季度", "month": "月", "year": "年"}.get(granularity, granularity)
        st.metric("时间粒度", gran_label)
    if generated_at:
        st.caption(f"生成时间: {generated_at[:19].replace('T', ' ')}")
    if periods:
        st.caption(f"时间范围: {periods[0]} ~ {periods[-1]}")

        nav_col1, nav_col2, nav_col3 = st.columns([1, 5, 1])
        with nav_col1:
            if st.button("← 上一期", disabled=focus_idx == 0, key="temporal_prev"):
                st.session_state.temporal_focus_period = periods[focus_idx - 1]
                st.rerun()
        with nav_col2:
            focus_period = st.select_slider(
                "拖动时间轴查看兴趣演进",
                options=periods,
                key="temporal_focus_period",
            )
        with nav_col3:
            if st.button("下一期 →", disabled=focus_idx == len(periods) - 1, key="temporal_next"):
                st.session_state.temporal_focus_period = periods[focus_idx + 1]
                st.rerun()

        focus_idx = periods.index(st.session_state.temporal_focus_period)
    else:
        focus_period = None

    focus_detail = period_options.get(focus_period, {}) if focus_period else {}
    focus_intensity = next((i for i in intensity if i.get("period") == focus_period), {})
    focus_top_themes = focus_detail.get("top_themes", [])
    focus_theme_names = {t["theme"] for t in focus_top_themes[:5]}

    current_weights = {}
    prev_weights = {}
    if periods and theme_timeline:
        prev_idx = max(0, focus_idx - 1)
        for item in theme_timeline:
            weights = item.get("weights", [])
            if focus_idx < len(weights):
                current_weights[item["theme"]] = weights[focus_idx]
            if prev_idx < len(weights):
                prev_weights[item["theme"]] = weights[prev_idx]
    theme_deltas = sorted(
        (
            {"theme": theme, "delta": weight - prev_weights.get(theme, 0), "weight": weight}
            for theme, weight in current_weights.items()
        ),
        key=lambda x: abs(x["delta"]),
        reverse=True,
    )

    theme_bucket = {}
    for bucket, label in (("core", "稳定核"), ("emerging", "新兴"), ("fading", "淡出"), ("spike", "阶段性")):
        for item in stability.get(bucket, []):
            theme_bucket[item["theme"]] = label

    bucket_colors = {
        "稳定核": "#2563EB",
        "新兴": "#16A34A",
        "淡出": "#7C3AED",
        "阶段性": "#F97316",
        "普通": "#94A3B8",
    }

    if focus_period:
        st.subheader("🎛️ 当前时段洞察")
        shift_hit = next(
            (
                s for s in significant_shifts
                if s.get("from") == focus_period or s.get("to") == focus_period
            ),
            None,
        )
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.metric("当前时段", focus_period)
        with d_col2:
            st.metric("笔记", focus_intensity.get("note_count", 0))
        with d_col3:
            st.metric("想法", focus_intensity.get("review_count", 0))
        with d_col4:
            st.metric("划线", focus_intensity.get("highlight_count", 0))
        if focus_top_themes:
            st.caption("本期主导主题: " + " · ".join(f"{t['theme']} {t['weight']:.1%}" for t in focus_top_themes[:4]))
        if shift_hit:
            st.info(f"变化显著时段: {shift_hit['from']} → {shift_hit['to']}，变化幅度 {shift_hit['delta']:.2f}")

        insight_col1, insight_col2 = st.columns([3, 2])
        with insight_col1:
            st.markdown("**⚡ 当前时段主题脉冲**")
            if focus_top_themes:
                df_focus = pd.DataFrame(focus_top_themes[:8])
                df_focus["类型"] = df_focus["theme"].map(theme_bucket).fillna("普通")
                fig_focus = px.bar(
                    df_focus.sort_values("weight"),
                    x="weight",
                    y="theme",
                    orientation="h",
                    color="类型",
                    color_discrete_map=bucket_colors,
                    labels={"weight": "本期占比", "theme": ""},
                    text="weight",
                )
                fig_focus.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig_focus.update_layout(
                    height=360,
                    xaxis_tickformat=".0%",
                    margin=dict(l=120, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_focus, use_container_width=True)
            else:
                st.caption("本期暂无主题数据。")

        with insight_col2:
            st.markdown("**🗓️ 时段详情**")
            detail = focus_detail or (period_details[-1] if period_details else {})
            if detail:
                st.caption(f"当前展示: {detail['period']}")
                st.markdown("**相邻时段变化**")
                if focus_idx == 0:
                    st.caption("第一期暂无上一时段对比。")
                else:
                    for item in theme_deltas[:5]:
                        arrow = "↑" if item["delta"] > 0 else "↓"
                        color = "#16A34A" if item["delta"] > 0 else "#7C3AED"
                        st.markdown(
                            f"<span style='color:{color};font-weight:700'>{arrow} {item['theme']}</span> "
                            f"{item['delta']:+.1%} · 当前 {item['weight']:.1%}",
                            unsafe_allow_html=True,
                        )
                st.markdown("**代表性笔记**")
                for note in detail.get("sample_notes", [])[:3]:
                    type_label = "想法" if note["type"] == "review" else "划线"
                    st.markdown(f"- *{note['content'][:100]}{'...' if len(note['content']) > 100 else ''}*")
                    st.caption(f"《{note['book_title']}》 | {type_label} | {note['theme']}")
            else:
                st.caption("暂无时段详情。")

    # LLM 叙事
    narrative = analysis.get("narrative", {})
    if narrative.get("unchanged") or narrative.get("shifts") or narrative.get("clues"):
        st.subheader("📖 演变叙事")
        if narrative.get("error"):
            st.warning(f"LLM 叙事部分失败: {narrative.get('clues', '')}")
        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            st.markdown("**不变的底色**")
            st.markdown(narrative.get("unchanged") or "—")
        with n_col2:
            st.markdown("**明显的转向**")
            st.markdown(narrative.get("shifts") or "—")
        with n_col3:
            st.markdown("**可能的内在线索**")
            st.markdown(narrative.get("clues") or "—")

    # 变与不变卡片
    st.subheader("🔄 变与不变")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.markdown("**稳定核**")
        for item in stability.get("core", [])[:8]:
            st.markdown(f"- `{item['theme']}` ({item['global_mean']:.1%})")
        if not stability.get("core"):
            st.caption("暂无")
    with s_col2:
        st.markdown("**新兴**")
        for item in stability.get("emerging", [])[:8]:
            fh = item.get("first_half", 0)
            sh = item.get("second_half", 0)
            st.markdown(f"- `{item['theme']}` ({fh:.1%} → {sh:.1%})")
        if not stability.get("emerging"):
            st.caption("暂无")
    with s_col3:
        st.markdown("**淡出**")
        for item in stability.get("fading", [])[:8]:
            fh = item.get("first_half", 0)
            sh = item.get("second_half", 0)
            st.markdown(f"- `{item['theme']}` ({fh:.1%} → {sh:.1%})")
        if not stability.get("fading"):
            st.caption("暂无")
    with s_col4:
        st.markdown("**阶段性**")
        for item in stability.get("spike", [])[:8]:
            peak = item.get("peak_period", "")
            st.markdown(f"- `{item['theme']}` @ {peak}")
        if not stability.get("spike"):
            st.caption("暂无")

    # 阅读强度
    if intensity:
        st.subheader("📊 阅读强度")
        df_int = pd.DataFrame(intensity)
        active_periods = [p == focus_period for p in df_int["period"]]
        highlight_line = ["#0F172A" if active else "rgba(148, 163, 184, 0.2)" for active in active_periods]
        highlight_width = [3 if active else 0 for active in active_periods]
        fig_int = go.Figure()
        fig_int.add_trace(go.Bar(
            x=df_int["period"],
            y=df_int["highlight_count"],
            name="划线",
            marker_color=["#38BDF8" if active else "rgba(147, 197, 253, 0.35)" for active in active_periods],
            marker_line_color=highlight_line,
            marker_line_width=highlight_width,
        ))
        fig_int.add_trace(go.Bar(
            x=df_int["period"],
            y=df_int["review_count"],
            name="想法",
            marker_color=["#1D4ED8" if active else "rgba(37, 99, 235, 0.35)" for active in active_periods],
            marker_line_color=highlight_line,
            marker_line_width=highlight_width,
        ))
        fig_int.update_layout(
            barmode="stack",
            xaxis_title="时段",
            yaxis_title="笔记数",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_int, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    top_n = st.slider("热力图/趋势线主题数", min_value=5, max_value=20, value=12, key="temporal_top_n")

    with chart_col1:
        st.subheader("🌡️ 主题热力图")
        if theme_timeline and periods:
            top_themes = theme_timeline[:top_n]
            z = [t["weights"] for t in top_themes]
            y_labels = [t["theme"] for t in top_themes]
            fig_heat = go.Figure(data=go.Heatmap(
                z=z,
                x=periods,
                y=y_labels,
                colorscale="Blues",
                zmin=0,
                zmax=max(max(row) for row in z) if z else 1,
                hovertemplate="时段: %{x}<br>主题: %{y}<br>占比: %{z:.1%}<extra></extra>",
            ))
            fig_heat.update_layout(
                xaxis_title="时段",
                height=max(400, top_n * 28),
                margin=dict(l=120),
            )
            if focus_period:
                fig_heat.add_vline(x=focus_period, line_width=3, line_dash="dash", line_color="#F97316")
            st.plotly_chart(fig_heat, use_container_width=True)

    with chart_col2:
        st.subheader("📈 主题趋势线")
        if theme_timeline and periods:
            fig_line = go.Figure()
            highlight_themes = set()
            for bucket in ("core", "emerging", "fading"):
                for item in stability.get(bucket, [])[:3]:
                    highlight_themes.add(item["theme"])

            colors = px.colors.qualitative.Set2
            line_themes = theme_timeline[:top_n]
            for i, item in enumerate(line_themes):
                theme = item["theme"]
                is_focus_theme = theme in focus_theme_names
                width = 4 if is_focus_theme else 3 if theme in highlight_themes else 1
                dash = "solid" if theme in highlight_themes else "dot"
                marker_sizes = [12 if p == focus_period else 5 for p in periods]
                fig_line.add_trace(go.Scatter(
                    x=periods,
                    y=item["weights"],
                    mode="lines+markers",
                    name=theme,
                    line=dict(width=width, dash=dash, color=colors[i % len(colors)]),
                    opacity=1 if is_focus_theme or theme in highlight_themes else 0.35,
                    marker=dict(size=marker_sizes, line=dict(width=1, color="white")),
                    hovertemplate="%{x}<br>%{y:.1%}<extra>" + theme + "</extra>",
                ))
            if focus_period:
                fig_line.add_vline(x=focus_period, line_width=3, line_dash="dash", line_color="#F97316")
            legend_rows = max(1, (len(line_themes) + 2) // 3)
            legend_margin = 50 + legend_rows * 28
            fig_line.update_layout(
                xaxis_title="时段",
                yaxis_tickformat=".0%",
                height=max(400, min(top_n, 8) * 40) + legend_margin,
                legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
                margin=dict(b=legend_margin),
            )
            st.plotly_chart(fig_line, use_container_width=True)


def main():
    """主函数"""
    st.set_page_config(
        page_title="微信读书笔记洞察",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 微信读书笔记主题洞察")

    pages = ["📊 概览", "📚 主题列表", "📝 笔记详情", "🔍 噪声洞察", "📈 时间演变"]

    # 侧边栏导航（放在刷新按钮之前，并用 key 持久化选中页）
    page = st.sidebar.radio(
        "导航",
        pages,
        key="nav_page",
        label_visibility="collapsed",
    )

    if st.sidebar.button("🔄 刷新数据"):
        load_data.clear()
        load_noise_analysis.clear()
        load_temporal_analysis.clear()
        st.rerun()

    # 加载数据
    with st.spinner("加载数据..."):
        notes, book_map, themes, labels, coords_2d = load_data()
        noise_analysis = load_noise_analysis(_noise_analysis_mtime())
        temporal_analysis = load_temporal_analysis(_temporal_analysis_mtime())

    # 页面切换
    if page == "📊 概览":
        view_overview(notes, themes, labels, coords_2d)
    elif page == "📚 主题列表":
        view_themes(themes, notes, book_map, labels)
    elif page == "🔍 噪声洞察":
        view_noise_analysis(noise_analysis, notes, themes, book_map)
    elif page == "📈 时间演变":
        view_temporal(temporal_analysis)
    else:
        view_notes(notes, themes, book_map, labels)


if __name__ == "__main__":
    main()
