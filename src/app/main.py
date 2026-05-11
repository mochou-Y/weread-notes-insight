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
import streamlit as st

from src.api.weread import DataLoader
from src.data.models import Theme


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


def main():
    """主函数"""
    st.set_page_config(
        page_title="微信读书笔记洞察",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 微信读书笔记主题洞察")

    # 加载数据
    with st.spinner("加载数据..."):
        notes, book_map, themes, labels, coords_2d = load_data()

    # 侧边栏导航
    page = st.sidebar.radio(
        "导航",
        ["📊 概览", "📚 主题列表", "📝 笔记详情"],
        label_visibility="collapsed",
    )

    # 页面切换
    if page == "📊 概览":
        view_overview(notes, themes, labels, coords_2d)
    elif page == "📚 主题列表":
        view_themes(themes, notes, book_map, labels)
    else:
        view_notes(notes, themes, book_map, labels)


if __name__ == "__main__":
    main()
