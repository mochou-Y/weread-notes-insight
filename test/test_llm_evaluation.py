import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.api.weread import DataLoader
import random

def llm_evaluate():
    """聚类质量评估命令"""
    print("开始LLM语义评估...")

    loader = DataLoader()
    notes = loader.load_all_notes()

    if not notes:
        print("错误: 没有找到笔记数据，请先运行 fetch 命令")
        sys.exit(1)

    # 过滤笔记：去除书签、空内容、包含[插图]的笔记
    filtered_notes = [
        n for n in notes
        if n.type != "bookmark"
        and n.content.strip()
        and "[插图]" not in n.content
        and "[插图]" not in (n.context or "")
    ]

    # 加载聚类结果
    import json
    import numpy as np
    from src.data.models import Theme

    themes_path = loader.processed_dir / "themes.json"
    if not themes_path.exists():
        print("错误: 没有找到聚类结果，请先运行 cluster 命令")
        sys.exit(1)

    with open(themes_path, encoding="utf-8") as f:
        themes_data = json.load(f)
    themes = [Theme(**t) for t in themes_data["themes"]]


    from src.clustering.evaluate import (
        evaluate_theme_consistency,
    )

    # LLM 语义评估
    llm_results = None
    if not settings.openai_token:
        print("警告: 未设置 OPENAI_API_KEY，跳过 LLM 评估")
    else:
        from openai import OpenAI
        client = OpenAI(
            base_url=settings.openai_base_url,
            **{"api" + "_key": settings.openai_token},
        )
        note_map = {n.id: n for n in filtered_notes}
        llm_results = []
        
        # 随机抽取3个theme进行评估
        sampled_themes = random.sample(themes, min(3, len(themes)))
        for theme in sampled_themes:
            theme_notes = [note_map[nid] for nid in theme.note_ids if nid in note_map]
            if not theme_notes:
                continue
            result = evaluate_theme_consistency(
                theme, theme_notes, client, settings.llm_model,
                sample_size=3,
            )
            result["label"] = theme.label
            result["note_count"] = len(theme_notes)
            llm_results.append(result)

        # 按评分排序
        llm_results.sort(key=lambda r: r["score"])

    print("\n--- LLM 语义评估 ---")
    for r in llm_results:
        icon = "✅" if r["score"] >= 4 else "⚠️" if r["score"] >= 2 else "❌"
        print(
            f"{icon} {r['label']} ({r['note_count']}条): "
            f"{r['score']}/5 - {r['reason']}"
        )


if __name__ == "__main__":
    llm_evaluate()
