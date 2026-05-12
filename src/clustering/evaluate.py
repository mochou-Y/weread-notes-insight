"""聚类质量评估模块"""

import random
from typing import Optional

import numpy as np
from openai import OpenAI
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config.settings import settings
from src.data.models import Note, Theme


def compute_technical_metrics(
    coords_2d: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """计算技术指标

    Args:
        coords_2d: UMAP 2D 坐标
        labels: 聚类标签

    Returns:
        包含各项指标的字典
    """
    total = len(labels)
    noise_count = (labels == -1).sum()
    clustered_count = total - noise_count

    mask = labels != -1
    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)

    # 基础统计
    result = {
        "total_notes": total,
        "clustered_notes": clustered_count,
        "noise_notes": noise_count,
        "coverage": clustered_count / total,
        "noise_rate": noise_count / total,
        "n_clusters": n_clusters,
    }

    # 主题大小统计
    cluster_sizes = [int((labels == i).sum()) for i in unique_labels]
    result["min_size"] = min(cluster_sizes)
    result["max_size"] = max(cluster_sizes)
    result["median_size"] = sorted(cluster_sizes)[len(cluster_sizes) // 2]
    result["cluster_sizes"] = cluster_sizes

    # 技术指标（仅对非噪声点计算，且至少需要2个簇）
    if n_clusters >= 2 and mask.sum() > 1:
        X = coords_2d[mask]
        y = labels[mask]

        result["silhouette"] = silhouette_score(
            X, y, metric="euclidean", sample_size=min(5000, len(y))
        )
        result["davies_bouldin"] = davies_bouldin_score(X, y)
        result["calinski_harabasz"] = calinski_harabasz_score(X, y)
    else:
        result["silhouette"] = None
        result["davies_bouldin"] = None
        result["calinski_harabasz"] = None

    return result


def evaluate_theme_consistency(
    theme: Theme,
    notes: list[Note],
    client: OpenAI,
    model: str,
    sample_size: int = 3,
) -> dict:
    """用 LLM 评估单个主题的语义一致性

    Args:
        theme: 主题
        notes: 该主题下的笔记列表
        client: OpenAI 客户端
        model: 模型名称
        sample_size: 抽样数量

    Returns:
        包含评分和理由的字典
    """
    print(f"开始评估主题'{theme.label}'下随机抽样的笔记语义一致性...")
    sample = random.sample(notes, min(sample_size, len(notes)))
    contents = [n.content[:200] for n in sample]

    prompt = f"""以下是同一个主题"{theme.label}"下的笔记摘录，请评估这些笔记的语义一致性。

笔记内容：
{chr(10).join(f'- {c}' for c in contents)}

请按以下格式回复（不要有其他内容）：
评分: 1-5（1=完全无关，5=高度一致）
理由: 一句话说明"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()

        # 解析评分和理由
        score = 0
        reason = ""
        for line in text.split("\n"):
            if line.startswith("评分"):
                try:
                    score = int("".join(c for c in line if c.isdigit())[:1])
                except ValueError:
                    score = 0
            elif line.startswith("理由"):
                reason = line.split(":", 1)[-1].strip() or line.split("：", 1)[-1].strip()

        return {"score": score, "reason": reason}
    except Exception as e:
        return {"score": 0, "reason": f"评估失败: {e}"}


def print_evaluation_report(
    metrics: dict,
    llm_results: Optional[list[dict]] = None,
):
    """打印评估报告

    Args:
        metrics: 技术指标
        llm_results: LLM 评估结果列表
    """
    print("\n=== 聚类质量评估 ===")

    # 技术指标
    print("\n--- 技术指标 ---")
    print(
        f"覆盖率: {metrics['coverage']:.1%} "
        f"({metrics['clustered_notes']}/{metrics['total_notes']} 条笔记被归入主题)"
    )
    print(f"噪声率: {metrics['noise_rate']:.1%} ({metrics['noise_notes']} 条)")
    print(f"主题数: {metrics['n_clusters']}")
    print(
        f"主题大小: 最小{metrics['min_size']}, "
        f"最大{metrics['max_size']}, "
        f"中位数{metrics['median_size']}"
    )

    if metrics["silhouette"] is not None:
        print(f"Silhouette Score: {metrics['silhouette']:.4f} [簇内紧凑度 vs 簇间分离度，-1~1, 越高越好]")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} [簇间相似度，越低越好]")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.1f} [簇间/簇内方差比，越高越好]")
    else:
        print("(簇数不足2，无法计算技术指标)")

    # 聚簇大小分布
    if metrics.get("cluster_sizes"):
        print("\n主题聚簇大小分布")
        sizes = metrics["cluster_sizes"]
        bucket_size = 10
        buckets: dict[str, int] = {}
        for s in sizes:
            lower = (s // bucket_size) * bucket_size
            upper = lower + bucket_size - 1
            key = f"{lower}-{upper}"
            buckets[key] = buckets.get(key, 0) + 1
        for key in sorted(buckets, key=lambda k: int(k.split("-")[0])):
            print(f"  {key}: {buckets[key]} themes")

    # LLM 评估结果
    if llm_results:
        print("\n--- LLM 语义评估 ---")
        for r in llm_results:
            icon = "✅" if r["score"] >= 4 else "⚠️" if r["score"] >= 2 else "❌"
            print(
                f"{icon} {r['label']} ({r['note_count']}条): "
                f"{r['score']}/5 - {r['reason']}"
            )
