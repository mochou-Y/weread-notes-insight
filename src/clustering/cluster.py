"""主题聚类模块"""

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import hdbscan
import jieba.analyse
import numpy as np
import umap
from openai import OpenAI

from config.settings import settings
from src.data.models import Note, Theme

logger = logging.getLogger(__name__)


class ThemeClusterer:
    """主题聚类器（支持UMAP降维）"""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_method: str = "eom",
        n_components: int = 15,
        use_umap: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.n_components = n_components
        self.use_umap = use_umap
        self.reducer_nd = None  # n维降维器
        self.reducer_2d = None  # 2维降维器（用于可视化）

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """使用UMAP降维到n_components维"""
        print(f"UMAP降维: {embeddings.shape[1]} -> {self.n_components}")
        self.reducer_nd = umap.UMAP(
            n_components=self.n_components,
            metric="cosine",
            random_state=42,
        )
        return self.reducer_nd.fit_transform(embeddings)

    def reduce_to_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """使用UMAP降维到2维（用于可视化）"""
        print(f"UMAP降维到2D: {embeddings.shape[1]} -> 2")
        self.reducer_2d = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        return self.reducer_2d.fit_transform(embeddings)

    def cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """执行HDBSCAN聚类，返回labels和2D坐标"""
        coords_2d = None
        # 先降维到n维用于聚类
        if self.use_umap and embeddings.shape[1] > self.n_components:
            embeddings_nd = self.reduce_dimensions(embeddings)
        else:
            embeddings_nd = embeddings

        # 降维到2维用于可视化
        if self.use_umap and embeddings.shape[1] > 2:
            coords_2d = self.reduce_to_2d(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(embeddings_nd)
        return labels, coords_2d

    def get_cluster_stats(self, labels: np.ndarray) -> dict:
        """获取聚类统计信息"""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        cluster_sizes = Counter(labels)
        cluster_sizes.pop(-1, None)  # 移除噪声点

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": dict(cluster_sizes),
        }


class ThemeLabeler:
    """主题标签生成器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.model = model or settings.llm_model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def label_theme_by_llm(self, notes: list[Note], max_notes: int = 20) -> str:
        """使用LLM生成主题标签"""
        # 取前N条笔记作为样本
        # TODO 这里取了前N条，是否需要判断选取的笔记是否有代表性？
        sample_notes = notes[:max_notes]
        contents = [note.content[:200] for note in sample_notes]  # 限制每条长度

        prompt = f"""以下是用户在微信读书中的笔记摘录，请用一个简短的词组（2-6个字）概括这些笔记的共同主题。

笔记内容：
{chr(10).join(f'- {c}' for c in contents)}

请直接输出主题词，不要有其他内容。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )

        return response.choices[0].message.content.strip()

    def label_theme_by_tfidf(self, notes: list[Note]) -> str:
        """使用jieba TF-IDF生成主题标签"""
        # 常见无意义词
        stop_words = {
            "一个", "一种", "这个", "那个", "这些", "那些", "什么", "怎么",
            "如何", "为什么", "因为", "所以", "但是", "然而", "如果", "虽然",
            "可以", "可能", "应该", "必须", "需要", "能够", "已经", "正在",
            "我们", "他们", "她们", "它们", "自己", "他人", "大家", "人们",
            "这样", "那样", "怎样", "多少", "哪里", "那里", "这里", "真的",
        }

        contents = [note.content for note in notes]
        text = " ".join(contents)

        # 使用jieba提取关键词，多提取一些以便过滤
        keywords = jieba.analyse.extract_tags(text, topK=8, withWeight=False)

        # 过滤停用词和单字词
        filtered = [w for w in keywords if w not in stop_words and len(w) > 1]

        if not filtered:
            return "未命名主题"

        # 组合前2个关键词作为标签
        return "/".join(filtered[:2])


class ThemeManager:
    """主题管理器"""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_method: str = "eom",
        n_components: int = 15,
        use_umap: bool = True,
    ):
        self.clusterer = ThemeClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            n_components=n_components,
            use_umap=use_umap,
        )
        self.labeler = ThemeLabeler()

    def discover_themes(
        self,
        notes: list[Note],
        embeddings: np.ndarray,
        use_llm: bool = True,
    ) -> tuple[list[Theme], np.ndarray, np.ndarray | None]:
        """发现主题

        Returns:
            themes: 主题列表
            labels: 聚类标签
            coords_2d: UMAP 2D坐标（用于可视化）
        """
        print(f"执行聚类 (min_cluster_size={self.clusterer.min_cluster_size}, min_samples={self.clusterer.min_samples}, method={self.clusterer.cluster_selection_method}, n_components={self.clusterer.n_components})...")
        vector_dim = embeddings.shape[1]
        print(f"原始向量维度: {vector_dim}")
        t0 = time.time()
        labels, coords_2d = self.clusterer.cluster(embeddings)
        stats = self.clusterer.get_cluster_stats(labels)
        print(f"发现 {stats['n_clusters']} 个主题，{stats['n_noise']} 个噪声点")
        elapsed = time.time() - t0
        m, s = divmod(int(elapsed), 60)
        print(f"聚类耗时: {f'{m}分钟' if m else ''}{s}秒")

        # 按聚类分组笔记
        themes = []
        unique_labels = sorted(set(labels) - {-1})

        # 准备聚类数据
        cluster_data = {}
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_notes = [notes[i] for i in cluster_indices]
            cluster_data[cluster_id] = cluster_notes

        # 生成主题标签
        labels_map = {}
        t1 = time.time()
        if use_llm:
            def label_cluster(cluster_id, cluster_notes):
                try:
                    label = self.labeler.label_theme_by_llm(cluster_notes)
                except Exception as e:
                    logger.warning(f"LLM标注失败，使用TF-IDF: {e}")
                    label = self.labeler.label_theme_by_tfidf(cluster_notes)
                return cluster_id, label

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(label_cluster, cid, cnotes): cid
                    for cid, cnotes in cluster_data.items()
                }
                for future in as_completed(futures):
                    cid, label = future.result()
                    labels_map[cid] = label
        else:
            for cluster_id, cluster_notes in cluster_data.items():
                labels_map[cluster_id] = self.labeler.label_theme_by_tfidf(cluster_notes)

        elapsed = time.time() - t1
        m, s = divmod(int(elapsed), 60)
        print(f"生成标签耗时: {f'{m}分钟' if m else ''}{s}秒")

        for cluster_id in unique_labels:
            cluster_notes = cluster_data[cluster_id]
            label = labels_map[cluster_id]
            theme = Theme(
                id=f"theme_{cluster_id}",
                label=label,
                note_ids=[note.id for note in cluster_notes],
            )
            themes.append(theme)
            print(f"  - {label}: {len(cluster_notes)} 条笔记")

        return themes, labels, coords_2d
