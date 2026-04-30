"""主题聚类模块"""

from collections import Counter
from typing import Optional

import hdbscan
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import settings
from src.data.models import Note, Theme


class ThemeClusterer:
    """主题聚类器"""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """执行HDBSCAN聚类"""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

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
        """使用TF-IDF生成主题标签"""
        contents = [note.content for note in notes]

        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(contents)

        # 获取TF-IDF最高的词
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        return top_words[0] if top_words else "未命名主题"


class ThemeManager:
    """主题管理器"""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_method: str = "eom",
    ):
        self.clusterer = ThemeClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        self.labeler = ThemeLabeler()

    def discover_themes(
        self,
        notes: list[Note],
        embeddings: np.ndarray,
        use_llm: bool = True,
    ) -> list[Theme]:
        """发现主题"""
        print(f"执行聚类 (min_cluster_size={self.clusterer.min_cluster_size}, min_samples={self.clusterer.min_samples}, method={self.clusterer.cluster_selection_method})...")
        vector_dim = embeddings.shape[1]
        print(f"向量维度 {vector_dim}")
        labels = self.clusterer.cluster(embeddings)
        stats = self.clusterer.get_cluster_stats(labels)
        print(f"发现 {stats['n_clusters']} 个主题，{stats['n_noise']} 个噪声点")

        # 按聚类分组笔记
        themes = []
        unique_labels = sorted(set(labels) - {-1})

        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_notes = [notes[i] for i in cluster_indices]

            # 生成主题标签
            if use_llm:
                try:
                    label = self.labeler.label_theme_by_llm(cluster_notes)
                except Exception:
                    label = self.labeler.label_theme_by_tfidf(cluster_notes)
            else:
                label = self.labeler.label_theme_by_tfidf(cluster_notes)

            theme = Theme(
                id=f"theme_{cluster_id}",
                label=label,
                note_ids=[note.id for note in cluster_notes],
            )
            themes.append(theme)
            print(f"  - {label}: {len(cluster_notes)} 条笔记")

        return themes, labels
