"""Embedding生成模块"""

from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from config.settings import settings
from src.data.models import Note


class Embedder:
    """Embedding生成器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.model = model or settings.embedding_model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def embed_text(self, text: str) -> list[float]:
        """生成单个文本的embedding"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """批量生成embedding"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"  生成embedding: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_notes(self, notes: list[Note]) -> np.ndarray:
        """为笔记列表生成embedding"""
        texts = []
        for note in notes:
            # 组合内容和上下文
            # TODO check
            text = note.content
            if note.context:
                text = f"{note.context}\n{note.content}"
            texts.append(text)

        embeddings = self.embed_texts(texts)
        return np.array(embeddings)


class EmbeddingStorage:
    """Embedding存储"""

    def __init__(self, embeddings_dir: str = "data/embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, name: str = "notes") -> None:
        """保存embedding到文件"""
        path = self.embeddings_dir / f"{name}.npy"
        np.save(path, embeddings)
        print(f"Embedding已保存到 {path}")

    def load(self, name: str = "notes") -> Optional[np.ndarray]:
        """从文件加载embedding"""
        path = self.embeddings_dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        return None

    def exists(self, name: str = "notes") -> bool:
        """检查embedding文件是否存在"""
        path = self.embeddings_dir / f"{name}.npy"
        return path.exists()
