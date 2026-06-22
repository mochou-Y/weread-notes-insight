"""Embedding生成模块"""

from pathlib import Path
import time
from typing import Optional

import numpy as np
import requests

from config.settings import settings
from src.data.models import Note


class Embedder:
    """Embedding生成器（使用SiliconFlow API）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.embedding_api_key or settings.openai_api_key
        self.base_url = base_url or settings.embedding_base_url
        self.model = model or settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size

    def embed_text(self, text: str) -> list[float]:
        """生成单个文本的embedding"""
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[float]]:
        """批量生成embedding"""
        all_embeddings = []
        batch_size = batch_size or self.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"  生成embedding: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")

            payload = {
                "model": self.model,
                "input": batch,
            }
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions

            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=180,
                    )
                    break
                except requests.Timeout:
                    if attempt == 2:
                        raise
                    wait_seconds = 2 ** attempt
                    print(f"  请求超时，{wait_seconds}秒后重试...")
                    time.sleep(wait_seconds)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise requests.HTTPError(
                    f"{exc}. Response body: {response.text}",
                    response=response,
                ) from exc
            data = response.json()

            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_notes(self, notes: list[Note], storage: Optional["EmbeddingStorage"] = None) -> np.ndarray:
        """为笔记列表生成embedding"""
        texts = []
        for note in notes:
            # 组合内容和上下文
            text = note.content
            if note.context:
                text = f"{note.context}\n{note.content}"
            texts.append(text)

        existing = storage.load_partial() if storage else None
        completed = len(existing) if existing is not None else 0
        embeddings = existing.tolist() if existing is not None else []

        if completed:
            print(f"  发现未完成embedding: {completed}/{len(texts)}，继续生成...")

        batch_size = getattr(self, "batch_size", settings.embedding_batch_size)
        for i in range(completed, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embed_texts(batch, batch_size=batch_size)
            embeddings.extend(batch_embeddings)
            if storage:
                storage.save_partial(np.array(embeddings))

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
        partial_path = self.embeddings_dir / f"{name}.partial.npy"
        if partial_path.exists():
            partial_path.unlink()
        print(f"Embedding已保存到 {path}")

    def save_partial(self, embeddings: np.ndarray, name: str = "notes") -> None:
        """保存未完成的embedding，便于网络中断后续跑"""
        path = self.embeddings_dir / f"{name}.partial.npy"
        np.save(path, embeddings)

    def load_partial(self, name: str = "notes") -> Optional[np.ndarray]:
        """加载未完成的embedding"""
        path = self.embeddings_dir / f"{name}.partial.npy"
        if path.exists():
            return np.load(path)
        return None

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
