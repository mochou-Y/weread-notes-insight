"""数据模型"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Note:
    """笔记数据模型"""

    id: str  # bookmarkId 或 reviewId
    book_id: str
    book_title: str
    book_author: str
    type: str  # "bookmark" | "highlight" | "review"
    content: str  # markText 或 content
    chapter: str
    create_time: datetime
    color_style: int = 0  # 0-4，仅划线有
    context: str = ""  # abstract（想法对应的划线内容）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_author": self.book_author,
            "type": self.type,
            "content": self.content,
            "chapter": self.chapter,
            "create_time": self.create_time.isoformat(),
            "color_style": self.color_style,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        """从字典创建"""
        return cls(
            id=data["id"],
            book_id=data["book_id"],
            book_title=data["book_title"],
            book_author=data["book_author"],
            type=data["type"],
            content=data["content"],
            chapter=data["chapter"],
            create_time=datetime.fromisoformat(data["create_time"]),
            color_style=data.get("color_style", 0),
            context=data.get("context", ""),
        )


@dataclass
class Book:
    """书籍数据模型"""

    book_id: str
    title: str
    author: str
    cover: str
    categories: list = field(default_factory=list)
    finished: bool = False
    review_count: int = 0
    note_count: int = 0  # highlight
    bookmark_count: int = 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "cover": self.cover,
            "categories": self.categories,
            "review_count": self.review_count,
            "note_count": self.note_count,
            "bookmark_count": self.bookmark_count,
        }


@dataclass
class Theme:
    """主题数据模型"""

    id: str  # theme_1, theme_2, ...
    label: str  # 主题标签
    note_ids: list = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "label": self.label,
            "note_ids": self.note_ids,
            "description": self.description,
        }


@dataclass
class GraphNode:
    """图谱节点"""

    id: str
    type: str  # "theme" | "note" | "book"
    label: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """图谱边"""

    source: str
    target: str
    type: str  # "contains" | "from" | "similar"
    score: Optional[float] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }
        if self.score is not None:
            result["score"] = self.score
        return result
