"""微信读书API调用模块"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from config.settings import settings
from src.data.models import Book, Note


class WereadAPI:
    """微信读书API客户端"""

    def __init__(self, cookie: Optional[str] = None):
        self.base_url = settings.weread_base_url
        self.cookie = cookie or settings.cookie
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": self.cookie,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json",
        })

    def get_notebook(self) -> list[Book]:
        """获取所有有笔记的书籍列表"""
        url = f"{self.base_url}/api/user/notebook"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()

        books = []
        for item in data.get("books", []):
            book_info = item.get("book", {})
            book = Book(
                book_id=book_info.get("bookId", ""),
                title=book_info.get("title", ""),
                author=book_info.get("author", ""),
                cover=book_info.get("cover", ""),
                categories=[
                    cat.get("title", "")
                    for cat in book_info.get("categories", [])
                ],
                finished=book_info.get("finished", 0) == 1,
                review_count=item.get("reviewCount", 0),
                bookmark_count=item.get("bookmarkCount", 0),
            )
            books.append(book)

        return books

    def get_bookmarks(self, book_id: str) -> list[Note]:
        """获取某本书的划线列表"""
        url = f"{self.base_url}/web/book/bookmarklist"
        params = {"bookId": book_id}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        notes = []
        book_info = data.get("book", {})
        book_title = book_info.get("title", "")
        book_author = book_info.get("author", "")

        for item in data.get("updated", []):
            note = Note(
                id=item.get("bookmarkId", ""),
                book_id=book_id,
                book_title=book_title,
                book_author=book_author,
                type="bookmark",
                content=item.get("markText", ""),
                chapter=item.get("chapterName", ""),
                create_time=datetime.fromtimestamp(item.get("createTime", 0)),
                color_style=item.get("colorStyle", 0),
            )
            notes.append(note)

        return notes

    def get_reviews(self, book_id: str) -> list[Note]:
        """获取某本书的想法列表"""
        url = f"{self.base_url}/web/review/list"
        params = {
            "bookId": book_id,
            "listType": 11,
            "mine": 1,
            "synckey": 0,
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        notes = []
        for item in data.get("reviews", []):
            review = item.get("review", {})
            book_info = review.get("book", {})
            note = Note(
                id=review.get("reviewId", ""),
                book_id=book_id,
                book_title=book_info.get("title", ""),
                book_author=book_info.get("author", ""),
                type="review",
                content=review.get("content", ""),
                chapter=review.get("chapterTitle", "") or review.get("chapterName", ""),
                create_time=datetime.fromtimestamp(review.get("createTime", 0)),
                context=review.get("abstract", ""),
            )
            notes.append(note)

        return notes

    def get_all_notes(self, book_id: str) -> list[Note]:
        """获取某本书的所有笔记（划线+想法）"""
        bookmarks = self.get_bookmarks(book_id)
        reviews = self.get_reviews(book_id)
        return bookmarks + reviews


class DataLoader:
    """数据加载和存储"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "bookmarks").mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_notebook(self, books: list[Book]) -> None:
        """保存书籍列表"""
        data = {
            "total": len(books),
            "books": [book.to_dict() for book in books],
            "updated_at": datetime.now().isoformat(),
        }
        path = self.raw_dir / "notebook.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_notebook(self) -> list[Book]:
        """加载书籍列表"""
        path = self.raw_dir / "notebook.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [Book(**book) for book in data.get("books", [])]

    def save_book_notes(self, book_id: str, notes: list[Note]) -> None:
        """保存某本书的笔记"""
        data = {
            "book_id": book_id,
            "total": len(notes),
            "notes": [note.to_dict() for note in notes],
            "updated_at": datetime.now().isoformat(),
        }
        path = self.raw_dir / "bookmarks" / f"{book_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_book_notes(self, book_id: str) -> list[Note]:
        """加载某本书的笔记"""
        path = self.raw_dir / "bookmarks" / f"{book_id}.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [Note.from_dict(note) for note in data.get("notes", [])]

    def load_all_notes(self) -> list[Note]:
        """加载所有笔记"""
        notes = []
        bookmarks_dir = self.raw_dir / "bookmarks"
        for path in bookmarks_dir.glob("*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            notes.extend([Note.from_dict(note) for note in data.get("notes", [])])
        return notes

    def get_synced_book_ids(self) -> set[str]:
        """获取已同步的书籍ID列表"""
        bookmarks_dir = self.raw_dir / "bookmarks"
        return {p.stem for p in bookmarks_dir.glob("*.json")}
