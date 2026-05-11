"""图谱可视化模块"""

from pathlib import Path
from typing import Optional

import numpy as np
from pyvis.network import Network

from src.data.models import Book, GraphEdge, GraphNode, Note, Theme


class GraphBuilder:
    """图谱构建器"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def build_nodes_from_themes(
        self,
        themes: list[Theme],
        notes: list[Note],
        books: list[Book],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """从主题构建图谱节点和边"""
        nodes = []
        edges = []

        # 创建笔记和书籍的映射
        note_map = {note.id: note for note in notes}
        book_map = {book.book_id: book for book in books}

        # 已添加的书籍节点
        added_books = set()

        # 添加主题节点
        for theme in themes:
            nodes.append(GraphNode(
                id=theme.id,
                type="theme",
                label=theme.label,
                metadata={"note_count": len(theme.note_ids)},
            ))

            # 添加该主题下的笔记节点和边
            for note_id in theme.note_ids:
                if note_id not in note_map:
                    continue

                note = note_map[note_id]

                # 添加笔记节点
                nodes.append(GraphNode(
                    id=note.id,
                    type="note",
                    label=note.content[:50] + "..." if len(note.content) > 50 else note.content,
                    metadata={
                        "book_title": note.book_title,
                        "chapter": note.chapter,
                        "create_time": note.create_time.strftime("%Y-%m-%d"),
                    },
                ))

                # 主题 -> 笔记 边
                edges.append(GraphEdge(
                    source=theme.id,
                    target=note.id,
                    type="contains",
                ))

                # 添加书籍节点（如果还没有）
                if note.book_id not in added_books:
                    book = book_map.get(note.book_id)
                    if book:
                        nodes.append(GraphNode(
                            id=book.book_id,
                            type="book",
                            label=book.title,
                            metadata={"author": book.author},
                        ))
                        added_books.add(note.book_id)

                # 笔记 -> 书籍 边
                if note.book_id in added_books:
                    edges.append(GraphEdge(
                        source=note.id,
                        target=note.book_id,
                        type="from",
                    ))

        return nodes, edges

    def add_similar_edges(
        self,
        edges: list[GraphEdge],
        notes: list[Note],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> list[GraphEdge]:
        """添加相似笔记边"""
        note_ids = [note.id for note in notes]

        # 只在同一聚类内计算相似度
        unique_labels = set(labels) - {-1}

        for cluster_label in unique_labels:
            indices = np.where(labels == cluster_label)[0]
            cluster_embeddings = embeddings[indices]

            # 计算相似度矩阵
            similarity_matrix = np.dot(cluster_embeddings, cluster_embeddings.T)

            # 找出相似的笔记对
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if similarity_matrix[i, j] > self.similarity_threshold:
                        edges.append(GraphEdge(
                            source=note_ids[indices[i]],
                            target=note_ids[indices[j]],
                            type="similar",
                            score=float(similarity_matrix[i, j]),
                        ))

        return edges


class GraphVisualizer:
    """图谱可视化器"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_file: str = "graph.html",
        title: str = "微信读书笔记主题图谱",
    ) -> str:
        """生成可视化图谱"""
        # 创建网络
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
        )

        # 节点颜色配置
        colors = {
            "theme": "#FF6B6B",
            "note": "#4ECDC4",
            "book": "#EA38C1",
        }

        # 添加节点
        for node in nodes:
            net.add_node(
                node.id,
                label=node.label[:30],
                color=colors.get(node.type, "#999999"),
                title=f"{node.label}\n类型: {node.type}",
                size=30 if node.type == "theme" else 15,
            )

        # 添加边
        edge_colors = {
            "contains": "#999999",
            "from": "#45B7D1",
            "similar": "#FF6B6B",
        }

        for edge in edges:
            net.add_edge(
                edge.source,
                edge.target,
                color=edge_colors.get(edge.type, "#999999"),
                title=f"关系: {edge.type}" + (f"\n相似度: {edge.score:.2f}" if edge.score else ""),
            )

        # 设置物理布局
        net.barnes_hut(
            gravity=-2000,
            central_gravity=0.3,
            spring_length=100,
        )

        # 保存文件
        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))

        print(f"图谱已保存到 {output_path}")
        return str(output_path)
