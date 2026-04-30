"""主入口模块"""

import argparse
import sys
from datetime import datetime

from config.settings import settings
from src.api.weread import DataLoader, WereadAPI


def cmd_fetch(args):
    """数据采集命令"""
    print("开始获取数据...")

    # 检查Cookie配置
    if not settings.cookie:
        print("错误: 请设置环境变量 WEREAD_COOKIE 或在配置中设置cookie")
        sys.exit(1)

    api = WereadAPI()
    loader = DataLoader()

    # 获取书籍列表
    print("获取书籍列表...")
    books = api.get_notebook()
    print(f"共 {len(books)} 本书有笔记")

    # 保存书籍列表
    loader.save_notebook(books)

    # 获取每本书的笔记
    limit = args.limit or len(books)
    synced_ids = loader.get_synced_book_ids()

    for i, book in enumerate(books[:limit]):
        if args.incremental and book.book_id in synced_ids:
            print(f"[{i+1}/{limit}] 跳过 {book.title} (已同步)")
            continue

        print(f"[{i+1}/{limit}] 获取 {book.title} 的笔记...")
        try:
            # TODO 目前当cookie过期时会得到空列表，不会报错提示或自动刷新
            notes = api.get_all_notes(book.book_id)
            loader.save_book_notes(book.book_id, notes)
            print(f"  - 获取 {len(notes)} 条笔记")
        except Exception as e:
            print(f"  - 错误: {e}")

    print("数据采集完成!")


def cmd_cluster(args):
    """主题聚类命令"""
    print("开始主题聚类...")

    # 检查API配置
    if not settings.openai_api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    loader = DataLoader()
    notes = loader.load_all_notes()

    if not notes:
        print("错误: 没有找到笔记数据，请先运行 fetch 命令")
        sys.exit(1)

    notes_exclude_bookmarks = [note for note in notes if note.type != 'bookmark']
    print(f"加载 {len(notes)} 条笔记")
    print(f"去除书签共 {len(notes_exclude_bookmarks)} 条笔记")

    # 生成或加载embedding
    from src.embedding.embedder import Embedder, EmbeddingStorage

    storage = EmbeddingStorage()
    embeddings = storage.load()

    if embeddings is None:
        print("生成embedding...")
        embedder = Embedder()
        embeddings = embedder.embed_notes(notes_exclude_bookmarks)
        storage.save(embeddings)
    else:
        print(f"加载已有embedding: {embeddings.shape}")

    # 执行聚类
    from src.clustering.cluster import ThemeManager

    manager = ThemeManager(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.method,
    )
    themes, labels = manager.discover_themes(notes_exclude_bookmarks, embeddings, use_llm=True)

    # 保存聚类结果
    import json
    import numpy as np

    result = {
        "total_notes": len(notes_exclude_bookmarks),
        "total_themes": len(themes),
        "themes": [theme.to_dict() for theme in themes],
        "updated_at": datetime.now().isoformat(),
    }

    output_path = loader.processed_dir / "themes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存 labels
    labels_path = loader.processed_dir / "labels.npy"
    np.save(labels_path, labels)
    print(f"聚类结果已保存到 {output_path}")
    print(f"Labels已保存到 {labels_path}")


def cmd_graph(args):
    """生成图谱命令"""
    print("开始生成主题图谱...")

    loader = DataLoader()
    notes = loader.load_all_notes()
    notes = [note for note in notes if note.type != 'bookmark']
    books = loader.load_notebook()

    if not notes:
        print("错误: 没有找到笔记数据，请先运行 fetch 命令")
        sys.exit(1)

    # 加载聚类结果
    import json

    themes_path = loader.processed_dir / "themes.json"
    if not themes_path.exists():
        print("错误: 没有找到聚类结果，请先运行 cluster 命令")
        sys.exit(1)

    with open(themes_path, encoding="utf-8") as f:
        themes_data = json.load(f)

    from src.data.models import Theme

    themes = [Theme(**t) for t in themes_data["themes"]]

    # 加载embedding和labels
    import numpy as np
    from src.embedding.embedder import EmbeddingStorage

    storage = EmbeddingStorage()
    embeddings = storage.load()

    if embeddings is None:
        print("错误: 没有找到embedding数据")
        sys.exit(1)

    # 加载已保存的 labels
    labels_path = loader.processed_dir / "labels.npy"
    if not labels_path.exists():
        print("错误: 没有找到labels数据，请先运行 cluster 命令")
        sys.exit(1)

    labels = np.load(labels_path)
    print(f"加载已有labels: {labels.shape}")

    # 构建图谱
    from src.graph.visualizer import GraphBuilder, GraphVisualizer

    builder = GraphBuilder()
    nodes, edges = builder.build_nodes_from_themes(themes, notes, books)

    # 添加相似笔记边
    edges = builder.add_similar_edges(edges, notes, embeddings, labels)

    # 可视化
    visualizer = GraphVisualizer()
    output_path = visualizer.visualize(nodes, edges, args.output)

    print(f"图谱生成完成: {output_path}")


def cmd_status(args):
    """查看状态命令"""
    loader = DataLoader()
    books = loader.load_notebook()
    all_notes = loader.load_all_notes()

    print("=" * 50)
    print("数据状态")
    print("=" * 50)
    print(f"书籍总数量: {len(books)}")
    print(f"本地保存笔记总数: {len(all_notes)}")

    if all_notes:
        # 统计笔记类型
        bookmarks = [n for n in all_notes if n.type == "bookmark"]
        highlights = [n for n in all_notes if n.type == "highlight"]
        reviews = [n for n in all_notes if n.type == "review"]
        print(f"  - 书签: {len(bookmarks)}")
        print(f"  - 划线: {len(highlights)}")
        print(f"  - 想法: {len(reviews)}")

        # 时间范围
        times = [n.create_time for n in all_notes]
        print(f"时间范围: {min(times).strftime('%Y-%m-%d')} ~ {max(times).strftime('%Y-%m-%d')}")

    print("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微信读书笔记洞察工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # fetch 命令
    fetch_parser = subparsers.add_parser("fetch", help="获取数据")
    fetch_parser.add_argument("--limit", type=int, help="限制获取书籍数量")
    fetch_parser.add_argument("--incremental", action="store_true", help="增量更新")

    # cluster 命令
    cluster_parser = subparsers.add_parser("cluster", help="主题聚类")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=3, help="HDBSCAN min_cluster_size参数")
    cluster_parser.add_argument("--min-samples", type=int, default=2, help="HDBSCAN min_samples参数")
    cluster_parser.add_argument("--method", type=str, default="leaf", choices=["eom", "leaf"], help="聚类选择方法: eom(粗粒度) 或 leaf(细粒度)")

    # graph 命令
    graph_parser = subparsers.add_parser("graph", help="生成主题图谱")
    graph_parser.add_argument("--output", type=str, default="graph.html", help="输出文件名")

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看数据状态")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "cluster":
        cmd_cluster(args)
    elif args.command == "graph":
        cmd_graph(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
