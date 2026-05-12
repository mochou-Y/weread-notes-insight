"""主入口模块"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

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


def cmd_embedding(args):
    """生成embedding命令"""
    print("开始生成embedding...")

    # 检查API配置
    if not settings.openai_api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    loader = DataLoader()
    notes = loader.load_all_notes()

    if not notes:
        print("错误: 没有找到笔记数据，请先运行 fetch 命令")
        sys.exit(1)

    # 过滤笔记：去除书签、空内容、包含[插图]的笔记
    filtered_notes = [
        n for n in notes
        if n.type != "bookmark"
        and n.content.strip()  # 过滤空内容
        and "[插图]" not in n.content  # 过滤内容包含[插图]
        and "[插图]" not in (n.context or "")  # 过滤context包含[插图]
    ]
    print(f"加载 {len(notes)} 条笔记")
    print(f"过滤后共 {len(filtered_notes)} 条笔记，去除书签、空内容、包含[插图]的笔记")

    # 生成embedding
    from src.embedding.embedder import Embedder, EmbeddingStorage

    storage = EmbeddingStorage()

    if storage.exists() and not args.force:
        print("embedding已存在，使用 --force 参数强制重新生成")
        sys.exit(0)

    embedder = Embedder()
    embeddings = embedder.embed_notes(filtered_notes)
    storage.save(embeddings)
    print("embedding生成完成!")


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

    # 过滤笔记：去除书签、空内容、包含[插图]的笔记
    filtered_notes = [
        n for n in notes
        if n.type != "bookmark"
        and n.content.strip()  # 过滤空内容
        and "[插图]" not in n.content  # 过滤内容包含[插图]
        and "[插图]" not in (n.context or "")  # 过滤context包含[插图]
    ]
    print(f"加载 {len(notes)} 条笔记")
    print(f"过滤后共 {len(filtered_notes)} 条笔记，去除书签、空内容、包含[插图]的笔记")


    # 加载embedding
    from src.embedding.embedder import EmbeddingStorage

    storage = EmbeddingStorage()
    embeddings = storage.load()

    if embeddings is None:
        print("错误: 没有找到embedding数据，请先运行 embedding 命令")
        sys.exit(1)

    print(f"加载已有embedding: {embeddings.shape}")

    # 执行聚类
    from src.clustering.cluster import ThemeManager

    manager = ThemeManager(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.method,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        use_umap=not args.no_umap,
        label_strategy=args.label_strategy,
    )
    themes, labels, coords_2d = manager.discover_themes(filtered_notes, embeddings, use_llm=True)

    # 保存聚类结果
    import json
    import numpy as np

    result = {
        "total_notes": len(filtered_notes),
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

    # 保存 UMAP 2D 坐标（用于可视化）
    if coords_2d is not None:
        coords_path = loader.processed_dir / "umap_coords.npy"
        np.save(coords_path, coords_2d)
        print(f"UMAP坐标已保存到 {coords_path}")

    print(f"聚类结果已保存到 {output_path}")
    print(f"Labels已保存到 {labels_path}")


def cmd_evaluate(args):
    """聚类质量评估命令"""
    print("开始评估聚类质量...")

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

    # 加载 labels
    labels_path = loader.processed_dir / "labels.npy"
    if not labels_path.exists():
        print("错误: 没有找到labels数据，请先运行 cluster 命令")
        sys.exit(1)
    labels = np.load(labels_path)

    # 加载 UMAP 2D 坐标
    coords_path = loader.processed_dir / "umap_coords.npy"
    if not coords_path.exists():
        print("错误: 没有找到UMAP坐标数据，请重新运行 cluster 命令")
        sys.exit(1)
    coords_2d = np.load(coords_path)

    # 确保数据量一致
    if len(labels) != len(filtered_notes):
        print(f"警告: labels数量({len(labels)})与过滤后笔记数量({len(filtered_notes)})不一致")
        min_len = min(len(labels), len(filtered_notes))
        labels = labels[:min_len]
        filtered_notes = filtered_notes[:min_len]
        coords_2d = coords_2d[:min_len]

    # 计算技术指标
    from src.clustering.evaluate import (
        compute_technical_metrics,
        evaluate_theme_consistency,
        print_evaluation_report,
    )

    metrics = compute_technical_metrics(coords_2d, labels)

    # LLM 语义评估
    llm_results = None
    if not args.no_llm:
        if not settings.openai_api_key:
            print("警告: 未设置 OPENAI_API_KEY，跳过 LLM 评估")
        else:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            note_map = {n.id: n for n in filtered_notes}
            llm_results = []
            for theme in themes:
                theme_notes = [note_map[nid] for nid in theme.note_ids if nid in note_map]
                if not theme_notes:
                    continue
                result = evaluate_theme_consistency(
                    theme, theme_notes, client, settings.llm_model,
                    sample_size=args.sample,
                )
                result["label"] = theme.label
                result["note_count"] = len(theme_notes)
                llm_results.append(result)

            # 按评分排序
            llm_results.sort(key=lambda r: r["score"])

    # 打印报告
    print_evaluation_report(metrics, llm_results)


def cmd_graph(args):
    """生成图谱命令"""
    print("开始生成主题图谱...")

    loader = DataLoader()
    notes = loader.load_all_notes()
    # 过滤笔记：去除书签、空内容、包含[插图]的笔记
    notes = [
        n for n in notes
        if n.type != "bookmark"
        and n.content.strip()  # 过滤空内容
        and "[插图]" not in n.content  # 过滤内容包含[插图]
        and "[插图]" not in (n.context or "")  # 过滤context包含[插图]
    ]
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
        notes = [
            n for n in all_notes
            if n.type != "bookmark"
            and n.content.strip()  # 过滤空内容
            and "[插图]" not in n.content  # 过滤内容包含[插图]
            and "[插图]" not in (n.context or "")  # 过滤context包含[插图]
        ]
        print(f"  - 去除书签、空内容、包含[插图]: {len(notes)}")

        # 时间范围
        times = [n.create_time for n in all_notes]
        print(f"时间范围: {min(times).strftime('%Y-%m-%d')} ~ {max(times).strftime('%Y-%m-%d')}")

    print("=" * 50)


def cmd_serve(args):
    """启动可视化服务"""
    import subprocess
    import sys

    print("启动 Streamlit 服务...")
    app_path = Path(__file__).parent / "app" / "main.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
    ])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微信读书笔记洞察工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # fetch 命令
    fetch_parser = subparsers.add_parser("fetch", help="获取数据")
    fetch_parser.add_argument("--limit", type=int, help="限制获取书籍数量")
    fetch_parser.add_argument("--incremental", action="store_true", help="增量更新")

    # embedding 命令
    embedding_parser = subparsers.add_parser("embedding", help="生成embedding向量")
    embedding_parser.add_argument("--force", action="store_true", help="强制重新生成")

    # cluster 命令
    cluster_parser = subparsers.add_parser("cluster", help="主题聚类")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=3, help="HDBSCAN min_cluster_size参数")
    cluster_parser.add_argument("--min-samples", type=int, default=2, help="HDBSCAN min_samples参数")
    cluster_parser.add_argument("--method", type=str, default="eom", choices=["eom", "leaf"], help="聚类选择方法: eom(粗粒度) 或 leaf(细粒度)")
    cluster_parser.add_argument("--n-components", type=int, default=15, help="UMAP降维目标维度")
    cluster_parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors参数，影响全局结构保留")
    cluster_parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist参数，影响簇的紧凑程度")
    cluster_parser.add_argument("--label-strategy", type=str, default="hybrid", choices=["llm", "tfidf", "hybrid"], help="标签生成策略: llm(全LLM), tfidf(全TF-IDF), hybrid(混合)")
    cluster_parser.add_argument("--no-umap", action="store_true", help="禁用UMAP降维")

    # evaluate 命令
    evaluate_parser = subparsers.add_parser("evaluate", help="评估聚类质量")
    evaluate_parser.add_argument("--no-llm", action="store_true", help="跳过LLM语义评估")
    evaluate_parser.add_argument("--sample", type=int, default=3, help="每个主题抽样数量（默认3）")

    # graph 命令
    graph_parser = subparsers.add_parser("graph", help="生成主题图谱")
    graph_parser.add_argument("--output", type=str, default="graph.html", help="输出文件名")

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看数据状态")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动可视化服务")
    serve_parser.add_argument("--port", type=int, default=8501, help="服务端口")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "embedding":
        cmd_embedding(args)
    elif args.command == "cluster":
        cmd_cluster(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "graph":
        cmd_graph(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
