"""噪声深度分析模块 — 从多主题交叉区挖掘交叉知识与用户画像"""

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import hdbscan
import numpy as np
from openai import OpenAI

from config.settings import settings
from src.api.weread import DataLoader
from src.clustering.cluster import ThemeLabeler
from src.data.models import Note, Theme


class NoiseAnalyzer:
    """噪声深度分析器

    对 HDBSCAN 标记为噪声（label=-1）的笔记进行三层挖掘：
    1. 子聚类 — 发现噪声内部的微主题
    2. 桥接分析 — 发现噪声笔记与已有主题的交叉关系
    3. 用户画像 — 综合产出结构化的用户认知画像
    """

    def __init__(
        self,
        bridge_threshold: float = 0.6,
        bridge_top_k: int = 3,
        subcluster_min_size: int = 3,
        subcluster_min_samples: int = 2,
    ):
        self.bridge_threshold = bridge_threshold
        self.bridge_top_k = bridge_top_k
        self.subcluster_min_size = subcluster_min_size
        self.subcluster_min_samples = subcluster_min_samples

        self.loader = DataLoader()
        self.labeler = ThemeLabeler()
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            **{"api" + "_key": settings.openai_token},
        )

        # load_data() 后填充
        self.notes: list[Note] = []
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.themes: list[Theme] = []
        self.noise_indices: Optional[np.ndarray] = None

    def load_data(self):
        """加载全部所需数据"""
        print("加载数据...")

        # 加载并过滤笔记
        all_notes = self.loader.load_all_notes()
        self.notes = [
            n for n in all_notes
            if n.type != "bookmark"
            and n.content.strip()
            and "[插图]" not in n.content
            and "[插图]" not in (n.context or "")
        ]

        # 加载 embedding
        from src.embedding.embedder import EmbeddingStorage

        storage = EmbeddingStorage()
        self.embeddings = storage.load()
        if self.embeddings is None:
            print("错误: 没有找到 embedding 数据，请先运行 embedding 命令")
            return False

        # 加载 labels
        labels_path = self.loader.processed_dir / "labels.npy"
        if not labels_path.exists():
            print("错误: 没有找到 labels 数据，请先运行 cluster 命令")
            return False
        self.labels = np.load(labels_path)

        # 加载 themes
        themes_path = self.loader.processed_dir / "themes.json"
        if not themes_path.exists():
            print("错误: 没有找到 themes 数据，请先运行 cluster 命令")
            return False
        with open(themes_path, encoding="utf-8") as f:
            themes_data = json.load(f)
        self.themes = [Theme(**t) for t in themes_data["themes"]]

        # 对齐数据
        min_len = min(len(self.labels), len(self.notes), len(self.embeddings))
        self.labels = self.labels[:min_len]
        self.notes = self.notes[:min_len]
        self.embeddings = self.embeddings[:min_len]

        # 提取噪声索引
        self.noise_indices = np.where(self.labels == -1)[0]
        print(f"笔记总数: {min_len}, 噪声笔记: {len(self.noise_indices)}")

        return True

    # ---- Step 1: 子聚类 ----

    def subcluster_noise(self) -> list[dict]:
        """对噪声笔记单独聚类，发现内部微主题"""
        print("\n=== 噪声子聚类 ===")

        noise_emb = self.embeddings[self.noise_indices]
        noise_notes = [self.notes[i] for i in self.noise_indices]

        print(f"噪声 embedding 形状: {noise_emb.shape}")
        print(f"HDBSCAN 参数: min_cluster_size={self.subcluster_min_size}, min_samples={self.subcluster_min_samples}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.subcluster_min_size,
            min_samples=self.subcluster_min_samples,
            metric="euclidean",
        )
        sub_labels = clusterer.fit_predict(noise_emb)

        unique_labels = sorted(set(sub_labels) - {-1})
        n_sub = len(unique_labels)
        still_noise = (sub_labels == -1).sum()
        print(f"发现 {n_sub} 个微主题, 仍有 {still_noise} 条未归类")

        # 分组并标注
        micro_themes = []
        for cid in unique_labels:
            mask = sub_labels == cid
            cluster_indices = np.where(mask)[0]
            cluster_notes = [noise_notes[i] for i in cluster_indices]

            # 标注
            try:
                label = self.labeler.label_theme_by_tfidf(cluster_notes)
            except Exception:
                label = f"微主题_{cid}"

            # 取样本
            sample_size = min(5, len(cluster_notes))
            samples = random.sample(cluster_notes, sample_size)
            sample_contents = [n.content[:200] for n in samples]

            micro_themes.append({
                "id": f"noise_theme_{cid}",
                "label": label,
                "size": len(cluster_notes),
                "note_ids": [n.id for n in cluster_notes],
                "sample_notes": sample_contents,
            })
            print(f"  - {label}: {len(cluster_notes)} 条")

        # 按大小降序
        micro_themes.sort(key=lambda x: x["size"], reverse=True)

        # 缓存
        self._save_cache("noise_micro_themes.json", micro_themes)

        return micro_themes

    # ---- Step 2: 桥接分析 ----

    def _detect_bridge_notes(self) -> tuple[list[dict], Counter, dict]:
        """检测桥接笔记（纯计算，不调用 LLM）"""
        note_id_to_idx = {n.id: i for i, n in enumerate(self.notes)}
        theme_centroids = {}
        theme_labels = {}

        for theme in self.themes:
            indices = [note_id_to_idx[nid] for nid in theme.note_ids if nid in note_id_to_idx]
            if not indices:
                continue
            emb = self.embeddings[indices]
            centroid = emb.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            theme_centroids[theme.id] = centroid
            theme_labels[theme.id] = theme.label

        noise_emb = self.embeddings[self.noise_indices]
        norms = np.linalg.norm(noise_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        noise_emb_normed = noise_emb / norms

        centroid_matrix = np.array([theme_centroids[tid] for tid in theme_centroids])
        theme_ids_ordered = list(theme_centroids.keys())
        sim_matrix = noise_emb_normed @ centroid_matrix.T

        bridge_pair_counter = Counter()
        bridge_notes_map: dict[tuple, list[dict]] = {}
        bridge_notes_detail: list[dict] = []

        for i in range(len(self.noise_indices)):
            sims = sim_matrix[i]
            top_k_indices = np.argsort(sims)[-self.bridge_top_k:][::-1]
            top_k_sims = sims[top_k_indices]

            qualified = [
                (theme_ids_ordered[top_k_indices[j]], top_k_sims[j])
                for j in range(len(top_k_indices))
                if top_k_sims[j] >= self.bridge_threshold
            ]

            if len(qualified) >= 2:
                pair = tuple(sorted([theme_labels[qualified[0][0]], theme_labels[qualified[1][0]]]))
                bridge_pair_counter[pair] += 1
                if pair not in bridge_notes_map:
                    bridge_notes_map[pair] = []
                note = self.notes[self.noise_indices[i]]
                bridge_themes = [theme_labels[tid] for tid, sim in qualified[:2]]
                note_detail = {
                    "note_id": note.id,
                    "book_id": note.book_id,
                    "book_title": note.book_title,
                    "content": note.content[:200],
                    "themes": bridge_themes,
                }
                bridge_notes_map[pair].append(note_detail)
                bridge_notes_detail.append(note_detail)

        return bridge_notes_detail, bridge_pair_counter, bridge_notes_map

    def analyze_bridges(self) -> list[dict]:
        """分析噪声笔记与已有主题的桥接关系"""
        print("\n=== 交叉桥分析 ===")

        bridge_notes_detail, bridge_pair_counter, bridge_notes_map = self._detect_bridge_notes()
        self._bridge_notes_detail = bridge_notes_detail
        print(f"检测到 {len(bridge_notes_detail)} 条桥接笔记")

        # 先缓存桥接笔记明细，避免 LLM 阶段中断导致丢失
        self._save_cache("noise_bridge_notes.json", bridge_notes_detail)

        # 排序并构建结果（LLM 分析）
        bridges = []
        top_n_for_llm = 10
        min_bridge_count = 4
        sorted_pairs = [(p, c) for p, c in bridge_pair_counter.most_common() if c >= min_bridge_count]

        for idx, (pair, count) in enumerate(sorted_pairs):
            notes_for_llm = bridge_notes_map[pair][:5]

            if idx < top_n_for_llm:
                insight = self._llm_bridge_insight(pair, notes_for_llm)
            else:
                insight = ""

            bridges.append({
                "themes": list(pair),
                "count": count,
                "insight": insight,
                "sample_notes": [n["content"] for n in notes_for_llm],
            })
            print(f"  - {pair[0]} <-> {pair[1]}: {count} 条桥接笔记")

        print(f"共发现 {len(bridges)} 个桥接模式")
        self._save_cache("noise_bridges.json", bridges)

        return bridges

    def identify_cross_domain_books(self, bridge_notes: list[dict]) -> list[dict]:
        """识别跨领域书籍：笔记横跨多个主题，或含大量桥接笔记"""
        note_to_theme: dict[str, str] = {}
        for theme in self.themes:
            for nid in theme.note_ids:
                note_to_theme[nid] = theme.label

        books: dict[str, dict] = {}

        def ensure_book(note: Note) -> dict:
            if note.book_id not in books:
                books[note.book_id] = {
                    "book_id": note.book_id,
                    "title": note.book_title,
                    "author": note.book_author,
                    "themes": set(),
                    "bridge_note_count": 0,
                    "bridge_pairs": set(),
                    "note_count": 0,
                    "sample_bridge_notes": [],
                }
            return books[note.book_id]

        for note in self.notes:
            b = ensure_book(note)
            b["note_count"] += 1
            if note.id in note_to_theme:
                b["themes"].add(note_to_theme[note.id])

        for bn in bridge_notes:
            note = next((n for n in self.notes if n.id == bn["note_id"]), None)
            if note is None:
                continue
            b = ensure_book(note)
            b["bridge_note_count"] += 1
            pair = tuple(sorted(bn["themes"]))
            b["bridge_pairs"].add(pair)
            for t in bn["themes"]:
                b["themes"].add(t)
            if len(b["sample_bridge_notes"]) < 3:
                b["sample_bridge_notes"].append(bn["content"])

        result = []
        for b in books.values():
            theme_count = len(b["themes"])
            if theme_count < 2 and b["bridge_note_count"] < 2:
                continue
            bridge_pair_count = len(b["bridge_pairs"])
            score = theme_count * 2 + b["bridge_note_count"] * 3 + bridge_pair_count
            result.append({
                "book_id": b["book_id"],
                "title": b["title"],
                "author": b["author"],
                "theme_count": theme_count,
                "themes": sorted(b["themes"]),
                "bridge_note_count": b["bridge_note_count"],
                "bridge_pairs": [list(p) for p in sorted(b["bridge_pairs"])],
                "note_count": b["note_count"],
                "cross_score": score,
                "sample_bridge_notes": b["sample_bridge_notes"],
            })

        result.sort(key=lambda x: -x["cross_score"])
        return result

    # ---- Step 3: 用户画像 ----

    def build_user_profile(self, micro_themes: list[dict], bridges: list[dict]) -> dict:
        """综合微主题和桥接分析，构建多维用户画像"""
        print("\n=== 构建用户画像 ===")

        # 知识域分布
        domain_weights = {}
        # 已有主题
        for theme in self.themes:
            domain_weights[theme.label] = len(theme.note_ids)
        # 噪声微主题
        for mt in micro_themes:
            domain_weights[f"[交叉] {mt['label']}"] = mt["size"]

        total = sum(domain_weights.values())
        knowledge_domains = [
            {"domain": d, "weight": round(c / total, 3)}
            for d, c in sorted(domain_weights.items(), key=lambda x: -x[1])
        ]

        # 交叉兴趣
        cross_interests = [
            {"pair": b["themes"], "strength": round(b["count"] / len(self.noise_indices), 3)}
            for b in bridges
        ]

        # 深度指标：每个主题被桥接的次数
        depth_counter = Counter()
        for b in bridges:
            for t in b["themes"]:
                depth_counter[t] += b["count"]
        depth_indicators = [
            {"domain": d, "bridge_count": c}
            for d, c in depth_counter.most_common()
        ]

        prev_profile = self._load_previous_output()

        # 认知风格 LLM 分析
        cognitive_style = self._llm_cognitive_style(micro_themes, bridges)
        cognitive_style = self._resolve_llm_field(
            cognitive_style, prev_profile, "cognitive_style",
            lambda: self._fallback_cognitive_style(depth_indicators, bridges),
        )

        bridge_notes = getattr(self, "_bridge_notes_detail", None)
        if bridge_notes is None:
            bridge_notes = self._load_cache("noise_bridge_notes.json")
        if not bridge_notes:
            print("桥接笔记缓存不存在，重新检测...")
            bridge_notes, _, _ = self._detect_bridge_notes()
            self._save_cache("noise_bridge_notes.json", bridge_notes)
            print(f"检测到 {len(bridge_notes)} 条桥接笔记")
        cross_domain_books = self.identify_cross_domain_books(bridge_notes)
        print(f"识别 {len(cross_domain_books)} 本跨领域书籍")

        print("提炼阅读母题...")
        master_theme = self._llm_master_theme(
            knowledge_domains, depth_indicators, cognitive_style,
            bridges, micro_themes, cross_domain_books,
        )
        master_theme = self._resolve_llm_field(
            master_theme, prev_profile, "master_theme",
            lambda: self._fallback_master_theme(
                knowledge_domains, depth_indicators, bridges,
            ),
        )
        if master_theme.get("title"):
            src = master_theme.get("source", "llm")
            print(f"  母题: {master_theme['title']} ({src})")

        profile = {
            "noise_stats": {
                "total": int(len(self.noise_indices)),
                "sub_clusters": len(micro_themes),
            },
            "micro_themes": micro_themes,
            "bridge_patterns": bridges,
            "cross_domain_books": cross_domain_books,
            "user_profile": {
                "knowledge_domains": knowledge_domains,
                "cross_interests": cross_interests,
                "cognitive_style": cognitive_style,
                "depth_indicators": depth_indicators,
                "master_theme": master_theme,
            },
            "generated_at": datetime.now().isoformat(),
        }

        # 保存最终结果
        output_dir = Path("log/insights_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "noise_cross_cognitive.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_path}")

        # 同时保存可读文本
        self._print_profile(profile)

        return profile

    # ---- 主流程 ----

    def run(self, mode: str = "all"):
        """运行分析，支持 subcluster / bridge / profile / all"""
        if not self.load_data():
            return

        micro_themes = None
        bridges = None

        if mode in ("subcluster", "all"):
            micro_themes = self.subcluster_noise()

        if mode in ("bridge", "all"):
            bridges = self.analyze_bridges()

        if mode in ("profile", "all"):
            # profile 依赖前两步，尝试从缓存加载
            if micro_themes is None:
                micro_themes = self._load_cache("noise_micro_themes.json")
                if micro_themes is None:
                    print("错误: 请先运行 subcluster 模式或 all 模式")
                    return
            if bridges is None:
                bridges = self._load_cache("noise_bridges.json")
                if bridges is None:
                    print("错误: 请先运行 bridge 模式或 all 模式")
                    return
            self.build_user_profile(micro_themes, bridges)

    # ---- LLM 辅助 ----

    def _llm_bridge_insight(self, pair: tuple[str, str], sample_notes: list[dict]) -> str:
        """用 LLM 分析桥接对的交叉含义"""
        notes_text = "\n".join(f"- {n['content']}" for n in sample_notes)
        prompt = f"""以下笔记同时涉及「{pair[0]}」和「{pair[1]}」两个领域。请用一两句话分析这两个领域在这位读者的思维中是如何交叉的，交叉点反映了什么深层的认知倾向。

笔记内容：
{notes_text}

请直接输出分析，不要有其他格式。"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"分析失败: {e}"

    def _llm_cognitive_style(self, micro_themes: list[dict], bridges: list[dict]) -> dict:
        """用 LLM 分析认知风格"""
        # 收集代表性内容
        theme_labels = [mt["label"] for mt in micro_themes[:10]]
        bridge_pairs = [f"{b['themes'][0]} <-> {b['themes'][1]}" for b in bridges[:8]]
        sample_contents = []
        for mt in micro_themes[:5]:
            sample_contents.extend(mt["sample_notes"][:2])
        for b in bridges[:3]:
            sample_contents.extend(b["sample_notes"][:2])

        notes_text = "\n".join(f"- {c}" for c in sample_contents[:20])

        prompt = f"""基于以下分析结果，请总结这位读者的认知风格。

噪声中的微主题: {', '.join(theme_labels)}
跨领域桥接模式: {', '.join(bridge_pairs)}
代表性笔记:
{notes_text}

请按以下格式回复：
认知关键词: [3-5个关键词，用逗号分隔]
认知风格说明: [一段话，描述这位读者的思维特点、关注倾向和深层认知模式]"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()

            keywords = []
            description = ""
            for line in text.split("\n"):
                if line.startswith("认知关键词"):
                    kw_str = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                    keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
                elif line.startswith("认知风格说明"):
                    description = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()

            if not description:
                description = text

            return {"keywords": keywords, "description": description}
        except Exception as e:
            return {"keywords": [], "description": f"分析失败: {e}"}

    def _llm_master_theme(
        self,
        knowledge_domains: list[dict],
        depth_indicators: list[dict],
        cognitive_style: dict,
        bridges: list[dict],
        micro_themes: list[dict],
        cross_domain_books: list[dict],
    ) -> dict:
        """从画像分析中提炼一个贯穿所有线索的阅读母题"""
        top_domains = [f"{d['domain']}({d['weight']:.0%})" for d in knowledge_domains[:8]]
        top_depth = [f"{d['domain']}({d['bridge_count']}次桥接)" for d in depth_indicators[:6]]
        bridge_insights = [
            f"{b['themes'][0]}↔{b['themes'][1]}: {b.get('insight', '')[:80]}"
            for b in bridges[:6] if b.get("insight")
        ]
        cross_books = [
            f"《{b['title']}》({b['theme_count']}主题/{b['bridge_note_count']}桥接)"
            for b in cross_domain_books[:6]
        ]
        sample_notes = []
        for b in bridges[:4]:
            sample_notes.extend(b.get("sample_notes", [])[:2])
        for mt in micro_themes[:2]:
            sample_notes.extend(mt.get("sample_notes", [])[:2])
        notes_text = "\n".join(f"- {c[:120]}" for c in sample_notes[:12])

        cognitive_desc = cognitive_style.get("description", "")
        cognitive_kw = ", ".join(cognitive_style.get("keywords", []))

        prompt = f"""你是一位阅读心理分析者。以下是一位读者跨大量书籍的笔记分析结果——包含知识域分布、主题桥接、认知风格与跨领域书籍等线索。

请从中提炼出 **唯一一个** 贯穿所有线索的「阅读母题」：这是读者在不同书里反复思考、所有阅读经验最终汇入的核心生命议题。不要罗列多个主题，要找到那个最深的、统一的切口。

## 分析材料

知识域 Top: {', '.join(top_domains)}
深度桥接领域: {', '.join(top_depth)}
认知关键词: {cognitive_kw}
认知风格: {cognitive_desc[:300]}
主要桥接洞察:
{chr(10).join(f'- {i}' for i in bridge_insights) or '- 无'}
跨领域书籍: {', '.join(cross_books) or '无'}
代表性笔记:
{notes_text}

## 输出格式（严格遵守，不要其他内容）

母题: [4-12字的母题命名，如「在局限中建构真实自我」]
核心命题: [一句话：读者反复追问的根本问题]
展开说明: [2-3句话：为何这是所有线索的汇点，它如何串联不同领域的阅读]
汇聚线索: [线索1, 线索2, 线索3, 线索4]
书中回响: [回响1; 回响2; 回响3]"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_master_theme(text)
        except Exception as e:
            return {"title": "", "statement": "", "narrative": "", "converging_clues": [], "manifestations": [], "error": str(e)}

    def _parse_master_theme(self, text: str) -> dict:
        """解析 LLM 返回的母题结构化文本"""
        result = {
            "title": "",
            "statement": "",
            "narrative": "",
            "converging_clues": [],
            "manifestations": [],
        }
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("母题"):
                result["title"] = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
            elif line.startswith("核心命题"):
                result["statement"] = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
            elif line.startswith("展开说明"):
                result["narrative"] = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
            elif line.startswith("汇聚线索"):
                kw_str = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                kw_str = kw_str.strip("[]")
                result["converging_clues"] = [k.strip() for k in kw_str.split(",") if k.strip()]
            elif line.startswith("书中回响"):
                kw_str = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                kw_str = kw_str.strip("[]")
                result["manifestations"] = [k.strip() for k in kw_str.split(";") if k.strip()]

        if not result["title"] and text:
            result["narrative"] = text
        return result

    def _load_previous_output(self) -> Optional[dict]:
        """加载已有的分析结果，供 LLM 失败时回退"""
        path = Path("log/insights_output/noise_cross_cognitive.json")
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _is_llm_failed_cognitive(cs: dict) -> bool:
        desc = cs.get("description", "")
        return bool(desc.startswith("分析失败")) or bool(cs.get("error"))

    @staticmethod
    def _is_llm_failed_master(mt: dict) -> bool:
        return not mt.get("title") or bool(mt.get("error"))

    def _resolve_llm_field(self, new_val: dict, prev_profile: Optional[dict], key: str, fallback_fn) -> dict:
        """LLM 失败时优先沿用上次成功结果，否则用规则 fallback"""
        is_failed = (
            self._is_llm_failed_cognitive(new_val)
            if key == "cognitive_style"
            else self._is_llm_failed_master(new_val)
        )
        if not is_failed:
            new_val["source"] = "llm"
            return new_val

        prev = (prev_profile or {}).get("user_profile", {}).get(key)
        prev_ok = prev and not (
            self._is_llm_failed_cognitive(prev)
            if key == "cognitive_style"
            else self._is_llm_failed_master(prev)
        )
        if prev_ok:
            print(f"  {key}: LLM 失败，沿用上次结果")
            prev = dict(prev)
            prev["source"] = "cached"
            prev["llm_error"] = new_val.get("error") or new_val.get("description", "")
            return prev

        print(f"  {key}: LLM 失败，使用规则 fallback")
        fb = fallback_fn()
        fb["source"] = "fallback"
        fb["llm_error"] = new_val.get("error") or new_val.get("description", "")
        return fb

    def _fallback_cognitive_style(self, depth_indicators: list[dict], bridges: list[dict]) -> dict:
        """LLM 不可用时的认知风格规则归纳"""
        keywords = list(dict.fromkeys(
            d["domain"].split("/")[0] for d in depth_indicators[:5]
        ))[:5]
        parts = []
        if bridges and bridges[0].get("insight"):
            parts.append(bridges[0]["insight"])
        if depth_indicators:
            hub = depth_indicators[0]
            parts.append(
                f"桥接最密集的领域是「{hub['domain']}」（{hub['bridge_count']}次），"
                "表明这是跨书阅读中最核心的交汇地带。"
            )
        description = " ".join(parts) if parts else "基于桥接数据的规则归纳。"
        return {"keywords": keywords, "description": description}

    def _fallback_master_theme(
        self,
        knowledge_domains: list[dict],
        depth_indicators: list[dict],
        bridges: list[dict],
    ) -> dict:
        """LLM 不可用时的阅读母题规则归纳"""
        hub = depth_indicators[0] if depth_indicators else None
        top_bridge = bridges[0] if bridges else None

        if top_bridge:
            t0, t1 = top_bridge["themes"]
            title = f"{t0} × {t1}"
            statement = (
                f"在不同书籍的笔记中，「{t0}」与「{t1}」"
                f"反复交汇（{top_bridge['count']}条桥接笔记），"
                "构成跨域阅读的核心追问。"
            )
            narrative = top_bridge.get("insight") or (
                f"「{hub['domain']}」是桥接最密集的领域"
                f"（{hub['bridge_count']}次），"
                f"与「{t0}↔{t1}」共同构成阅读轨迹的主轴。"
                if hub else ""
            )
        elif hub:
            title = hub["domain"]
            statement = f"「{hub['domain']}」是跨书笔记中桥接最密集的领域，是理解阅读轨迹的入口。"
            narrative = ""
        else:
            dom = knowledge_domains[0]["domain"] if knowledge_domains else "跨域阅读"
            title = dom.replace("[交叉] ", "")
            statement = ""
            narrative = "基于知识域分布的规则归纳。"

        converging_clues = []
        if hub:
            converging_clues.append(f"桥接枢纽：{hub['domain']}（{hub['bridge_count']}次）")
        for b in bridges[:2]:
            converging_clues.append(f"{b['themes'][0]} ↔ {b['themes'][1]}（{b['count']}条）")
        for d in knowledge_domains[:2]:
            converging_clues.append(f"{d['domain']}（{d['weight']:.0%}）")

        manifestations = []
        for b in bridges[:3]:
            if b.get("insight"):
                manifestations.append(b["insight"][:80])
            else:
                manifestations.append(f"{b['themes'][0]} ↔ {b['themes'][1]}（{b['count']}条）")

        return {
            "title": title,
            "statement": statement,
            "narrative": narrative,
            "converging_clues": converging_clues[:4],
            "manifestations": manifestations,
        }

    # ---- 缓存辅助 ----

    def _save_cache(self, filename: str, data):
        path = self.loader.processed_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"缓存已保存到 {path}")

    def _load_cache(self, filename: str):
        path = self.loader.processed_dir / filename
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ---- 打印摘要 ----

    def _print_profile(self, profile: dict):
        """打印可读的用户画像摘要"""
        up = profile["user_profile"]

        print("\n" + "=" * 60)
        print("用户画像摘要")
        print("=" * 60)

        print(f"\n噪声笔记: {profile['noise_stats']['total']} 条 → {profile['noise_stats']['sub_clusters']} 个微主题")

        print("\n--- 知识域 Top 10 ---")
        for d in up["knowledge_domains"][:10]:
            print(f"  {d['domain']}: {d['weight']:.1%}")

        print("\n--- 交叉兴趣 Top 5 ---")
        for ci in up["cross_interests"][:5]:
            print(f"  {ci['pair'][0]} <-> {ci['pair'][1]}: {ci['strength']:.1%}")

        print("\n--- 深度领域 ---")
        for di in up["depth_indicators"][:5]:
            print(f"  {di['domain']}: {di['bridge_count']} 次桥接")

        if up["cognitive_style"]["keywords"]:
            print(f"\n--- 认知风格 ---")
            print(f"  关键词: {', '.join(up['cognitive_style']['keywords'])}")
            print(f"  {up['cognitive_style']['description']}")

        mt = up.get("master_theme", {})
        if mt.get("title"):
            print(f"\n--- 阅读母题 ---")
            print(f"  {mt['title']}")
            if mt.get("statement"):
                print(f"  {mt['statement']}")

        # 保存文本版本
        import io
        buf = io.StringIO()
        buf.write(f"噪声笔记深度分析\n")
        buf.write(f"生成时间: {profile['generated_at']}\n\n")
        buf.write(f"噪声笔记: {profile['noise_stats']['total']} 条 → {profile['noise_stats']['sub_clusters']} 个微主题\n\n")

        buf.write("=== 知识域分布 ===\n")
        for d in up["knowledge_domains"][:15]:
            buf.write(f"  {d['domain']}: {d['weight']:.1%}\n")

        buf.write("\n=== 交叉兴趣 ===\n")
        for ci in up["cross_interests"][:10]:
            buf.write(f"  {ci['pair'][0]} <-> {ci['pair'][1]}: {ci['strength']:.1%}\n")

        buf.write("\n=== 深度领域 ===\n")
        for di in up["depth_indicators"][:10]:
            buf.write(f"  {di['domain']}: {di['bridge_count']} 次桥接\n")

        buf.write("\n=== 认知风格 ===\n")
        if up["cognitive_style"]["keywords"]:
            buf.write(f"  关键词: {', '.join(up['cognitive_style']['keywords'])}\n")
        buf.write(f"  {up['cognitive_style']['description']}\n")

        mt = up.get("master_theme", {})
        if mt.get("title"):
            buf.write("\n=== 阅读母题 ===\n")
            buf.write(f"  {mt['title']}\n")
            if mt.get("statement"):
                buf.write(f"  {mt['statement']}\n")
            if mt.get("narrative"):
                buf.write(f"  {mt['narrative']}\n")
            if mt.get("converging_clues"):
                buf.write(f"  汇聚线索: {', '.join(mt['converging_clues'])}\n")
            if mt.get("manifestations"):
                buf.write(f"  书中回响: {'; '.join(mt['manifestations'])}\n")

        buf.write("\n=== 桥接洞察 ===\n")
        for b in profile["bridge_patterns"][:10]:
            buf.write(f"\n{b['themes'][0]} <-> {b['themes'][1]} ({b['count']}条)\n")
            buf.write(f"  {b['insight']}\n")

        buf.write("\n=== 噪声微主题 ===\n")
        for mt in profile["micro_themes"][:15]:
            buf.write(f"  {mt['label']}: {mt['size']}条\n")

        buf.write("\n=== 跨领域书籍 ===\n")
        for book in profile.get("cross_domain_books", [])[:15]:
            buf.write(
                f"  《{book['title']}》: {book['theme_count']} 个主题, "
                f"{book['bridge_note_count']} 条桥接笔记\n"
            )

        output_dir = Path("log/insights_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = output_dir / "noise_cross_cognitive.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"\n文本摘要已保存到 {txt_path}")
