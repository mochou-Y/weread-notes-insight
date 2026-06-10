"""时间维度分析 — 按时段追踪主题分布与阅读兴趣演变"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from config.settings import settings
from src.api.weread import DataLoader
from src.data.models import Book, Note, Theme


def _note_weight(note: Note) -> float:
    weight = 3.0 if note.type == "review" else 1.0
    if note.color_style > 0:
        weight *= 1.2
    return weight


def _period_key(dt: datetime, granularity: str) -> str:
    if granularity == "year":
        return str(dt.year)
    if granularity == "month":
        return f"{dt.year}-{dt.month:02d}"
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def _merge_sparse_periods(
    period_counts: dict[str, int],
    min_notes: int = 10,
) -> dict[str, str]:
    """将稀疏时段合并到相邻时段，返回 old_period -> merged_period 映射"""
    sorted_periods = sorted(period_counts.keys())
    if not sorted_periods:
        return {}

    merged_counts = dict(period_counts)
    mapping = {p: p for p in sorted_periods}

    changed = True
    while changed:
        changed = False
        for i, period in enumerate(sorted(merged_counts.keys())):
            if merged_counts[period] >= min_notes:
                continue
            keys = sorted(merged_counts.keys())
            idx = keys.index(period)
            target = keys[idx + 1] if idx + 1 < len(keys) else keys[idx - 1]
            if target == period:
                continue
            merged_counts[target] = merged_counts.get(target, 0) + merged_counts[period]
            del merged_counts[period]
            for orig, cur in list(mapping.items()):
                if cur == period:
                    mapping[orig] = target
            changed = True
            break

    return mapping


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


class TemporalAnalyzer:
    """时间维度分析器 — 基于全局主题追踪跨时段兴趣演变"""

    def __init__(self, min_period_notes: int = 10):
        self.min_period_notes = min_period_notes
        self.granularity = settings.time_granularity or "quarter"

        self.loader = DataLoader()
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        self.notes: list[Note] = []
        self.labels: Optional[np.ndarray] = None
        self.themes: list[Theme] = []
        self.books: list[Book] = []
        self.label_to_theme: dict[int, str] = {}

    def load_data(self) -> bool:
        """加载并对齐笔记、标签、主题"""
        print("加载数据...")

        all_notes = self.loader.load_all_notes()
        self.notes = [
            n for n in all_notes
            if n.type != "bookmark"
            and n.content.strip()
            and "[插图]" not in n.content
            and "[插图]" not in (n.context or "")
        ]

        from src.embedding.embedder import EmbeddingStorage

        storage = EmbeddingStorage()
        embeddings = storage.load()
        if embeddings is None:
            print("错误: 没有找到 embedding 数据，请先运行 embedding 命令")
            return False

        labels_path = self.loader.processed_dir / "labels.npy"
        if not labels_path.exists():
            print("错误: 没有找到 labels 数据，请先运行 cluster 命令")
            return False
        self.labels = np.load(labels_path)

        themes_path = self.loader.processed_dir / "themes.json"
        if not themes_path.exists():
            print("错误: 没有找到 themes 数据，请先运行 cluster 命令")
            return False
        with open(themes_path, encoding="utf-8") as f:
            themes_data = json.load(f)
        self.themes = [Theme(**t) for t in themes_data["themes"]]

        self.books = self.loader.load_notebook()

        min_len = min(len(self.labels), len(self.notes), len(embeddings))
        self.labels = self.labels[:min_len]
        self.notes = self.notes[:min_len]

        self.label_to_theme = {-1: "未归类"}
        for theme in self.themes:
            try:
                cluster_id = int(theme.id.split("_")[1])
            except (IndexError, ValueError):
                continue
            self.label_to_theme[cluster_id] = theme.label

        print(f"笔记总数: {min_len}, 主题数: {len(self.themes)}")
        return True

    def _theme_for_index(self, idx: int) -> str:
        label = int(self.labels[idx])
        return self.label_to_theme.get(label, "未归类")

    def _build_period_mapping(self) -> tuple[list[str], dict[int, str]]:
        """构建时段列表及笔记索引到合并后时段的映射"""
        raw_counts: Counter[str] = Counter()
        index_period: dict[int, str] = {}

        for i, note in enumerate(self.notes):
            period = _period_key(note.create_time, self.granularity)
            index_period[i] = period
            raw_counts[period] += 1

        merge_map = _merge_sparse_periods(dict(raw_counts), self.min_period_notes)
        if merge_map:
            index_period = {i: merge_map.get(p, p) for i, p in index_period.items()}

        periods = sorted(set(index_period.values()))
        return periods, index_period

    def _compute_theme_timeline(
        self,
        periods: list[str],
        index_period: dict[int, str],
    ) -> tuple[list[dict], dict[str, dict[str, float]]]:
        """计算各主题在各时段的加权占比"""
        period_theme_weight: dict[str, Counter[str]] = {p: Counter() for p in periods}
        period_total: Counter[str] = Counter()

        for i, note in enumerate(self.notes):
            period = index_period[i]
            theme = self._theme_for_index(i)
            w = _note_weight(note)
            period_theme_weight[period][theme] += w
            period_total[period] += w

        all_themes = sorted({t.label for t in self.themes} | {"未归类"})
        theme_timeline = []
        period_weights: dict[str, dict[str, float]] = {}

        for theme in all_themes:
            weights = []
            for period in periods:
                total = period_total[period] or 1.0
                w = period_theme_weight[period].get(theme, 0.0) / total
                weights.append(round(w, 4))
            if sum(weights) == 0:
                continue
            theme_timeline.append({"theme": theme, "weights": weights})
            for period, w in zip(periods, weights):
                period_weights.setdefault(period, {})[theme] = w

        theme_timeline.sort(key=lambda x: -sum(x["weights"]))
        return theme_timeline, period_weights

    def _classify_stability(
        self,
        theme_timeline: list[dict],
        periods: list[str],
    ) -> dict[str, list[dict]]:
        """分类稳定核、新兴、淡出、阶段性主题"""
        n = len(periods)
        if n == 0:
            return {"core": [], "emerging": [], "fading": [], "spike": []}

        mid = max(1, n // 2)
        recent_window = max(2, min(4, n // 3 or 1))
        core, emerging, fading, spike = [], [], [], []

        for item in theme_timeline:
            theme = item["theme"]
            weights = np.array(item["weights"], dtype=float)
            if weights.sum() == 0:
                continue

            global_mean = float(weights.mean())
            cv = float(weights.std() / global_mean) if global_mean > 0 else 999.0
            first_half = float(weights[:mid].mean())
            second_half = float(weights[mid:].mean())
            early = float(weights[:recent_window].mean())
            recent = float(weights[-recent_window:].mean())
            above_threshold = sum(1 for w in weights if w > global_mean * 0.5)
            peak = float(weights.max())
            peak_periods = sum(1 for w in weights if w > global_mean * 1.5)

            entry = {"theme": theme, "global_mean": round(global_mean, 4)}

            if peak > global_mean * 3 and peak_periods <= 2 and global_mean > 0.01:
                entry["peak_period"] = periods[int(weights.argmax())]
                spike.append(entry)

            if above_threshold >= max(1, int(n * 0.7)) and cv < 0.8 and global_mean > 0.02:
                core.append(entry)
            elif (
                second_half > first_half * 1.5 and second_half > 0.015
            ) or (
                recent > early * 2.0 and recent > 0.015
            ):
                entry["first_half"] = round(first_half, 4)
                entry["second_half"] = round(second_half, 4)
                entry["recent"] = round(recent, 4)
                emerging.append(entry)
            elif (
                first_half > second_half * 1.5 and first_half > 0.015
            ) or (
                early > recent * 2.0 and early > 0.015
            ):
                entry["first_half"] = round(first_half, 4)
                entry["second_half"] = round(second_half, 4)
                entry["recent"] = round(recent, 4)
                fading.append(entry)

        for bucket in (core, emerging, fading, spike):
            bucket.sort(key=lambda x: -x["global_mean"])

        return {"core": core, "emerging": emerging, "fading": fading, "spike": spike}

    def _compute_drift(self, theme_timeline: list[dict], periods: list[str]) -> float:
        """相邻时段主题分布的平均 cosine 距离"""
        if len(periods) < 2 or not theme_timeline:
            return 0.0

        n_periods = len(periods)
        matrix = np.array([t["weights"] for t in theme_timeline], dtype=float)
        if matrix.size == 0:
            return 0.0

        dists = []
        for i in range(n_periods - 1):
            dists.append(_cosine_distance(matrix[:, i], matrix[:, i + 1]))
        return round(float(np.mean(dists)), 4)

    def _compute_intensity(
        self,
        periods: list[str],
        index_period: dict[int, str],
    ) -> list[dict]:
        """每时段笔记数与加权笔记数"""
        result = []
        for period in periods:
            indices = [i for i, p in index_period.items() if p == period]
            note_count = len(indices)
            weighted = sum(_note_weight(self.notes[i]) for i in indices)
            review_count = sum(1 for i in indices if self.notes[i].type == "review")
            highlight_count = note_count - review_count
            result.append({
                "period": period,
                "note_count": note_count,
                "weighted_count": round(weighted, 1),
                "review_count": review_count,
                "highlight_count": highlight_count,
            })
        return result

    def _compute_category_timeline(
        self,
        periods: list[str],
        index_period: dict[int, str],
    ) -> list[dict]:
        """每时段书籍类别 Top 分布"""
        book_map = {b.book_id: b for b in self.books}
        period_cats: dict[str, Counter[str]] = {p: Counter() for p in periods}

        for i, note in enumerate(self.notes):
            period = index_period[i]
            book = book_map.get(note.book_id)
            if not book or not book.categories:
                period_cats[period]["未分类"] += _note_weight(note)
                continue
            for cat in book.categories:
                period_cats[period][cat] += _note_weight(note)

        result = []
        for period in periods:
            total = sum(period_cats[period].values()) or 1.0
            top = period_cats[period].most_common(5)
            result.append({
                "period": period,
                "top_categories": [
                    {"category": cat, "weight": round(w / total, 4)}
                    for cat, w in top
                ],
            })
        return result

    def _build_period_details(
        self,
        periods: list[str],
        index_period: dict[int, str],
        period_weights: dict[str, dict[str, float]],
    ) -> list[dict]:
        """各时段 Top 主题与代表性笔记"""
        details = []
        for period in periods:
            theme_weights = period_weights.get(period, {})
            top_themes = sorted(theme_weights.items(), key=lambda x: -x[1])[:5]

            indices = [i for i, p in index_period.items() if p == period]
            scored = sorted(
                indices,
                key=lambda i: (_note_weight(self.notes[i]), self.notes[i].type == "review"),
                reverse=True,
            )
            sample_notes = []
            for i in scored[:3]:
                note = self.notes[i]
                sample_notes.append({
                    "content": note.content[:200],
                    "book_title": note.book_title,
                    "type": note.type,
                    "theme": self._theme_for_index(i),
                    "create_time": note.create_time.strftime("%Y-%m-%d"),
                })

            details.append({
                "period": period,
                "top_themes": [{"theme": t, "weight": w} for t, w in top_themes],
                "sample_notes": sample_notes,
            })
        return details

    def _find_significant_shifts(
        self,
        periods: list[str],
        period_weights: dict[str, dict[str, float]],
        top_n: int = 3,
    ) -> list[dict]:
        """找出变化最显著的相邻时段对"""
        if len(periods) < 2:
            return []

        shifts = []
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i + 1]
            w1 = period_weights.get(p1, {})
            w2 = period_weights.get(p2, {})
            all_themes = set(w1) | set(w2)
            delta = sum(abs(w2.get(t, 0) - w1.get(t, 0)) for t in all_themes)
            shifts.append({"from": p1, "to": p2, "delta": round(delta, 4)})

        shifts.sort(key=lambda x: -x["delta"])
        return shifts[:top_n]

    def _llm_temporal_narrative(
        self,
        periods: list[str],
        theme_timeline: list[dict],
        stability: dict,
        period_details: list[dict],
        significant_shifts: list[dict],
    ) -> dict:
        """LLM 生成阅读兴趣演变叙事"""
        top_by_period = []
        for detail in period_details:
            themes_str = ", ".join(
                f"{t['theme']}({t['weight']:.0%})" for t in detail["top_themes"][:5]
            )
            top_by_period.append(f"{detail['period']}: {themes_str}")

        core = [s["theme"] for s in stability["core"][:6]]
        emerging = [s["theme"] for s in stability["emerging"][:6]]
        fading = [s["theme"] for s in stability["fading"][:6]]

        shift_lines = [
            f"{s['from']} → {s['to']} (变化幅度 {s['delta']:.2f})"
            for s in significant_shifts
        ]

        sample_lines = []
        for detail in period_details[-3:]:
            for note in detail.get("sample_notes", [])[:2]:
                sample_lines.append(f"[{detail['period']}] {note['content'][:100]}")

        prompt = f"""你是一位温和、善于倾听的阅读朋友。以下是一位读者跨时段的笔记主题分析（按{self.granularity}聚合）。

## 各时段主题 Top5
{chr(10).join(top_by_period)}

## 变与不变
稳定核（长期核心兴趣）: {', '.join(core) or '暂无'}
新兴（近期新关注）: {', '.join(emerging) or '暂无'}
淡出（曾经热衷现已减弱）: {', '.join(fading) or '暂无'}

## 变化最显著的时段
{chr(10).join(shift_lines) or '暂无'}

## 近期代表性笔记
{chr(10).join(f'- {l}' for l in sample_lines[:8])}

请用温和朋友的语气，输出三段分析（每段 2-4 句话）：
不变的底色: [长期稳定的核心关注点，读者思维中不变的部分]
明显的转向: [兴趣如何随时间迁移，有哪些新兴或淡出的主题]
可能的内在线索: [这些变化背后可能反映的生活阶段、问题意识或认知成长]

严格遵守上述三段标题格式，不要其他内容。"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_narrative(text)
        except Exception as e:
            return {
                "unchanged": "",
                "shifts": "",
                "clues": f"分析失败: {e}",
                "error": str(e),
            }

    @staticmethod
    def _parse_narrative(text: str) -> dict:
        result = {"unchanged": "", "shifts": "", "clues": ""}
        current = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("不变的底色"):
                current = "unchanged"
                val = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                if val:
                    result["unchanged"] = val
            elif line.startswith("明显的转向"):
                current = "shifts"
                val = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                if val:
                    result["shifts"] = val
            elif line.startswith("可能的内在线索"):
                current = "clues"
                val = line.split(":", 1)[-1].strip() if ":" in line else line.split("：", 1)[-1].strip()
                if val:
                    result["clues"] = val
            elif current and line:
                result[current] = (result[current] + " " + line).strip()
        return result

    def _print_report(self, profile: dict) -> None:
        """输出可读文本报告"""
        buf = StringIO()
        buf.write("时间维度分析\n")
        buf.write(f"生成时间: {profile['generated_at']}\n")
        buf.write(f"粒度: {profile['granularity']}\n")
        buf.write(f"时段: {', '.join(profile['periods'])}\n")
        buf.write(f"主题漂移度: {profile.get('drift_score', 0)}\n\n")

        buf.write("=== 阅读强度 ===\n")
        for item in profile["intensity"]:
            buf.write(
                f"  {item['period']}: {item['note_count']} 条"
                f" (想法 {item['review_count']}, 划线 {item['highlight_count']})\n"
            )

        stability = profile["stability"]
        buf.write("\n=== 稳定核 ===\n")
        for s in stability["core"][:10]:
            buf.write(f"  {s['theme']} (均占比 {s['global_mean']:.1%})\n")

        buf.write("\n=== 新兴 ===\n")
        for s in stability["emerging"][:10]:
            buf.write(f"  {s['theme']} ({s.get('first_half', 0):.1%} → {s.get('second_half', 0):.1%})\n")

        buf.write("\n=== 淡出 ===\n")
        for s in stability["fading"][:10]:
            buf.write(f"  {s['theme']} ({s.get('first_half', 0):.1%} → {s.get('second_half', 0):.1%})\n")

        narrative = profile.get("narrative", {})
        if narrative.get("unchanged") or narrative.get("shifts"):
            buf.write("\n=== 演变叙事 ===\n")
            if narrative.get("unchanged"):
                buf.write(f"不变的底色: {narrative['unchanged']}\n")
            if narrative.get("shifts"):
                buf.write(f"明显的转向: {narrative['shifts']}\n")
            if narrative.get("clues"):
                buf.write(f"可能的内在线索: {narrative['clues']}\n")

        text = buf.getvalue()
        print(text)

        output_dir = Path("log/insights_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = output_dir / "temporal_evolution.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"文本报告已保存到 {txt_path}")

    def run(self) -> Optional[dict]:
        """执行完整时间分析流程"""
        if not self.load_data():
            return None

        print(f"\n=== 时间分桶 ({self.granularity}) ===")
        periods, index_period = self._build_period_mapping()
        print(f"时段数: {len(periods)} ({periods[0]} ~ {periods[-1]})")

        theme_timeline, period_weights = self._compute_theme_timeline(periods, index_period)
        stability = self._classify_stability(theme_timeline, periods)
        drift_score = self._compute_drift(theme_timeline, periods)
        intensity = self._compute_intensity(periods, index_period)
        category_timeline = self._compute_category_timeline(periods, index_period)
        period_details = self._build_period_details(periods, index_period, period_weights)
        significant_shifts = self._find_significant_shifts(periods, period_weights)

        print(f"稳定核: {len(stability['core'])}, 新兴: {len(stability['emerging'])}, "
              f"淡出: {len(stability['fading'])}, 阶段性: {len(stability['spike'])}")
        print(f"主题漂移度: {drift_score}")

        print("\n=== LLM 演变叙事 ===")
        narrative = self._llm_temporal_narrative(
            periods, theme_timeline, stability, period_details, significant_shifts,
        )

        profile = {
            "granularity": self.granularity,
            "periods": periods,
            "theme_timeline": theme_timeline,
            "stability": stability,
            "drift_score": drift_score,
            "intensity": intensity,
            "category_timeline": category_timeline,
            "period_details": period_details,
            "significant_shifts": significant_shifts,
            "narrative": narrative,
            "generated_at": datetime.now().isoformat(),
        }

        output_dir = Path("log/insights_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "temporal_evolution.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_path}")

        self._print_report(profile)
        return profile
