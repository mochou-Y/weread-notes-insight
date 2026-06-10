"""情绪洞察分析模块 — 过滤负面/高唤醒笔记并挖掘用户画像价值"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config.settings import settings
from src.api.weread import DataLoader
from src.data.models import Note, Theme


NEGATIVE_EMOTIONS = {
    "焦虑", "愤怒", "挫败", "恐惧", "悲伤", "羞耻", "厌恶", "无力感",
    "失望", "痛苦", "孤独", "不安", "压抑", "沮丧", "困惑", "紧张",
}


class EmotionAnalyzer:
    """情绪洞察分析器

    对有效笔记做 LLM 情绪分析，筛选负面或高唤醒情绪笔记，
    再从这些笔记中提炼用户画像价值。
    """

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.loader = DataLoader()
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.notes: list[Note] = []
        self.themes: list[Theme] = []
        self.note_to_theme: dict[str, str] = {}

    def load_data(self, sample_limit: Optional[int] = None) -> bool:
        """加载并过滤笔记，可用 sample_limit 限制调试样本量"""
        all_notes = self.loader.load_all_notes()
        self.notes = [
            n for n in all_notes
            if n.type != "bookmark"
            and n.content.strip()
            and "[插图]" not in n.content
            and "[插图]" not in (n.context or "")
        ]
        self.notes.sort(key=lambda n: n.create_time)

        if sample_limit is not None:
            self.notes = self.notes[:sample_limit]

        if not self.notes:
            print("错误: 没有找到可分析的笔记，请先运行 fetch 命令")
            return False

        self._load_theme_mapping()
        print(f"加载可分析笔记: {len(self.notes)} 条")
        return True

    def run(self, sample_limit: Optional[int] = None) -> Optional[dict]:
        """运行完整情绪洞察流程"""
        if not self.load_data(sample_limit=sample_limit):
            return None

        print("开始批量情绪分析...")
        analyzed = self._analyze_notes_in_batches()
        selected = [item for item in analyzed if self._is_selected(item)]
        print(f"筛选出负面或高唤醒笔记: {len(selected)} 条")

        profile_value = self._llm_profile_value(selected)
        result = self._build_result(analyzed, selected, profile_value)
        self._save_outputs(result)
        self._print_summary(result)
        return result

    # ---- LLM 分析 ----

    def _analyze_notes_in_batches(self) -> list[dict]:
        results = []
        total = len(self.notes)
        for start in range(0, total, self.batch_size):
            batch = self.notes[start:start + self.batch_size]
            print(f"  分析批次 {start + 1}-{start + len(batch)} / {total}")
            batch_results = self._llm_classify_batch(batch)
            results.extend(batch_results)
        return results

    def _llm_classify_batch(self, notes: list[Note]) -> list[dict]:
        notes_text = "\n".join(
            self._format_note_for_prompt(idx, note)
            for idx, note in enumerate(notes, 1)
        )
        prompt = f"""你是一位严谨的阅读笔记情绪分析师。请分析以下微信读书笔记，判断笔记中读者本人表现出的情绪。

注意：
1. 不要把书中人物或作者的情绪直接等同为读者情绪，除非笔记体现出读者的认同、共鸣、评价、投射或强烈关注。
2. review/想法 比纯 highlight/划线 更能代表读者本人；划线需要谨慎判断。
3. 重点识别负面情绪或高唤醒情绪，如焦虑、愤怒、挫败、恐惧、羞耻、无力感、不安、紧张、强烈抗拒、震惊、激动等。
4. valence 只能是 positive、neutral、negative。
5. arousal 只能是 low、medium、high。
6. intensity 为 1-5 的整数。
7. is_selected 当 valence=negative 或 arousal=high 时为 true。

笔记：
{notes_text}

请只返回 JSON 数组，不要 Markdown，不要解释。数组每项格式：
{{
  "note_id": "原 note_id",
  "emotion": "主要情绪，中文短词",
  "valence": "positive|neutral|negative",
  "arousal": "low|medium|high",
  "intensity": 1,
  "is_selected": true,
  "reason": "为什么这样判断，1句话",
  "profile_value": "这条笔记对理解用户画像有什么价值，1句话"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  批次 API 调用失败: {e}")
            return [self._fallback_note_result(note, str(e)) for note in notes]

        try:
            parsed = self._parse_json_array(text)
            return self._normalize_batch_results(notes, parsed)
        except Exception as e:
            print(f"  批次 JSON 解析失败: {e}，保存原始返回并改为逐条重试")
            self._save_raw_response(text)
            return self._retry_notes_individually(notes)

    def _retry_notes_individually(self, notes: list[Note]) -> list[dict]:
        results = []
        for note in notes:
            result = self._llm_classify_single(note)
            results.append(result)
        return results

    def _llm_classify_single(self, note: Note) -> dict:
        prompt = f"""请分析以下一条微信读书笔记中读者本人表现出的情绪。

判断规则：
1. 不要把书中人物或作者的情绪直接等同为读者情绪，除非笔记体现出读者认同、共鸣、评价、投射或强烈关注。
2. valence 只能是 positive、neutral、negative。
3. arousal 只能是 low、medium、high。
4. intensity 为 1-5 的整数。
5. is_selected 当 valence=negative 或 arousal=high 时为 true。

笔记：
{self._format_note_for_prompt(1, note)}

请只返回 JSON 对象，不要 Markdown，不要解释。格式：
{{
  "note_id": "{note.id}",
  "emotion": "主要情绪，中文短词",
  "valence": "positive|neutral|negative",
  "arousal": "low|medium|high",
  "intensity": 1,
  "is_selected": true,
  "reason": "为什么这样判断，1句话",
  "profile_value": "这条笔记对理解用户画像有什么价值，1句话"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            text = response.choices[0].message.content.strip()
            parsed = self._parse_json_object(text)
            return self._merge_note_result(note, parsed)
        except Exception as e:
            print(f"    单条分析失败 {note.id}: {e}")
            return self._fallback_note_result(note, str(e))

    def _llm_profile_value(self, selected: list[dict]) -> dict:
        if not selected:
            return {
                "core_concerns": [],
                "emotional_triggers": [],
                "sensitive_topics": [],
                "coping_or_defense_patterns": [],
                "values_and_needs": [],
                "latent_questions": [],
                "summary": "未筛选出负面或高唤醒情绪笔记。",
                "source": "empty",
            }

        sample = sorted(selected, key=lambda x: -x.get("intensity", 0))[:40]
        notes_text = "\n".join(
            f"- [{item.get('emotion')}/{item.get('arousal')}/强度{item.get('intensity')}] "
            f"《{item.get('book_title')}》: {item.get('content', '')[:180]}\n"
            f"  画像价值: {item.get('profile_value', '')}"
            for item in sample
        )
        emotion_counts = Counter(item.get("emotion", "未知") for item in selected)
        book_counts = Counter(item.get("book_title", "未知") for item in selected)
        theme_counts = Counter(item.get("theme", "未归类") for item in selected)

        prompt = f"""以下是从一位读者的微信读书笔记中筛选出的负面或高唤醒情绪笔记。请挖掘这些笔记对用户画像和用户理解的价值。

情绪分布: {dict(emotion_counts.most_common(12))}
高频书籍: {dict(book_counts.most_common(10))}
高频主题: {dict(theme_counts.most_common(10))}

代表性笔记：
{notes_text}

请只返回 JSON 对象，不要 Markdown，不要解释。格式：
{{
  "core_concerns": ["核心关切1", "核心关切2"],
  "emotional_triggers": ["情绪触发器1", "情绪触发器2"],
  "sensitive_topics": ["敏感议题1", "敏感议题2"],
  "coping_or_defense_patterns": ["可能的应对/防御模式1"],
  "values_and_needs": ["价值观或需求1"],
  "latent_questions": ["用户反复追问的隐性问题1"],
  "summary": "一段话总结这些情绪笔记如何帮助理解用户"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            text = response.choices[0].message.content.strip()
            parsed = self._parse_json_object(text)
            parsed["source"] = "llm"
            return parsed
        except Exception as e:
            print(f"画像价值分析失败: {e}")
            return self._fallback_profile_value(selected, str(e))

    # ---- 结果构建 ----

    def _build_result(self, analyzed: list[dict], selected: list[dict], profile_value: dict) -> dict:
        emotion_counts = Counter(item.get("emotion", "未知") for item in selected)
        trigger_themes = self._build_trigger_themes(selected)
        book_distribution = [
            {"book_title": title, "count": count}
            for title, count in Counter(item.get("book_title", "未知") for item in selected).most_common(20)
        ]

        result = {
            "stats": {
                "total_notes": len(self.notes),
                "analyzed_notes": len(analyzed),
                "selected_notes": len(selected),
                "negative_count": sum(1 for item in selected if item.get("valence") == "negative"),
                "high_arousal_count": sum(1 for item in selected if item.get("arousal") == "high"),
            },
            "emotion_distribution": [
                {"emotion": emotion, "count": count}
                for emotion, count in emotion_counts.most_common()
            ],
            "trigger_themes": trigger_themes,
            "book_distribution": book_distribution,
            "selected_notes": selected,
            "user_profile_value": profile_value,
            "generated_at": datetime.now().isoformat(),
        }
        return result

    def _build_trigger_themes(self, selected: list[dict]) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in selected:
            grouped[item.get("theme") or "未归类"].append(item)

        themes = []
        for theme, items in grouped.items():
            emotions = [e for e, _ in Counter(i.get("emotion", "未知") for i in items).most_common(5)]
            profile_values = [i.get("profile_value", "") for i in items if i.get("profile_value")]
            themes.append({
                "theme": theme,
                "count": len(items),
                "emotions": emotions,
                "profile_value": profile_values[0] if profile_values else "",
                "sample_notes": [i.get("content", "")[:160] for i in items[:3]],
            })
        themes.sort(key=lambda x: -x["count"])
        return themes

    def _save_outputs(self, result: dict):
        output_dir = Path("log/insights_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "emotion_insights.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON 报告已保存到 {json_path}")

        txt_path = output_dir / "emotion_insights.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self._render_text_report(result))
        print(f"文本报告已保存到 {txt_path}")

    def _save_raw_response(self, text: str):
        path = self.loader.processed_dir / "emotion_last_raw_response.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  原始 LLM 返回已保存到 {path}")

    def _render_text_report(self, result: dict) -> str:
        stats = result["stats"]
        profile = result.get("user_profile_value", {})
        lines = [
            "情绪洞察报告",
            f"生成时间: {result.get('generated_at', '')}",
            "",
            "=== 概览 ===",
            f"分析笔记: {stats['analyzed_notes']} / {stats['total_notes']}",
            f"筛选笔记: {stats['selected_notes']}",
            f"负面情绪: {stats['negative_count']}",
            f"高唤醒情绪: {stats['high_arousal_count']}",
            "",
            "=== 情绪分布 ===",
        ]
        for item in result.get("emotion_distribution", [])[:20]:
            lines.append(f"  {item['emotion']}: {item['count']}")

        lines.extend(["", "=== 高频触发主题 ==="])
        for item in result.get("trigger_themes", [])[:12]:
            lines.append(f"  {item['theme']} ({item['count']}条): {', '.join(item['emotions'])}")
            if item.get("profile_value"):
                lines.append(f"    画像价值: {item['profile_value']}")

        lines.extend(["", "=== 用户画像价值 ==="])
        for key, title in [
            ("core_concerns", "核心关切"),
            ("emotional_triggers", "情绪触发器"),
            ("sensitive_topics", "敏感议题"),
            ("coping_or_defense_patterns", "应对/防御模式"),
            ("values_and_needs", "价值观与需求"),
            ("latent_questions", "隐性问题"),
        ]:
            values = profile.get(key, [])
            if values:
                lines.append(f"{title}: {', '.join(values)}")
        if profile.get("summary"):
            lines.append(f"总结: {profile['summary']}")

        lines.extend(["", "=== 代表性笔记 ==="])
        selected = sorted(
            result.get("selected_notes", []),
            key=lambda x: (x.get("intensity", 0), x.get("arousal") == "high"),
            reverse=True,
        )
        for item in selected[:30]:
            lines.append(
                f"\n《{item.get('book_title', '')}》 "
                f"[{item.get('emotion', '')}/{item.get('valence', '')}/{item.get('arousal', '')}/强度{item.get('intensity', '')}]"
            )
            lines.append(f"  内容: {item.get('content', '')[:240]}")
            if item.get("reason"):
                lines.append(f"  判断: {item['reason']}")
            if item.get("profile_value"):
                lines.append(f"  画像价值: {item['profile_value']}")

        return "\n".join(lines) + "\n"

    def _print_summary(self, result: dict):
        stats = result["stats"]
        print("\n" + "=" * 60)
        print("情绪洞察摘要")
        print("=" * 60)
        print(f"分析笔记: {stats['analyzed_notes']} 条")
        print(f"筛选笔记: {stats['selected_notes']} 条")
        print(f"负面情绪: {stats['negative_count']} 条")
        print(f"高唤醒情绪: {stats['high_arousal_count']} 条")
        print("\n情绪 Top 10:")
        for item in result.get("emotion_distribution", [])[:10]:
            print(f"  {item['emotion']}: {item['count']}")
        summary = result.get("user_profile_value", {}).get("summary")
        if summary:
            print(f"\n画像总结: {summary}")

    # ---- 辅助方法 ----

    def _load_theme_mapping(self):
        themes_path = self.loader.processed_dir / "themes.json"
        if not themes_path.exists():
            return
        try:
            with open(themes_path, encoding="utf-8") as f:
                themes_data = json.load(f)
            self.themes = [Theme(**t) for t in themes_data.get("themes", [])]
            for theme in self.themes:
                for note_id in theme.note_ids:
                    self.note_to_theme[note_id] = theme.label
        except (OSError, json.JSONDecodeError, TypeError):
            self.themes = []
            self.note_to_theme = {}

    def _format_note_for_prompt(self, idx: int, note: Note) -> str:
        context = f"\n  context: {note.context[:180]}" if note.context else ""
        return (
            f"{idx}. note_id: {note.id}\n"
            f"  type: {note.type}\n"
            f"  book: 《{note.book_title}》/{note.book_author}\n"
            f"  content: {note.content[:260]}{context}"
        )

    def _normalize_batch_results(self, notes: list[Note], parsed: list[dict]) -> list[dict]:
        parsed_by_id = {
            str(item.get("note_id")): item
            for item in parsed
            if isinstance(item, dict) and item.get("note_id")
        }
        normalized = []
        for note in notes:
            item = parsed_by_id.get(note.id, {})
            normalized.append(self._merge_note_result(note, item))
        return normalized

    def _merge_note_result(self, note: Note, item: dict) -> dict:
        emotion = str(item.get("emotion") or "中性").strip()
        valence = str(item.get("valence") or "neutral").strip().lower()
        arousal = str(item.get("arousal") or "low").strip().lower()
        if valence not in {"positive", "neutral", "negative"}:
            valence = "neutral"
        if arousal not in {"low", "medium", "high"}:
            arousal = "low"
        try:
            intensity = int(item.get("intensity", 1))
        except (TypeError, ValueError):
            intensity = 1
        intensity = max(1, min(5, intensity))
        is_selected = bool(item.get("is_selected")) or valence == "negative" or arousal == "high"

        return {
            "note_id": note.id,
            "book_id": note.book_id,
            "book_title": note.book_title,
            "book_author": note.book_author,
            "type": note.type,
            "chapter": note.chapter,
            "create_time": note.create_time.isoformat(),
            "content": note.content,
            "context": note.context,
            "theme": self.note_to_theme.get(note.id, "未归类"),
            "emotion": emotion,
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity,
            "is_selected": is_selected,
            "reason": str(item.get("reason") or "").strip(),
            "profile_value": str(item.get("profile_value") or "").strip(),
        }

    def _fallback_note_result(self, note: Note, error: str) -> dict:
        return self._merge_note_result(note, {
            "emotion": "分析失败",
            "valence": "neutral",
            "arousal": "low",
            "intensity": 1,
            "is_selected": False,
            "reason": f"LLM 分析失败: {error}",
            "profile_value": "",
        })

    @staticmethod
    def _is_selected(item: dict) -> bool:
        emotion = item.get("emotion", "")
        return (
            bool(item.get("is_selected"))
            or item.get("valence") == "negative"
            or item.get("arousal") == "high"
            or emotion in NEGATIVE_EMOTIONS
        )

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        text = EmotionAnalyzer._strip_code_fence(text)
        text = EmotionAnalyzer._remove_control_chars(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(text[start:end + 1])
        if not isinstance(data, list):
            raise ValueError("LLM 返回不是 JSON 数组")
        return data

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        text = EmotionAnalyzer._strip_code_fence(text)
        text = EmotionAnalyzer._remove_control_chars(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("LLM 返回不是 JSON 对象")
        return data

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        return "".join(
            ch for ch in text
            if ch in "\n\r\t" or ord(ch) >= 32
        )

    def _fallback_profile_value(self, selected: list[dict], error: str) -> dict:
        emotion_counts = Counter(item.get("emotion", "未知") for item in selected)
        theme_counts = Counter(item.get("theme", "未归类") for item in selected)
        return {
            "core_concerns": [theme for theme, _ in theme_counts.most_common(5)],
            "emotional_triggers": [emotion for emotion, _ in emotion_counts.most_common(5)],
            "sensitive_topics": [],
            "coping_or_defense_patterns": [],
            "values_and_needs": [],
            "latent_questions": [],
            "summary": "LLM 画像归纳失败，当前仅保留基于情绪和主题频次的规则摘要。",
            "source": "fallback",
            "llm_error": error,
        }
