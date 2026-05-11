"""测试jieba TF-IDF标签生成效果"""

import json
from pathlib import Path

import jieba.analyse

# 常见无意义词
STOP_WORDS = {
    "一个", "一种", "这个", "那个", "这些", "那些", "什么", "怎么",
    "如何", "为什么", "因为", "所以", "但是", "然而", "如果", "虽然",
    "可以", "可能", "应该", "必须", "需要", "能够", "已经", "正在",
    "我们", "他们", "她们", "它们", "自己", "他人", "大家", "人们",
    "这样", "那样", "怎样", "多少", "哪里", "那里", "这里", "真的",
}


def load_all_notes():
    """加载所有笔记"""
    notes = {}
    bookmarks_dir = Path("data/raw/bookmarks")

    for file_path in bookmarks_dir.glob("*.json"):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            for note in data.get("notes", []):
                notes[note["id"]] = note

    return notes


def label_by_jieba(notes_data: list[str]) -> str:
    """使用jieba TF-IDF生成标签"""
    text = " ".join(notes_data)
    keywords = jieba.analyse.extract_tags(text, topK=8, withWeight=False)

    # 过滤停用词和单字词
    filtered = [w for w in keywords if w not in STOP_WORDS and len(w) > 1]

    if not filtered:
        return "未命名主题"
    return "/".join(filtered[:2])


def main():
    # 加载主题数据
    with open("data/processed/themes.json", encoding="utf-8") as f:
        themes_data = json.load(f)

    # 加载所有笔记
    print("加载笔记数据...")
    all_notes = load_all_notes()
    print(f"共加载 {len(all_notes)} 条笔记")

    # 测试theme_46到theme_86
    print("\n" + "=" * 80)
    print(f"{'主题ID':<12} {'旧标签':<40} {'新标签(jieba)':<30} {'笔记数':<8}")
    print("=" * 80)

    for theme in themes_data["themes"]:
        theme_id = int(theme["id"].split("_")[1])
        if 46 <= theme_id <= 86:
            # 获取笔记内容
            note_contents = []
            for note_id in theme["note_ids"]:
                if note_id in all_notes:
                    content = all_notes[note_id].get("content", "")
                    if content:
                        note_contents.append(content)

            if note_contents:
                new_label = label_by_jieba(note_contents)
                old_label = theme["label"]
                print(f"{theme['id']:<12} {old_label:<40} {new_label:<30} {len(theme['note_ids']):<8}")


if __name__ == "__main__":
    main()
