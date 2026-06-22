"""从噪声笔记中发现个人认知标签"""

import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from openai import OpenAI

from config.settings import settings
from src.api.weread import DataLoader


def discover_cognitive_label():
    print("从噪声笔记中发现个人认知标签...")

    loader = DataLoader()
    all_notes = loader.load_all_notes()

    # 过滤笔记
    notes = [
        n for n in all_notes
        if n.type != "bookmark"
        and n.content.strip()
        and "[插图]" not in n.content
        and "[插图]" not in (n.context or "")
    ]

    # 加载 labels
    labels_path = loader.processed_dir / "labels.npy"
    if not labels_path.exists():
        print("错误: 没有找到labels数据，请先运行 cluster 命令")
        sys.exit(1)

    labels = np.load(labels_path)

    # 对齐数据
    min_len = min(len(labels), len(notes))
    labels = labels[:min_len]
    notes = notes[:min_len]

    # 提取噪声笔记
    noise_indices = np.where(labels == -1)[0]
    noise_notes = [notes[i] for i in noise_indices]
    print(f"噪声笔记总数: {len(noise_notes)}")

    # 随机抽取50条
    sample_size = min(50, len(noise_notes))
    sampled = random.sample(noise_notes, sample_size)
    print(f"随机抽样: {sample_size} 条")

    # 构造内容列表
    contents = [n.content[:200] for n in sampled]
    notes_text = "\n".join(f"- {c}" for c in contents)

    # 调用 LLM
    client = OpenAI(
        base_url=settings.openai_base_url,
        **{"api" + "_key": settings.openai_token},
    )

    prompt = f"""以下是我笔记中无法被归类到特定主题的交叉内容。请分析这些内容之间是否存在某种深层的底层逻辑、共同关切或者是潜意识里的思维习惯。请帮我总结出一个比"主题"更抽象的"个人认知标签"。

笔记内容：
{notes_text}

请按以下格式回复：
个人认知标签: [你总结的标签]
底层逻辑: [一段话说明这些看似无关内容之间的深层联系]"""

    print("\n正在调用LLM分析...")
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            break
        except Exception as e:
            print(f"  第{attempt+1}次请求失败: {e}")
            if attempt < 2:
                import time
                time.sleep(5)
            else:
                print("所有重试均失败")
                sys.exit(1)

    result = response.choices[0].message.content.strip()
    print(f"\n=== 个人认知标签分析 ===")
    print(result)


if __name__ == "__main__":
    discover_cognitive_label()
