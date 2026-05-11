"""项目配置"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class Settings:
    """配置类"""

    # 微信读书API
    weread_base_url: str = "https://weread.qq.com"
    cookie: str = ""  # 从环境变量或配置文件读取

    # Embedding API配置（SiliconFlow）
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.com/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"

    # LLM配置（OpenAI兼容接口）
    openai_api_key: str = ""
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.5-flash"

    # 聚类参数
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    similarity_threshold: float = 0.85

    # 时间粒度
    time_granularity: str = "quarter"  # month, quarter, year

    # 数据路径
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    output_dir: str = "output"

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        return cls(
            cookie=os.getenv("WEREAD_COOKIE", ""),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.com/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            llm_model=os.getenv("LLM_MODEL", "qwen3.5-flash"),
        )


settings = Settings.from_env()
