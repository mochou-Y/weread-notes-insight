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

    # OpenAI兼容接口配置
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

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
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )


settings = Settings.from_env()
