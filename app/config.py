from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # 将操作系统中的环境变量读取一下

class Settings(BaseModel):  # BaseModel后续这个类会转换为JSON/字典方便使用
    # deepseek
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", 'https://api.deepseek.com/v1')
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "sk-f359d11bb18e4d00b05b735d394f770e")
    deepseek_embedding_model_name: str = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek_chat")


    # qianwen
    qianwen_base_url: str = os.getenv("QIANWEN_BASE_URL", 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    qianwen_api_key: str = os.getenv("QIANWEN_API_KEY", "sk-3eca7ef0c1b14bc19ef14cc1f1a1457b")
    qianwen_embedding_model_name: str = os.getenv("QIANWEN_EMBEDDING_MODEL_NAME", 'text-embedding-v2')


    # chatgpt
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "sk-f359d11bb18e4d00b05b735d394f770e")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")



    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

settings = Settings()
