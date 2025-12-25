import os
from langchain_chroma import Chroma

def get_vectorstore(embeddings):
    """获取向量存储，带错误处理和降级机制"""
    try:
        # 尝试使用持久化存储
        persist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        # 确保目录可写
        if not os.access(persist_dir, os.W_OK):
            os.chmod(persist_dir, 0o777)
        
        print(f"Using ChromaDB at: {persist_dir}")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_name="documents"
        )
    except Exception as e:
        print(f"Persistent ChromaDB failed: {e}")
        print("Falling back to in-memory ChromaDB")
        # 降级到内存模式
        return Chroma(
            embedding_function=embeddings,
            collection_name="documents"
        )
