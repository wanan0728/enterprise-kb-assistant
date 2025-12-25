# 下面是直接连接chatgpt的语法
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from app.config import settings
#
# def get_llm():
#     return ChatOpenAI(
#         model=settings.model_name,
#         api_key=settings.openai_api_key,
#         temperature=0.2,  # 大模型温度
#         streaming=True,   # 支持流式输出
#     )
#
# def get_embeddings():
#     return OpenAIEmbeddings(api_key=settings.openai_api_key)


# 下面是针对deepseek和千问嵌入式模型的代码
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from app.config import settings
from app.rag.vectorstore import get_vectorstore
from langchain_openai import OpenAIEmbeddings


def get_llm():
    return ChatOpenAI(

        # deepseek
        # base_url=settings.deepseek_base_url,
        # model="deepseek-chat",
        # api_key=settings.deepseek_api_key,

        # 千问
        base_url=settings.qianwen_base_url,
        model="qwen-max",
        api_key=settings.qianwen_api_key,

        temperature=0.2,
        streaming=True,
    )

def get_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=settings.qianwen_api_key,
    )


def get_vs():
    return get_vectorstore(get_embeddings())

# 测试
if __name__ == "__main__":
    resp = get_llm().invoke('你是谁')
    print(resp.content)

