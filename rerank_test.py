import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests

# 加载 + 切片 + 存入
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="你的硅基流动API_Key",
    base_url="https://api.siliconflow.cn/v1"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_rerank"
)

# 粗筛：先检索 10 条
query = "准入类和水平评价类职业资格有什么区别"
candidates = vectorstore.similarity_search(query, k=10)
print(f"粗筛出 {len(candidates)} 条\n")

# Rerank：调硅基流动的重排序 API
def rerank(query, docs, top_n=3):  #表示 Rerank 后只返回分数最高的 3 条
    url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": f"Bearer 你的硅基流动API_Key",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": [doc.page_content for doc in docs],
        "top_n": top_n
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

result = rerank(query, candidates, top_n=3)

print("=== Rerank 后的排序 ===")
for item in result["results"]:
    idx = item["index"]
    score = item["relevance_score"]
    print(f"分数：{score:.4f}")
    print(f"内容：{candidates[idx].page_content[:80]}...")
    print()