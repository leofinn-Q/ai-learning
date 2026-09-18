import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# 加载 + 切片
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)

# 向量检索器
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_hybrid"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 关键词检索器（BM25）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 混合检索器：两者结果合并
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5] #制向量检索和关键词检索各占多少权重
)

# 测试
query = "准入类和水平评价类职业资格有什么区别"

print("=== 纯向量检索 ===")
for i, doc in enumerate(vector_retriever.invoke(query), 1):
    print(f"{i}. {doc.page_content[:60]}...")

print("\n=== 纯关键词检索（BM25）===")
for i, doc in enumerate(bm25_retriever.invoke(query), 1):
    print(f"{i}. {doc.page_content[:60]}...")

print("\n=== 混合检索 ===")
for i, doc in enumerate(ensemble_retriever.invoke(query), 1):
    print(f"{i}. {doc.page_content[:60]}...")