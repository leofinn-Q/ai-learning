import chromadb
from openai import OpenAI

# 硅基流动的 Embedding 客户端
embed_client = OpenAI(
    #api_key="你的硅基流动API_Key",
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return response.data[0].embedding

# 准备文档
documents = [
    "数字游民是指利用互联网远程工作、可以自由选择居住地的人。",
    "数字游民常见职业包括程序员、设计师、自由撰稿人、跨境电商、自媒体等。",
    "数字游民追求工作与生活的平衡，更看重生活质量、体验和自由度。",
    "今天天气很好，适合去公园散步。",
    "Python 是一种广泛使用的编程语言，适合数据分析和人工智能开发。",
    "RAG 是检索增强生成，先从文档里找相关内容，再让模型基于这些内容回答。",
    "向量数据库用来存储文本的向量表示，并支持快速相似度检索。",
    "黄金通常被视为避险资产，与股市相关性较低。",
]

# 手动算好每条文档的向量
embeddings = [get_embedding(doc) for doc in documents]

# 创建 Chroma 客户端
client = chromadb.PersistentClient(path="./chroma_db_day4")
collection = client.get_or_create_collection(name="docs_with_custom_embedding")

# 存入文档 + 向量
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("已存入", collection.count(), "条文本\n")

# 检索  n_results返回的结果数量
def search(query, n_results=1):
    query_vector = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    print(f"问题：{query}")
    print("最相关的", n_results, "条：")
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        print(f"  {i+1}. {doc}")
        print(f"     距离：{distance:.4f}")
    print("-" * 50)

# 测试三个问题
#search("数字游民靠什么赚钱")
search("程序员适合做什么远程工作")
#search("什么是 RAG")
#search("今天适合出门吗")