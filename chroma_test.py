import chromadb
#chromadb 是一个向量数据库，可以存储文本、图片等数据，并支持向量检索。它可以与 OpenAI 的 Embedding 模型结合使用，实现语义搜索和相似度检索。
# 创建客户端，数据存在本地文件夹
client = chromadb.PersistentClient(path="./chroma_db")

# 创建一个集合（类似数据库里的表）
collection = client.get_or_create_collection(name="my_docs")

# 准备几条文本
documents = [
    "数字游民是指利用互联网远程工作、可以自由选择居住地的人。",
    "数字游民常见职业包括程序员、设计师、自由撰稿人、跨境电商、自媒体等。",
    "数字游民追求工作与生活的平衡，更看重生活质量、体验和自由度。",
    "今天天气很好，适合去公园散步。",
    "Python 是一种广泛使用的编程语言，适合数据分析和人工智能开发。"
]

# 给每条文本一个唯一 ID
ids = [f"doc_{i}" for i in range(len(documents))]

# 存入数据库
collection.add(
    documents=documents,
    ids=ids
)

print("已存入", collection.count(), "条文本")

results = collection.query(
    query_texts=["数字游民靠什么赚钱"],
    n_results=3
)

print("\n最相关的 3 条：")
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}")
    print(f"   距离：{results['distances'][0][i]:.4f}")