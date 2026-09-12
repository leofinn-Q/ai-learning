import chromadb
from openai import OpenAI

# 硅基流动：算 Embedding
embed_client = OpenAI(
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

# DeepSeek：生成回答
chat_client = OpenAI(
    api_key="3",
    base_url="https://api.deepseek.com"
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

# 存入 Chroma
client = chromadb.PersistentClient(path="./chroma_db_day5")
collection = client.get_or_create_collection(name="rag_docs")

# 如果集合是空的，才存入，避免重复
if collection.count() == 0:
    embeddings = [get_embedding(doc) for doc in documents]
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print("已存入", collection.count(), "条文本")
else:
    print("已有", collection.count(), "条文本，跳过存入")

# 检索函数
def retrieve(query, n_results=3):
    query_vector = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results["documents"][0]

# 生成回答
def ask(query):
    # 第一步：检索
    docs = retrieve(query)
    context = "\n".join(docs)

    # 第二步：拼 messages
    messages = [
        {
            "role": "system",
           # "content": "你是一个助手，只根据提供的参考资料回答问题。如果参考资料里没有答案，就说“资料中没有相关信息”。"
           "content": "你是一个助手，回答我问的问题。"
           
        },
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{query}"
        }
    ]

    # 第三步：调用 DeepSeek
    response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    print(f"\n问题：{query}")
    print(f"检索到的资料：\n{context}")
    print(f"\n回答：{response.choices[0].message.content}")
    print("=" * 50)

# 测试
#ask("数字游民靠什么赚钱")
ask("黄金价格怎么样")
#ask("什么是 RAG")
#ask("今天适合出门吗")
#ask("黄金价格怎么样")