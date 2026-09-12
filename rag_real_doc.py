import chromadb
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 两个客户端
embed_client = OpenAI(
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

chat_client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return response.data[0].embedding

# 读取文档
with open("my_doc.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("文档长度：", len(text), "字")

# 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,#每个切片最多 200 个字
    chunk_overlap=50,#相邻两个切片之间重叠 50 个字
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

chunks = splitter.split_text(text)
print("切片数量：", len(chunks))

# 存入 Chroma
client = chromadb.PersistentClient(path="./chroma_db_real")
collection = client.get_or_create_collection(name="real_doc")

# 清空旧数据，避免重复
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

embeddings = [get_embedding(chunk) for chunk in chunks]
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("已存入", collection.count(), "个切片\n")

# 检索 + 生成
def ask(query, n_results=3):
    query_vector = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    docs = results["documents"][0]
    context = "\n---\n".join(docs)

    messages = [
        {
            "role": "system",
            "content": "你是一个助手，只根据提供的参考资料回答问题。如果参考资料里没有答案，就说“资料中没有相关信息”。"
        },
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{query}"
        }
    ]

    response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    print(f"\n问题：{query}")
    print(f"回答：{response.choices[0].message.content}")
    print("=" * 50)

# 测试
ask("这份文档主要讲了什么")
ask("文档里提到了哪些职业")
ask("文档里有没有提到黄金价格")