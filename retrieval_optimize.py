import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 加载文档
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

# Embedding 模型
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

# 测试不同参数组合
configs = [
    {"chunk_size": 50, "chunk_overlap": 0, "k": 1},
    #{"chunk_size": 50, "chunk_overlap": 30, "k": 2},
    #{"chunk_size": 50, "chunk_overlap": 50, "k": 2},
    #{"chunk_size": 50, "chunk_overlap": 50, "k": 4},
]

query = "文档里提到了哪些职业"

for cfg in configs:
    print(f"\n{'='*50}")
    print(f"配置：chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}, k={cfg['k']}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"切片数量：{len(chunks)}")

    # 每次用不同的 collection 名，避免冲突
    collection_name = f"test_{cfg['chunk_size']}_{cfg['chunk_overlap']}_{cfg['k']}"
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_optimize"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg["k"]})
    results = retriever.invoke(query)

    print(f"检索到 {len(results)} 条：")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content[:60]}...")