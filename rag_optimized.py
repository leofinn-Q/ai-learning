import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import requests

# ========== 配置 ==========
SILICONFLOW_KEY = "你的硅基流动API_Key"
DEEPSEEK_KEY = "你的DeepSeek_API_Key"

# ========== 加载 + 切片 ==========
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"切片数量：{len(chunks)}")

# ========== Embedding ==========
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=SILICONFLOW_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# ========== 向量检索器 ==========
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_optimized"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ========== BM25 检索器 ==========
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# ========== 混合检索器 ==========
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# ========== Rerank ==========
def rerank(query, docs, top_n=3):
    url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_KEY}",
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

# ========== LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com"
)

# ========== Prompt ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，只根据提供的参考资料回答问题。回答时必须在句末标注来源编号，格式如 [来源1]。如果资料里没有答案，就说“资料中没有相关信息”。"),
    ("placeholder", "{chat_history}"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

# ========== 对话历史 ==========
chat_history = []

# ========== 完整流程 ==========
def ask(question):
    # 1. 混合检索粗筛
    candidates = ensemble_retriever.invoke(question)
    print(f"粗筛出 {len(candidates)} 条")

    # 2. Rerank 精排
    if len(candidates) > 0:
        rerank_result = rerank(question, candidates, top_n=3)
        top_docs = [candidates[item["index"]] for item in rerank_result["results"]]
    else:
        top_docs = []

    # 3. 格式化，带来源编号
    context = "\n\n".join(
        f"[来源{i}] {doc.page_content}"
        for i, doc in enumerate(top_docs, 1)
    )

    # 4. 拼消息
    messages = prompt.format_messages(
        chat_history=chat_history,
        context=context,
        question=question
    )

    # 5. 调用模型
    response = llm.invoke(messages)

    # 6. 更新历史
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    return response.content, top_docs

# ========== 测试 ==========
questions = [
    "准入类和水平评价类职业资格有什么区别",
    "那退出目录后证书还有效吗",
    "文档里有没有提到黄金价格"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问题：{q}")
    answer, docs = ask(q)
    print(f"回答：{answer}")
    print("\n参考来源：")
    for i, doc in enumerate(docs, 1):
        print(f"  [来源{i}] {doc.page_content[:60]}...")