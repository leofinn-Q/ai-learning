import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    persist_directory="./chroma_citation"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="你的DeepSeek_API_Key",
    base_url="https://api.deepseek.com"
)

# 带编号的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，只根据提供的参考资料回答问题。回答时必须在句末标注来源编号，格式如 [来源1]。如果资料里没有答案，就说“资料中没有相关信息”。"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

# 给每段资料加编号
def format_docs_with_citation(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[来源{i}] {doc.page_content}")
    return "\n\n".join(formatted)

def ask(question):
    # 检索
    docs = retriever.invoke(question)

    # 格式化，带编号
    context = format_docs_with_citation(docs)

    # 拼 messages
    messages = prompt.format_messages(context=context, question=question)

    # 调用模型
    response = llm.invoke(messages)

    # 打印结果
    print(f"\n问题：{question}")
    print(f"回答：{response.content}")
    print("\n参考资料来源：")
    for i, doc in enumerate(docs, 1):
        print(f"  [来源{i}] {doc.page_content[:80]}...")
    print("=" * 50)

# 测试
ask("准入类和水平评价类职业资格有什么区别")
ask("退出目录后证书还有效吗")