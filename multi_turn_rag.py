import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# 加载文档 + 切片 + 存入 Chroma
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_multi_turn"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com"
)

# 多轮对话的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，只根据提供的参考资料回答问题。如果参考资料里没有答案，就说“资料中没有相关信息”。"),
    ("placeholder", "{chat_history}"),#把历史插入到消息中间
    ("user", "参考资料：\n{context}\n\n问题：{question}")   
])

def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

# 手动管理对话历史
chat_history = []

def ask(question):
    # 检索
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # 拼历史 + 当前问题
    messages = prompt.format_messages(
        chat_history=chat_history, #存历史对话，每轮追加
        context=context,
        question=question
    )

    # 调用模型
    response = llm.invoke(messages)

    # 更新历史
    chat_history.append(HumanMessage(content=question))  #HumanMessage标记用户
    chat_history.append(AIMessage(content=response.content))#AIMessage 标记ai信息

    return response.content

# 测试连续追问
print("第一轮：")
print(ask("这份文档主要讲了什么"))
print()

print("第二轮（追问）：")
print(ask("那它和水平评价类有什么区别？"))
print()

print("第三轮（再追问）：")
print(ask("退出目录后证书还有效吗？"))