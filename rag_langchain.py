import os
os.environ["USER_AGENT"] = "my-ai-learning/1.0"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()

# 2. 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(docs)
print("切片数量：", len(chunks))

# 3. 创建 Embedding 模型（用硅基流动）
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key="sk-ozejnqcpokgzaskeoioyvjjvraeqvnzwljovmlzngshylpwf",
    base_url="https://api.siliconflow.cn/v1"
)

# 4. 存入 Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain"
)

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. 创建模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-9c4a44f536384096a8d1afb4e8ee8913",
    base_url="https://api.deepseek.com"
)

# 7. 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，只根据提供的参考资料回答问题。如果参考资料里没有答案，就说“资料中没有相关信息”。"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

# 8. 把检索结果格式化成文本
def format_docs(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

# 9. 串联成链
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 10. 测试
questions = [
    "这份文档主要讲了什么",
    "文档里提到了哪些职业",
    "文档里有没有提到黄金价格"
]

for q in questions:
    print(f"\n问题：{q}")
    print(f"回答：{chain.invoke(q)}")
    print("=" * 50)