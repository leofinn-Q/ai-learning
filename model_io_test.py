from langchain_openai import ChatOpenAI

# 创建模型对象
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com",
    temperature=0.7
)

# 方式一：直接传字符串
response = llm.invoke("用一句话介绍什么是 RAG")
print("方式一：", response.content)
print()

# 方式二：传消息列表
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个专业的数据分析师，只输出表格。"),
    HumanMessage(content="用表格对比 Python 和 Java。")
]

response = llm.invoke(messages)
print("方式二：", response.content)
print()

# 方式三：批量调用
responses = llm.batch([
    "什么是 RAG",
    "什么是 Embedding",
    "什么是 Chroma"
])

for i, r in enumerate(responses, 1):
    print(f"批量 {i}：", r.content)

for chunk in llm.stream("写一段 200 字的短文，介绍纳斯达克"):
    print(chunk.content, end="", flush=True)