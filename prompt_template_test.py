from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com"
)

# 创建模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个摘要助手，用一句话总结。"),
    ("user", "总结这段文字：{text}")
])

# 填入变量
formatted = prompt.invoke({"text": "今天天气很好，我想去公园散步。"})
print("格式化后的消息：")
for msg in formatted.messages:
    print(f"  {msg.type}: {msg.content}")
print()

# 直接和 llm 串联
chain = prompt | llm
response = chain.invoke({"text": "今天天气很好，我想去公园散步。"})
print("总结结果：", response.content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用{style}回答。"),
    ("user", "{question}")
])

chain = prompt | llm
response = chain.invoke({
    "role": "数据分析师",
    "style": "表格形式",
    "question": "对比 Python 和 Java"
})
print(response.content)