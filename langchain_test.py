from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com",
    temperature=0.7
)

response = llm.invoke("用一句话介绍什么是数字游民")
print(response.content)
