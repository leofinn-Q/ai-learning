from openai import OpenAI

client = OpenAI(
    api_key="your key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
       {"role": "user", "content": "用三句话解释什么是 RAG"}
    ]
)
while True:
    question = input("你问：")
    if question == "退出":
        break
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": question}]
    )
    print("AI：", response.choices[0].message.content)
    print("-" * 30)
    
print(response.choices[0].message.content)