from openai import OpenAI

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

def stream_ask(question):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": question}],
        #stream=True
    )
    print(response.choices[0].message.content)
   # for chunk in response:
   #     print("原始 chunk：", chunk)
    #    content = chunk.choices[0].delta.content
   #     if content:
   #         print(content, end="", flush=True)
    print()

while True:
    q = input("\n你问：")
    if q == "退出":
        break
    print("AI：", end="")
    stream_ask(q)
    print("-" * 30)