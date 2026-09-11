from openai import OpenAI

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

while True:
    style = input("你要的风格（直接回车用默认）：")
    if style == "退出":
        break
    question = input("你的问题：")
    if question == "退出":
        break

    if style == "":
        style = "你是一个乐于助人的助手，用简洁的语言回答。"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": style},
            {"role": "user", "content": question}
        ]
    )
    print("AI：", response.choices[0].message.content)
    print("-" * 30)