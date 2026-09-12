from openai import OpenAI

client = OpenAI(
     api_key="sk-9c4a44f536384096a8d1afb4e8ee8913",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
   messages=[
    #system 决定角色和风格
    {"role": "system", "content": "你是一个金融交易员"},
    #user 决定具体任务
    {"role": "user", "content": "参考资料：纳指 标普 黄金等。\n\n问题：这些基金哪些比较好,有什么优势？"}
]
)

print(response.choices[0].message.content)