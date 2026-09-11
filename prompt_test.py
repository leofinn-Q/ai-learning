from openai import OpenAI

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
   messages=[
    #system 决定角色和风格
    {"role": "system", "content": "你是一个严格的语文老师，指出语法问题并改正。"},
    #user 决定具体任务
    {"role": "user", "content": "批改这句话：他昨天去了学校，学习了数学。"}
]
)

print(response.choices[0].message.content)