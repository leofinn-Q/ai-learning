from openai import OpenAI

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

#Token 是计费单位
#输入和输出都计费
#可以用 max_tokens 控制输出长度
#短问题和长问题的 token 差异
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "写一篇 500 字的文章，介绍数字游民"}
    ],
    #输出token数限制
    max_tokens=50
)

print("回答：", response.choices[0].message.content)
print("输出 token：", response.usage.completion_tokens)

#print("回答：", response.choices[0].message.content)
#print("输入 token：", response.usage.prompt_tokens)
#print("输出 token：", response.usage.completion_tokens)
#print("总 token：", response.usage.total_tokens)