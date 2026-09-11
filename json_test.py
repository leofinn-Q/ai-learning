import json

from openai import OpenAI

client = OpenAI(
    api_key="your key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
    {"role": "system", "content": "从下面这段话里提取所有商品和价格，返回 JSON 数组：格式：{\"name\": \"\", \"price\": \"\"}"},
    {"role": "user", "content": "商品：苹果，价格：5元；商品：香蕉，价格：3元。"}]
)

content = response.choices[0].message.content.strip()

#过滤掉代码块标记
if content.startswith("```"):
    content = content.split("```")[1]
    if content.startswith("json"):
        content = content[4:]
    content = content.strip()

try:
    data = json.loads(content)
    for item in data:
        print("商品：", item["name"])
        print("价格：", item["price"])
except json.JSONDecodeError:
    print("解析失败，原始返回：", content)