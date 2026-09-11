from openai import OpenAI
import time

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

start = time.time()

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "写一段 300 字左右的文章，介绍数字游民的生活方式。"}
    ],
    #告诉 API 不要等全部生成完，而是边生成边返回
    stream=True
)
#流式输出，边生成边打印
for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        # end="" 表示不换行
        # flush=True 立刻刷新屏幕，不等缓冲区满 
        print(content, end="", flush=True)
print(f"\n耗时：{time.time() - start:.2f} 秒")

