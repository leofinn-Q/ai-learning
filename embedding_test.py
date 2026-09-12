from openai import OpenAI
import numpy as np

client = OpenAI(
    api_key="",
    base_url="https://api.siliconflow.cn/v1"
)

def get_embedding(text):
    response = client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

text1 = "数字游民靠什么赚钱"
text2 = "数字游民的收入来源"
text3 = "今天天气怎么样"

v1 = get_embedding(text1)
v2 = get_embedding(text2)
v3 = get_embedding(text3)

print("text1 和 text2 相似度：", cosine_similarity(v1, v2))
print("text1 和 text3 相似度：", cosine_similarity(v1, v3))