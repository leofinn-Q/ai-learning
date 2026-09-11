from openai import OpenAI
import json

client = OpenAI(
     api_key="your key",
    base_url="https://api.deepseek.com"
)

def clean_json(content):
    """清理模型返回中可能带有的 markdown 代码块标记"""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return content

def analyze_text(text):
    """调用模型，返回摘要和关键词"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个文本分析助手。只返回 JSON，不要解释，不要 markdown。格式：{\"summary\": \"一句话摘要\", \"keywords\": [\"关键词1\", \"关键词2\"]}"
            },
            {
                "role": "user",
                "content": f"分析这段文字：{text}"
            }
        ]
    )
    return response.choices[0].message.content

def main():
    print("=" * 40)
    print("摘要 + 关键词提取工具")
    print("=" * 40)

    text = input("\n请粘贴一段文字（输入后按回车）：\n")

    if not text.strip():
        print("没有输入内容，退出。")
        return

    print("\n正在分析...\n")

    raw = analyze_text(text)
    cleaned = clean_json(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("解析失败，原始返回：")
        print(raw)
        return

    print("=" * 40)
    print("摘要：", data.get("summary", "无"))
    print("关键词：", "、".join(data.get("keywords", [])))
    print("=" * 40)

    # 保存结果到文件
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 result.json")

if __name__ == "__main__":
    main()