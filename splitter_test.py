from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
数字游民是指利用互联网远程工作、可以自由选择居住地的人。
他们通常依靠笔记本电脑和网络完成工作，在全球各地边旅行边生活。
常见职业包括程序员、设计师、自由撰稿人、跨境电商、自媒体等。
他们追求工作与生活的平衡，更看重生活质量、体验和自由度。
想要成为数字游民，需要先培养一项可远程变现的技能，比如编程、设计、写作、营销等。
然后建立稳定的收入来源，再考虑长期流动。
从短期试水开始，比如先旅居一个月，了解签证和税务规则，避免法律风险。
"""

# 策略一：按固定字数切
print("=== 策略一：chunk_size=50, overlap=0 ===")
splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks1 = splitter1.split_text(text)
for i, c in enumerate(chunks1, 1):
    print(f"切片{i}（{len(c)}字）：{c}")
print()

# 策略二：加重叠
print("=== 策略二：chunk_size=50, overlap=20 ===")
splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks2 = splitter2.split_text(text)
for i, c in enumerate(chunks2, 1):
    print(f"切片{i}（{len(c)}字）：{c}")
print()

# 策略三：按段落切
print("=== 策略三：chunk_size=100, overlap=0 ===")
splitter3 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks3 = splitter3.split_text(text)
for i, c in enumerate(chunks3, 1):
    print(f"切片{i}（{len(c)}字）：{c}")