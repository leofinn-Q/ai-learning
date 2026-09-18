from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com"
)

# 创建 JSON 解析器
parser = JsonOutputParser()

# 创建模板，注意 {format_instructions}
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个商品提取助手。{format_instructions}"),
    ("user", "提取这段文字里的商品和价格：{text}")
])

# 把解析器的格式说明填进模板
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# 串联：prompt → llm → parser
chain = prompt | llm | parser

# 调用
result = chain.invoke({"text": "我买了苹果 5 元，香蕉 3 元，橙子 4 元。"})
print("解析结果：", result)
print("类型：", type(result))
print("商品：", result["商品"])
print("价格：", result["价格"])