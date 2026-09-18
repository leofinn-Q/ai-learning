from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# 1. 加载纯文本
print("=== 加载 txt ===")
loader = TextLoader("my_doc.txt", encoding="utf-8")
docs = loader.load()
print("文档数量：", len(docs))
print("内容前 100 字：", docs[0].page_content[:100])
print("metadata：", docs[0].metadata)
print()

# 2. 加载 PDF（如果你有 PDF 文件）
# print("=== 加载 PDF ===")
# pdf_loader = PyPDFLoader("example.pdf")
# pdf_docs = pdf_loader.load()
# print("页数：", len(pdf_docs))
# print("第一页前 100 字：", pdf_docs[0].page_content[:100])
# print()

# 3. 加载网页
print("=== 加载网页 ===")
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()
print("文档数量：", len(web_docs))
print("内容前 100 字：", web_docs[0].page_content[:100])
print("metadata：", web_docs[0].metadata)