from sentence_transformers import SentenceTransformer

# 下载并缓存模型
model = SentenceTransformer('BAAI/bge-m3')
print("模型下载完成")