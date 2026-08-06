"""向量化示例脚本

演示如何使用 bge-small-zh-v1.5 对文本进行向量化
"""

from sentence_transformers import SentenceTransformer

# 1. 加载预训练模型（首次运行会自动下载）
print("加载模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"模型维度: {model.get_sentence_embedding_dimension()}")

# 2. 示例文档（构建索引时）
documents = [
    "虚拟电厂是一种分布式能源聚合系统，通过智能控制协调多个分布式能源资源。",
    "可调节负荷包括工业负荷、商业负荷和居民负荷，可以根据电网需求进行调整。",
    "碳减排量的计算需要考虑发电量、电网排放因子等多个因素。",
]

print("\n向量化文档...")
doc_embeddings = model.encode(documents, normalize_embeddings=True)
print(f"文档向量形状: {doc_embeddings.shape}")  # (3, 512)

# 3. 查询向量化（检索时）
query = "什么是虚拟电厂？"
print(f"\n查询: {query}")

# bge 模型建议为查询加前缀
query_embedding = model.encode(f"为检索生成表示：{query}", normalize_embeddings=True)
print(f"查询向量形状: {query_embedding.shape}")  # (512,)

# 4. 计算相似度（Qdrant 内部做的事）
import numpy as np

similarities = np.dot(doc_embeddings, query_embedding)
print("\n相似度分数:")
for i, score in enumerate(similarities):
    print(f"  文档 {i+1}: {score:.4f}")
    print(f"    {documents[i][:50]}...")

# 5. 返回最相似的文档
best_idx = np.argmax(similarities)
print(f"\n最相似的文档: 文档 {best_idx+1}")
print(f"  分数: {similarities[best_idx]:.4f}")
print(f"  内容: {documents[best_idx]}")
