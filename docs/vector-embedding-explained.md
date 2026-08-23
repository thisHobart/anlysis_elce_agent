# 向量化原理与流程详解

## 📚 基础概念

### 什么是向量化（Embedding）？

**向量化**是将文本转换为数字向量的过程，使得语义相似的文本在向量空间中距离更近。

**示例**：
```
"虚拟电厂" → [0.23, -0.15, 0.67, ..., 0.42]  (512 个数字)
"VPP"      → [0.25, -0.13, 0.65, ..., 0.40]  (语义相似，向量接近)
"苹果"     → [-0.89, 0.34, -0.12, ..., 0.78]  (语义不同，向量远)
```

### 为什么需要向量化？

传统关键词搜索的问题：
- ❌ "虚拟电厂" 搜不到 "VPP"
- ❌ "调度原理" 搜不到 "控制算法"
- ❌ 无法理解同义词和相关概念

向量化后的语义搜索：
- ✅ 理解同义词："虚拟电厂" ≈ "VPP"
- ✅ 理解相关概念："调度" ≈ "控制" ≈ "协调"
- ✅ 跨语言检索："virtual power plant" ≈ "虚拟电厂"（如果模型支持）

---

## 🔬 bge-small-zh-v1.5 模型

### 模型信息

| 属性 | 值 |
|------|-----|
| **开发者** | 北京智源人工智能研究院（BAAI） |
| **模型名称** | bge-small-zh-v1.5 |
| **HuggingFace** | https://huggingface.co/BAAI/bge-small-zh-v1.5 |
| **模型大小** | ~100MB |
| **向量维度** | 512 |
| **语言** | 中文优化 |
| **训练数据** | 数亿条中文句子对 |
| **任务类型** | 语义检索（Retrieval） |

### 为什么选择这个模型？

1. **中文优化**: 专门针对中文语义理解优化
2. **轻量高效**: 100MB 大小，推理速度快
3. **开源免费**: Apache 2.0 许可证
4. **效果优秀**: 在中文检索任务上表现优异
5. **社区活跃**: HuggingFace 上有详细文档和示例

### 模型性能对比

| 模型 | 维度 | 大小 | 速度 | 效果 |
|------|------|------|------|------|
| bge-small-zh-v1.5 | 512 | 100MB | ⚡⚡⚡ 快 | ⭐⭐⭐ 好 |
| bge-base-zh-v1.5 | 768 | 400MB | ⚡⚡ 中 | ⭐⭐⭐⭐ 很好 |
| bge-large-zh-v1.5 | 1024 | 1.3GB | ⚡ 慢 | ⭐⭐⭐⭐⭐ 最好 |

**我们选择 small 版本**：在效果和性能之间取得良好平衡。

---

## 🔄 完整的向量化流程

### 阶段 1：模型下载与缓存

**首次运行** `build_index.py` 或 `search()` 时：

```python
from sentence_transformers import SentenceTransformer

# 自动从 HuggingFace 下载模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
```

**下载位置**：
```
~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/
├── snapshots/
│   └── <commit-hash>/
│       ├── config.json
│       ├── pytorch_model.bin  (模型权重)
│       ├── tokenizer_config.json
│       └── vocab.txt
```

**Windows 路径**：
```
C:\Users\37779\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\
```

**后续运行**：直接从缓存加载，不再下载。

### 阶段 2：文档向量化（构建索引）

```python
# 1. 读取并切分文档
documents = [
    "虚拟电厂是一种分布式能源聚合系统...",
    "可调节负荷包括工业负荷、商业负荷...",
    # ... 126-226 个文档块
]

# 2. 批量向量化
embeddings = model.encode(
    documents,
    normalize_embeddings=True,  # L2 归一化
    show_progress_bar=True      # 显示进度
)

# 3. 结果
print(embeddings.shape)  # (126-226, 512)
# 每个文档块 → 512 维向量
```

**向量化过程（模型内部）**：

```
文本 "虚拟电厂是..."
    ↓
分词 ["虚拟", "电厂", "是", ...]
    ↓
Token IDs [1234, 5678, 91, ...]
    ↓
BERT 编码器 (12 层 Transformer)
    ↓
池化层 (CLS token 或 mean pooling)
    ↓
512 维向量 [0.23, -0.15, 0.67, ..., 0.42]
    ↓
L2 归一化
    ↓
最终向量 (长度为 1)
```

### 阶段 3：存储到 Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(host="localhost", port=6333)

# 批量写入
for i, (doc, vector) in enumerate(zip(documents, embeddings)):
    client.upsert(
        collection_name="vpp_knowledge",
        points=[PointStruct(
            id=i,
            vector=vector.tolist(),  # 转为列表
            payload={
                "content": doc,
                "source": "vpp_concept.md",
                "section": "概述",
            }
        )]
    )
```

**Qdrant 中的数据结构**：

```json
{
  "id": 0,
  "vector": [0.23, -0.15, 0.67, ..., 0.42],  // 512 个数字
  "payload": {
    "content": "虚拟电厂是一种...",
    "source": "vpp_concept.md",
    "section": "概述"
  }
}
```

### 阶段 4：查询向量化（检索时）

```python
# 用户问题
query = "什么是虚拟电厂？"

# 向量化查询（使用同一个模型）
query_vector = model.encode(
    f"为检索生成表示：{query}",  # bge 模型建议的前缀
    normalize_embeddings=True
)

print(query_vector.shape)  # (512,)
```

**为什么要加前缀 "为检索生成表示："？**

bge 模型在训练时使用了这个前缀，加上它可以提升检索效果：
- 查询：`"为检索生成表示：" + 问题`
- 文档：直接向量化（不加前缀）

这是 bge 模型的**特定优化**，其他模型可能不需要。

### 阶段 5：向量搜索（Qdrant）

```python
# Qdrant 执行 COSINE 相似度搜索
results = client.query_points(
    collection_name="vpp_knowledge",
    query=query_vector.tolist(),
    limit=3,
    with_payload=True
)

# 返回最相似的 3 个文档块
for point in results.points:
    print(f"相似度: {point.score:.4f}")
    print(f"内容: {point.payload['content']}")
```

**COSINE 相似度计算**：

```
相似度 = cos(θ) = (向量A · 向量B) / (|向量A| × |向量B|)

由于向量已归一化（长度为 1），简化为：
相似度 = 向量A · 向量B (点积)

范围: [-1, 1]
- 1.0  = 完全相同
- 0.9+ = 非常相似
- 0.5+ = 相关
- 0.0  = 无关
- -1.0 = 完全相反
```

---

## 🔍 实际示例

### 示例 1：概念检索

**查询**: "什么是虚拟电厂？"

**向量化结果**（简化）：
```
query_vector: [0.25, -0.13, 0.65, 0.89, ...]
```

**Qdrant 搜索结果**：
```
1. 相似度: 0.9523
   来源: vpp_concept.md / 概述
   内容: "虚拟电厂（Virtual Power Plant，VPP）是一种..."

2. 相似度: 0.8234
   来源: business_model.md / 定义
   内容: "虚拟电厂通过聚合分布式能源资源..."

3. 相似度: 0.7891
   来源: adjustable_load.md / 背景
   内容: "在虚拟电厂系统中，可调节负荷是重要组成..."
```

### 示例 2：跨文档检索

**查询**: "如何计算碳减排量？"

**向量化结果**（简化）：
```
query_vector: [-0.12, 0.78, -0.34, 0.56, ...]
```

**Qdrant 搜索结果**：
```
1. 相似度: 0.9105
   来源: carbon_formula.md / 计算公式
   内容: "碳减排量 = 发电量 × 电网排放因子 × 减排系数..."

2. 相似度: 0.7654
   来源: business_model.md / 环保效益
   内容: "通过减少火电发电量，虚拟电厂可实现碳减排..."

3. 相似度: 0.7123
   来源: 微电网...平台.pdf / 第15页
   内容: "系统自动统计各类能源的碳排放..."
```

---

## 🎯 关键要点总结

### 1. **不需要训练模型**
- ✅ 直接使用 HuggingFace 预训练模型
- ✅ 首次运行自动下载，后续使用缓存
- ✅ 模型已在大规模中文数据上训练完成

### 2. **向量化是自动的**
- ✅ `build_index.py` 自动向量化所有文档
- ✅ `search()` 自动向量化查询
- ✅ 使用同一个模型保证一致性

### 3. **流程简单**
```
文档 → 模型 → 向量 → Qdrant
查询 → 模型 → 向量 → 搜索 → 结果
```

### 4. **性能优化**
- ✅ 批量向量化（一次处理多个文档）
- ✅ 向量归一化（加速相似度计算）
- ✅ Qdrant 索引（毫秒级检索）

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install sentence-transformers qdrant-client

# 2. 启动 Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. 构建索引（自动下载模型、向量化文档、存储到 Qdrant）
python tools/knowledge-search/build_index.py

# 4. 测试检索
python -c "from app.services.knowledge import search; print(search('什么是虚拟电厂'))"

# 5. 运行演示脚本
python examples/vector_embedding_demo.py
```

---

## 📖 延伸阅读

- [bge-small-zh-v1.5 模型卡片](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Sentence Transformers 文档](https://www.sbert.net/)
- [Qdrant 向量搜索教程](https://qdrant.tech/documentation/tutorials/)
- [语义检索原理](https://arxiv.org/abs/1908.10084)

---

**文档版本**: 1.0  
**最后更新**: 2026-08-05
