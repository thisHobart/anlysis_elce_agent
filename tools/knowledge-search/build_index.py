"""
VPP 知识库索引构建脚本

从 references/*.md 和 *.pdf 读取文档 -> 切分 -> 嵌入 -> 写入 Qdrant

使用方法:
    python tools/knowledge-search/build_index.py

依赖:
    pip install qdrant-client sentence-transformers pypdf
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import get_settings

# 配置
REFERENCES_DIR = Path(__file__).parent / "references"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def extract_pdf_text(pdf_path: Path) -> str:
    """从 PDF 提取纯文本"""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[ERROR] pypdf 未安装，无法处理 PDF 文件")
        print("  请运行: pip install pypdf")
        return ""

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append(f"[第{i+1}页]\n{text}")
    return "\n\n".join(pages)


def recursive_split_text(text: str, separators: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """递归字符分割：按分隔符列表逐级拆解，每段不超过 chunk_size"""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
    if not separators:
        return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    parts = text.split(sep)

    if len(parts) <= 2 and all(len(p.strip()) > chunk_size for p in parts if p.strip()):
        return recursive_split_text(text, separators[1:], chunk_size)

    result = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        result.extend(recursive_split_text(cleaned, separators[1:], chunk_size))
    return result


def merge_small_chunks(chunks: list[str], min_size: int = 200, max_size: int = CHUNK_SIZE) -> list[str]:
    """合并过小的相邻块"""
    if not chunks:
        return []
    merged = []
    buf = chunks[0]
    for c in chunks[1:]:
        if len(buf) < min_size and len(buf) + len(c) <= max_size:
            buf += "\n" + c
        else:
            merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)
    return merged


def add_overlap(chunks: list[str], overlap_chars: int = CHUNK_OVERLAP) -> list[str]:
    """相邻块之间添加重叠"""
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for c in chunks[1:]:
        prev = result[-1]
        overlap = prev[-overlap_chars:] if len(prev) >= overlap_chars else prev
        result.append(overlap + c)
    return result


def chunk_document(text: str, source_file: str) -> list[dict]:
    """
    递归字符分割：## 标题 → \n\n → \n → 。 → ，
    目标 200-400 字，相邻块 50 字重叠。

    Returns:
        [{"content": str, "source": str, "section": str}, ...]
    """
    separators = ["\n\n", "\n", "。", "，"]

    # 移除代码块
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 按 ## 标题分割，保持章节上下文
    sections = re.split(r"(^##\s+.*?$)", text, flags=re.MULTILINE)
    chunks = []
    current_section = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            current_section = part.replace("## ", "").strip()
        else:
            part = re.sub(r"^---+$", "", part, flags=re.MULTILINE).strip()
            if not part:
                continue

            raw = recursive_split_text(part, separators)
            raw = merge_small_chunks(raw, min_size=200)
            raw = add_overlap(raw, overlap_chars=50)

            for chunk_text in raw:
                chunks.append(
                    {
                        "content": chunk_text,
                        "source": source_file,
                        "section": current_section or "(概述)",
                    }
                )

    return chunks


def main():
    """构建知识库索引"""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # 加载配置
    settings = get_settings()

    print("[INFO] 配置:")
    print(f"  模型: {settings.knowledge_model_path}")
    print(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  集合: {settings.knowledge_collection}")
    print(f"  文档目录: {REFERENCES_DIR}")
    print()

    # 导入依赖
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        print("  请运行: pip install qdrant-client sentence-transformers pypdf")
        sys.exit(1)

    # 加载模型
    print(f"[INFO] 加载模型: {settings.knowledge_model_path}")
    try:
        model = SentenceTransformer(settings.knowledge_model_path)
        dim = model.get_sentence_embedding_dimension()
        print(f"[INFO] 模型维度: {dim}")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        sys.exit(1)

    # 连接 Qdrant
    print(f"[INFO] 连接 Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    except Exception as e:
        print(f"[ERROR] Qdrant 连接失败: {e}")
        print("  请确保 Qdrant 服务正在运行")
        print("  Docker 启动: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # 重建集合
    collection_name = settings.knowledge_collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[INFO] 已删除旧集合: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[INFO] 已创建集合: {collection_name} (size={dim}, distance=Cosine)")
    print()

    # 读取文档 -> 切分（支持 .md 和 .pdf）
    all_chunks = []
    md_files = sorted(REFERENCES_DIR.glob("*.md"))
    pdf_files = sorted(REFERENCES_DIR.glob("*.pdf"))
    all_files = list(md_files) + list(pdf_files)

    if not all_files:
        print(f"[WARN] {REFERENCES_DIR} 目录下没有可索引的文档！")
        sys.exit(1)

    for doc_file in all_files:
        print(f"[INFO] 读取: {doc_file.name}")
        if doc_file.suffix.lower() == ".pdf":
            text = extract_pdf_text(doc_file)
            if not text:
                print(f"  [SKIP] PDF 提取失败")
                continue
        else:
            text = doc_file.read_text(encoding="utf-8")

        chunks = chunk_document(text, doc_file.name)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} 个知识块")

    if not all_chunks:
        print("[WARN] 没有可索引的知识块")
        sys.exit(1)

    print(f"\n共计 {len(all_chunks)} 个知识块，开始嵌入...")

    # 生成 embeddings
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # 写入 Qdrant
    batch_size = 100
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "section": chunk["section"],
                },
            )
        )

        if len(points) >= batch_size or i == len(all_chunks) - 1:
            client.upsert(
                collection_name=collection_name,
                points=points,
            )
            print(f"  [写入] {i+1}/{len(all_chunks)}", end="\r")
            points = []

    print(f"\n\n[DONE] 完成！共 {len(all_chunks)} 个知识块写入 {collection_name}")
    print(f"\n测试检索:")
    print(f"  python -c \"from app.services.knowledge import search; print(search('什么是虚拟电厂'))\"")


if __name__ == "__main__":
    main()
