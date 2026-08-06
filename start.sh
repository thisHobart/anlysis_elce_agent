#!/bin/bash
# VPP Agent 快速启动脚本

echo "=========================================="
echo "VPP Agent 启动脚本"
echo "=========================================="
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "❌ 未找到 .env 文件"
    echo "正在从 .env.example 创建 .env..."
    cp .env.example .env
    echo "✅ .env 文件已创建，请根据实际环境修改配置"
    echo ""
fi

# 检查依赖
echo "检查依赖..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ 依赖未安装，正在安装..."
    pip install -e ".[dev]"
fi

echo "✅ 依赖检查完成"
echo ""

# 显示配置信息
echo "当前配置："
echo "  - 环境: $(grep VPP_ENVIRONMENT .env | cut -d= -f2)"
echo "  - LLM 启用: $(grep VPP_LLM_ENABLED .env | cut -d= -f2)"
echo "  - 数据库: $(grep VPP_DB_HOST .env | cut -d= -f2):$(grep VPP_DB_PORT .env | cut -d= -f2)"
echo ""

# 启动服务
echo "=========================================="
echo "启动服务..."
echo "=========================================="
echo ""
echo "API 文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/health"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 使用 uvicorn 启动（开发模式，自动重载）
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
