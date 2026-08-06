@echo off
REM VPP Agent 快速启动脚本 (Windows)

echo ==========================================
echo VPP Agent 启动脚本
echo ==========================================
echo.

REM 检查 .env 文件
if not exist .env (
    echo [错误] 未找到 .env 文件
    echo 正在从 .env.example 创建 .env...
    copy .env.example .env
    echo [成功] .env 文件已创建，请根据实际环境修改配置
    echo.
)

REM 检查依赖
echo 检查依赖...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [错误] 依赖未安装，正在安装...
    pip install -e ".[dev]"
)

echo [成功] 依赖检查完成
echo.

REM 显示配置信息
echo 当前配置：
for /f "tokens=2 delims==" %%a in ('findstr VPP_ENVIRONMENT .env') do echo   - 环境: %%a
for /f "tokens=2 delims==" %%a in ('findstr VPP_LLM_ENABLED .env') do echo   - LLM 启用: %%a
for /f "tokens=2 delims==" %%a in ('findstr VPP_DB_HOST .env') do echo   - 数据库: %%a
echo.

REM 启动服务
echo ==========================================
echo 启动服务...
echo ==========================================
echo.
echo API 文档: http://localhost:8000/docs
echo 健康检查: http://localhost:8000/health
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 使用 uvicorn 启动（开发模式，自动重载）
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
