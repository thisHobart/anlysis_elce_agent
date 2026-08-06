# VPP Agent 部署完成报告

## 部署日期
2026-08-05

## 部署状态
✅ **服务已成功启动并运行**

---

## 服务信息

### 服务地址
- **基础 URL**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### 服务状态
```json
{
    "status": "ok",
    "environment": "development"
}
```

---

## 前端调用示例

### 1. 基本查询

```javascript
// JavaScript/TypeScript
const response = await fetch('http://localhost:8000/v1/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 'user-123',
    question: '什么是虚拟电厂？',
    role: 'viewer'
  })
});

const data = await response.json();
console.log('回答:', data.answer);
console.log('FAQ ID:', data.faq_id);
```

### 2. 带大屏联动的查询

```javascript
const response = await fetch('http://localhost:8000/v1/query', {
  method: 'POST',
  headers: {
    'Content-Type': application/json',
  },
  body: JSON.stringify({
    session_id: 'user-123',
    question: '当前总发电功率是多少？',
    role: 'viewer',
    request_action: true  // 请求大屏联动
  })
});

const data = await response.json();

// 处理回答
console.log('回答:', data.answer);

// 处理大屏联动
if (data.screen_action) {
  console.log('大屏动作:', data.screen_action);
  // 执行大屏联动逻辑
  handleScreenAction(data.screen_action);
}
```

### 3. 多轮对话

```javascript
// 保持同一个 session_id 实现多轮对话
const sessionId = `session-${Date.now()}`;

// 第一轮
const response1 = await fetch('http://localhost:8000/v1/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: sessionId,
    question: '山东鲁能新能源的实时负荷是多少？',
    role: 'viewer'
  })
});
const data1 = await response1.json();
console.log('Q1:', data1.answer);

// 第二轮（带上下文）
const response2 = await fetch('http://localhost:8000/v1/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: sessionId,
    question: '它的可调能力怎么样？',  // "它" 会自动理解为前面提到的企业
    role: 'viewer'
  })
});
const data2 = await response2.json();
console.log('Q2:', data2.answer);
```

### 4. 健康检查

```javascript
// 前端启动时检查服务是否可用
const response = await fetch('http://localhost:8000/health');
const data = await response.json();

if (data.status === 'ok') {
  console.log('Agent 服务正常');
} else {
  console.error('Agent 服务异常');
}
```

---

## API 接口详情

### POST /v1/query

**请求体**:
```json
{
  "session_id": "user-123",        // 必填：会话ID，用于多轮对话
  "question": "当前总发电功率是多少？",  // 必填：用户问题
  "role": "viewer",                // 可选：用户角色 viewer/operator/admin
  "params": {},                    // 可选：额外参数
  "request_action": true           // 可选：是否请求大屏联动
}
```

**响应体**:
```json
{
  "session_id": "user-123",
  "answer": "根据 Mock API 数据，当前总发电功率为123.45兆瓦...",
  "intent": "faq",
  "route": "faq",
  "faq_id": "FAQ-06",
  "classification_source": "llm",   // llm 或 rules
  "compose_source": "llm",          // llm、template 或 fixed
  "data": {...},                    // 查询的数据
  "screen_action": {                // 大屏联动指令（如果 request_action=true）
    "type": "highlight",
    "target": "power_gauge"
  },
  "validation_errors": [],          // 验证错误（如果有）
  "trace": [...]                    // 执行轨迹（调试用）
}
```

---

## 支持的功能

### ✅ 已实现功能

1. **FAQ 问答** (29 个 FAQ)
   - 虚拟电厂介绍
   - 实时数据查询
   - 历史统计查询
   - 设备状态查询

2. **多步查询编排** (P7)
   - 企业名 → 企业ID → 企业数据
   - FAQ-09: 企业实时负荷
   - FAQ-15: 企业可调能力

3. **知识检索** (P6)
   - 5 个参考文档
   - 语义搜索 + RAG

4. **数据库查询** (P5)
   - 22+ 预置查询命令
   - 只读安全保证

5. **LLM 增强** (P3)
   - 意图识别
   - 实体提取
   - 自然语言生成
   - 支持降级到规则

6. **多轮对话**
   - 上下文记忆
   - 实体累积
   - 会话隔离

7. **安全控制** (P0)
   - RBAC 角色控制
   - SQL 注入防护
   - 数据源不可用时拒绝编造

8. **大屏联动**
   - 根据角色控制
   - 结构化动作指令

---

## 配置说明

当前配置位于 `.env` 文件：

### 关键配置项

```bash
# 环境标识
VPP_ENVIRONMENT=development

# LLM 配置
VPP_LLM_ENABLED=true
VPP_LLM_BASE_URL=http://llm.example.invalid:11434/v1
VPP_LLM_MODEL=qwen3.6:27b

# 数据库配置
VPP_DB_HOST=db.example.invalid
VPP_DB_PORT=3306
VPP_DB_USER=readonly_user
VPP_DB_PASSWORD=
VPP_DB_NAME=example_db

# 知识库配置
VPP_QDRANT_HOST=localhost
VPP_QDRANT_PORT=6333
VPP_KNOWLEDGE_COLLECTION=vpp_knowledge

# 前端 CORS
VPP_ALLOWED_ORIGINS=http://localhost:3000
```

**重要**: 如果前端域名不是 `http://localhost:3000`，需要修改 `VPP_ALLOWED_ORIGINS`

---

## 启动和停止

### 启动服务

**Windows**:
```bash
start.bat
```

**Linux/Mac**:
```bash
./start.sh
```

**手动启动**:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 停止服务

按 `Ctrl + C` 停止服务

---

## 测试文件

已创建测试脚本 `test_api_live.py`，可以快速验证 API：

```bash
python test_api_live.py
```

测试内容包括：
1. 健康检查
2. FAQ 查询
3. 带大屏动作的查询
4. 多轮对话
5. 知识检索

---

## 文档资源

- **部署指南**: [docs/deployment-guide.md](docs/deployment-guide.md)
- **API 文档**: http://localhost:8000/docs （服务启动后访问）
- **测试报告**: [docs/non-api-test-summary.md](docs/non-api-test-summary.md)
- **P7 完成报告**: [docs/p7-multi-step-completion.md](docs/p7-multi-step-completion.md)

---

## 常见问题

### 1. 前端无法访问（CORS 错误）

**解决**: 修改 `.env` 文件中的 `VPP_ALLOWED_ORIGINS`，添加你的前端域名：

```bash
VPP_ALLOWED_ORIGINS=http://localhost:3000,http://your-frontend.com
```

然后重启服务。

### 2. LLM 调用失败

**解决**: 
- 检查 `VPP_LLM_BASE_URL` 是否可达
- 检查 LLM 服务是否正常运行
- 增加超时时间 `VPP_LLM_TIMEOUT_SECONDS=30`

### 3. 数据库连接失败

**解决**:
- 检查数据库配置是否正确
- 测试网络连通性
- 服务会自动降级，仍可使用其他功能

### 4. 端口被占用

**解决**:
```bash
# Windows 查找占用端口的进程
netstat -ano | findstr :8000

# 修改端口
python -m uvicorn app.main:app --port 8001
```

---

## 监控和日志

### 查看日志

服务日志会输出到终端，包括：
- 请求日志
- 错误日志
- LLM 调用日志
- 数据库查询日志

### 健康检查

前端可以定期调用健康检查接口：

```javascript
setInterval(async () => {
  try {
    const response = await fetch('http://localhost:8000/health');
    const data = await response.json();
    if (data.status !== 'ok') {
      console.warn('Agent 服务异常');
    }
  } catch (error) {
    console.error('Agent 服务不可达');
  }
}, 60000); // 每分钟检查一次
```

---

## 下一步

### 立即可用 ✅
- ✅ 服务已启动运行
- ✅ API 接口可调用
- ✅ 前端可以集成

### 生产环境部署
1. 使用 Nginx 反向代理
2. 配置 HTTPS
3. 使用 Gunicorn 多进程
4. 配置监控和告警
5. 设置日志轮转

参考：[docs/deployment-guide.md](docs/deployment-guide.md) 的生产环境章节

---

**部署完成时间**: 2026-08-05  
**服务状态**: ✅ 运行中  
**API 地址**: http://localhost:8000
