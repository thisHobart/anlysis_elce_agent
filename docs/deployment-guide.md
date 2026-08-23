# VPP Agent 部署指南

## 快速部署

### 1. 准备环境配置

首先复制并配置环境变量：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，根据你的实际环境调整配置。

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 启动服务

#### 开发模式（带自动重载）

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 生产模式

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 验证服务

访问以下 URL 验证服务启动成功：

- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## 配置说明

### 核心配置

| 配置项 | 说明 | 示例 |
|-------|------|------|
| `VPP_APP_NAME` | 应用名称 | `VPP LangGraph Agent` |
| `VPP_ENVIRONMENT` | 环境标识 | `development` / `production` |
| `VPP_ALLOWED_ORIGINS` | CORS 允许的前端域名（逗号分隔） | `http://localhost:3000,https://vpp.example.com` |
| `VPP_ENABLE_SCREEN_ACTIONS` | 是否启用大屏联动 | `true` / `false` |
| `VPP_LOG_LEVEL` | 日志级别 | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### LLM 配置（P3）

| 配置项 | 说明 | 示例 |
|-------|------|------|
| `VPP_LLM_ENABLED` | 是否启用 LLM | `true` / `false` |
| `VPP_LLM_BASE_URL` | OpenAI 兼容 API 地址 | `http://llm.example.invalid:11434/v1` |
| `VPP_LLM_API_KEY` | API Key | `None` 或实际密钥 |
| `VPP_LLM_MODEL` | 模型名称 | `qwen3.6:27b` |
| `VPP_LLM_TIMEOUT_SECONDS` | 超时时间（秒） | `20` |
| `VPP_LLM_STRUCTURED_MODE` | 结构化输出模式 | `json_prompt` / `native` |

### 数据库配置（P5）

| 配置项 | 说明 | 示例 |
|-------|------|------|
| `VPP_DB_HOST` | MySQL 主机地址 | `db.example.invalid` |
| `VPP_DB_PORT` | MySQL 端口 | `3306` |
| `VPP_DB_USER` | 数据库用户（只读） | `readonly_user` |
| `VPP_DB_PASSWORD` | 数据库密码 | `` |
| `VPP_DB_NAME` | 数据库名称 | `example_db` |

### 知识库配置（P6）

| 配置项 | 说明 | 示例 |
|-------|------|------|
| `VPP_QDRANT_HOST` | Qdrant 主机地址 | `localhost` |
| `VPP_QDRANT_PORT` | Qdrant 端口 | `6333` |
| `VPP_KNOWLEDGE_COLLECTION` | 知识库集合名称 | `vpp_knowledge` |
| `VPP_KNOWLEDGE_TOP_K` | 检索结果数量 | `3` |
| `VPP_KNOWLEDGE_MODEL_PATH` | 嵌入模型路径 | `BAAI/bge-small-zh-v1.5` |

---

## API 接口说明

### 1. 查询接口 (POST /v1/query)

**请求体**:

```json
{
  "session_id": "user-123",
  "question": "当前总发电功率是多少？",
  "role": "viewer",
  "params": {},
  "request_action": true
}
```

**字段说明**:

- `session_id` (必填): 会话ID，用于多轮对话上下文
- `question` (必填): 用户问题
- `role` (可选): 用户角色 `viewer` / `operator` / `admin`，默认 `viewer`
- `params` (可选): 额外参数，如 `{"enterprise_name": "山东鲁能新能源"}`
- `request_action` (可选): 是否请求大屏联动，默认 `false`

**响应体**:

```json
{
  "session_id": "user-123",
  "question": "当前总发电功率是多少？",
  "answer": "根据 Mock API 数据，当前总发电功率为123.45兆瓦。",
  "route": "faq",
  "faq_id": "FAQ-06",
  "intent": "faq",
  "entities": {},
  "context": {},
  "screen_action": {
    "type": "highlight",
    "target": "power_gauge"
  },
  "trace": ["prepared", "classified:llm:faq:FAQ-06", ...],
  "classification_source": "llm",
  "compose_source": "llm"
}
```

### 2. 健康检查 (GET /health)

**响应体**:

```json
{
  "status": "healthy",
  "environment": "development",
  "llm_enabled": true,
  "db_available": true,
  "knowledge_available": true
}
```

---

## 前端集成示例

### JavaScript/TypeScript

```typescript
// 定义请求和响应类型
interface QueryRequest {
  session_id: string;
  question: string;
  role?: 'viewer' | 'operator' | 'admin';
  params?: Record<string, any>;
  request_action?: boolean;
}

interface QueryResponse {
  session_id: string;
  question: string;
  answer: string;
  route: string;
  faq_id?: string;
  intent: string;
  entities: Record<string, any>;
  context: Record<string, any>;
  screen_action?: {
    type: string;
    target: string;
  };
  trace: string[];
  classification_source: string;
  compose_source: string;
}

// 调用 Agent API
async function queryAgent(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch('http://localhost:8000/v1/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

// 使用示例
async function example() {
  try {
    const result = await queryAgent({
      session_id: 'user-123',
      question: '当前总发电功率是多少？',
      role: 'viewer',
      request_action: true,
    });

    console.log('回答:', result.answer);
    
    // 处理大屏联动
    if (result.screen_action) {
      console.log('大屏动作:', result.screen_action);
      // 触发大屏联动逻辑
      handleScreenAction(result.screen_action);
    }
  } catch (error) {
    console.error('查询失败:', error);
  }
}

// 多轮对话示例
async function multiTurnConversation() {
  const sessionId = `session-${Date.now()}`;

  // 第一轮
  const result1 = await queryAgent({
    session_id: sessionId,
    question: '山东鲁能新能源的实时负荷是多少？',
  });
  console.log('Q1:', result1.answer);

  // 第二轮（带上下文）
  const result2 = await queryAgent({
    session_id: sessionId,
    question: '它的可调能力怎么样？',
  });
  console.log('Q2:', result2.answer);
}
```

### Python

```python
import requests
from typing import Optional, Dict, Any

class VPPAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def query(
        self,
        session_id: str,
        question: str,
        role: str = "viewer",
        params: Optional[Dict[str, Any]] = None,
        request_action: bool = False,
    ) -> Dict[str, Any]:
        """查询 Agent"""
        url = f"{self.base_url}/v1/query"
        payload = {
            "session_id": session_id,
            "question": question,
            "role": role,
            "params": params or {},
            "request_action": request_action,
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# 使用示例
if __name__ == "__main__":
    client = VPPAgentClient()
    
    # 健康检查
    health = client.health_check()
    print(f"服务状态: {health}")
    
    # 单轮查询
    result = client.query(
        session_id="user-123",
        question="当前总发电功率是多少？",
        request_action=True,
    )
    print(f"回答: {result['answer']}")
    
    # 多轮对话
    session_id = f"session-{int(time.time())}"
    
    result1 = client.query(session_id, "山东鲁能新能源的实时负荷是多少？")
    print(f"Q1: {result1['answer']}")
    
    result2 = client.query(session_id, "它的可调能力怎么样？")
    print(f"Q2: {result2['answer']}")
```

---

## Docker 部署（可选）

### 1. 创建 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 复制代码
COPY app ./app
COPY .env.example ./.env

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. 构建镜像

```bash
docker build -t vpp-agent:latest .
```

### 3. 运行容器

```bash
docker run -d \
  --name vpp-agent \
  -p 8000:8000 \
  -e VPP_LLM_ENABLED=true \
  -e VPP_DB_HOST=db.example.invalid \
  vpp-agent:latest
```

---

## 生产环境建议

### 1. 使用进程管理器

推荐使用 `gunicorn` 或 `supervisor`:

```bash
# 安装 gunicorn
pip install gunicorn

# 启动（4 个 worker）
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile /var/log/vpp-agent/access.log \
  --error-logfile /var/log/vpp-agent/error.log
```

### 2. 配置 Nginx 反向代理

```nginx
upstream vpp_agent {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name vpp-api.example.com;

    location / {
        proxy_pass http://vpp_agent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持（如需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时配置
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

### 3. 监控和日志

- 使用 Prometheus + Grafana 监控
- 配置日志轮转
- 设置告警规则

### 4. 安全加固

- 启用 HTTPS
- 配置防火墙规则
- 限制 CORS 域名
- 添加 API 认证（JWT）
- 限流保护

---

## 故障排查

### 常见问题

#### 1. 服务无法启动

```bash
# 检查端口是否被占用
netstat -ano | findstr :8000

# 检查依赖是否安装
pip list | grep fastapi

# 查看详细错误日志
uvicorn app.main:app --log-level debug
```

#### 2. LLM 调用失败

- 检查 `VPP_LLM_BASE_URL` 是否可达
- 验证 API Key 是否正确
- 增加超时时间 `VPP_LLM_TIMEOUT_SECONDS`

#### 3. 数据库连接失败

```bash
# 测试数据库连接
mysql -h db.example.invalid -P 3306 -u readonly_user -p example_db
```

#### 4. CORS 错误

- 确认前端域名已添加到 `VPP_ALLOWED_ORIGINS`
- 多个域名用逗号分隔

---

## 下一步

1. ✅ 复制 `.env.example` 为 `.env` 并配置
2. ✅ 安装依赖 `pip install -e ".[dev]"`
3. ✅ 启动服务 `uvicorn app.main:app --reload`
4. ✅ 访问 http://localhost:8000/docs 测试 API
5. ✅ 前端集成调用

---

**文档版本**: 1.0  
**更新日期**: 2026-08-05
