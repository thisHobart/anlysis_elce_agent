# VPP Agent 服务器部署完整指南

## 目录
1. [方案选择](#方案选择)
2. [方案 1: 直接部署](#方案-1-直接部署到-linux-服务器)
3. [方案 2: Docker 部署](#方案-2-docker-部署推荐)
4. [访问测试](#访问测试)
5. [常见问题](#常见问题)

---

## 方案选择

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 直接部署 | 简单直接 | 环境依赖复杂 | 小型项目、熟悉 Linux |
| Docker 部署 | 环境一致、易迁移 | 需要学习 Docker | 推荐、生产环境 |

---

## 方案 1: 直接部署到 Linux 服务器

### 前置要求
- Linux 服务器（Ubuntu 20.04+ 或 CentOS 7+）
- 至少 2 核 CPU、4GB 内存
- 已安装 Python 3.11+
- 有 sudo 权限

### 步骤 1: 上传代码

**方式 A: 使用 Git（推荐）**
```bash
# SSH 连接到服务器
ssh user@your-server-ip

# 安装 Git
sudo apt update
sudo apt install git

# 克隆代码
cd /opt
sudo git clone https://github.com/your-repo/vpp-langgraph-agent.git
cd vpp-langgraph-agent
```

**方式 B: 使用 SCP 上传**
```bash
# 在本地 Windows 上
# 先打包代码
tar -czf vpp-agent.tar.gz vpp-langgraph-agent/

# 上传到服务器
scp vpp-agent.tar.gz user@your-server-ip:/opt/

# 在服务器上解压
ssh user@your-server-ip
cd /opt
tar -xzf vpp-agent.tar.gz
```

### 步骤 2: 安装 Python 和依赖

```bash
# 安装 Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# 进入项目目录
cd /opt/vpp-langgraph-agent

# 创建虚拟环境
python3.11 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -e ".[dev]"
```

### 步骤 3: 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置
nano .env
```

**重要配置修改**：
```bash
# 环境标识
VPP_ENVIRONMENT=production

# 前端域名（多个用逗号分隔）
VPP_ALLOWED_ORIGINS=http://your-frontend.com,https://your-frontend.com

# LLM 配置（确保服务器可访问）
VPP_LLM_ENABLED=true
VPP_LLM_BASE_URL=http://llm-server-ip:11434/v1

# 数据库配置
VPP_DB_HOST=db.example.invalid
VPP_DB_PORT=3306

# 日志级别
VPP_LOG_LEVEL=INFO
```

### 步骤 4: 测试运行

```bash
# 激活虚拟环境（如果还没激活）
source .venv/bin/activate

# 测试启动
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 另开一个终端测试
curl http://localhost:8000/health
# 应该返回: {"status":"ok","environment":"production"}

# Ctrl+C 停止测试
```

### 步骤 5: 配置 systemd 服务（开机自启）

创建服务文件：
```bash
sudo nano /etc/systemd/system/vpp-agent.service
```

内容：
```ini
[Unit]
Description=VPP LangGraph Agent
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/vpp-langgraph-agent
Environment="PATH=/opt/vpp-langgraph-agent/.venv/bin"
ExecStart=/opt/vpp-langgraph-agent/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=append:/var/log/vpp-agent/output.log
StandardError=append:/var/log/vpp-agent/error.log

[Install]
WantedBy=multi-user.target
```

创建日志目录：
```bash
sudo mkdir -p /var/log/vpp-agent
sudo chown www-data:www-data /var/log/vpp-agent
```

启动服务：
```bash
# 重载配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start vpp-agent

# 设置开机自启
sudo systemctl enable vpp-agent

# 查看状态
sudo systemctl status vpp-agent

# 查看日志
sudo journalctl -u vpp-agent -f
```

### 步骤 6: 安装配置 Nginx

```bash
# 安装 Nginx
sudo apt install -y nginx

# 创建配置
sudo nano /etc/nginx/sites-available/vpp-agent
```

配置内容：
```nginx
server {
    listen 80;
    server_name your-domain.com;  # 改为你的域名或服务器IP

    # 日志
    access_log /var/log/nginx/vpp-agent-access.log;
    error_log /var/log/nginx/vpp-agent-error.log;

    # API 接口
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置（LLM 可能需要较长时间）
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        # CORS 处理（如果 .env 配置不够）
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type' always;

        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }

    # 健康检查（不记录日志）
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

启用配置：
```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/vpp-agent /etc/nginx/sites-enabled/

# 删除默认配置
sudo rm /etc/nginx/sites-enabled/default

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx

# 设置开机自启
sudo systemctl enable nginx
```

### 步骤 7: 配置防火墙

```bash
# 开放 HTTP 端口
sudo ufw allow 80/tcp

# 开放 HTTPS 端口（可选）
sudo ufw allow 443/tcp

# 开放 SSH 端口（重要！）
sudo ufw allow 22/tcp

# 启用防火墙
sudo ufw enable

# 查看状态
sudo ufw status
```

### 步骤 8: 配置 HTTPS（推荐）

```bash
# 安装 certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书（自动配置 Nginx）
sudo certbot --nginx -d your-domain.com

# 测试自动续期
sudo certbot renew --dry-run
```

---

## 方案 2: Docker 部署（推荐）

### 前置要求
- Linux 服务器
- 已安装 Docker 和 Docker Compose

### 步骤 1: 安装 Docker

```bash
# 更新包索引
sudo apt update

# 安装依赖
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# 添加 Docker 官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker 仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER
# 需要重新登录生效
```

### 步骤 2: 上传代码

```bash
# 方式同方案 1，上传整个项目目录
cd /opt/vpp-langgraph-agent
```

### 步骤 3: 配置环境变量

```bash
# 复制并编辑 .env
cp .env.example .env
nano .env
```

### 步骤 4: 构建和启动

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f vpp-agent

# 查看运行状态
docker-compose ps
```

### 步骤 5: 测试

```bash
# 健康检查
curl http://localhost/health

# 测试 API
curl -X POST http://localhost/v1/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","question":"什么是虚拟电厂？","role":"viewer"}'
```

### Docker 常用命令

```bash
# 查看日志
docker-compose logs -f

# 停止服务
docker-compose stop

# 启动服务
docker-compose start

# 重启服务
docker-compose restart

# 停止并删除容器
docker-compose down

# 重新构建并启动
docker-compose up -d --build

# 进入容器
docker exec -it vpp-agent bash
```

---

## 访问测试

### 1. 健康检查

```bash
# 本地测试
curl http://localhost/health

# 远程测试（替换为你的服务器 IP 或域名）
curl http://your-server-ip/health
```

### 2. 查看 API 文档

浏览器访问：
- **本地**: http://localhost/docs
- **远程**: http://your-server-ip/docs 或 http://your-domain.com/docs

### 3. 测试 API 调用

```bash
curl -X POST http://your-server-ip/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-001",
    "question": "什么是虚拟电厂？",
    "role": "viewer"
  }'
```

### 4. 前端测试

```javascript
// 在前端代码中
const response = await fetch('http://your-server-ip/v1/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: 'user-123',
    question: '当前总发电功率是多少？',
    role: 'viewer'
  })
});

const data = await response.json();
console.log(data.answer);
```

---

## 监控和维护

### 查看日志

**systemd 部署**：
```bash
# 查看实时日志
sudo journalctl -u vpp-agent -f

# 查看最近 100 行
sudo journalctl -u vpp-agent -n 100

# 查看今天的日志
sudo journalctl -u vpp-agent --since today
```

**Docker 部署**：
```bash
# 查看实时日志
docker-compose logs -f vpp-agent

# 查看最近 100 行
docker-compose logs --tail=100 vpp-agent
```

### 重启服务

**systemd**：
```bash
sudo systemctl restart vpp-agent
```

**Docker**：
```bash
docker-compose restart vpp-agent
```

### 更新代码

**systemd**：
```bash
cd /opt/vpp-langgraph-agent
git pull
source .venv/bin/activate
pip install -e ".[dev]"
sudo systemctl restart vpp-agent
```

**Docker**：
```bash
cd /opt/vpp-langgraph-agent
git pull
docker-compose up -d --build
```

---

## 常见问题

### 1. 无法访问服务

**检查防火墙**：
```bash
sudo ufw status
sudo ufw allow 80/tcp
```

**检查服务状态**：
```bash
# systemd
sudo systemctl status vpp-agent

# Docker
docker-compose ps
```

**检查端口占用**：
```bash
sudo netstat -tulpn | grep :8000
```

### 2. CORS 错误

修改 `.env` 文件：
```bash
VPP_ALLOWED_ORIGINS=http://your-frontend.com,https://your-frontend.com
```

重启服务。

### 3. 数据库连接失败

**检查网络连通性**：
```bash
telnet db.example.invalid 3306
```

**检查配置**：
```bash
cat .env | grep DB_
```

### 4. LLM 调用超时

增加超时时间：
```bash
# .env
VPP_LLM_TIMEOUT_SECONDS=60
```

### 5. 内存不足

使用更少的 worker：
```bash
# systemd 服务文件中
ExecStart=.../uvicorn app.main:app --workers 2
```

### 6. 查看资源使用

```bash
# CPU 和内存
top
htop

# Docker
docker stats vpp-agent
```

---

## 给其他人的使用说明

部署完成后，给前端/后端开发者提供以下信息：

### 服务信息
```
API 地址: http://your-server-ip/v1/query
API 文档: http://your-server-ip/docs
健康检查: http://your-server-ip/health
```

### 调用示例
```javascript
fetch('http://your-server-ip/v1/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: 'user-123',
    question: '你的问题',
    role: 'viewer'
  })
})
.then(r => r.json())
.then(data => console.log(data.answer));
```

### 支持联系
```
技术支持: your-email@example.com
问题反馈: https://github.com/your-repo/issues
```

---

## 安全建议

1. **使用 HTTPS**（生产环境必须）
2. **定期更新系统和依赖**
3. **限制 SSH 登录**（使用密钥，禁用密码）
4. **配置日志轮转**（避免磁盘被日志占满）
5. **监控服务状态**（设置告警）
6. **定期备份配置和数据**

---

## 下一步优化

1. **添加 API 认证**（如果需要对外暴露）
2. **配置负载均衡**（多实例）
3. **添加监控**（Prometheus + Grafana）
4. **配置日志收集**（ELK Stack）
5. **自动化部署**（CI/CD）

---

**部署完成后，记得测试所有功能！**
