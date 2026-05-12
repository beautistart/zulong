# 祖龙 (ZULONG) CI/CD 配置指南

**版本**: v1.0  
**创建日期**: 2026-03-30  
**适用环境**: GitHub Actions

---

## 📋 目录

1. [CI/CD 概述](#cicd-概述)
2. [CI 流程](#ci-流程)
3. [CD 流程](#cd-流程)
4. [环境配置](#环境配置)
5. [部署脚本](#部署脚本)
6. [最佳实践](#最佳实践)

---

## 🎯 CI/CD 概述

### 工作流程

```
开发流程:
代码提交 → CI 检查 → 单元测试 → 集成测试 → 构建镜像 → 部署开发环境

发布流程:
创建 Release → 构建生产镜像 → 部署 Staging → 人工审批 → 部署生产
```

### 环境划分

**1. 开发环境 (Develop)**:
- 自动部署
- 用于日常开发
- 最新代码

**2. 测试环境 (Staging)**:
- 发布前验证
- 模拟生产环境
- 冒烟测试

**3. 生产环境 (Production)**:
- 人工审批
- 正式发布
- 蓝绿部署

---

## 🔧 CI 流程

### 配置文件

`.github/workflows/ci.yml`

### 工作流程

```yaml
name: CI - 持续集成

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

### 工作步骤

**1. 代码质量检查**:
```bash
# Flake8 代码规范检查
flake8 zulong/ --count --select=E9,F63,F7,F82

# Black 格式化检查
black --check zulong/

# Mypy 类型检查
mypy zulong/ --ignore-missing-imports
```

**2. 单元测试**:
```bash
# 运行单元测试 + 覆盖率
pytest tests/ -v --cov=zulong --cov-report=xml
```

**3. 集成测试**:
```bash
# 运行集成测试（需要外部服务）
pytest tests/integration/ -v --mock-sensors
```

**4. Docker 镜像构建**:
```bash
# 构建并推送镜像
docker build -t zulong-system:latest .
docker push ghcr.io/zulong/zulong-system:latest
```

**5. 性能测试** (仅 main 分支):
```bash
# 性能基准测试
pytest tests/performance/ -v --benchmark-json=output.json
```

### 状态检查

**通过标准**:
- ✅ 代码质量检查通过
- ✅ 单元测试通过率 100%
- ✅ 集成测试通过率 100%
- ✅ Docker 镜像构建成功
- ✅ 覆盖率 >80%

---

## 🚀 CD 流程

### 配置文件

`.github/workflows/cd.yml`

### 触发条件

```yaml
on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: '部署环境'
        required: true
        default: 'staging'
```

### 工作流程

**1. 构建生产镜像**:
```bash
# 使用生产 Dockerfile
docker build -f Dockerfile.prod -t zulong-system:v1.0 .
docker push ghcr.io/zulong/zulong-system:v1.0
```

**2. 部署到 Staging**:
```bash
# 部署到测试环境
kubectl set image deployment/zulong zulong=zulong-system:v1.0
kubectl rollout status deployment/zulong
```

**3. 冒烟测试**:
```bash
# 运行冒烟测试
pytest tests/smoke/ --staging
```

**4. 人工审批** (生产环境):
- GitHub 自动创建审批按钮
- 需要管理员审批

**5. 部署到生产**:
```bash
# 部署到生产环境
kubectl set image deployment/zulong zulong=zulong-system:v1.0
kubectl rollout status deployment/zulong
```

**6. 健康检查**:
```bash
# 验证服务
curl -f https://zulong.example.com/health
```

**7. 失败回滚**:
```bash
# 回滚到上一个版本
kubectl rollout undo deployment/zulong
```

---

## ⚙️ 环境配置

### GitHub Secrets

**必需配置**:

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `SLACK_WEBHOOK_URL` | Slack 通知 Webhook | `https://hooks.slack.com/...` |
| `DEPLOY_KEY` | 部署密钥 | `ssh-rsa AAAA...` |
| `PROD_SERVER` | 生产服务器地址 | `prod.example.com` |

**配置步骤**:
1. 仓库 → Settings → Secrets and variables → Actions
2. New repository secret
3. 添加变量

### GitHub Environments

**配置环境**:

1. **staging**:
   - 自动部署
   - URL: https://staging.zulong.example.com

2. **production**:
   - 需要审批
   - 审批人：管理员
   - URL: https://zulong.example.com

**配置步骤**:
1. 仓库 → Settings → Environments
2. New environment
3. 配置审批规则

---

## 📜 部署脚本

### 使用方式

```bash
# 显示帮助
./scripts/deploy.sh --help

# 部署到生产环境
./scripts/deploy.sh

# 构建并部署
./scripts/deploy.sh --build production

# 查看日志
./scripts/deploy.sh --logs dev

# 健康检查
./scripts/deploy.sh --health

# 备份数据
./scripts/deploy.sh --backup
```

### 脚本功能

**1. 构建镜像**:
```bash
./scripts/deploy.sh --build
```

**2. 启动服务**:
```bash
./scripts/deploy.sh --up
```

**3. 停止服务**:
```bash
./scripts/deploy.sh --down
```

**4. 重启服务**:
```bash
./scripts/deploy.sh --restart
```

**5. 查看状态**:
```bash
./scripts/deploy.sh --status
```

**6. 查看日志**:
```bash
./scripts/deploy.sh --logs
```

**7. 健康检查**:
```bash
./scripts/deploy.sh --health
```

**8. 备份数据**:
```bash
./scripts/deploy.sh --backup
```

**9. 恢复数据**:
```bash
./scripts/deploy.sh --restore ./backup/20260330_120000
```

---

## 📊 监控与通知

### Slack 通知

**配置**:
```yaml
notify:
  steps:
    - name: 发送 Slack 通知
      uses: slackapi/slack-github-action@v1.23.0
      with:
        payload: |
          {
            "text": "部署${{ needs.deploy-production.result }}",
            "environment": "${{ github.event.inputs.environment }}",
            "version": "${{ github.ref_name }}",
            "actor": "${{ github.actor }}"
          }
```

**通知内容**:
- 部署开始
- 部署成功/失败
- 健康检查结果
- 性能指标

### 邮件通知

**配置 Alertmanager**:
```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.example.com:587'
```

---

## 🎯 最佳实践

### 1. 分支策略

**Git Flow**:
```
main (生产)
  ↑
staging (测试)
  ↑
develop (开发)
  ↑
feature/* (功能)
```

**保护规则**:
- main: 需要 PR + 审批
- staging: 需要 PR
- develop: 直接推送

### 2. 版本管理

**语义化版本**:
```
v1.0.0
├─ Major: 不兼容更新
├─ Minor: 新功能
└─ Patch: Bug 修复
```

**打标签**:
```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### 3. 回滚策略

**自动回滚**:
```yaml
rollback:
  if: failure()
  steps:
    - name: 回滚
      run: kubectl rollout undo deployment/zulong
```

**手动回滚**:
```bash
# 回滚到指定版本
kubectl rollout undo deployment/zulong --to-revision=2
```

### 4. 蓝绿部署

**配置**:
```yaml
# 两个相同的部署
deployment-green
deployment-blue

# 通过 Service 切换流量
service:
  selector:
    deployment: green  # 或 blue
```

**切换流程**:
1. 部署新版本到 blue
2. 运行测试
3. 切换 Service 到 blue
4. 观察监控
5. 删除 green

### 5. 金丝雀发布

**配置**:
```yaml
# 90% 流量到 green, 10% 到 blue
istio VirtualService:
  route:
    - destination: green
      weight: 90
    - destination: blue
      weight: 10
```

**渐进式发布**:
1. 1% → 监控
2. 10% → 监控
3. 50% → 监控
4. 100% → 完成

---

## 🔧 故障排查

### 问题 1: CI 失败

**症状**: GitHub Actions 显示失败

**解决方案**:
```bash
# 查看日志
# Actions → 失败的工作流 → 查看日志

# 本地重现
pytest tests/ -v
black --check zulong/
mypy zulong/
```

### 问题 2: 部署失败

**症状**: 部署脚本报错

**解决方案**:
```bash
# 查看部署日志
./scripts/deploy.sh --logs

# 检查 Docker 状态
docker-compose ps

# 检查服务健康
./scripts/deploy.sh --health
```

### 问题 3: 镜像推送失败

**症状**: `denied: requested access to the resource is denied`

**解决方案**:
```bash
# 重新登录
docker login ghcr.io -u USERNAME -p TOKEN

# 检查权限
# Settings → Actions → General → Workflow permissions
```

---

## 📚 相关文档

- [Docker 部署指南](DOCKER_DEPLOYMENT.md)
- [监控配置指南](MONITORING_SETUP.md)
- [生产环境指南](PRODUCTION_SETUP.md)

---

**文档版本**: v1.0  
**最后更新**: 2026-03-30  
**维护者**: 祖龙 (ZULONG) 系统架构组
