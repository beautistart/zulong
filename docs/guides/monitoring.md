# 祖龙 (ZULONG) 监控配置指南

**版本**: v1.0  
**创建日期**: 2026-03-30  
**适用环境**: 生产环境

---

## 📋 目录

1. [监控架构](#监控架构)
2. [Prometheus 配置](#prometheus-配置)
3. [Grafana 仪表盘](#grafana-仪表盘)
4. [告警规则](#告警规则)
5. [自定义指标](#自定义指标)
6. [故障排查](#故障排查)

---

## 🏗️ 监控架构

### 组件概览

```
监控体系:
├─ Prometheus      # 指标采集与存储
├─ Grafana         # 可视化仪表盘
├─ Loki            # 日志聚合（可选）
└─ Alertmanager    # 告警管理（可选）

数据流:
应用 → Metrics → Prometheus → Grafana
应用 → Logs → Loki → Grafana
Prometheus → Alertmanager → 通知渠道
```

### 指标分类

**1. 系统指标**:
- CPU 使用率
- 内存使用率
- GPU 显存使用
- GPU 温度
- 磁盘空间

**2. 应用指标**:
- 推理延迟（P50, P90, P95, P99）
- 吞吐量（QPS）
- 缓存命中率
- 错误率

**3. 业务指标**:
- 任务成功率
- 路径规划时间
- 视觉检测准确率

---

## 📊 Prometheus 配置

### 基础配置

配置文件：`monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s      # 采集间隔
  evaluation_interval: 15s  # 规则评估间隔

scrape_configs:
  - job_name: 'zulong'
    static_configs:
      - targets: ['zulong:8000']
    metrics_path: '/metrics'
```

### 启动 Prometheus

```bash
# 使用 Docker Compose
docker-compose up -d prometheus

# 访问 Prometheus
# http://localhost:9090
```

### 验证指标采集

```bash
# 检查指标
curl http://localhost:8000/metrics

# 示例输出:
# HELP zulong_inference_latency_seconds 推理延迟
# TYPE zulong_inference_latency_seconds histogram
# zulong_inference_latency_seconds_bucket{le="0.5"} 100
# zulong_inference_latency_seconds_bucket{le="1.0"} 200
```

---

## 📈 Grafana 仪表盘

### 启动 Grafana

```bash
# 使用 Docker Compose
docker-compose up -d grafana

# 访问 Grafana
# http://localhost:3000
# 默认账号：admin / zulong123
```

### 配置数据源

**自动配置**（推荐）:
```yaml
# monitoring/grafana/datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
```

**手动配置**:
1. 访问 Grafana
2. Configuration → Data Sources
3. Add data source → Prometheus
4. URL: `http://prometheus:9090`
5. Save & Test

### 导入仪表盘

**方法 1: 使用预配置仪表盘**

```bash
# 仪表盘已配置在 docker-compose.yml 中
# 自动挂载到 /etc/grafana/provisioning/dashboards
```

**方法 2: 手动导入**

1. 访问 Grafana
2. Dashboards → Import
3. 上传 JSON 文件
4. 选择数据源
5. Import

### 可用仪表盘

**1. 系统资源监控** (`system.json`)
- CPU、内存、GPU 使用率
- GPU 温度监控
- 资源使用趋势

**2. 性能监控** (`performance.json`)
- 推理延迟分布（P50, P90, P95, P99）
- 缓存命中率
- 路径规划时间
- 吞吐量（QPS）

**3. 业务监控** (`business.json`)
- 任务成功率
- 错误类型分布
- 用户活跃度

---

## 🚨 告警规则

### 配置告警

配置文件：`monitoring/alerts.yml`

```yaml
groups:
  - name: zulong-alerts
    rules:
      # GPU 显存超限告警
      - alert: HighGPUMemoryUsage
        expr: zulong_gpu_memory_usage_bytes / zulong_gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 显存使用率超过 90%"
```

### 告警级别

**Severity 级别**:
- `critical` - 严重（立即处理）
- `warning` - 警告（需要关注）
- `info` - 信息（可选处理）

### 告警规则列表

| 告警名称 | 表达式 | 阈值 | 持续时间 | 级别 |
|---------|--------|------|---------|------|
| HighGPUMemoryUsage | GPU 显存使用率 | >90% | 5m | warning |
| HighInferenceLatency | P95 推理延迟 | >2.0s | 5m | warning |
| HighErrorRate | 错误率 | >5% | 5m | critical |
| LowCacheHitRate | 缓存命中率 | <30% | 10m | info |
| HighMemoryPressure | 内存压力 | >80% | 5m | warning |
| HighTaskFailureRate | 任务失败率 | >10% | 5m | critical |
| HighCPUUsage | CPU 使用率 | >80% | 10m | warning |
| ServiceDown | 服务可用性 | ==0 | 1m | critical |
| HighPlanningTime | P95 规划时间 | >50ms | 5m | warning |
| LowDiskSpace | 磁盘可用空间 | <10% | 5m | warning |

### 配置通知渠道

**1. 邮件通知**:
```yaml
# Alertmanager 配置
receivers:
  - name: 'email'
    email_configs:
      - to: 'admin@example.com'
        from: 'alert@example.com'
        smarthost: 'smtp.example.com:587'
```

**2. Slack 通知**:
```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/...'
        channel: '#alerts'
```

**3. Webhook 通知**:
```yaml
receivers:
  - name: 'webhook'
    webhook_configs:
      - url: 'http://your-webhook/alerts'
```

---

## 📏 自定义指标

### 在应用中暴露指标

**1. 定义指标**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Counter - 计数
zulong_requests_total = Counter(
    'zulong_requests_total',
    'Total requests',
    ['method', 'endpoint']
)

# Histogram - 分布
zulong_inference_latency = Histogram(
    'zulong_inference_latency_seconds',
    'Inference latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Gauge - 瞬时值
zulong_gpu_memory = Gauge(
    'zulong_gpu_memory_usage_bytes',
    'GPU memory usage'
)
```

**2. 记录指标**:

```python
import time

@zulong_inference_latency.time()
async def infer():
    """推理函数"""
    start = time.time()
    # 推理逻辑...
    return result

# 更新 Gauge
zulong_gpu_memory.set(get_gpu_memory())

# 更新 Counter
zulong_requests_total.labels(method='POST', endpoint='/infer').inc()
```

**3. 暴露指标端点**:

```python
from fastapi import FastAPI
from prometheus_client import generate_latest

app = FastAPI()

@app.get("/metrics")
def metrics():
    """Prometheus 指标端点"""
    return generate_latest()
```

### 指标命名规范

**格式**: `<namespace>_<component>_<name>_<unit>`

**示例**:
- `zulong_inference_latency_seconds`
- `zulong_cache_hit_rate`
- `zulong_gpu_memory_usage_bytes`

**指标类型**:
- `Counter` - 只增不减（请求数、错误数）
- `Gauge` - 可增可减（内存、温度）
- `Histogram` - 分布统计（延迟、时间）
- `Summary` - 分位数统计

---

## 🔧 故障排查

### 问题 1: 指标无法采集

**症状**: Prometheus 显示 `up=0`

**解决方案**:
```bash
# 检查服务状态
docker-compose ps

# 检查日志
docker-compose logs zulong

# 验证指标端点
curl http://localhost:8000/metrics
```

### 问题 2: Grafana 无数据

**症状**: 仪表盘显示 "No data"

**解决方案**:
```bash
# 检查数据源配置
# Grafana → Configuration → Data Sources

# 验证 Prometheus 连接
# Grafana → Data Sources → Prometheus → Save & Test

# 检查指标名称
# Prometheus → Graph → 输入指标名称
```

### 问题 3: 告警不触发

**症状**: 条件满足但告警未触发

**解决方案**:
```bash
# 检查告警规则
# Prometheus → Alerts

# 验证表达式
# Prometheus → Graph → 输入表达式

# 检查持续时间
# 确认 'for' 条件是否满足
```

### 问题 4: 内存占用过高

**症状**: Prometheus 内存使用持续增长

**解决方案**:
```yaml
# 调整保留时间
# prometheus.yml:
global:
  scrape_interval: 30s  # 降低采集频率

# 限制存储时间
command:
  - '--storage.tsdb.retention.time=7d'
```

---

## 📚 最佳实践

### 1. 指标设计

**选择合适的指标类型**:
- 计数 → Counter
- 瞬时值 → Gauge
- 分布 → Histogram

**合理设置 buckets**:
```python
# 推理延迟（秒）
buckets=[0.1, 0.5, 1.0, 2.0, 5.0]

# 规划时间（毫秒）
buckets=[0.005, 0.01, 0.02, 0.05, 0.1]
```

### 2. 采集频率

**平衡精度与开销**:
```yaml
# 生产环境
scrape_interval: 15s

# 开发环境
scrape_interval: 5s

# 长期存储
scrape_interval: 60s
```

### 3. 告警优化

**避免告警风暴**:
- 设置合理的 `for` 持续时间
- 使用分组告警
- 配置告警抑制

**告警分级**:
- Critical - 电话通知
- Warning - 邮件/Slack
- Info - 仅记录

### 4. 仪表盘设计

**分层展示**:
- Level 1: 概览（关键指标）
- Level 2: 详情（组件指标）
- Level 3: 调试（详细日志）

**颜色规范**:
- 绿色 - 正常
- 黄色 - 警告
- 红色 - 严重

---

## 📊 监控示例

### 查询示例

**1. 查询推理延迟**:
```promql
# P95 推理延迟
histogram_quantile(0.95, rate(zulong_inference_latency_seconds_bucket[5m]))

# 平均推理延迟
rate(zulong_inference_latency_seconds_sum[5m]) / rate(zulong_inference_latency_seconds_count[5m])
```

**2. 查询缓存命中率**:
```promql
# 缓存命中率
zulong_cache_hit_rate

# 缓存命中次数
rate(zulong_cache_hits_total[5m])

# 缓存未命中次数
rate(zulong_cache_misses_total[5m])
```

**3. 查询 GPU 使用**:
```promql
# GPU 显存使用率
zulong_gpu_memory_used_bytes / zulong_gpu_memory_total_bytes

# GPU 温度
zulong_gpu_temperature_celsius
```

### 告警示例

**1. 推理延迟过高**:
```yaml
- alert: HighInferenceLatency
  expr: histogram_quantile(0.95, rate(zulong_inference_latency_seconds_bucket[5m])) > 2.0
  for: 5m
  annotations:
    summary: "P95 推理延迟超过 2 秒"
```

**2. 缓存命中率过低**:
```yaml
- alert: LowCacheHitRate
  expr: zulong_cache_hit_rate < 0.3
  for: 10m
  annotations:
    summary: "缓存命中率低于 30%"
```

---

## 📚 相关文档

- [Docker 部署指南](DOCKER_DEPLOYMENT.md)
- [生产环境指南](PRODUCTION_SETUP.md)
- [性能基准报告](PERFORMANCE_BENCHMARK.md)

---

**文档版本**: v1.0  
**最后更新**: 2026-03-30  
**维护者**: 祖龙 (ZULONG) 系统架构组
