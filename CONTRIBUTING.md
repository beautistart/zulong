# 祖龙 (ZULONG) - 贡献指南

感谢您考虑为祖龙项目做出贡献！

## 📄 许可证说明

本项目采用分层许可证策略：

- **核心代码**（`zulong/l2/`, `zulong/memory/memory_graph.py` 等）：AGPL-3.0
- **接口前端**（`zulong-ide/`, `zulong/config/` 等）：MIT
- **文档**（`docs/`, `README.md`）：CC BY-NC-SA 4.0

详见 [LICENSE](../LICENSE) 文件。

## 🤝 如何贡献

### 1. Fork 和 Clone

```bash
# Fork 本仓库到您的 GitHub
# 然后 clone
git clone https://github.com/YOUR_USERNAME/zulong.git
cd zulong

# 添加上游仓库
git remote add upstream https://github.com/beautistart/zulong.git
```

### 2. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 3. 进行修改

- 遵循代码风格（见下文）
- 添加必要的测试
- 更新相关文档

### 4. 提交代码

```bash
git add .
git commit -m "feat: 添加XX功能"
```

提交信息格式：
- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `refactor:` 重构
- `test:` 测试相关

### 5. 推送和创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

## 📝 贡献者许可协议 (CLA)

对于重要的贡献（超过100行代码或重要的功能贡献），您需要签署我们的 CLA。

CLA 确保：
1. 您授予祖龙项目永久、非独占的使用权
2. 您的贡献可以在闭源版本中使用
3. 保护项目免受未来的法律问题

我们会在您的 PR 被接受后联系您签署 CLA。

## 🎨 代码风格

### Python

- 遵循 PEP 8
- 使用 4 空格缩进
- 最大行长度 100 字符
- 使用类型提示
- 添加文档字符串

```python
def example_function(param1: str, param2: int) -> bool:
    """
    示例函数
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    """
    return True
```

### TypeScript

- 使用 2 空格缩进
- 使用 TypeScript 类型
- 组件使用函数式写法
- 添加必要的注释

## 🧪 测试

运行测试：

```bash
# Python 测试
pytest tests/

# TypeScript 测试
cd zulong-ide
npm test
```

请确保您的修改不破坏现有测试，并添加新的测试。

## 📚 文档

如果您的修改涉及功能变更，请更新相关文档：

- API 变更：更新 `docs/TSD_v3.0.md`
- 用户指南：更新 `docs/Zulong_IDE使用指南.md`
- README：更新 `README.md`

## 🔍 Code Review

所有 PR 都需要经过 code review：

1. 确保代码质量
2. 确保测试覆盖
3. 确保文档更新
4. 确保许可证兼容

## ❓ 问题讨论

- Bug 报告：使用 GitHub Issues
- 功能建议：使用 GitHub Discussions
- 技术讨论：加入我们的 Discord（待建立）

## 🙏 感谢

感谢所有贡献者！您的贡献让祖龙变得更好。

---

祖龙 - 让AI拥有真正的记忆
