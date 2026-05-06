# Zulong IDE 全面汉化方案

## Context

Zulong IDE 是面向中文用户的 VS Code 插件（基于 Cline v3.82.0）。当前所有 ~400 条 UI 文本为硬编码英文，分布在 ~50 个源文件中。项目没有 i18n 框架。用户要求将所有界面内容直接替换为中文。

## 方案

直接在源码中将英文字符串替换为中文，不引入 i18n 框架。使用 Python 批处理脚本高效执行，每批完成后验证 TypeScript 编译。

## 实施步骤（6 批次）

按风险从低到高排列：

### 批次 1: 静态数据文件（~65 条）
- `webview-ui/src/components/chat/chat-view/shared/buttonConfig.ts` — 按钮标签（批准/拒绝/重试/取消等）
- `webview-ui/src/components/chat/auto-approve-menu/constants.ts` — 自动批准选项（读取/编辑/命令/浏览器/MCP）
- `webview-ui/src/components/chat/FeatureTip.tsx` — 13 条功能提示
- `webview-ui/src/components/onboarding/data-steps.ts` — 入门步骤文本

### 批次 2: package.json + Walkthrough（~40 条）
- `package.json` — 命令标题、walkthrough 描述
- `walkthrough/step1.md` ~ `step5.md` — 教程内容全文翻译

### 批次 3: 欢迎页 / 入门 / 账户（~25 条）
- `webview-ui/src/components/welcome/WelcomeView.tsx`
- `webview-ui/src/components/welcome/HomeHeader.tsx`
- `webview-ui/src/components/onboarding/OnboardingView.tsx`
- `webview-ui/src/components/account/AccountWelcomeView.tsx`

### 批次 4: 设置面板（~100 条）
- `webview-ui/src/components/settings/SettingsView.tsx` — 标签页名称
- `webview-ui/src/components/settings/sections/FeatureSettingsSection.tsx` — 功能开关标签和描述
- `webview-ui/src/components/settings/sections/BrowserSettingsSection.tsx`
- `webview-ui/src/components/settings/sections/TerminalSettingsSection.tsx`
- `webview-ui/src/components/settings/sections/GeneralSettingsSection.tsx`
- `webview-ui/src/components/settings/sections/ApiConfigurationSection.tsx`
- `webview-ui/src/components/settings/sections/RemoteConfigSection.tsx`
- `webview-ui/src/components/settings/sections/AboutSection.tsx`

### 批次 5: 聊天界面（~65 条）
- `webview-ui/src/components/chat/ChatView.tsx` — 输入框占位符
- `webview-ui/src/components/chat/ChatRow.tsx` — "Zulong 想要..." 操作消息（~24 条）
- `webview-ui/src/components/chat/ChatTextArea.tsx` — 输入框提示、模式切换
- `webview-ui/src/components/chat/BrowserSessionRow.tsx`
- `webview-ui/src/components/chat/SubagentStatusRow.tsx`
- `webview-ui/src/components/chat/ErrorRow.tsx` — 错误提示
- `webview-ui/src/components/chat/auto-approve-menu/AutoApproveBar.tsx`
- `webview-ui/src/components/chat/auto-approve-menu/AutoApproveModal.tsx`

### 批次 6: 历史 + MCP + 看板弹窗（~50 条）
- `webview-ui/src/components/history/HistoryView.tsx`
- `webview-ui/src/components/history/HistoryPreview.tsx`
- `webview-ui/src/components/history/HistoryViewItem.tsx`
- `webview-ui/src/components/mcp/configuration/McpConfigurationView.tsx`
- `webview-ui/src/components/mcp/configuration/tabs/installed/*.tsx`
- `webview-ui/src/components/common/ClineKanbanLaunchModal.tsx`

## 安全规则

Python 脚本使用精确字符串匹配（非正则），每个替换为 `(文件路径, 原文, 译文)` 三元组。

**不触碰**：import 语句、变量名、组件名、CSS 类名、URL、文件路径、枚举值、对象键名

## 特殊处理

- 模板字符串中的三元表达式值（如 `"Act"` → `"执行"`）需精确定位
- 日期/数字格式中的 `"en-US"` locale → `"zh-CN"`
- package.json 用 Python json 模块读写确保格式安全
- 中文不区分单复数，合并翻译

## 验证

每批完成后：
1. `npx tsc --noEmit` 编译检查（主项目 + webview-ui）
2. `git diff` 确认仅字符串变更

全部完成后：
1. `npm run build:webview` 构建
2. `vsce package` 打包 VSIX
3. 安装到 VS Code 验证界面
