# Zulong IDE - Project Intelligence

> This file is auto-read by Qoder to provide project context.

## Project Overview

Zulong IDE is a VS Code extension based on Cline v3.82.0, serving as the frontend for the Zulong multi-layer adaptive intelligent agent system. UI has been fully localized to Chinese (zh-CN).

## Build Pipeline

```bash
# Full build sequence (run from zulong-ide/ directory):
npm run protos                    # Generate TypeScript proto files
cd webview-ui && npm install && npm run build && cd ..  # Build React webview
node esbuild.mjs --production     # Build extension with esbuild
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license  # Package VSIX
code --install-extension zulong-ide-0.1.0.vsix --force  # Install
```

## Key Architecture

- **Frontend**: React + Vite (`webview-ui/`)
- **Extension**: TypeScript + esbuild (`zulong-ide/src/`)
- **Backend**: Python FastAPI + WebSocket (`zulong/ide/`)
- **Communication**: WebSocket at `ws://127.0.0.1:8090/ide`
- **Config**: `config/zulong_config.yaml` (port: 8090)

## Critical File Map

| File | Role |
|------|------|
| `zulong-ide/src/core/api/providers/zulong.ts` | ZulongHandler - WebSocket API provider |
| `zulong-ide/src/core/api/transport/zulong-websocket.ts` | WebSocket transport layer |
| `zulong-ide/webview-ui/src/components/settings/providers/ZulongProvider.tsx` | Zulong settings UI |
| `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` | Provider selection & conditional render |
| `zulong-ide/src/shared/providers/providers.json` | Provider list (Zulong is first entry) |
| `zulong/ide/ide_server.py` | Python backend entry (FastAPI + WebSocket) |
| `zulong/ide/ide_fc_runner.py` | FC loop executor |
| `zulong/ide/ide_tool_registry.py` | Tool registry & smart routing |

## Coding Conventions

- **Component function names must NOT be translated** (e.g., keep `RefreshButton`, not `刷新Button`)
- All user-facing strings are in Chinese (zh-CN)
- Removed features: Account page, Kanban modal (Cline native features not needed)
- Navigation tabs: Chat / MCP / History / Settings (4 tabs)

## Known Issues

- **Provider selection bug**: Selecting "Zulong (祖龙)" in settings dropdown may show Anthropic settings instead of Zulong WebSocket config. Investigate `handleProviderChange` in `ApiOptions.tsx`.

## TypeScript Check

```bash
cd zulong-ide && npx tsc --noEmit
```

## Documentation

- Usage guide: `docs/Zulong_IDE使用指南.md`
- Memory export: `docs/Qoder_Quest_Memory_Export.md`
- Deep analysis: `docs/祖龙系统深度技术分析报告.md`
