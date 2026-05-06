# IDE Slim: Zulong-Only Provider Architecture

## Context

Zulong IDE is a VS Code extension forked from Cline v3.82.0. It carries **43 third-party AI provider handlers** (Anthropic, OpenAI, Bedrock, etc.) totaling ~600KB of dead code, plus matching transform files, settings UI components, account/OAuth controllers, and model refresh handlers. In Zulong mode, **all AI inference runs on the Python backend** via WebSocket; the IDE only needs the `ZulongHandler` relay. This task strips the IDE to a "shell + hands/feet" architecture where all non-Zulong provider code is removed.

**Expected result**: ~3.5MB of dead code removed, cleaner codebase, faster builds, zero functional regression (only Zulong provider was ever used).

---

## Step 1: Delete Provider Handler Files

**Directory**: `zulong-ide/src/core/api/providers/`

DELETE all files **except** `zulong.ts`:

```
aihubmix.ts, anthropic.ts, asksage.ts, baseten.ts, bedrock.ts, cerebras.ts,
claude-code.ts, cline.ts, deepseek.ts, dify.ts, doubao.ts, fireworks.ts,
gemini.ts, gemini-mock.test.ts, groq.ts, hicap.ts, huawei-cloud-maas.ts,
huggingface.ts, litellm.ts, lmstudio.ts, minimax.ts, mistral.ts, moonshot.ts,
nebius.ts, nousresearch.ts, oca.ts, ollama.ts, openai.ts, openai-codex.ts,
openai-native.ts, openrouter.ts, qwen.ts, qwen-code.ts, requesty.ts,
sambanova.ts, sapaicore.ts, together.ts, types.ts, vercel-ai-gateway.ts,
vertex.ts, vscode-lm.ts, wandb.ts, xai.ts, zai.ts
```

Also delete the `__tests__/` subdirectory.

**Why types.ts is safe**: Only exports `OpenRouterErrorResponse` and `LanguageModelChatSelector`, imported exclusively by deleted provider files (openrouter.ts, cline.ts, vscode-lm.ts).

**KEEP**: `zulong.ts` (255 lines, active WebSocket handler)

---

## Step 2: Delete Transform Files

**Directory**: `zulong-ide/src/core/api/transform/`

DELETE all files **except** `stream.ts`:

```
anthropic-format.ts, gemini-format.ts, mistral-format.ts, o1-format.ts,
ollama-format.ts, openai-format.ts, openai-response-format.ts,
openrouter-stream.ts, r1-format.ts, tool-call-processor.ts,
vercel-ai-gateway-stream.ts, vscode-lm-format.ts, vscode-lm-format.test.ts
```

Also delete the `__tests__/` subdirectory if present.

**Why tool-call-processor.ts is safe**: Used by 17 provider files, all of which are being deleted. Not imported by zulong.ts or stream.ts.

**KEEP**: `stream.ts` (exported types `ApiStream`, `ApiStreamChunk`, `ApiStreamUsageChunk` used by Task system and zulong.ts)

---

## Step 3: Delete Settings UI Provider Components

**Directory**: `zulong-ide/webview-ui/src/components/settings/providers/`

DELETE all files **except** `ZulongProvider.tsx`:

```
AihubmixProvider.tsx, AnthropicProvider.tsx, AskSageProvider.tsx,
BasetenProvider.tsx, BedrockProvider.tsx, CerebrasProvider.tsx,
ClaudeCodeProvider.tsx, ClineProvider.tsx, DeepSeekProvider.tsx,
DifyProvider.tsx, DoubaoProvider.tsx, FireworksProvider.tsx,
GeminiProvider.tsx, GroqProvider.tsx, HicapProvider.tsx,
HuaweiCloudMaasProvider.tsx, HuggingFaceProvider.tsx, LiteLlmProvider.tsx,
LMStudioProvider.tsx, MiniMaxProvider.tsx, MistralProvider.tsx,
MoonshotProvider.tsx, NebiusProvider.tsx, NousresearchProvider.tsx,
OcaModelPicker.tsx, OcaProvider.tsx, OllamaProvider.tsx,
OpenAiCodexProvider.tsx, OpenAICompatible.tsx, OpenAINative.tsx,
OpenRouterProvider.tsx, QwenCodeProvider.tsx, QwenProvider.tsx,
RequestyProvider.tsx, SambanovaProvider.tsx, SapAiCoreProvider.tsx,
TogetherProvider.tsx, VercelAIGatewayProvider.tsx, VertexProvider.tsx,
VSCodeLmProvider.tsx, WandbProvider.tsx, XaiProvider.tsx, ZAiProvider.tsx
```

Also DELETE `OpenRouterModelPicker.tsx` from `webview-ui/src/components/settings/` (only used by deleted OpenRouterProvider).

**KEEP**: `ZulongProvider.tsx` (66 lines, server URL + auto-approve config)

---

## Step 4: Delete Account/OAuth Files

### 4a: Account Controller
**Directory**: `zulong-ide/src/core/controller/account/`
DELETE entire directory (15 files). All handlers are registered dynamically via protobus gRPC; no direct imports from outside.

### 4b: OCA Account Controller
**Directory**: `zulong-ide/src/core/controller/ocaAccount/`
DELETE entire directory (3 files).

### 4c: Account UI Components
**Directory**: `zulong-ide/webview-ui/src/components/account/`
DELETE these files:
```
AccountView.tsx, AccountWelcomeView.tsx, CreditBalance.tsx,
CreditsHistoryTable.tsx, StyledCreditDisplay.tsx
```

**KEEP**: `RemoteConfigToggle.tsx` and `helpers.ts` (imported by `RemoteConfigSection.tsx`)

---

## Step 5: Refactor `api/index.ts`

**File**: `zulong-ide/src/core/api/index.ts` (518 lines -> ~50 lines)

Changes:
1. Remove 42 provider imports (lines 6-47), keep only ZulongHandler import
2. Simplify `createHandlerForProvider()` switch to:
   ```typescript
   function createHandlerForProvider(
     apiProvider: string | undefined,
     options: Omit<ApiConfiguration, "apiProvider">,
     mode: Mode,
   ): ApiHandler {
     // Zulong is the only supported provider
     return new ZulongHandler({
       onRetryAttempt: options.onRetryAttempt,
       zulongServerUrl: options.zulongServerUrl,
     })
   }
   ```
3. Keep `buildApiHandler()` function and all exported interfaces (`ApiHandler`, `ApiHandlerModel`, `ApiProviderInfo`, `SingleCompletionHandler`, `CommonApiHandlerOptions`)
4. Keep `ApiStream` and `ApiStreamUsageChunk` re-export from `./transform/stream`

---

## Step 6: Refactor `providers.json`

**File**: `zulong-ide/src/shared/providers/providers.json` (176 lines -> 8 lines)

Replace entire content with:
```json
{
  "list": [
    {
      "value": "zulong",
      "label": "Zulong (祖龙)"
    }
  ]
}
```

---

## Step 7: Refactor `ApiOptions.tsx`

**File**: `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` (587 lines -> ~100 lines)

Changes:
1. Remove 42 provider imports (lines 15-56), keep only `ZulongProvider` import
2. Remove `OpenRouterModelPicker` import (line 14). Define `DROPDOWN_Z_INDEX` as a literal number (e.g., `const DROPDOWN_Z_INDEX = 110`)
3. Remove Ollama polling logic (`_ollamaModels` state, `requestLocalModels`, `useInterval` - lines 108-133)
4. Remove `declare module "vscode"` block (lines 84-91)
5. Remove ALL conditional provider renders **except** the Zulong block (lines 359-361). Delete lines 363-528
6. Keep: Provider search/dropdown UI (it now has only 1 option but the UI still works), error message display, styled components

---

## Step 8: Refactor `providerUtils.ts`

**File**: `zulong-ide/webview-ui/src/components/settings/utils/providerUtils.ts` (970 lines -> ~200 lines)

Changes:
1. Remove all non-Zulong model imports from `@shared/api` (lines 1-72). Keep only `ApiConfiguration`, `ApiProvider`, `ModelInfo`
2. Simplify `getModelsForProvider()`: return `undefined` for all cases (Zulong uses backend-managed models)
3. Simplify `normalizeApiConfiguration()`: only handle `"zulong"` case + default
4. Simplify `syncModeConfigurations()`: only `"zulong"` case + default
5. Simplify `getProviderInfo()`: only `"zulong"` case + default

---

## Step 9: Update `protobus-services.ts` (Auto-Generated)

**File**: `zulong-ide/src/generated/hosts/vscode/protobus-services.ts`

Since this file is auto-generated but we're fundamentally changing the architecture, manually edit:

1. Remove Account service imports (lines 6-20, 15 lines)
2. Remove `AccountServiceHandlers` const (lines 204-220)
3. Remove OcaAccount service imports (lines 110-113)
4. Remove `OcaAccountServiceHandlers` const (lines 316-320)
5. Remove from `serviceHandlers` export:
   - `"cline.AccountService": AccountServiceHandlers,`
   - `"cline.OcaAccountService": OcaAccountServiceHandlers,`

**Note**: Model service handlers are kept even though most are dead code. They compile fine (controller/models/ files stay) and removing them requires updating proto definitions. This is deferred.

---

## Step 10: Deferred Items (NOT in this task)

These items are intentionally deferred to keep scope manageable:

1. **Prompts layer** (`src/core/prompts/`, 2.9MB, 112 files): Complex dependency from `Task.ts` line 2001 `getSystemPrompt()`. Requires deeper analysis.
2. **Model refresh controllers** (`src/core/controller/models/`, 30 files): Mostly dead but registered in protobus. Removing requires proto regeneration.
3. **Shared API type cleanup** (`@shared/api`): `ApiHandlerSettings` still has 100+ provider-specific fields. Harmless but messy.

---

## Verification

### TypeScript Check
```bash
cd zulong-ide && npx tsc --noEmit
```
Fix any remaining type errors iteratively until clean.

### Full Build
```bash
cd zulong-ide
npm run protos
cd webview-ui && npm install && npm run build && cd ..
node esbuild.mjs --production
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license
```

### Functional Test
1. Install the built VSIX: `code --install-extension zulong-ide-0.1.0.vsix --force`
2. Open VS Code, go to Zulong settings tab
3. Verify "Zulong (祖龙)" is the only provider option
4. Verify WebSocket connection URL field is displayed
5. Verify no console errors in Developer Tools

---

## File Impact Summary

| Action | Count | Location |
|--------|-------|----------|
| DELETE provider handlers | 44 | `src/core/api/providers/` |
| DELETE transform files | 13 | `src/core/api/transform/` |
| DELETE settings UI | 43+1 | `webview-ui/.../providers/` + `OpenRouterModelPicker.tsx` |
| DELETE account controller | 15 | `src/core/controller/account/` |
| DELETE oca account | 3 | `src/core/controller/ocaAccount/` |
| DELETE account UI | 5 | `webview-ui/.../account/` |
| REFACTOR | 5 | `api/index.ts`, `providers.json`, `ApiOptions.tsx`, `providerUtils.ts`, `protobus-services.ts` |
| **Total files removed** | **~123** | |
