# 任务二三 · Agent 架构调研与设计说明

> 本文回答三个问题：
> ① 市面上主流 agent（Codex、Claude Code、DeepSeek Harness）到底是怎么做的；
> ② 它们的共性设计模式是什么；
> ③ 本项目「识车档案 CarDex」的任务二、任务三**吸收了什么、简化了什么、为什么**。

---

## 第一部分：调研方法与证据等级

| 对象 | 证据来源 | 等级 |
| --- | --- | --- |
| **DeepSeek Harness** | 本机安装的 DSH Desktop 2.0.2，`resources/app.asar.unpacked/node_modules/@deepseek-ai/` 下 **199 个包**，其中约 150 个带中文 README | **一手**（源码 + 官方文档） |
| **OpenAI Codex** | 本机安装的 Codex Desktop 0.147.0-alpha.6.5 产生的运行产物：`~/.codex/sessions/**/rollout-*.jsonl`、`models.json`、`logs_2.sqlite`（tracing span 与出站请求体） | **一手**（运行产物），标注 `[实测]` |
| **Claude Code / Agent SDK** | `code.claude.com/docs`（官方文档，可访问）+ AWS Bedrock 官方文档中的 Anthropic Messages API 参数 | **官方文档**，`platform.claude.com` / `github.com` 在本网络不可达 |

**证据等级说明**：本报告中 DSH 的结论来自源码与官方 README（最可靠）；Codex 的结论来自本机实测产物，
少数来自搜索引擎摘要的 PR/Issue 编号，已标注；Claude Code 的结论来自官方文档。
**凡是查不到的，下文都明确写"未证实"，不做推测性陈述。**

---

## 第二部分：三家 harness 的架构对比

### 2.1 分层结构

| 维度 | **DeepSeek Harness** | **OpenAI Codex** | **Claude Code / Agent SDK** |
| --- | --- | --- | --- |
| 形态 | Cordis 插件系统（199 个包），Electron/TUI/Web 都是外壳 | Rust 引擎 + app-server 协议层，TUI/IDE/Desktop 是前端 | CLI 二进制 + SDK 包装（SDK 会 spawn 并监管 `claude` 子进程） |
| 循环所在 | `dsh-agent-loop` —— **整个仓库唯一含具体循环逻辑的包** | `codex-core` 的 `session::{turn,handlers}` | Claude Code 内部（SDK 只消费消息流） |
| 前端如何接 | 订阅事件（`session/event`、`agent/*`） | app-server 协议方法（`thread/start`、`turn/start`、`thread/resume`） | 消费消息流（init / assistant / tool_result / result） |
| 状态存放 | 会话 + 事件日志，持久化是独立插件 | append-only 的 `rollout-*.jsonl` | `~/.claude/projects/<project>/<session-id>.jsonl` |

**共同点 1：分层彻底。** 三家都把「循环」「工具」「会话」「前端」拆成独立层，
前端只订阅事件流。这带来一个直接好处：**刷新页面 ≠ 重启智能体**。

### 2.2 Agent Loop 的终止条件

| harness | 主终止条件 | 硬上限 | 上下文耗尽时 |
| --- | --- | --- | --- |
| DSH | 模型不再产生 `tool_calls` | **"没有内置轮次预算"**（README 明说，需靠 `agent/turn-stopping` 扩展点自己加） | `dsh-compaction` 在 `agent/pre-step` 观测压力后压缩，继续 |
| Codex | 模型不再产生工具调用 | **未查证到固定最大轮数** | 自动摘要后继续（base instructions 写明 "time never runs out"） |
| Claude Code | 响应中**不含 tool call** | `maxTurns`（**只计含工具调用的轮次**）、`maxBudgetUsd`（含子智能体开销） | 自动压缩（先清旧工具输出，再摘要） |

**共同点 2：主终止条件是"模型不再调工具"，不是"模型说完成了"。**
三家都是靠**结构信号**（有没有 tool call）而非自然语言判断结束。

**共同点 3：轮数上限是兜底，不是主要机制。**
DSH 明确说自己"没有内置轮次预算"；Codex 查不到固定上限；
只有 Claude Code 提供 `maxTurns`，且它是**只计工具轮**的——纯聊天不计。

### 2.3 工具机制

| 维度 | DSH | Codex | Claude Code |
| --- | --- | --- | --- |
| 定义格式 | `ToolDefinition` = schema + 必填 `output{schema, render}` + `execute(args, exec)` | `{"type":"function", ...}`（JSON Schema）或 `{"type":"custom"}`（freeform，走 grammar） | `{"type":"custom","name","description","input_schema": <JSON Schema>}` |
| 参数校验 | JSON Schema + 可选 `defineTool()` 类型化辅助 | function 走 JSON Schema；freeform 走 Lark grammar | JSON Schema；属性名须匹配 `^[a-zA-Z0-9_.-]{1,64}$` |
| **执行流水线** | **6 段**：`tools/pre-execute`（可重排的门禁）→ 单调 `guard` → `tools/execute`（超时/重试环绕）→ 工具主体 → `tools/post-execute` → `finalizeContent` → `tools/result`（只观测） | `exec_policy` 判定（审批 + 沙箱）后执行 | hooks：`PreToolUse` → 规则 → 模式 → `PostToolUse`/`PostToolUseFailure`/`PostToolBatch` |
| 结果格式 | `{isError:false, value, content, meta?, additionalContexts?}` / `{isError:true, error:{message, info?}, ...}` | `function_call_output` / `custom_tool_call_output` | `{content:[blocks], structuredContent?, isError?}` + `tool_use_id` |
| 并发 | `isConcurrencySafe(args)` 精确分类；独占调用成屏障，并行安全调用走有界滚动池（默认 10），**但策略/持久结果/结果上下文仍保持模型顺序** | `supports_parallel_tool_calls=true` | 只读工具可并发，写类串行 |
| 输出截断 | —— | `truncation_policy={"mode":"tokens","limit":10000}` | 官方工程博客：默认约 25000 token |
| 取消 | 协作式 `AbortSignal`；未派发的调用收到合成 `ABORTED_BEFORE_DISPATCH` 结果 | 有 `op.dispatch.interrupt` span | —— |

**共同点 4：工具执行不是一行 `handler(**args)`，而是一条可观测、可干预的流水线。**
DSH 的 6 段流水线是最完整的一手证据。每一段都有明确职责，
且**失败一律作为结构化结果回填**，而不是抛异常中断循环。

**共同点 5：工具结果有明确的成功/失败结构，且失败必须让模型看得懂。**
三家都有 `isError` / `ok:false` 之类的字段，配错误信息，让模型据此改道。

### 2.4 权限与沙箱

| harness | 机制 |
| --- | --- |
| DSH | `tools/pre-execute` 提供可扩展的**拒绝/询问**门禁；`ctx.tools.guard()` 提供**单调**拥有方策略（后续监听器不能把拒绝改回允许）；`dsh-sandbox` / `dsh-sandbox-policy` / `dsh-permission-presets` / `dsh-plan-mode` |
| Codex | 每回合快照 `approval_policy` + `sandbox_policy{writable_roots, network_access}` + `permission_profile`；**升级走工具参数**（`sandbox_permissions:"require_escalated"` + `justification` + `prefix_rule`），不是全局开关 |
| Claude Code | **固定 6 步漏斗**：hooks → deny 规则 → ask 规则 → permission mode → allow 规则 → `canUseTool` 回调。**hook 的 allow 不能绕过 deny**；规则写成 `Bash(npm test *)`、`Edit(//secrets/**)` |

**共同点 6：权限是分层的、单调的、可审计的。**
"单调"指：一旦被拒绝，后置环节不能翻案（DSH 的 guard 明确写"后续 waterfall 监听器无法将守卫的拒绝重新变为允许"）。

### 2.5 上下文管理

| 维度 | DSH | Codex | Claude Code |
| --- | --- | --- | --- |
| 压缩 | `dsh-compaction` + `dsh-compaction-basic` + `dsh-compaction-tool-result-*`，在 `agent/pre-step` 观测压力 | 自动摘要，且"仍能看到全部历史用户请求" | **先丢弃较旧的工具输出，再摘要对话**；保留最近 5 个文件；反复压缩仍填满则**报错停止**（防 thrashing） |
| 动态上下文 | `systemPrompt.context()` 贡献的**有序动态上下文**，会变成模型历史里带来源的 runtime-context 快照 | user 消息里的 `<environment_context>`（cwd/shell/日期/时区/权限清单） | system prompt 里的环境上下文段 |
| 项目记忆 | `dsh-skill-filesystem` + `dsh-agent-instructions` | `AGENTS.md`（`world_state.state.agents_md`） | `CLAUDE.md` / `CLAUDE.local.md`，**从根到 cwd 全部拼接**，每次请求注入 |
| 子智能体 | `dsh-subagent` + `dsh-subagent-*`，**在循环之外**作为服务实现，用 `ctx.agents.create()` | `/root` 团队 agent，4 个并发槽位，`fork_turns` 语义 | `Agent` 工具，**独立上下文窗口**，只回传最终文本（官方量化约 1–2k token） |

**共同点 7：子智能体首先是一种上下文压缩手段。**
Claude Code 官方量化：子智能体自己烧几万 token，回传给父级只有 1–2k。

### 2.6 系统提示词的组织

| harness | 组织方式 |
| --- | --- |
| **DSH** | `dsh-system-prompt` 注册表：**有序段** + 工具 schema + 具名变量。段顺序区间有约定：`-100` harness 身份、`0` 部署 persona、`100–199` 工具引导。支持 `complete:true` 段（成为唯一完整提示词）。变量 `{{name}}` 严格插值（未知引用直接抛错）。**工具 schema 是组装结果的一部分**——"模型获知自己能做什么是一个连贯整体" |
| Codex | 三层：模型级 `base_instructions`（本机 17730 字符）+ developer 层（应用能力、多 agent 说明）+ user 层 `<environment_context>`；项目级 `AGENTS.md`；技能 `<skills>` + 每技能一份 `SKILL.md`（渐进披露） |
| Claude Code | 三种起点：不设 → 极简默认；`preset:"claude_code"` → 完整提示；或完全自定义。官方建议取"正确的海拔"——既不要硬编码 if-else，也不要过度笼统 |

**共同点 8：提示词是"组装"出来的，不是拼字符串。**
三家都有明确的**分层/分段**结构与**动态上下文**机制。

---

## 第三部分：本项目吸收了什么

对照任务书第五节的要求，逐条说明设计映射。

### 3.1 吸收的模式

| # | 借鉴的模式 | 本项目落地 | 对应任务书要求 |
| --- | --- | --- | --- |
| 1 | **只有循环是逻辑，其余是环节** | `agent/loop.py` 是唯一的循环实现；权限、门槛、记账、会话状态全部通过注册表的钩子与事件挂载 | 5.1 完整流程 |
| 2 | **工具执行流水线**（DSH 6 段 → 本项目 7 步） | `agent/tools/registry.py`：查表 → 解析 JSON → Schema 校验 → 前置门禁 → 执行（超时+取消）→ 后置处理 → 写调用记录 | 5.2 工具设计 |
| 3 | **失败结构化回填，不伪造成功** | 每个失败返回 `{ok:false, error:{code, message, hint}}`，错误码见 `agent/errors.py`（20 个） | 5.2 失败处理 / 5.5 异常反馈 |
| 4 | **工具定义 = schema + 执行体 + 策略** | `ToolDefinition` 含 `parameters`（JSON Schema）、`timeout_s`、`concurrency_safe`、`requires_image`、`tags` | 5.2 明确名称/用途/参数/返回/失败 |
| 5 | **主终止条件 = 没有 tool_calls** | `loop._run_steps()` 中 `if not calls: return` | 5.1 完整流程 |
| 6 | **轮数上限只作兜底** | `max_rounds=8`、`max_tool_calls=16`，并在超限时**明确报错而非假装完成** | 5.1 |
| 7 | **同一工具重复失败要熔断** | 同工具 + 同错误码 + 同参数连续 3 次 → `MAX_ROUNDS_EXCEEDED` 并给出提示 | 5.5 异常处理 |
| 8 | **工具输出截断** | 超过 6000 字符截断，并在截断处写明"请用更精确参数重试" | 工程健壮性 |
| 9 | **会话 = append-only 事件日志** | `agent/session.py` 每会话一个 JSONL，事件类型：`session_meta` / `turn_context` / `user_message` / `assistant_message` / `tool_result` / `error` / `turn_end`；支持重放恢复 | 5.4 多轮对话 |
| 10 | **一次 turn 一个策略快照** | `turn_context` 事件记录本轮的图片、阈值、工具清单、离线标记 | 5.5 阈值由验证集确定 |
| 11 | **提示词按段组装 + 动态上下文** | `agent/prompts.py`：`-100` 身份 / `0` 人设 / `100-120` 业务规则；`RuntimeContext` 每轮求值（日期、模型、阈值、会话状态） | 5.5 低置信度处理 |
| 12 | **前置门禁（可扩展的 allow/deny）** | `registry.add_gate()`：需要图片的工具在会话没有图片时**直接拒绝**，不把无意义调用送进模型 | 5.5 异常处理 |
| 13 | **工具调用记录** | 每次调用写 `agent/data/tool_calls.jsonl`（会话、轮次、调用 ID、工具、参数、结果、耗时、错误码） | 3.2「提供工具调用记录」 |
| 14 | **结构化决策建议** | `classify_car` 返回 `decision.action` = `accept` / `confirm_with_user` / `report_unknown` + 对应指引 | 5.5 unknown 与低置信度 |
| 15 | **业务工具与视觉得分离开** | 8 个工具分三类：视觉 1 个、知识库 2 个、档案 5 个 | 5.2 至少 3 个工具 |

### 3.2 明确**没有**吸收的部分（及原因）

| 未吸收 | 原因 |
| --- | --- |
| 子智能体 / Task 工具 | 本任务书未要求；8 个工具的任务规模不需要上下文隔离 |
| 上下文自动压缩 | 单次会话消息量很小（< 20 条），压缩带来的复杂度大于收益。**但截断与熔断都做了** |
| 沙箱 / 审批弹窗 | 本项目工具的副作用仅限本机 SQLite 与读图片，无 shell / 文件写入。**门禁机制保留了**（`add_gate`），只是默认策略宽松 |
| fork / rewind / checkpoint | 超出任务书范围；会话重放已实现，fork 是锦上添花 |
| 多模型路由 / 降级 | 任务书规定模型名固定为 `deepseek-v4-flash` |
| Code Mode（`run_code` 单工具） | DSH 与 Codex 都在往"用代码编排工具"演进，但那是为**几十个工具 + 复杂编排**设计的。本项目 8 个工具，原生 Function Calling 更直观、也更容易向老师解释 |
| MCP 协议 | 超出范围 |

> **关于 Code Mode 的说明**：DSH 的 `dsh-tools` 支持 `mode: native | code | both`，
> Codex 新版也有 `exec`（V8 isolate + `tools.*` 编排）。
> 这是两家都在走的方向，但**本项目刻意选择了 `native`**——
> 任务书要考察的是"模型提出函数名和参数、程序真正执行函数"这条链路，
> 用代码编排会把这层遮蔽掉。

---

## 第四部分：本项目的具体设计

### 4.1 目录与模块

```
agent/
├── config.py     配置：模型名、路径、API Key 来源、各类阈值
├── errors.py     20 个错误码 + AgentError / ToolError
├── schema.py     轻量 JSON Schema 校验器（零额外依赖）
├── llm.py        DeepSeekClient（真实）+ ScriptedClient（离线替身，测试替身）
├── prompts.py    系统提示词分段组装 + 动态运行时上下文
├── session.py    会话状态 + append-only 事件日志 + 重放恢复
├── tools/
│   ├── registry.py  工具注册表 + 7 步执行流水线
│   ├── vision.py    classify_car       ← 任务书要求的视觉分类工具
│   ├── profile.py   query_car_info / list_supported_cars
│   └── garage.py    add_car_record / search_my_records / update_record_note
│                    / delete_car_record / my_garage_stats
└── loop.py       Agent Loop（唯一的循环实现）
```

### 4.2 一轮请求的时序

```
用户输入（可带图片）
  │
  ├─ 写 user_message 事件；记录 turn_context 快照
  │
  └─ 步骤循环（最多 max_rounds=8）
       ├─ 组装：系统提示词（分段+动态上下文） + 历史 + 8 个工具 schema
       ├─ 调用 DeepSeek（model=deepseek-v4-flash, tool_choice=auto）
       ├─ 没有 tool_calls → 返回最终回答，结束
       ├─ 有 tool_calls → 先落库 assistant 消息，再逐个执行
       │    └─ 注册表流水线：查表→解析→校验→门禁→执行→后置→记账
       │         · classify_car 会真实加载 C1+C2 集成模型并前向推理
       │         · 结果吸收进 session（last_prediction / record_count）
       └─ 结果以 role=tool 回填 → 下一轮
  │
  └─ 写 turn_end 事件；返回 TurnResult（含完整工具调用记录）
```

### 4.3 与任务书第五节的逐条对照

| 任务书要求 | 出处 | 实现 |
| --- | --- | --- |
| 统一通过 DeepSeek API，模型名固定 `deepseek-v4-flash` | 5.1 | `config.MODEL_NAME`，`DeepSeekClient` 走 `/chat/completions` |
| 完成工具定义、参数传递、工具执行、结果回传、最终回答 | 5.1 | 见 4.2 时序 |
| 模型不直接接收图片；图片由自训练视觉模型处理 | 5.1 | `classify_car` 的 schema **只有文本参数**；`loop` 只把"已上传图片"这个事实写进动态上下文 |
| **视觉识别必须通过工具调用执行** | 5.1 | 图片路径不进提示词；模型必须调 `classify_car` 才拿到结果。离线替身的规则 0 也是"有图先调 classify_car" |
| 视觉分类工具：真实加载并调用本组训练的模型，返回类别和 confidence | 5.2 | `agent/tools/vision.py`，复用任务一的 `CarEnsemble`（C1+C2 集成），返回 car_id / car_name / confidence / top_k / is_unknown |
| 其他工具至少 2 个 | 5.2 | 提供 **7 个**（知识库 2 + 档案 5） |
| 每个工具明确名称、用途、输入参数、返回结果、失败处理 | 5.2 | `registry.describe()` 可导出全部；失败统一为 `{ok:false, error:{code,message,hint}}` |
| 连续调用 ≥3 个工具的完整任务 | 5.3 | 已实测：`classify_car → query_car_info → add_car_record`，一次请求内完成 |
| 多轮对话，沿用上一轮结果 | 5.4 | 会话保存 `last_prediction`；`add_car_record` 的 `car_id` 可省略，自动沿用。已实测"第二轮只说『帮我记录下来』，无需重新上传" |
| 低置信度：展示 Top-K / 请用户确认 / 提示重拍 | 5.5 | `classify_car` 返回 `low_confidence` 与 `top_candidates`，`decision.action = confirm_with_user` 并给出指引；提示词有专门一段约束模型行为 |
| 阈值须通过验证集确定并在报告中说明 | 5.5 | 「拒识阈值 t=0.25」由 `src/calibrate.py` 扫 204 组选出；「低置信度提示阈值 c=0.70」由 `tools/derive_confidence_threshold.py` 做选择性预测分析得到（≥0.70 的预测准确率 98.5%，低于它只有 74.9%） |
| unknown 时告知"未识别为本组支持的车型"，不编造、不继续业务 | 5.5 | 工具返回 `is_unknown` + `decision.action=report_unknown`；`add_car_record` 会**主动拒绝**写入 unknown；提示词有专门一段 |
| 处理图片读取失败、API 失败、工具失败 | 5.5 | `IMAGE_NOT_FOUND` / `IMAGE_UNREADABLE` / `MODEL_NOT_LOADED` / `API_REQUEST_FAILED` / `API_KEY_MISSING` / `TOOL_TIMEOUT` / `TOOL_FAILED` 等 20 个错误码 |

---

## 第五部分：两条工程决策的说明

### 5.1 为什么要有「离线替身」（ScriptedClient）

`DEEPSEEK_API_KEY` 未配置时，系统**不会崩溃**，而是自动切到离线替身模式：

- 用规则模拟模型的工具调用决策，**驱动完全相同的 Agent Loop**；
- 因此可以在**没有 Key、没有网络**的环境下验证：工具注册、参数校验、
  执行流水线、错误回传、多轮会话、Web 前端时间线是否全部正常；
- **它不是语言模型**，产出的文字只是"能跑通"的证据，**不能当作模型效果**。
  代码里每个替身回复都带「（离线演示模式：文字由脚本化替身生成。）」标注，
  UI 上也会显示醒目提示。

这是**测试替身（test double）**的常规做法，目的是把"环境依赖"与"代码正确性"解耦。

### 5.2 为什么把「决策建议」写进工具返回值

任务书 5.5 节要求模型在 unknown / 低置信度时采取特定行为。
有两种实现方式：

| 方案 | 做法 | 问题 |
| --- | --- | --- |
| A | 全部写在系统提示词里，指望模型每次都遵守 | 提示词很长时容易被忽略；行为不可验证 |
| B | **工具直接把"该走哪个分支"作为结构化字段返回** | —— |

本项目采用 B（并保留提示词里的对应段落作为双保险）：

```json
{
  "ok": true,
  "car_id": "unknown",
  "is_unknown": true,
  "decision": {
    "action": "report_unknown",
    "guidance": "模型判定该车不属于本组支持的 50 款车型。请如实告知用户…"
  }
}
```

好处：① 分支逻辑由**代码**决定，是确定的、可测试的；
② 提示词可以更短；③ 报告里能直接引用工具返回值证明"确实做了低置信度处理"。

---

## 第六部分：一手证据摘录

### 6.1 DSH 的工具执行流水线（`dsh-tools/README.zh.md` 原文）

> 工具插件注册各自的 schema 和执行器；agent loop 依次让每次调用经过
> `tools/pre-execute`（可扩展的允许／拒绝门禁）→ 已注册的单调守卫 →
> `tools/execute`（供超时／重试／指标插件使用的环绕分发包装层）→
> `tools/post-execute`（检查／替换结果、附加上下文）→ 由工具定义持有的
> `finalizeContent` 边界 → 仅观测的 `tools/result` 通知。

### 6.2 DSH 关于「轮次预算」的原话

> **没有内置轮次预算**：工具调用或 steering 会让当前轮次继续；
> 限制失控轮次的策略必须从既有生命周期扩展点（如 `agent/turn-stopping`）执行取消。

### 6.3 Codex 的 rollout 记录类型（本机实测）

| 记录类型 | 关键字段 |
| --- | --- |
| `session_meta` | `session_id, cwd, originator, base_instructions, dynamic_tools, context_window` |
| `turn_context` | `turn_id, cwd, approval_policy, sandbox_policy, permission_profile, model` |
| `world_state` | `agents_md, environments, current_date, timezone, filesystem` |
| `response_item` | `payload.type=message`（role: developer/user/assistant） |
| `event_msg` | `task_started / task_complete / user_message` |

（本项目的 `session.py` 事件类型就是参照这个结构设计的。）

### 6.4 Claude Code 的权限评估顺序（官方文档）

> `1 hooks → 2 deny 规则 → 3 ask 规则 → 4 permission mode → 5 allow 规则 → 6 canUseTool 回调`
>
> 关键细节：hook 的 `allow` **不能**跳过 deny/ask 规则；`bypassPermissions` 也挡不住 deny 规则。

---

## 附录：调研产出物清单

| 文件 | 内容 |
| --- | --- |
| 本文 | 三家对比 + 本项目映射 |
| `E:\Hanako\OH-WorkSpace\codex-agent-research.md` | Codex 调研完整报告（子代理产出） |
| `E:\Hanako\OH-WorkSpace\codex-exec-doc.txt` | Codex `exec` 工具的原始描述 |
| `E:\Hanako\OH-WorkSpace\codex-request-body.txt` | Codex 出站请求体结构摘录 |
| `E:\Hanako\OH-WorkSpace\research-raw/` | Claude Code 文档抓取原文与关键词扫描结果 |

> Codex 与 Claude 的原始调研产物由子代理写入共享工作区，**不属于本项目提交内容**，
> 仅作为调研过程的证据留存。
