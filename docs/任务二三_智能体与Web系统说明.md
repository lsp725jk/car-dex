# 识车档案 CarDex — 任务二 / 任务三 说明

> 本文档说明**智能体与工具开发**（任务二）与 **Web 系统开发与集成**（任务三）的实现、
> 参数含义、运行方式与验证结果。设计依据见 [`任务二三_agent架构调研与设计.md`](任务二三_agent架构调研与设计.md)。

---

## 一、十秒启动

```bash
# 1) 安装依赖（任务一已装过的可跳过）
pip install -r requirements.txt

# 2)（可选）配置 API Key —— 不配也能跑，会进入离线替身模式
copy agent\.env.example agent\.env.local      # 然后编辑填入自己的 Key

# 3) 构建车型档案库（首次运行需要，约 1 秒）
python tools/build_car_profiles.py

# 4) 启动 Web 系统
python web/app.py
#   → 浏览器打开 http://127.0.0.1:5000
```

**命令行调试**（不启动 Web）：

```bash
python -m agent.cli --image data/val/0975/0975_0007.jpg --once "这是什么车？介绍一下，然后加进我的收藏"
python -m agent.cli --tools          # 看工具清单
python -m agent.cli                  # 交互式对话
```

---

## 二、目录结构

```
agent/                         任务二：智能体与工具
├── __init__.py                包说明（模块划分）
├── config.py                  配置：模型名、路径、API Key 来源、各类阈值
├── errors.py                  20 个错误码 + AgentError / ToolError
├── schema.py                  轻量 JSON Schema 校验器（零额外依赖）
├── llm.py                     DeepSeekClient（真实）+ ScriptedClient（离线替身）
├── prompts.py                 系统提示词分段组装 + 动态运行时上下文
├── session.py                 会话状态 + append-only 事件日志 + 重放恢复
├── loop.py                    Agent Loop（唯一的循环实现）
├── cli.py                     命令行入口
├── .env.example               配置示例（占位符，不含真实 Key）
├── tools/                     工具
│   ├── registry.py            注册表 + 7 步执行流水线
│   ├── vision.py              classify_car
│   ├── profile.py             query_car_info / list_supported_cars
│   └── garage.py              add_car_record / search_my_records /
│                              update_record_note / delete_car_record / my_garage_stats
└── data/                      运行时数据
    ├── car_profiles.json      50 款车型档案（由 tools/build_car_profiles.py 生成）
    ├── confidence_threshold.json  低置信度提示阈值（由验证集推导）
    ├── garage.db              看车档案（SQLite，运行时生成）
    ├── tool_calls.jsonl       工具调用记录（每次调用一行）
    └── sessions/*.jsonl       会话事件日志

web/                           任务三：Web 系统
├── app.py                     Flask 应用（含 SSE 流式接口）
├── templates/index.html       单页界面
└── static/{style.css,app.js}  样式与前端逻辑

data/uploads/                  用户上传的图片
```

---

## 三、任务二：智能体与工具开发

### 3.1 八个工具

| # | 工具 | 类型 | 作用 | 关键参数 |
| --- | --- | --- | --- | --- |
| 1 | `classify_car` | **视觉** | 真实加载 C1+C2 集成模型识别车型 | `image_path?` `top_k?` |
| 2 | `query_car_info` | 知识 | 按 ID 或名称查车型档案 | `car_id_or_name`（必填） |
| 3 | `list_supported_cars` | 知识 | 列出 50 款支持车型，可筛选 | `body_group?` `brand?` |
| 4 | `add_car_record` | 档案 | 写入看车档案（`car_id` 可省略） | `car_id?` `note?` `confidence?` |
| 5 | `search_my_records` | 档案 | 按关键词/ID/品牌/车身类型检索 | `keyword?` `car_id?` `brand?` `body_group?` `scope?` `limit?` |
| 6 | `update_record_note` | 档案 | 修改档案备注 | `record_id`（必填）`note?` |
| 7 | `delete_car_record` | 档案 | 删除档案记录 | `record_id`（必填） |
| 8 | `my_garage_stats` | 档案 | 档案统计与覆盖度 | `scope?` |

**任务书要求"至少 1 个视觉工具 + 至少 2 个其他工具" → 本项目提供 1 + 7 = 8 个。**

用 `python -m agent.cli --tools` 可打印全部工具的完整参数表。

### 3.2 工具执行流水线（7 步）

`agent/tools/registry.py::ToolRegistry.execute()`：

| 步 | 环节 | 失败时的错误码 |
| --- | --- | --- |
| 1 | 查表：`name → ToolDefinition` | `UNKNOWN_TOOL`（并回告可用工具名） |
| 2 | 解析参数：JSON 字符串 → dict | `INVALID_ARGUMENTS` |
| 3 | 校验参数：JSON Schema | `INVALID_ARGUMENTS`（附参数定义） |
| 4 | **前置门禁**（可扩展钩子） | `DENIED_BY_POLICY` |
| 5 | 执行主体（带超时 + 协作式取消） | `TOOL_TIMEOUT` / `TOOL_FAILED` / `ABORTED` |
| 6 | 后置处理（可替换/补充结果） | — |
| 7 | 写 `tool_calls.jsonl` 调用记录 | — |

**失败一律如实上报**，返回结构：

```json
{
  "ok": false,
  "error": {
    "code": "CAR_NOT_FOUND",
    "message": "资料库里没有与「保时捷911」匹配的车型",
    "hint": "本资料库只收录 50 款车型，可用 list_supported_cars 查看全部清单",
    "candidates": [ ... ]        // 有歧义时给候选，便于模型修正
  },
  "retryable": false
}
```

挂载的两个钩子（演示流水线的可扩展性）：

- **前置门禁**：需要图片的工具在会话没有图片时**直接拒绝**，不把无意义调用送进模型；
- **后置处理**：给失败结果统一补 `retryable` 标记（超时/内部错误/N+1 类错误标为可重试）。

### 3.3 连续调用 3 个工具（任务书 5.3）

**实测**：用户上传车照 +「这是什么车？介绍一下，然后加进我的收藏。」

```
classify_car({})                     → 日产_轩逸 / 0975 / confidence 0.9857
   ↓ car_id 作为下一步输入
query_car_info({car_id_or_name:"0975"})  → 品牌=日产，车身=轿车/紧凑型三厢轿车，看点…
   ↓ 同一个 car_id
add_car_record({})                   → 写入档案，记录 #6
   ↓
最终回答：已经把 日产_轩逸 写入你的看车档案…
```

**衔接关系**：`classify_car` 的输出 `car_id` 是 `query_car_info` 的输入；
`query_car_info` 确认车型存在后，`add_car_record` 再用同一个 `car_id` 写入。
**三个工具都由程序按顺序实际执行**，日志见 `agent/data/tool_calls.jsonl`。

### 3.4 多轮对话（任务书 5.4）

**实测**：

| 轮次 | 用户输入 | 是否重新上传图片 | 工具调用 |
| --- | --- | --- | --- |
| 1 | 这是什么车？介绍一下，然后加进我的收藏。 | 是 | classify_car → query_car_info → add_car_record |
| 2 | 帮我记录下来 | **否** | add_car_record（省略 car_id，沿用 `session.last_prediction`） |
| 3 | 我的看车档案里都有什么？ | 否 | my_garage_stats |

**实现要点**：`Agent._absorb()` 在每次工具成功后把结果吸收进会话状态：

```python
if tool_name == "classify_car":
    session.last_prediction = {"car_id":…, "car_name":…, "confidence":…, "is_unknown":…}
elif tool_name == "add_car_record":
    session.record_count = payload["total_records_in_session"]
```

`add_car_record` 的 `car_id` 因此可以省略；`prompts.py` 里也明确告诉模型
「用户第二轮只说『帮我记下来』时，直接调用即可，**不要**要求他重新上传图片」。

**换新图片会重新识别**：会话记录 `last_prediction["image"]`，
若与当前 `session.image_path` 不一致，视为上一次的结论已过期，必须重新调用 `classify_car`。

### 3.5 低置信度与 unknown（任务书 5.5）

本系统有**三个**由验证集确定的数值，用途各不相同：

| 名称 | 取值 | 含义 | 来源 |
| --- | --- | --- | --- |
| 先验校正 τ | **+0.05** | 推理时从 logits 中减去 `τ·log(π)`，调节 unknown 的触发难易 | `src/finalize.py` 在验证集上扫 204 组 (τ,t) 选出 |
| **拒识阈值 t** | **0.25** | `max(softmax) < t` → 改判为 `unknown`。**属于模型决策** | 同上 |
| **低置信度提示阈值 c** | **0.70** | 预测**仍为具体车型**但置信度 < c 时，智能体应展示 Top-K / 请用户确认 / 提示重拍。**只影响措辞，不改变判定** | `tools/derive_confidence_threshold.py` 做选择性预测分析得到 |

**c = 0.70 的推导依据**（验证集 2,422 张，选择性预测分析）：

| 提示阈值 c | 自动接受样本数 | 覆盖率 | **这批样本的准确率** | 剩余样本的准确率 |
| --- | --- | --- | --- | --- |
| 0.60 | 2122 | 87.6% | 97.97% | 69.67% |
| 0.65 | 2062 | 85.1% | 98.35% | 72.22% |
| **0.70** | **2008** | **82.9%** | **98.51%** | **74.88%** |
| 0.75 | 1923 | 79.4% | 99.12% | 76.55% |
| 0.85 | 1671 | 69.0% | 99.40% | 83.49% |

即：**置信度 ≥ 0.70 的预测，98.5% 是对的；低于 0.70 的，只有 74.9% 是对的**。
取 0.70 可以在「自动接受 82.9% 的样本」与「接受样本错误率仅 1.5%」之间取得平衡。

`classify_car` 会把结论编码成结构化建议返回：

```json
{
  "ok": true,
  "car_id": "unknown",
  "is_unknown": true,
  "confidence": 0.42,
  "top_candidates": [ {"car_id":"0768","car_name":"福特_福克斯","confidence":0.21}, … ],
  "decision": {
    "action": "report_unknown",
    "guidance": "模型判定该车不属于本组支持的 50 款车型。请如实告知用户「未识别为本组支持的车型」，不要据此编造具体车型，也不要继续按某个确定车型执行业务操作。可以展示 top_candidates 供参考，并邀请用户手动指定车型或换一张更清晰的照片。"
  }
}
```

三种 `action`：

| action | 触发条件 | 智能体该做什么 |
| --- | --- | --- |
| `accept` | `confidence ≥ 0.70` 且非 unknown | 继续查询档案、写入记录 |
| `confirm_with_user` | `confidence < 0.70` 且非 unknown | 展示 Top-K、请用户确认、提示重拍；**确认前不写入档案** |
| `report_unknown` | `is_unknown = true` | 告知「未识别为本组支持的车型」；**不编造、不继续业务** |

**额外保护**：`add_car_record` 会**主动拒绝**把 unknown 写入档案（错误码 `UNKNOWN_CAR`），
即使模型判断失误也不会污染用户的档案。

### 3.6 异常处理清单

| 场景 | 错误码 | 用户可见的表现 |
| --- | --- | --- |
| 没有上传图片就要求识别 | `DEPENDENCY_MISSING`（门禁直接拒绝） | 「请先上传一张车辆照片」 |
| 上传的不是图片 / 图片损坏 | `IMAGE_UNREADABLE` | 上传时就报错，不会进入对话 |
| 图片文件被删除 | `IMAGE_NOT_FOUND` | 工具卡片显示失败，附修正建议 |
| 视觉模型权重缺失 | `MODEL_NOT_LOADED` | 明确告知期望的路径 |
| 未配置 API Key | `API_KEY_MISSING` | 启动时切离线模式，页面顶部横幅提示 |
| DeepSeek 接口失败 | `API_REQUEST_FAILED`（带 HTTP 状态与对应建议） | 气泡里显示错误卡片 |
| 接口返回被截断 | `API_RESPONSE_INVALID` | 提示缩短回答或提高 max_tokens |
| 模型返回空内容 | `MODEL_REFUSED` | 提示重新发起 |
| 工具执行超时 | `TOOL_TIMEOUT` | 工具卡片标红 + 建议简化请求 |
| 工具内部异常 | `TOOL_FAILED` | 卡片标红，如实显示异常类型 |
| 同一工具连续同样失败 3 次 | 熔断 → `MAX_ROUNDS_EXCEEDED` | 「同一个工具以相同方式连续失败多次，已停止以避免无效循环」 |
| 超过 8 轮 / 16 次工具调用 | `MAX_ROUNDS_EXCEEDED` / `MAX_TOOL_CALLS_EXCEEDED` | 明确报错，**不假装完成** |

---

## 四、任务三：Web 系统

### 4.1 界面构成

| 区域 | 内容 |
| --- | --- |
| 顶部横幅 | 离线替身模式警告（仅在未配置 API Key 时出现） |
| 顶栏 | 品牌、会话 ID、模型名、运行模式、视觉模型、新会话、暗色切换 |
| 左 · 对话区 | 消息气泡 + **工具调用卡片**（可点开看完整参数与返回 JSON）+ 元信息（轮数/工具次数/耗时） |
| 左 · 输入区 | 上传按钮、待上传图片预览（可移除）、输入框、发送 |
| 右 · 运行状态 | 模型、模式、API Key（脱敏）、视觉模型、三个阈值、工具数 |
| 右 · 我的看车档案 | 缩略图、车型名/ID、品牌·车身类型·置信度·时间、备注、删除；支持关键词与范围筛选 |
| 右 · 可用工具 | 8 个工具的名称、用途、参数标签 |

### 4.2 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 页面 |
| GET | `/api/runtime` | 运行状态（模型、模式、阈值、工具数） |
| GET | `/api/tools` | 工具清单（含参数 schema） |
| POST | `/api/session` | 新建会话 |
| GET | `/api/session/<sid>` | 读取会话状态（不存在则创建同 id 的，便于刷新恢复） |
| POST | `/api/upload` | 上传图片（校验扩展名 + 真实可解析 + 12 MB 上限） |
| POST | `/api/chat` | 对话，**SSE 流式**返回事件 |
| GET | `/api/records` | 查询看车档案 |
| DELETE | `/api/records/<id>` | 删除档案记录（限制只能删本会话的） |
| GET | `/api/events/<sid>` | 会话事件时间线（审计/调试） |
| GET | `/uploads/<name>` | 访问已上传图片 |

### 4.3 SSE 事件流

`/api/chat` 逐个推送，前端据此实时渲染：

| 事件 | 载荷 | 前端行为 |
| --- | --- | --- |
| `session` | `session_id` | 更新顶栏会话 chip |
| `turn_start` | `turn`（本轮快照：图片、阈值、工具列表） | 显示「本轮已绑定图片」 |
| `model_request` | `round` `messages` `tools` | 内部记录 |
| `tool_call` | `tool` `arguments` `call_id` | 新增一张"调用中"的工具卡片 |
| `tool_result` | `ok` `code` `elapsed_ms` `summary` `result` | 卡片变为成功/失败，补上摘要、耗时与结果 JSON |
| `assistant_message` | `content` | 显示模型的中间发言 |
| `error` | `code` `message` `info.hint` | 插入红色错误块 |
| `turn_end` / `result` | `reply` `rounds` `tool_calls` `elapsed_ms` | 收尾，显示元信息，刷新档案面板 |

### 4.4 与任务书第六节的逐条对照

| 验收要求 | 实现 | 验证方式 |
| --- | --- | --- |
| 图片上传与显示 | `POST /api/upload` + 输入区预览 + 档案卡片缩略图 | 浏览器实测：上传 667×500 车照成功 |
| 用户输入与聊天 | 输入框 + SSE 流式回复 | 浏览器实测 |
| 视觉识别（智能体调用本组模型） | `classify_car` 工具卡片显示耗时 8.7 s，说明真实前向推理 | 工具卡片 + `tool_calls.jsonl` |
| 多轮对话（同 Session） | 会话 ID 贯穿；`last_prediction` 跨轮保留 | 实测第 2、3 轮无需重新上传 |
| 业务功能（查询/记录/推荐） | 8 个工具：查档案、写档案、检索、统计、清单 | 右侧档案面板同步刷新 |
| 异常反馈 | 上表 12 类场景 | 上传非图片、未上传就识别、unknown 均实测 |
| 模型/智能体/工具/页面连通 | 一条链路：页面 → SSE → Agent Loop → 注册表 → 视觉模型 → SQLite | 端到端实测 |
| 环境依赖与启动步骤 | 见第一节与 `README.md` | — |
| **API Key 用环境变量或本地配置** | `load_config()` 优先读 `DEEPSEEK_API_KEY`，其次 `agent/.env.local`；日志只打印脱敏值 | `agent/.env.example` 用占位符 |
| 保证本地能启动 | `python web/app.py` 一条命令 | 实测启动成功 |

---

## 五、运行状态与限制说明

### 5.1 当前环境没有 API Key

本机未设置 `DEEPSEEK_API_KEY`，因此系统默认运行在**离线替身模式**：

- 页面顶部有醒目横幅；
- 每条替身回复都带「（离线演示模式：文字由脚本化替身生成。）」；
- 消息元信息里标注「离线替身」。

**离线替身（`ScriptedClient`）是什么**：一个**测试替身**，用规则模拟模型的工具调用决策，
驱动**完全相同的 Agent Loop 与工具流水线**。它的用途是让"代码是否正确"与"有没有 Key"
这两件事解耦——没有 Key 也能验证工具注册、参数校验、执行、错误回传、多轮会话、前端渲染。

**它不是语言模型，产出的文字不能当作模型效果。**
配置 Key 后重启即切换为真实 `deepseek-v4-flash`，代码路径完全相同。

### 5.2 视觉模型是真实的

与替身不同，**`classify_car` 在任何模式下都是真的**：
它加载任务一训练出的 `improved_resnet18_224` + `improved_resnet50_224` 集成，
执行真实前向推理（实测单次约 8.7 秒，纯 CPU）。

### 5.3 已知限制

| 限制 | 说明 |
| --- | --- |
| 单机单进程 | 会话锁在进程内；多 worker 部署需要把会话状态外置 |
| 无上下文压缩 | 单会话消息量小（< 20 条），未实现压缩；已实现工具结果截断（6000 字符）与熔断 |
| 车型档案的信息量有限 | 教师数据集只给类别 ID 与名称，没有排量/价格/尺寸，因此档案只包含真实可得的信息，并在 `data_source` 字段注明来源，**不编造参数** |
| 并发 | 同一会话串行执行；不同会话可并发，但视觉模型推理受 CPU 限制 |

---

## 六、验证清单

```bash
# 1) 工具清单（应输出 8 个工具）
python -m agent.cli --tools

# 2) 连续调用三个工具
python -m agent.cli --image data/val/0975/0975_0007.jpg \
       --once "这是什么车？介绍一下，然后加进我的收藏"

# 3) unknown 处理
python -m agent.cli --image data/val/unknown/0022_0022.jpg --once "这是什么车？"

# 4) 工具调用记录（每次调用一行 JSONL）
type agent\data\tool_calls.jsonl

# 5) 会话事件时间线
python -m agent.cli --events <session_id>

# 6) Web 端
python web/app.py     # 打开 http://127.0.0.1:5000
```
