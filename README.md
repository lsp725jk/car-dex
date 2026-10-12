# 识车档案 CarDex —— 私人车型资料库

> 视觉智能体综合实训 · 项目选题方向：**个人知识库（私人车型资料库 / 汽车知识检索 / 对话式看车档案）**

「识车档案」是面向私人用户的车型资料库：用户上传一张车照，系统用**本组自训练的视觉模型**
识别车型，再由 DeepSeek-V4-Flash 智能体调用工具完成车型档案查询、看车记录写入与个人车库检索，
形成一份可检索、可积累的「对话式看车档案」。

本仓库完成 **任务一（数据分析与视觉模型训练）**、**任务二（智能体与工具开发）**
与 **任务三（Web 系统开发与集成）**。

---

## 0. 快速开始

### 方式一：双击启动（推荐）

```
双击 start.bat
```

它会自动完成：找 Python → 确认 API Key（没有会当场提示粘贴一次，写进已被 gitignore 忽略的
`agent/.env.local`）→ 环境自检 → 起服务 → 打开浏览器。

需要加参数时用 `start.ps1`（`start.bat` 会把参数原样转过去）：

```powershell
.\start.ps1                          # 起 Web 系统（默认，→ http://127.0.0.1:5000）
.\start.ps1 -Mode check -DeepCheck   # 只做全量环境自检（含真实 API + 真实视觉推理实测）
.\start.ps1 -Mode cli                # 命令行智能体
.\start.ps1 -Port 8000 -NoBrowser    # 换端口、不开浏览器
.\start.ps1 -Offline                 # 无网/无 Key 时演示（页面会标注「离线替身」）
```

> `start.ps1` 必须以 **UTF-8 with BOM** 保存，否则 Windows PowerShell 5.1 会把中文读成乱码。

### 方式二：手动命令

```bash
pip install -r requirements.txt

# （可选）配置 DeepSeek API Key —— 不配也能跑，会进入离线替身模式
copy agent\.env.example agent\.env.local      # 编辑后填入自己的 Key

# 构建车型档案库（首次需要）
python tools/build_car_profiles.py

# 启动 Web 系统（任务三）
python web/app.py          # → http://127.0.0.1:5000
```

命令行调试（任务二）：

```bash
python -m agent.cli --tools
python -m agent.cli --image data/val/0975/0975_0007.jpg \
       --once "这是什么车？介绍一下，然后加进我的收藏"
```

### 启动前 / 提交前的六道检查

```bash
python tools/selfcheck.py          # 41 项：依赖、交付文件、权重、工具装配、API + 网络 + 视觉实测
python tools/e2e_test.py           # 61 项：工具层 20 + 真实模型端到端 41（含联网）
python tools/verify_agent.py       # 115 项：任务书第五、六节逐条 + 界面 / 浏览器 / 档次库
python tools/verify_browser.py     # 21 项：真实浏览器抓取链路（需能启动 Chromium）
python tools/verify_trims.py       # 17 项：档次知识库全链路（真实模型 + 真实浏览器 + 反编造）
python src/verify_submission.py    # 41 项：任务一交付物合规
```

实测结果见 [`docs/10_真实API与端到端测试报告.md`](docs/10_真实API与端到端测试报告.md)、
[`docs/11_Web界面与会话管理.md`](docs/11_Web界面与会话管理.md)、
[`docs/12_车型档次知识库.md`](docs/12_车型档次知识库.md)。

### Web 界面能做什么

| 能力 | 位置 | 说明 |
| --- | --- | --- |
| **会话管理** | 左侧可收缩侧边栏 | 列表 / 切换（完整还原历史与工具卡片）/ 重命名 / 删除 / 筛选 |
| **设置** | 侧边栏左下角 **⚙** | API 配置（Key 只写不读、可先测试再保存）+ 模型配置 + 决策阈值 + 联网通道 |
| **联网搜索** | 输入框左侧 **🌐 联网搜索**（与上传图片并列） | 逐会话开关，一并控制下面两个工具 |
| ↳ `web_search` | — | 搜索公网信息（价格 / 新闻 / 口碑） |
| ↳ `browse_page` | — | **真实浏览器**打开网页读渲染后的正文；`extract="tables"` 可结构化抽取配置表 |
| **档次知识库** | 档案库文件 `agent/data/car_trims.json` | 各配置版本的**价格 / 参数 / 档次差异**；联网查到后存进来，以后直接查不必再联网 |
| ↳ `save_car_trims` | — | 把搜到的档次信息存进档案库（**必须带来源 URL**，会核验是否真打开过） |
| ↳ `compare_car_trims` | — | 比出各档次**真正有差别**的参数与"贵在哪" |
| 车型识别 | 输入框左侧 **📎 上传图片** | 走本组自训练的 51 类视觉模型，由智能体自己决定何时调用 |
| 看车档案 | 右栏 | 按品牌 / 车身类型检索，可删记录；可记下"用户看中的是哪一档" |

详见 [`docs/11_Web界面与会话管理.md`](docs/11_Web界面与会话管理.md)
与 [`docs/12_车型档次知识库.md`](docs/12_车型档次知识库.md)。

| 任务 | 主要产物 | 说明文档 |
| --- | --- | --- |
| 任务一 | `classes.txt`、`predictions.csv`、`outputs/checkpoints/*` | `docs/任务一_数据分析与视觉模型训练.md` |
| 任务二 | `agent/`（12 个工具 + Agent Loop） | `docs/任务二三_智能体与Web系统说明.md` |
| 任务三 | `web/`（Flask + SSE 单页界面） | `docs/11_Web界面与会话管理.md` |
| 档次库 | `agent/data/car_trims.json` | `docs/12_车型档次知识库.md` |
| **技术学习** | — | **`docs/13_技术栈全解.md`**（每一项技术/库/算法/协议的详细说明 + 学习路径） |
| **GPU 加速** | — | **`docs/14_GPU加速实战.md`**（设备无关改造 + 12.8× 提速实录 + 踩坑） |
| **微调** | — | **`docs/15_微调实战指南.md`** + `src/finetune.py`（三种冻结模式 / LLRD / 渐进解冻 / 自定义数据） |
| 调研 | — | `docs/任务二三_agent架构调研与设计.md`（Codex / Claude Code / DeepSeek Harness 对比） |
| 测试 | `tools/selfcheck.py`、`tools/e2e_test.py`、`tools/verify_browser.py`、`tools/verify_trims.py` | `docs/10_真实API与端到端测试报告.md` |

---

## 1. 任务一交付物位置

| 交付物 | 路径 | 说明 |
| --- | --- | --- |
| 类别清单 `classes.txt` | `classes.txt` | 项目根目录，恰好 50 行四位车型 ID，升序，UTF-8 无 BOM，不含 unknown |
| 预测结果 `predictions.csv` | `predictions.csv` | 表头 `image_id,predicted_label,confidence`，2,422 条记录 |
| 任务一完整报告 | `docs/任务一_数据分析与视觉模型训练.md` | 数据分析 / 选类理由 / 模型设计 / 实验结果 / 错误分析 / 阈值标定 |
| 任务二三说明 | `docs/任务二三_智能体与Web系统说明.md` | 工具清单、执行流水线、连续调用、多轮对话、低置信度、Web 接口 |
| Agent 架构调研 | `docs/任务二三_agent架构调研与设计.md` | Codex / Claude Code / DSH 对比 + 本项目设计映射 |
| **说明文档集（14 篇 Markdown）** | 从 `docs/00_文档索引.md` 起 | 逐文件说明、逐参数说明、命令手册、指标公式、FAQ 排错、技术栈全解、任务二三交接 |
| **说明文档（HTML 单文件版）** | `docs/识车档案_CarDex_说明文档.html` | 全部文档合并成一页，自带侧边栏导航 + 实时搜索 + 暗色模式，双击即可离线打开 |
| **说明文档（HTML 分篇版）** | `docs/html/index.html` | 每篇一个独立页面、互相跳转，适合日常查阅 |
| 类别选择报告 | `outputs/reports/selection_report.txt` | 选类理由、50 类明细、unknown 来源类别 |
| 数据分析报告 | `outputs/reports/data_analysis.txt` | 类别分布、图片尺寸、重复检测、unknown 构成 |
| 实验对比 | `outputs/reports/experiment_comparison.md` | 各实验指标总表 + 逐轮训练过程 + 决策标定扫描 |
| 最终模型选择 | `outputs/reports/final_model_choice.md` | C1 / C2 / C1+C2 三方比选与冠军逐类 F1 |
| 训练与实验结果 | `outputs/runs/*/` | 日志、逐轮指标、验证报告、混淆矩阵、错误样本、标定结果 |
| 模型权重 | `outputs/checkpoints/*/` | `best.pt` / `last.pt` / `epoch_XX.pt`（**不随作业提交**） |
| 工具调用记录 | `agent/data/tool_calls.jsonl` | 每次工具调用一行（会话/轮次/工具/参数/结果/耗时/错误码） |
| 会话事件日志 | `agent/data/sessions/*.jsonl` | append-only，可重放恢复 |

### 任务一主要结果（验证集 2,422 张，51 类等权 Macro-F1）

| 实验 | 骨干 / 输入 / 配方 | 未标定 | 最优 τ | 拒识阈值 t | **标定后 Macro-F1** | 用时 |
| --- | --- | --- | --- | --- | --- | --- |
| A 基线 | ResNet18 @160，轻量增强 | 0.8475 | 0.00 | 0.40 | 0.8512 | 28 min |
| B 类别平衡 | A + 逆频率类别权重 | 0.8428 | −0.40 | 0.50 | 0.8597 | 32 min |
| C1 改进 | ResNet18 @224 + 强增强 + EMA + MixUp/CutMix + TTA | 0.9482 | +0.10 | 0.20 | 0.9510 | 87 min |
| C2 骨干容量 | ResNet50 @224（同配方） | 0.9550 | −0.20 | 0.20 | 0.9592 | 175 min |
| **C3 集成 ★ 最终交付** | **C1 + C2 软投票集成** | 0.9587 | **+0.05** | **0.25** | **0.9604** | — |

**最终交付模型**：`improved_resnet18_224` + `improved_resnet50_224` 软投票集成 +
水平翻转 TTA + 验证集标定的先验校正/拒识阈值，验证集 **Macro-F1 = 0.9604**（相对基线 **+0.1092**）。
`predictions.csv` 已由该配置生成，用本地真值独立核算得 **Macro-F1 = 0.9604**（`src/score_predictions.py`），
并通过 41 项提交前自查（`src/verify_submission.py`，0 失败）。

---

## 2. 目录结构

```
car-kb-agent/
├── classes.txt                     # ★ 类别清单（项目根目录）
├── predictions.csv                 # ★ 批量推理结果
├── README.md
├── requirements.txt
├── configs/
│   ├── class_info.json             # 教师提供的 827 类类别说明
│   ├── selection.json              # 人工策划的 50 类清单 + 选类原则
│   ├── class_map.json              # 索引 0~49 → 车型 ID，索引 50 → unknown
│   ├── unknown_sources.json        # unknown 样本来源类别清单
│   ├── data_stats.json             # 抽取后的数据统计
│   ├── train_baseline.json         # A 基线训练配置
│   ├── train_weighted.json         # B 对比实验：类别平衡损失
│   ├── train_improved.json         # C1 改进模型配置
│   ├── train_improved_r50.json     # C2 骨干容量对比配置
│   └── train_moderate.json         # 备用中等增强配方（未跑）
├── docs/                           # 报告 + 说明文档集
│   ├── 00_文档索引.md … 09_任务二三交接说明.md
│   ├── 识车档案_CarDex_说明文档.html  # 上述文档合并成的单文件网页
│   └── 任务一_数据分析与视觉模型训练.md  # 实训报告正文
├── tools/
│   └── build_docs_html.py          # 由 Markdown 生成单文件 HTML 文档
├── scripts/
│   └── run_after_c2.ps1            # 一次性运维：等 C2 训练完自动收尾
├── data/                           # 抽取出的图片（按类别目录组织，不提交）
│   ├── train/<四位ID>/   train/unknown/
│   └── val/<四位ID>/     val/unknown/
├── src/
│   ├── cardex/                     # 公共代码包
│   │   ├── paths.py                # 路径与类别映射
│   │   ├── data.py                 # Dataset + 预处理 + 数据增强 + MixUp/CutMix
│   │   ├── models.py               # 模型工厂（ImageNet 预训练权重）
│   │   ├── engine.py               # 训练/验证引擎 + EMA + checkpoint
│   │   ├── metrics.py              # 51 类 Macro-F1 等指标
│   │   ├── inference.py            # 单张/批量推理（任务二复用）
│   │   └── utils.py                # 日志、种子、线程
│   ├── inventory.py                # 步骤1 数据集清点（827 类统计）
│   ├── list_candidates.py          # 步骤1 候选类别明细
│   ├── select_classes.py           # 步骤2 选类 + 生成 classes.txt + unknown 规划
│   ├── prepare_data.py             # 步骤3 从 zip 抽取出训练/验证图片
│   ├── data_analysis.py            # 步骤4 数据集分析
│   ├── train.py                    # 训练入口
│   ├── evaluate.py                 # 验证集评估 + 混淆矩阵 + 错误分析
│   ├── calibrate.py                # 决策标定：先验校正 τ + 拒识阈值 t（验证集扫描）
│   ├── threshold_analysis.py       # 拒识阈值专项扫描
│   ├── visualize_errors.py         # 错误样本并排可视化（训练集 vs 误判验证集）
│   ├── infer.py                    # 单张图片推理
│   ├── predict_batch.py            # 批量推理 → predictions.csv
│   ├── ensemble.py                 # 多模型软投票集成
│   ├── verify_reload.py            # 跨进程验证"模型保存与重新加载"的可复现性
│   ├── score_predictions.py        # 用本地真值核算 predictions.csv 的 Macro-F1
│   ├── make_reports.py             # 汇总实验对比表与图表
│   ├── finalize.py                 # 候选模型/集成比选 + 重生成 predictions.csv
│   ├── final_check.py              # 交付物总览核验（一次性打印全部状态与指标）
│   └── verify_submission.py        # 提交前自查清单自动校验（41 项）
└── outputs/
    ├── checkpoints/<run>/          # best.pt / last.pt / epoch_XX.pt
    ├── runs/<run>/                 # train.log / summary.json / val_report.json / 混淆矩阵 / 标定
    └── reports/                    # 各类分析报告与图表
```

---

## 3. 环境依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.11+（本项目实测 3.13.5 / Anaconda） | |
| PyTorch | 2.13.0（**CPU-only 构建**） | 本项目在 **CPU** 上完成全部训练（未使用 GPU） |
| torchvision | 0.28.0 | 提供 ImageNet 预训练骨干 |
| numpy / scikit-learn / Pillow / matplotlib / tqdm | 见 `requirements.txt` | 指标计算与可视化 |
| markdown / pygments | 可选 | **仅用于生成 HTML 说明文档**，训练推理不需要 |

```bash
pip install -r requirements.txt
pip install markdown pygments        # 可选：只在重新生成 HTML 文档时需要
```

**预训练权重**：本项目只使用 torchvision 官方 ImageNet 预训练权重（`resnet18` / `resnet50`），
不下载任何现成汽车分类模型，也不调用第三方在线图片识别接口。
权重需已缓存于 `%USERPROFILE%\.cache\torch\hub\checkpoints\`：

```
resnet18-f37072fd.pth          <= ResNet18_Weights.IMAGENET1K_V1
resnet50-0676ba61.pth          <= ResNet50_Weights.IMAGENET1K_V1   （注意：不是 V2）
```

> ⚠️ `ResNet50_Weights` 有两个版本；若误用 V2 会触发联网下载 `resnet50-11ad3fa6.pth`，
> 在无网环境下会失败。`src/cardex/models.py` 已固定使用与本地缓存一致的枚举并有注释说明。

---

## 4. 复现步骤

```bash
# 0) 准备教师提供的数据集压缩包，路径见 src/prepare_data.py 的 ZIP_PATH
# 1) 清点 827 类，生成类别统计
python src/inventory.py

# 2) 依据选类原则确定 50 类，生成 classes.txt / class_map.json / unknown_sources.json
python src/select_classes.py

# 3) 从压缩包抽取本组需要的图片到 data/
python src/prepare_data.py            # 加 --force 可清空重抽

# 4) 数据集分析（生成报告与样例图）
python src/data_analysis.py

# 5) 训练（5 组实验）
python src/train.py --config configs/train_baseline.json            # A 基线
python src/train.py --config configs/train_weighted.json            # B 类别平衡
python src/train.py --config configs/train_improved.json            # C1 改进
python src/train.py --config configs/train_improved_r50.json        # C2 骨干容量

# 6) 评估（51 类 Macro-F1 / 混淆矩阵 / 错误样本）
python src/evaluate.py --checkpoint outputs/checkpoints/improved_resnet50_224/best.pt --tta-hflip

# 7) 决策标定：先验校正 τ + 拒识阈值 t（写入 outputs/runs/<run>/calibration.json）
python src/calibrate.py --run improved_resnet50_224 --tta-hflip

# 8) 错误样本可视化（训练集 vs 被误判的验证集，用于分析易混淆原因）
python src/visualize_errors.py --run improved_resnet50_224 --class-id 0975

# 8.5) 最终模型比选：C1 / C2 / C1+C2 集成 → 自动选出冠军并重生成 predictions.csv
python src/finalize.py --candidates improved_resnet18_224 improved_resnet50_224 \
       --ensembles improved_resnet18_224+improved_resnet50_224

# 9) 推理
python src/infer.py --checkpoint outputs/checkpoints/improved_resnet50_224/best.pt --image 图片路径

# 9a) 对教师指定测试目录生成正式 predictions.csv（拿到目录后执行）
python src/predict_batch.py --checkpoint outputs/checkpoints/improved_resnet50_224/best.pt \
       --input-dir <教师指定测试图片目录> --output predictions.csv --tta-hflip

# 9b) 本次交付：测试目录尚未下发，先用本地验证集演练
python src/predict_batch.py --checkpoint outputs/checkpoints/improved_resnet50_224/best.pt \
       --split val --output predictions.csv --tta-hflip

# 10) 汇总与自查
python src/make_reports.py                 # 实验对比表 + 训练曲线 + 混淆矩阵图
python src/score_predictions.py            # 用本地真值核算 predictions.csv 的 Macro-F1
python src/verify_reload.py --checkpoint outputs/checkpoints/improved_resnet50_224/best.pt \
       --n 240 --tta-hflip --out outputs/tmp_a.json   # 跨进程验证模型重加载（跑两次再 --compare）
python src/verify_submission.py            # 提交前 41 项自动校验
python src/final_check.py                  # 交付物总览核验（一次看全部状态）

# 11) （可选）重新生成 HTML 说明文档
pip install markdown pygments              # 仅在需要时安装
python tools/build_docs_html.py            # → docs/识车档案_CarDex_说明文档.html
```

---

## 5. 类别与标签映射约定

* `classes.txt`：恰好 50 行，四位车型 ID，升序，**不含 unknown**。
* 模型输出 51 类：**索引 0~49 按 `classes.txt` 的行顺序**对应 50 个目标车型，**索引 50 固定为 `unknown`**。
* `predictions.csv` 中 `predicted_label` 写四位 ID（保留前导零）或小写 `unknown`。
* 该映射由 `configs/class_map.json` 唯一确定，**训练、保存、Web 推理、批量推理共用同一份**。

---

## 6. 免责与合规

* 类别名称以教师提供的 `class_info.json` 中 `name_from_new` 字段为准。
* `train/unknown/` 与 `val/unknown/` 的样本全部来自**组外类别**，且 train / val 来自压缩包内
  不同的官方划分，互不重复，完全不使用测试图片。
* API Key 通过环境变量或本地 `.env` 管理，**不写入代码、报告或截图**（任务二、任务三涉及）。
