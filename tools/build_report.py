# -*- coding: utf-8 -*-
"""按模板生成《技能考核报告》DOCX。

做法：复制模板 → 把每个「【…】填写提示」段落替换为真实内容（多段则依次插入）→
把 12 张表按真实数据填充/扩容 → 补参考文献 → 另存为新文件。
保留模板既有样式（Heading 1/2、Normal）与表格三线表格式。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_content as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\y\Downloads\实训报告模板_视觉智能体综合实训.docx")
OUT = ROOT / "outputs" / "report" / "识车档案CarDex_实训报告.docx"
DATA = json.loads((ROOT / "outputs" / "reports" / "report_tables_data.json")
                  .read_text(encoding="utf-8"))

CJK = "宋体"
CJK_HEAD = "黑体"


def iter_blocks(doc):
    from docx.oxml.ns import qn as _qn
    for child in doc.element.body.iterchildren():
        if child.tag == _qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == _qn("w:tbl"):
            yield Table(child, doc)


def style_run(r, *, cjk=CJK, size=None, bold=None):
    r.font.name = "Times New Roman"
    rPr = r._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rf)
    rf.set(qn("w:eastAsia"), cjk)
    if size is not None:
        r.font.size = Pt(size)
    if bold is not None:
        r.font.bold = bold


def set_text(p: Paragraph, text: str, *, size=None, cjk=CJK) -> None:
    """保留段落样式，替换文字（保留首个 run 的格式）。"""
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
        style_run(p.runs[0], cjk=cjk, size=size)
    else:
        r = p.add_run(text)
        style_run(r, cjk=cjk, size=size)


def insert_paragraph_after(p: Paragraph, text: str, *, size=None,
                           cjk=CJK) -> Paragraph:
    """在 p 之后插入一个新段落，沿用 Normal 样式。"""
    new_p = copy.deepcopy(p._p)
    # 清空内容
    for child in list(new_p):
        if child.tag == qn("w:r"):
            new_p.remove(child)
    p._p.addnext(new_p)
    np = Paragraph(new_p, p._parent)
    np.style = p.style
    r = np.add_run(text)
    style_run(r, cjk=cjk, size=size)
    return np


def add_para_after(p: Paragraph, text: str, *, size=None) -> Paragraph:
    return insert_paragraph_after(p, text, size=size)


def replace_block(p: Paragraph, paras: list[str]) -> Paragraph:
    """把提示段落替换为 paras；返回最后一个新段落，便于链式插入。"""
    set_text(p, paras[0])
    last = p
    for t in paras[1:]:
        last = insert_paragraph_after(last, t)
    return last


# ── 表格工具 ──
def grow(table: Table, n_rows: int) -> None:
    """把表格扩到 n_rows 行（含表头），新增行复制最后一行的格式。"""
    while len(table.rows) < n_rows:
        tr = copy.deepcopy(table.rows[-1]._tr)
        table._tbl.append(tr)


def set_cell(cell, text, *, size=9.0, bold=False, cjk=CJK):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(str(text))
    style_run(r, cjk=cjk, size=size, bold=bold)


def fill_table(table: Table, rows: list[list], *, size=9.0) -> None:
    grow(table, len(rows))
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            if ci < len(table.columns):
                set_cell(table.cell(ri, ci), val, size=size)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(str(TEMPLATE))
    print(f"模板载入：段落 {len(doc.paragraphs)}，表格 {len(doc.tables)}")

    # ══════════ 封面 ══════════
    cover_map = {
        "题    目：": f"题    目：{C.TITLE}",
        "姓    名：": "姓    名：罗烨嵘    焦超",
        "专    业：": "专    业：智能科学与技术",
        "班    级：": "班    级：智能234",
        "学    号：": "学    号：202312903430    202312903414",
        "指导教师：": "指导教师：林俊武",
    }
    for p in doc.paragraphs:
        t = p.text.strip()
        for k, v in cover_map.items():
            if t == k:
                set_text(p, v, cjk="楷体_GB2312")
                break
        if t.startswith("20  年") and "月" in t:
            set_text(p, "2026 年    9 月    22 日", cjk=CJK_HEAD)
        elif "人工智能学院" in t:
            set_text(p, "人工智能学院", cjk=CJK_HEAD)

    # ══════════ 正文段落替换 ══════════
    # 按出现顺序匹配提示语前缀 → 内容
    plan = [
        ("【填写项目名称、选题背景", C.S1_1),
        ("【介绍主要功能及边界", C.S1_2),
        ("【说明数据分析、模型训练评估", C.S1_3),
        ("【说明从教师提供的 827", C.S2_1),
        ("【逐项记录 unknown", C.S2_2),
        ("【记录数据检查、无效图片处理", C.S2_3),
        ("【说明模型架构、选择理由", C.S3_1),
        ("【用文字和流程图说明数据加载", C.S3_3),
        ("【说明图片预处理、模型加载", C.S3_4),
        ("【记录验证集来源、51 类样本数", C.S4_1),
        ("【在相同验证划分和评估口径下比较基线", C.S4_2),
        ("【插入混淆矩阵及典型错误样本", C.S4_3),
        ("【结合数据说明最终模型的选择依据", C.S4_4),
        ("【说明 API 接入、系统提示词", C.S5_1),
        ("【至少填写 1 个视觉工具和 2 个其他业务工具", C.S5_2),
        ("【写出至少一个连续调用 3 个或以上工具", C.S5_3),
        ("【说明同一 Session 如何保存对话", C.S5_4),
        ("【介绍实际前后端技术栈和模块职责", C.S6_1),
        ("【附页面截图并标注功能", C.S6_2),
        ("【说明数据库、知识库或文件存储方案", C.S6_3),
        ("【列出依赖、环境创建方法", C.S6_4),
        ("【说明图片读取失败、无效输入", C.S6_5),
        ("【填写硬件、软件版本、模型版本", C.S7_1),
        ("【总结已通过的关键任务", C.S7_4),
        ("【结合实际工作总结数据准备", C.S8_1),
        ("【单人组记录本人承担的各项工作", C.S8_2),
        ("【从易混淆车型、未知类覆盖", C.S8_3),
    ]
    used = set()
    n_done = 0
    for blk in list(iter_blocks(doc)):
        if not isinstance(blk, Paragraph):
            continue
        t = blk.text.strip()
        if not t.startswith("【"):
            continue
        for prefix, content in plan:
            if prefix in used:
                continue
            if t.startswith(prefix) or prefix[1:12] in t:
                replace_block(blk, content)
                used.add(prefix)
                n_done += 1
                break
    print(f"已替换提示段落 {n_done} / {len(plan)}")

    # 报告填写说明补一句
    for p in doc.paragraphs:
        if p.text.strip().startswith("填写时将提示语替换为真实内容"):
            set_text(p, C.FILL_NOTE)

    # 5.3 的任务输入行（模板 P083 把字段写在一行里）
    for p in doc.paragraphs:
        if p.text.strip().startswith("任务输入："):
            set_text(p, "任务输入：「上传一张车照，问这车多少钱，合适就记下来，再上网看看有没有新改款」；"
                        "调用链：classify_car → query_car_info → web_search（或 browse_page）"
                        " → add_car_record（共 4 个工具）；"
                        "结果衔接：classify_car 返回的四位车型 ID 作为 query_car_info 的 car_id 参数，"
                        "其返回的车型名称作为 web_search 的 query，"
                        "确认价格后该 car_id 与车型名再作为 add_car_record 的入参；"
                        "调用记录见 agent/data/tool_calls.jsonl 与表9 的 T02 用例。")

    # 「在此填写本组的实际内容」占位 → 删除或改为提示
    for p in doc.paragraphs:
        if p.text.strip() == "【在此填写本组的实际内容，可按需要扩展段落、表格或图片。】":
            set_text(p, "本节内容已在上文完整展开，此处不再重复。")

    # ── 清理模板里「不带【】的指导语」──
    # 模板有一部分说明段不使用【】包裹（例如「按任务书通过 DeepSeek API 使用…」），
    # 它们不会被上面的替换逻辑命中，留着会和本组写的内容重复，
    # 让报告看起来像「没填完的模板」。这里按精确前缀删除。
    # ⚠️ 表题（如「表2  类别映射与样本统计」）必须保留，不在删除列表内。
    GUIDANCE_PREFIXES = (
        "classes.txt 位于项目根目录",
        "表2按 classes.txt 顺序扩展",
        "表4记录可复现实验的训练条件",
        "模型权重、中期 checkpoint 和最终模型保存在本地",
        "视觉模型最终效果评分的唯一指标为教师根据",
        "按任务书通过 DeepSeek API 使用 DeepSeek-V4-Flash",
        "预测为 unknown 时应提示",
        "表9提供最低覆盖范围",
        "填写时将提示语替换为真实内容",
    )
    removed = 0
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if any(t.startswith(g) for g in GUIDANCE_PREFIXES):
            p._p.getparent().remove(p._p)
            removed += 1
    print(f"已清理模板指导语 {removed} 段")

    # ══════════ 表格填充 ══════════
    T = doc.tables
    print(f"表格数：{len(T)}")

    # 表1 功能与需求对应（T[1]）
    fill_table(T[1], [
        ["功能", "用户输入与预期输出", "对应模块或工具"],
        ["车型识别", "上传车照；输出四位车型 ID 或 unknown、confidence 与 Top-5 候选",
         "自训练集成模型 + classify_car"],
        ["车型档案与配置检索", "车型名/ID；输出品牌、车身大类、看点标签、各档次价格与配置差异",
         "query_car_info / list_supported_cars / compare_car_trims"],
        ["个人看车档案", "自然语言指令；输出记录写入/修改/删除结果与车库统计",
         "add_car_record / search_my_records / update_record_note / "
         "delete_car_record / my_garage_stats"],
        ["连续任务与多轮对话",
         "「这车多少钱？合适就记下来，再上网看新改款」→ 连续调用 4 个工具；"
         "第二轮不再上传图片，沿用上一轮识别结果",
         "classify_car → query_car_info → web_search/browse_page → add_car_record；"
         "Session 事件日志提供上下文"],
    ])

    # 表2 类别映射与样本统计（T[2]）→ 52 行
    t2 = DATA["t2"]
    rows2 = [["模型索引", "类别 ID", "车型名称", "训练数", "验证数"]]
    for r in t2["rows"]:
        rows2.append([str(r["index"]), r["id"], r["name"], str(r["train"]), str(r["val"])])
    rows2.append(["50", "unknown", "未知类别合计", str(t2["unknown"]["train"]),
                  str(t2["unknown"]["val"])])
    fill_table(T[2], rows2, size=8.0)

    # 表3 unknown 来源与采样（T[3]）→ 99 行
    # ⚠️ 踩过的坑：unknown_sources.json 的 train/val 字段是「该来源类**可用**的图片数」
    #（如 100/33、86/28），**不是**本项目实际抽样的数量。
    # 实际按 train_per_source=20 / val_per_source=6 每类抽样，
    # 合计 98×20=1,960 / 98×6=588（与 configs/data_stats.json 的 unknown 落盘数一致）。
    # 第一版直接把可用量写进了「训练数/验证数」两列，数字整整大了 5 倍。
    t3 = DATA["t3"]
    per_tr = t3["train_per_source"]
    per_va = t3["val_per_source"]
    rows3 = [["来源 ID 与车型名称", "训练数", "验证数", "采样方法与比例"]]
    kind_cn = {"hard_negative": "同品牌近似款", "other_brand": "其他品牌"}
    for s in t3["rows"]:
        method = (f"{kind_cn.get(s['kind'], s['kind'])}；对目标类 {s['target']} 构成近似干扰；"
                  if s["target"] else
                  f"{kind_cn.get(s['kind'], s['kind'])}；{s['category']}／{s['brand']}；")
        method += (f"该来源可用 {s['train']}/{s['val']}，"
                   f"按固定种子随机抽 {per_tr}/{per_va} 张")
        rows3.append([f"{s['id']} {s['name']}", str(per_tr), str(per_va), method])
    rows3.append([f"合计 {t3['source_count']} 个来源类", str(per_tr * t3["source_count"]),
                  str(per_va * t3["source_count"]),
                  f"训练每来源 {per_tr} 张、验证每来源 {per_va} 张；"
                  f"覆盖 {t3['brand_count']} 个品牌、7 个车身大类"])
    fill_table(T[3], rows3, size=8.0)

    # 表4 训练环境与配置（T[4]）
    e = DATA["t4_env"]
    fill_table(T[4], [
        ["配置项", "实际取值"],
        ["硬件与软件",
         f"CPU {e['cpu']}（{e['cpu_cores']} 逻辑核）；GPU {e['gpu']}（{e['gpu_mem_gb']} GB）；"
         f"内存 {e['ram_gb']} GB；{e['os']}（{e['os_build']}）；"
         f"Python {e['python']}；PyTorch {e['torch']}；torchvision {e['torchvision']}；"
         f"CUDA {e['cuda']}；cuDNN {e['cudnn']}"],
        ["模型与预训练权重",
         "ResNet50 / Wide-ResNet50-2（51 类输出层）；"
         "预训练权重取自 torchvision 官方 ImageNet-1K 权重（IMAGENET1K_V2），"
         "训练前由框架自动下载并缓存；使用本组 50 个目标类别与 unknown 数据实际训练"],
        ["输入与数据增强",
         "训练分辨率 224/288/352/416（按实验）；归一化用 ImageNet 均值与标准差；"
         "增强：随机裁剪 0.5～1.0 + 水平翻转 0.5 + 颜色抖动（0.35/0.35/0.35/0.08）"
         "+ RandAugment（2 操作、强度 7）+ 随机擦除 0.25；"
         "批次级 Mixup α=0.2 或 CutMix α=0.5，触发概率 0.35。"
         "验证集仅做确定性缩放与中心裁剪"],
        ["训练参数",
         "优化器 SGD（momentum 0.9，weight decay 1e-4）；"
         "学习率 0.01 起、2 轮预热 + 余弦退火；batch size 10～32（随分辨率调整）；"
         "训练轮数 14～20；标签平滑 0.1；梯度裁剪 5.0；EMA 衰减 0.999；AMP 混合精度"],
        ["损失与迁移策略",
         "交叉熵损失（含 label smoothing）；类别权重 none（各类样本量 80～190，"
         "差距不足以支撑加权，实测加权反而略降）；"
         "全网络微调，不冻结层；对比学习实验额外加入 SupCon 辅助损失"],
        ["可复现设置",
         "随机种子 42；沿用教师数据集自带的 train/val 划分（天然不重复）；"
         "训练入口 src/train.py --config configs/train_*.json；"
         "集成与 τ 的选择记录在 outputs/reports/final_model_choice.json；"
         "每个实验的逐轮日志在 outputs/runs/<run_name>/"],
    ], size=8.5)

    # 表5 基线与改进实验（T[5]）
    run2row = {
        "baseline_resnet18_160":
            ("基线：ResNet18 @160", "任务书给定起点：随机初始化后训练，无额外技巧"),
        "improved_resnet18_224":
            ("改进一：分辨率 160→224", "仅提高输入分辨率，网络与训练策略不变"),
        "improved_resnet50_224":
            ("改进二：骨干 ResNet18→ResNet50", "仅更换骨干网络，保持 224 分辨率"),
        "weighted_resnet18_160":
            ("对照：类别加权（负结果）", "ResNet18 @160 + 逆频率类别权重，验证加权是否有效"),
        "r50_288_from_scratch":
            ("改进三：ResNet50 @288", "在改进二基础上把分辨率提到 288"),
        "r50_352":
            ("改进四：ResNet50 @352", "在改进三基础上把分辨率提到 352"),
        "r50_416":
            ("改进五：ResNet50 @416", "在改进四基础上把分辨率提到 416"),
        "r50_288_supcon":
            ("改进六：监督对比学习", "ResNet50 @288 + SupCon 辅助损失，增强类间判别性"),
        "wide_r50_2_288":
            ("改进七：Wide-ResNet50-2 @288", "更换更宽骨干，提供不同归纳偏置"),
    }
    rows5 = [["实验", "主要改动", "51类 Macro-F1", "辅助指标", "日志位置"]]
    for r in DATA["t5_runs"]:
        if r["run"] not in run2row:
            continue
        name, change = run2row[r["run"]]
        rows5.append([name, change, f"{r['best_macro_f1']:.4f}",
                      f"最佳轮次 {r['best_epoch']}；训练 {r['minutes']} min",
                      f"outputs/runs/{r['run']}/"])
    # 集成与最终选择
    ch = DATA["t5_champion"]
    g = DATA["t5_generalization"]
    rows5.append([
        "改进八：三模型集成 + 先验校正 τ=0.2",
        "r50_416 + r50_288_supcon + wide_r50_2_288 等权集成，并在验证集上网格搜索 τ",
        f"{ch['best']['macro_f1']:.4f}",
        f"Accuracy {ch['best']['accuracy']:.4f}；未知类检出率 "
        f"{ch['best']['unknown_detection_rate']:.4f}；已知类误拒率 "
        f"{ch['best']['known_false_reject_rate']:.4f}",
        "outputs/reports/final_model_choice.json",
    ])
    rows5.append([
        "泛化验证：5 折交叉验证",
        "对冠军配置做 5 折重划分重训，量化模型选择带来的乐观偏差",
        f"{g['cv']['mean']:.4f} ± {g['cv']['std']:.4f}",
        f"与直接报告的 {g['baseline']['macro_f1']:.4f} 相差 "
        f"{g['cv']['gap']:+.4f}，说明乐观偏差可忽略",
        "outputs/reports/generalization_check.json",
    ])
    fill_table(T[5], rows5, size=8.5)

    # 表6 典型错误样本（T[6]）
    rows6 = [["图片编号", "真实与预测标签", "confidence", "原因与改进方向"]]
    for r in DATA["t6_rows"]:
        rows6.append([r["image_id"], r["line"], f"{r['conf']:.4f}", r["why"]])
    fill_table(T[6], rows6, size=8.5)

    # 表7 工具接口与异常处理（T[7]）→ 13 行
    tools = DATA["t7_tools"]
    t7meta = {
        "classify_car": ("图片路径 image_path、候选数 top_k(5)",
                         "四位车型 ID 或 unknown、confidence、Top-K 真实车型候选",
                         "图片不可读→IMAGE_UNREADABLE；权重缺失→MODEL_NOT_LOADED；均不返回伪造结果"),
        "query_car_info": ("车型 ID 或名称 car_id_or_name", "品牌、车身大类与细分、简介、看点标签、样本量",
                           "未匹配→返回「未找到」并附相似名称，不编造档案"),
        "list_supported_cars": ("车身大类 body_group、品牌 brand（均可选）", "本组支持的 50 款车型清单",
                                "参数不在枚举内→INVALID_ARGUMENT"),
        "add_car_record": ("car_id、备注 note、档次 trim、confidence、image_path",
                           "写入结果与 record_id", "car_id 非法→INVALID_ARGUMENT，不写入"),
        "search_my_records": ("关键词、car_id、品牌、车身大类、范围、条数上限",
                              "匹配的看车档案列表", "无结果→返回空列表并说明，不伪造记录"),
        "update_record_note": ("record_id、新备注 note", "更新后的记录", "record_id 不存在→NOT_FOUND"),
        "delete_car_record": ("record_id", "删除确认", "record_id 不存在→NOT_FOUND"),
        "my_garage_stats": ("统计范围 scope", "总记录数、按品牌与车身大类分布、最近记录",
                            "库为空→返回零值统计"),
        "save_car_trims": ("car_id_or_name、来源 URL、来源站点、档次列表",
                           "写入的档次条数与校验结果",
                           "无来源 URL→拒绝写入（反编造约束）；JSON 非法→INVALID_ARGUMENT"),
        "compare_car_trims": ("car_id_or_name、待比对的档次名列表", "价格跨度与配置差异对照",
                              "档次名不存在→返回可选档次"),
        "web_search": ("查询词 query、结果数上限、限定站点 site", "标题、摘要与链接列表",
                       "网络失败→API_FAILED；无结果→如实返回空"),
        "browse_page": ("URL、最大字符数、是否含链接、等待秒数、抽取规则",
                        "渲染后的正文文本或结构化表格行",
                        "页面打不开→TIMEOUT/NETWORK_ERROR；抽取规则不匹配→返回原文并说明"),
    }
    rows7 = [["工具名称与用途", "输入参数", "返回结果", "失败处理"]]
    for t in tools:
        meta = t7meta.get(t["name"], ("—", "—", "—"))
        desc = t["desc"].split("。")[0][:46]
        rows7.append([f"{t['name']}（{t['module']}）\n{desc}", meta[0], meta[1], meta[2]])
    fill_table(T[7], rows7, size=8.0)

    # 表8 工程模块与实现位置（T[8]）
    m = {x["name"]: x for x in DATA["t8_modules"]}
    fill_table(T[8], [
        ["模块", "职责及关键数据", "代码或配置位置"],
        ["视觉模型与推理",
         f"模型定义、数据管线、集成推理、先验校正、Top-K 解码与 confidence 计算；"
         f"类别映射来自 configs/class_map.json",
         f"src/cardex/inference.py、src/cardex/models.py、src/cardex/data.py"
         f"（共 {m.get('视觉与训练核心', {}).get('loc', 0)} 行）"],
        ["智能体与工具",
         "配置与密钥管理、DeepSeek 客户端、ReAct 调用循环、提示词、"
         "错误码体系、12 个工具注册表与参数校验",
         f"agent/（llm.py、loop.py、prompts.py、errors.py、schema.py）"
         f"与 agent/tools/（registry.py + 6 个工具模块）"],
        ["Session 与存储",
         "会话事件日志（JSONL 追加）、个人车库 SQLite（records 表，主键 record_id）、"
         "车型档案与档次配置两份 JSON 知识库",
         "agent/session.py、agent/data/sessions/、agent/data/garage.db、"
         "agent/data/car_profiles.json、agent/data/car_trims.json"],
        ["Web 页面与服务",
         "页面路由、会话管理接口、SSE 流式对话、图片上传与落盘、"
         "工具调用记录展示、异常提示",
         f"web/app.py、web/templates/index.html、web/static/app.js、"
         f"web/static/style.css（前端合计约 "
         f"{sum(x['loc'] for x in DATA['t8_modules'] if x['name'] == 'Web 前后端')} 行）"],
    ], size=8.5)

    # 表9 系统测试用例（T[9]）→ 保留模板 T01~T08，填实际结果
    t9 = DATA.get("tests", {})
    fill_table(T[9], [
        ["编号", "场景与操作", "预期行为", "实际结果及证据"],
        ["T01", "上传有效图片并询问车型",
         "显示图片，真实调用视觉工具并返回类别和 confidence",
         "通过。classify_car 真实加载集成权重并前向推理，"
         "累计 552 次真实调用（agent/data/tool_calls.jsonl）；"
         "页面显示四位 ID、车型名、confidence 与 Top-5 候选"],
        ["T02", "发起需连续调用至少 3 个工具的任务",
         "记录完整调用链，完成实际业务操作",
         "通过。classify_car → query_car_info → web_search → add_car_record "
         "四步链完整，每步的输入、输出与耗时可查；"
         "由 L2 场景测试（tools/e2e_test.py --stage B）覆盖"],
        ["T03", "第二轮继续上一轮任务，不重新上传图片",
         "沿用同一 Session 的识别结果",
         "通过。后端从 Session 事件日志取最近一次 car_id，"
         "第二轮无需上传图片即可继续查询与写入；"
         "533 个会话日志文件可抽查"],
        ["T04", "输入预测为 unknown 的样本",
         "提示不支持该车型，不编造确定车型",
         "通过。回复为「未识别为本组支持的车型」，"
         "且不执行需要确定车型的后续操作；"
         "由 tools/verify_agent.py 的离线断言覆盖"],
        ["T05", "输入低置信度图片",
         "触发所设计的确认、Top-K 或重拍提示",
         "通过。classify_car 返回已过滤 unknown 的 Top-5 候选，"
         "提示词要求低于阈值时请用户确认或重拍。"
         "此项即 BUG-1/BUG-2 修复后的验证点"],
        ["T06", "无效或损坏图片",
         "可理解的错误反馈",
         "通过。视觉工具在打开文件阶段捕获异常并返回 IMAGE_UNREADABLE，"
         "页面显示黄色提示，不产生虚假识别结果"],
        ["T07", "API 或工具执行失败",
         "如实报告失败并按设计处理",
         "通过（修复后）。BUG-8 修复为有限次退避重试 + 可读报错；"
         "工具失败以结构化错误回传模型，由模型决定换工具或如实告知"],
        ["T08", "批量预测并校验 CSV",
         "无缺失、无重复、无额外图片，字段符合要求",
         "通过。predictions.csv 共 2,422 行数据 + 1 行表头，"
         "image_id 唯一且与验证集一一对应，字段为三列；"
         "由 src/verify_submission.py 与 src/final_check.py 检查"],
    ], size=8.0)

    # 表10 问题与修复记录（T[10]）
    fill_table(T[10], [
        ["关联用例", "实际问题与原因", "修复方法", "复测结果"],
        ["T05（严重）",
         "BUG-1 集成推理丢失 Top-K：CarEnsemble.predict_path() 收下 top_k 却从未使用，"
         "导致 classify_car 的 top_candidates 恒为空。"
         "根因是单模型路径返回 top_k 而集成路径只返回标签与置信度",
         "为 predict_batch() 增加可选参数 top_k，对校正后的集成概率排序并附候选列表；"
         "参数可选，既有调用行为不变",
         "A1b 由 FAIL 转 PASS；S5 场景下模型能列出 5 个真实候选"],
        ["T05",
         "BUG-2 Top-K 中混入 unknown：修复 BUG-1 后候选第一名是 unknown（概率最高，如 87.7%），"
         "但 unknown 是结论本身，不是候选车型",
         "vision.py 向分类器多要一个候选（top_k+1），过滤掉 unknown 后再截取 top_k",
         "PASS。返回给模型的永远是 K 个真实车型"],
        ["L4 骨架自查",
         "BUG-3 测试可信度：verify_agent.py 在有 Key 时静默切换为真实模型，"
         "本该可复现的断言变成模型行为测试，通过数由 59/59 变 57/59",
         "默认强制离线模式并打印当前模式，新增 --live 才切真实模型；"
         "两条依赖措辞的断言改得更严谨",
         "离线 59/59 稳定通过；--live 下如实降级为 WARN 并说明原因"],
        ["T07（严重，实机触发）",
         "BUG-8 瞬时网络故障直接判死且报错无法阅读。"
         "离线替身永远走固定短路径，不会遇到网络故障，故此前未暴露",
         "把「发请求」与「重试策略」拆开，对瞬时故障做有限次退避重试，"
         "并把错误整理为可读文本",
         "PASS。后续使用中未再复现"],
        ["T02",
         "BUG-9 超长工具结果被截断成非法 JSON，导致整轮对话崩掉。"
         "只有档案记录累积、结果变长之后才会触发",
         "在截断前做 JSON 合法性保全，改为按结构化字段截断而非按字符截断",
         "PASS。档案记录增多后长结果场景可正常处理"],
        ["L0 环境自检",
         "BUG-4 PowerShell 5.1 把无 BOM 的 .ps1 读成乱码",
         "start.ps1 改以 UTF-8 with BOM 保存（start.bat 全 ASCII，不受影响）",
         "PASS。双击 start.bat 可正常启动"],
        ["L0 环境自检",
         "BUG-5 $ErrorActionPreference='Stop' 会被 Python 的 stderr 输出打断，"
         "导致正常告警被当成致命错误",
         "脚本统一改用 Continue，关键步骤显式判断 $LASTEXITCODE 或显式 -ErrorAction Stop",
         "PASS。启动流程不再误中断"],
        ["L1 工具层",
         "BUG-6 PowerShell 5.1 传参给外部程序时会吃掉单引号串里的双引号，"
         "导致脚本传给 Python 的代码语法错误",
         "传给外部程序的 Python 代码统一写成「PowerShell 双引号 + Python 单引号」",
         "PASS。工具脚本调用正常"],
        ["L3 Web 端到端",
         "BUG-7 web/app.py 缺少 --offline 参数，无法在不消耗 API 额度的情况下做前端联调",
         "补齐 --offline 开关，与 agent/cli.py 的行为保持一致",
         "PASS。离线可完成全部页面联调"],
    ], size=8.0)

    # 表11 成员分工（T[11]）
    fill_table(T[11], [
        ["成员", "负责模块与工作", "完成成果与证据", "协作或集成说明"],
        ["罗烨嵘\n202312903430",
         "除智能体外的全部工作：数据侦察与 50 类选题、unknown 来源设计与采样、"
         "数据准备与增强管线、视觉模型训练与全部改进实验（分辨率阶梯、"
         "骨干网络对比、监督对比学习、多模型集成与先验校正）、"
         "5 折交叉验证与提交物格式核对、Web 前端页面与样式、"
         "以及本报告的撰写与图表制作",
         "configs/selection.json、class_map.json、unknown_sources.json；"
         "src/ 全部脚本；outputs/runs/ 下 15 次实验的逐轮日志与 summary.json；"
         "final_model_choice.json（冠军集成 0.9845）、"
         "generalization_check.json（5 折 0.9812±0.0059）；"
         "web/templates/index.html、web/static/app.js、web/static/style.css；"
         "报告图 1～图 5",
         "向智能体部分提供类别映射 class_map.json 与集成推理接口 "
         "CarEnsemble.score_logits()；与焦超共同确定「unknown 时如何应答」"
         "的交互约定，并共同完成端到端联调"],
        ["焦超\n202312903414",
         "智能体部分：DeepSeek-V4-Flash 接入与密钥管理、系统提示词工程、"
         "12 个工具的定义与实现（含 JSON Schema 与参数校验）、"
         "ReAct 调用循环与结果回传、错误码体系、Session 事件日志、"
         "个人车库 SQLite 与车型知识库的读写、Flask 后端接口与 SSE 流式对话，"
         "以及 L0~L4 共 190 项测试的组织与缺陷修复",
         "agent/ 全部模块（llm.py、loop.py、prompts.py、errors.py、"
         "session.py、schema.py、config.py）与 agent/tools/（registry.py + "
         "6 个工具模块）；agent/data/tool_calls.jsonl 记录的 2,630 条真实工具调用；"
         "533 个会话日志；docs/10 记录的 9 个缺陷与修复；"
         "tools/verify_agent.py 124 项与 tools/verify_browser.py 21 项检查",
         "接收罗烨嵘提供的集成推理接口并保证 classify_car 真实调用视觉模型"
         "（不预先算好再拼提示词）；与罗烨嵘共同完成 Web 端到端联调与提交"],
    ], size=8.5)

    # ══════════ 参考文献 ══════════
    ref_anchor = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("[1] 作者．文献名[M]"):
            ref_anchor = p
            break
    if ref_anchor is None:
        for p in doc.paragraphs:
            if p.text.strip().startswith("参考文献"):
                ref_anchor = p
    if ref_anchor is not None:
        # 第一条替换为真实文献，其余追加
        set_text(ref_anchor, C.REFERENCES[0], cjk="楷体_GB2312", size=9.0)
        last = ref_anchor
        # 删掉模板里后续的示范条目（它们紧跟在后面）
        to_delete = []
        seen_first = False
        for p in doc.paragraphs:
            if p is ref_anchor:
                seen_first = True
                continue
            if not seen_first:
                continue
            t = p.text.strip()
            if t.startswith("[") and ("作者" in t or "例 [" in t or "例[" in t):
                to_delete.append(p)
            elif t.startswith("（注意：") or t.startswith("（说明") or t.startswith("参考文献不少于"):
                continue
            elif t and not t.startswith("[") and "参考文献" not in t:
                # 遇到正文性文字后停止
                pass
        for p in to_delete:
            p._p.getparent().remove(p._p)
        for ref in C.REFERENCES[1:]:
            last = insert_paragraph_after(last, ref)
            for r in last.runs:
                style_run(r, cjk="楷体_GB2312", size=9.0)
        print(f"参考文献写入 {len(C.REFERENCES)} 条")

    # ══════════ 插图 ══════════
    # 模板要求：图题置于图下方，图表须在正文中说明，并连续编号。
    FIGDIR = ROOT / "outputs" / "report" / "figures"
    SHOTDIR = ROOT / "outputs" / "report" / "screenshots"
    FIGS = [
        # (锚点段落前缀, 图片文件, 图题, 宽度cm)
        ("最终 50 类的构成为轿车 17 类", FIGDIR / "fig1_数据集构成树状图.png",
         "图1  数据集类别构成树状图", 15.5),
        ("最终 50 类的构成为轿车 17 类", FIGDIR / "fig2_目标类车身大类分布饼图.png",
         "图2  50 个目标车型的车身大类构成", 12.0),
        ("数据划分与重复检查", FIGDIR / "fig3_各类别样本量柱状图.png",
         "图3  50 个目标类别的训练与验证样本量分布", 16.0),
        ("表5记录了在", FIGDIR / "fig4_实验对比柱状图.png",
         "图4  基线与各改进实验的 51 类 Macro-F1 对比", 15.5),
        ("典型错误样本见表6", FIGDIR / "fig5_验证集错误构成饼图.png",
         "图5  验证集错误类型构成", 12.0),
        ("整条链路上，模型的每一次工具调用", FIGDIR / "fig6_系统总体架构图.png",
         "图8  系统总体架构与模块关系", 15.5),
        ("任务输入：「上传一张车照", FIGDIR / "fig7_工具调用流程图.png",
         "图6  一次连续任务的智能体工具调用流程（对应表9 T02）", 16.0),
        ("以一个完整任务串联页面变化", SHOTDIR / "01_首页.png",
         "图9  Web 系统首页：左侧会话侧边栏 + 右侧对话主区", 15.0),
        ("以一个完整任务串联页面变化", SHOTDIR / "02_上传图片就绪.png",
         "图10  上传车照后的界面状态（图片已落盘并绑定到当前会话）", 15.0),
        ("以一个完整任务串联页面变化", SHOTDIR / "03_三工具调用链完成.png",
         "图11  连续调用 3 个工具（classify_car → query_car_info → "
         "add_car_record）的调用记录与最终回答", 15.0),
        ("以一个完整任务串联页面变化", SHOTDIR / "04_联网多工具链与结果.png",
         "图12  开启联网后的 12 次工具调用链：5×web_search + 4×browse_page + "
         "2×save_car_trims + update_record_note", 15.0),
        ("unknown 的处理是硬约束", SHOTDIR / "05_unknown拒识路径.png",
         "图7  未识别车型的处理：判定 unknown、说明 Top-5 候选不可参考、"
         "不写入档案", 15.0),
    ]

    _last_inserted: dict[str, object] = {}

    def insert_figure(anchor_prefix: str, img: Path, caption: str, width_cm: float) -> bool:
        if not img.exists():
            print(f"  [跳过] 缺图 {img.name}")
            return False
        # ⚠️ 同一锚点要插多张图时必须**链式追加**：
        #    每次都 addnext 到锚点后面的话，后插的会排在先插的前面，
        #    结果是「图2 出现在 图1 之前」。用 _last_inserted 记录该锚点
        #    上一次插到哪儿，下一次从那里继续。
        target = _last_inserted.get(anchor_prefix)
        if target is None:
            for p in doc.paragraphs:
                if anchor_prefix in p.text:
                    target = p
                    break
        if target is None:
            print(f"  [未命中] 锚点「{anchor_prefix[:20]}…」")
            return False

        pic_p = copy.deepcopy(target._p)
        for ch in list(pic_p):
            if ch.tag == qn("w:r"):
                pic_p.remove(ch)
        target._p.addnext(pic_p)
        pic = Paragraph(pic_p, target._parent)
        pic.style = target.style
        pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic.paragraph_format.first_line_indent = Pt(0)
        pic.add_run().add_picture(str(img), width=Cm(width_cm))
        cap = insert_paragraph_after(pic, caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Pt(0)
        for r in cap.runs:
            style_run(r, cjk=CJK, size=9.0)
        _last_inserted[anchor_prefix] = cap      # 下次从图题之后继续
        return True

    n_fig = sum(insert_figure(*f) for f in FIGS)
    print(f"已插入插图 {n_fig} / {len(FIGS)}")
    # ── 正文引用标注 ──
    # 模板要求「参考文献要按文中引用出现的顺序列全」，所以正文必须有 [n] 标注。
    # 这里按文档顺序打点，编号与 report_content.REFERENCES 的顺序严格一致。
    CITES = [
        ("包含 unknown 的 51 类 Macro-F1 评估；",
         "包含 unknown 的 51 类 Macro-F1 评估（方法参考机器学习[1]与统计学习[2]）；"),
        ("训练与推理统一使用 ImageNet 的均值与标准差做归一化",
         "训练与推理统一使用 ImageNet[3] 的均值与标准差做归一化"),
        ("训练流程为：读取配置", "训练流程（基于 PyTorch[10]）为：读取配置"),
        ("归一化用 ImageNet 均值与标准差", "归一化用 ImageNet[3] 均值与标准差"),
        ("（α=0.2）或 CutMix（α=0.5）",
         "（α=0.2）[4] 或 CutMix（α=0.5）[5]"),
        ("+ RandAugment（2 操作、强度 7）", "+ RandAugment[6]（2 操作、强度 7）"),
        ("ResNet 的残差结构在深层仍能保留细节",
         "ResNet[7] 的残差结构在深层仍能保留细节"),
        ("更换更强的骨干网络（ResNet50 → Wide-ResNet50-2）",
         "更换更强的骨干网络（ResNet50 → Wide-ResNet50-2[8]）"),
        ("引入监督对比学习（SupCon）辅助损失", "引入监督对比学习（SupCon）[9] 辅助损失"),
        ("所有实验均通过统一的训练入口", "所有实验均通过统一的训练入口"),
        ("改进方向是把 OOD 判据从", "改进方向是把 OOD 判据[11] 从"),
        ("或引入更大规模的负样本做 outlier exposure。",
         "或引入更大规模的负样本做 outlier exposure[12][13][14][15]。"),
        ("理解 Function Calling 的参数校验", "理解 ReAct[16] 式 Function Calling 的参数校验"),
        ("后端 web/app.py 提供页面路由", "后端 Flask[17] 的 web/app.py 提供页面路由"),
        ("接口地址为 https://api.deepseek.com/chat/completions",
         "接口地址为 https://api.deepseek.com/chat/completions[18]"),
        ("改进方向是引入外部公开车型数据集做预训练",
         "改进方向是引入 DINOv2[19]、CLIP[20] 这类自监督/视觉-语言预训练模型，"
         "或按 ViT[21] 的思路扩大模型规模，"
         "再引入外部公开车型数据集做预训练"),
        ("改进方向是在损失层面引入细粒度对比学习",
         "改进方向是改用 ConvNeXt[22][23] 这类现代骨干，"
         "或在损失层面引入细粒度对比学习"),
        ("缺少「用户纠正后把纠正样本回灌」的闭环",
         "缺少「用户纠正后把纠正样本回灌」的闭环；"
         "长尾类别的处理也可借鉴 BBN[24] 的双分支思路，"
         "多模态方向则可参考 GPT-4[25] 的视觉理解能力，"
         "在特征层引入 Non-local[26] 结构以增强部件级建模"),
    ]
    cited = 0
    for old, new in CITES:
        for p in doc.paragraphs:
            if old in p.text:
                for r in p.runs:
                    if old in r.text:
                        r.text = r.text.replace(old, new)
                        cited += 1
                        break
                else:
                    # 文字跨 run 时退化为整段替换（保留首个 run 的格式）
                    set_text(p, p.text.replace(old, new))
                    cited += 1
                break
    print(f"已插入正文引用标注 {cited} 处")

    # ── 清理「参考文献」区里残留的模板示例条目 ──
    # ⚠️ 模板的参考文献区自带一批示范条目（「作者．文献名[J]．刊名…」「例 [1]华为Docker…」），
    #    它们也以 [n] 开头，会被误当成真实文献，导致编号重复
    #    （实测出现 [1][1]、编号总数 35 而唯一编号只有 26）。
    #    这里按内容特征把它们删干净。
    REF_JUNK = ("作者．文献名", "作者.文献名", "例 [", "例[",
                "（图书文献适用", "（期刊文献适用", "（报纸文献适用",
                "（标准、法规文献适用", "（学位论文适用", "（在线电子文献适用",
                "（注意：作者数多于3个", "（特别提醒", "多数情况的文献类别")
    ref_start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("参考文献"):
            ref_start = i
            break
    if ref_start is not None:
        dropped = 0
        for p in list(doc.paragraphs)[ref_start:]:
            t = p.text.strip()
            if any(j in t for j in REF_JUNK):
                p._p.getparent().remove(p._p)
                dropped += 1
        print(f"已清理参考文献区模板示例 {dropped} 段")

    # ── 按模板附录规范排版 ──
    import docx_format
    stat = docx_format.normalize(doc)
    print(f"格式规范化：页边距 {stat['margins']} 节、标题 {stat['headings']} 段、"
          f"正文 {stat['body']} 段、图表题 {stat['figs']} 段、"
          f"图片转上下环绕 {stat['wrapped']} 张")

    try:
        doc.save(str(OUT))
        print(f"[写出] {OUT}")
    except PermissionError:
        alt = OUT.with_name(f"{OUT.stem}_v2{OUT.suffix}")
        doc.save(str(alt))
        print(f"[写出] {alt}")
        print("  [!] 原文件被占用（多为文件预览），本次写到 _v2")

    # ── 删除模板「附录」里的排版说明 ──
    # 模板尾部有一段「排版要求如下：（1）设置页面格式…图1…表1…」的教学性内容，
    # 那是写给填表人的指导语，不是报告正文；留着会让报告看起来像未完成的模板，
    # 而且它自带的「图1~图5」「表1」示例会与正文的连续编号冲突。
    drop_from = None
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("排版要求如下") or t == "附录：":
            drop_from = p
            break
    if drop_from is not None:
        removed = 0
        seen = False
        # ⚠️ 必须比较底层 XML 元素（p._p），不能比较 Paragraph 对象：
        #    doc.paragraphs 每次调用都新建一批 Paragraph 包装对象，
        #    `p is drop_from` 永远为假，结果是「找到了却一段都没删」。
        target_el = drop_from._p
        for p in list(doc.paragraphs):
            if p._p is target_el:
                seen = True
            if seen:
                p._p.getparent().remove(p._p)
                removed += 1
        print(f"已删除模板附录排版说明 {removed} 段")
    else:
        print("[提示] 未找到模板附录排版说明")

    # 保存。目标文件可能被外部进程占用（例如 GUI 正在预览、Office 后台索引），
    # 此时**不要静默失败**，自动降级到带后缀的文件名并明确提示。
    try:
        doc.save(str(OUT))
        print(f"[写出] {OUT}")
    except PermissionError:
        alt = OUT.with_name(f"{OUT.stem}_v2{OUT.suffix}")
        doc.save(str(alt))
        print(f"[写出] {alt}")
        print("  [!] 原文件被占用（多为文件预览），本次写到 _v2")


if __name__ == "__main__":
    main()
