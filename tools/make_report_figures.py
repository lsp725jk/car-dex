# -*- coding: utf-8 -*-
"""生成实训报告用的 7 张图（树状图 / 饼图 / 柱状图 / 流程结构图）。

所有数值均取自 outputs/reports/report_tables_data.json，即项目实测数据。
输出到 outputs/report/figures/。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "outputs" / "reports" / "report_tables_data.json")
                  .read_text(encoding="utf-8"))
OUT = ROOT / "outputs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

C_MAIN = "#2E5C8A"
C_ACC = "#C0504D"
C_GREY = "#7F7F7F"
PALETTE = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6", "#F79646",
           "#2C4D75", "#7F7F7F"]


def save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  [图] {name}")


# ══════════════════════════════════════════════════════════════
# 图1 数据集构成树状图
# ══════════════════════════════════════════════════════════════
def fig1_tree() -> None:
    """数据集类别构成树状图。

    ⚠️ 画树最容易出的问题是「线条乱走」，本函数用三条硬约束避免：
      1. 每个父节点拥有**独占的竖向轨道 x**，轨道必须落在
         「父节点右边界」与「子节点左边界」之间的空隙里 ——
         第一版把 unknown 分支的轨道放在子项起点的右边，
         横线只能往回折，看起来就是线在乱走；
      2. 轨道之间留足间距，避免视觉上并成一条；
      3. 连接一律走「父右 → 轨道 → 子左」三段直角线，不画斜线。
    """
    t2, t3 = DATA["t2"], DATA["t3"]
    groups = Counter(r["group"] for r in t2["rows"])
    kn_tr = sum(r["train"] for r in t2["rows"])
    kn_va = sum(r["val"] for r in t2["rows"])
    # ⚠️ unknown 必须用**实际采样量**（98 类 × 20/6 = 1960/588），
    #    不是 unknown_sources.json 里的「可用量」（合计 9,925/3,280）。
    per_tr, per_va = t3["train_per_source"], t3["val_per_source"]
    n_src = t3["source_count"]
    unk_tr, unk_va = per_tr * n_src, per_va * n_src

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=9.5, fc="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    linewidth=1.5, edgecolor=color, facecolor=fc,
                                    zorder=4))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="#1a1a1a", zorder=5, linespacing=1.6)

    def elbow(x_from, y_from, rail_x, y_to, x_to, color):
        """父右边界 → 竖向轨道 → 子左边界，三段直角线。"""
        ax.plot([x_from, rail_x], [y_from, y_from], color=color,
                linewidth=1.0, zorder=1, solid_capstyle="round")
        ax.plot([rail_x, rail_x], [y_from, y_to], color=color,
                linewidth=1.0, zorder=1, solid_capstyle="round")
        ax.plot([rail_x, x_to], [y_to, y_to], color=color,
                linewidth=1.0, zorder=1, solid_capstyle="round")

    # ── 布局常量（改这里就能整体调整，不要在调用处写魔数）──
    ROOT_X, ROOT_W = 1.55, 2.60          # 根节点：右边界 = 2.85
    L1_X, L1_W = 5.60, 3.40              # 一级节点：左 3.90 / 右 7.30
    L1_RAIL = 3.25                       # 根→一级 的轨道（2.85 与 3.90 之间）
    L2_X = 9.60                          # 二级文字左边界
    RAIL = {0.72: 8.05, 0.42: 8.62}      # 一级 y → 该分支独占轨道（7.30 与 9.60 之间）

    # 根
    box(ROOT_X, 0.50, ROOT_W, 0.20,
        "教师数据集\n827 类 / 103,785 张", C_MAIN, fs=10.5, fc="#EAF0F7")

    # 一级
    lv1 = [
        (0.72, f"本组 50 个目标类（已知）\n{kn_tr} 训练 / {kn_va} 验证",
         "#4F81BD", "#F2F7FC"),
        (0.42, f"unknown 来源类 {n_src} 类\n{unk_tr} 训练 / {unk_va} 验证",
         C_ACC, "#FBF0F0"),
        (0.14, "未使用类别 679 类\n62,543 训练 / 20,669 验证", C_GREY, "#F5F5F5"),
    ]
    for y, txt, col, fc in lv1:
        elbow(ROOT_X + ROOT_W / 2, 0.50, L1_RAIL, y, L1_X - L1_W / 2, col)
        box(L1_X, y, L1_W, 0.155, txt, col, fs=9.5, fc=fc)

    # 二级：6 个车身大类挂在「50 个目标类」下，2 个来源类型挂在 unknown 下
    ax.text(12.85, 0.965, "细分类别数", ha="right", fontsize=8.4, color=C_GREY)
    lv2 = [
        (0.72, 0.905, f"轿车 {groups['轿车']} 类"),
        (0.72, 0.825, f"SUV {groups['SUV']} 类"),
        (0.72, 0.745, f"MPV {groups['MPV']} 类"),
        (0.72, 0.665, f"跑车 {groups['跑车']} 类"),
        (0.72, 0.585, f"微面 {groups['微面']} 类"),
        (0.72, 0.505, f"轻客 {groups['轻客']} 类"),
        (0.42, 0.345, f"同品牌近似款 {t3['hard_negative_count']} 类"),
        (0.42, 0.255, f"其他品牌 {t3['other_brand_count']} 类"),
    ]
    col_of = {y: c for y, _, c, _ in lv1}
    for py, cy, txt in lv2:
        elbow(L1_X + L1_W / 2, py, RAIL[py], cy, L2_X - 0.10, col_of[py])
        ax.text(L2_X, cy, txt, ha="left", va="center", fontsize=9.2, color="#222")

    ax.text(0.15, 0.985, "图1  数据集类别构成树状图", ha="left", fontsize=11.5,
            fontweight="bold", color="#1a1a1a")
    save(fig, "fig1_数据集构成树状图.png")


# ══════════════════════════════════════════════════════════════
# 图2 目标类车身大类分布（饼图）
# ══════════════════════════════════════════════════════════════
def fig2_pie_group() -> None:
    t2 = DATA["t2"]
    g = Counter(r["group"] for r in t2["rows"])
    order = ["SUV", "轿车", "MPV", "跑车", "微面", "轻客"]
    vals = [g[k] for k in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    w, tx = ax.pie(
        vals, labels=[f"{k}\n{v} 类（{v/50:.0%}）" for k, v in zip(order, vals)],
        colors=PALETTE[:len(order)], autopct=None, startangle=95,
        wedgeprops={"edgecolor": "white", "linewidth": 1.6},
        textprops={"fontsize": 10})
    ax.set_title("图2  50 个目标车型的车身大类构成", fontsize=12,
                 fontweight="bold", pad=14)
    ax.text(0, -1.42, f"合计 50 类，覆盖 37 个品牌", ha="center",
            fontsize=9, color="#555")
    save(fig, "fig2_目标类车身大类分布饼图.png")


# ══════════════════════════════════════════════════════════════
# 图3 各类别样本量（柱状图）
# ══════════════════════════════════════════════════════════════
def fig3_bar_samples() -> None:
    rows = DATA["t2"]["rows"]
    labels = [r["id"] for r in rows]
    tr = [r["train"] for r in rows]
    va = [r["val"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.bar(x - 0.21, tr, 0.42, label="训练样本", color="#4F81BD",
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.21, va, 0.42, label="验证样本", color="#C0504D",
           edgecolor="white", linewidth=0.4)
    ax.axhline(np.mean(tr), color="#4F81BD", linestyle="--", linewidth=1.0)
    ax.text(len(rows) - 0.4, np.mean(tr) + 3, f"训练均值 {np.mean(tr):.0f}",
            ha="right", fontsize=8.5, color="#4F81BD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.6)
    ax.set_xlabel("类别 ID（按 classes.txt 行序）", fontsize=10)
    ax.set_ylabel("图片数量（张）", fontsize=10)
    ax.set_title("图3  50 个目标类别的训练与验证样本量分布", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_各类别样本量柱状图.png")


# ══════════════════════════════════════════════════════════════
# 图4 各实验 Macro-F1 对比（柱状图）
# ══════════════════════════════════════════════════════════════
def fig4_bar_experiments() -> None:
    runs = {r["run"]: r for r in DATA["t5_runs"]}
    seq = [
        ("baseline_resnet18_160", "基线\nR18@160"),
        ("improved_resnet18_224", "分辨率\nR18@224"),
        ("improved_resnet50_224", "换骨干\nR50@224"),
        ("weighted_resnet18_160", "类别加权\n(负结果)"),
        ("r50_288_from_scratch", "R50@288"),
        ("r50_352", "R50@352"),
        ("r50_416", "R50@416"),
        ("r50_288_supcon", "R50@288\n+SupCon"),
        ("wide_r50_2_288", "WideR50-2\n@288"),
    ]
    names, vals = [], []
    for k, lab in seq:
        if k in runs:
            names.append(lab)
            vals.append(runs[k]["best_macro_f1"])
    champ = DATA["t5_champion"]["best"]["macro_f1"]
    names.append("三模型集成\n+τ=0.2")
    vals.append(champ)

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    colors = ["#B0B0B0"] + ["#4F81BD"] * 4 + ["#9BBB59"] * 3 + ["#C0504D"] * 2
    bars = ax.bar(np.arange(len(vals)), vals, 0.62, color=colors,
                  edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0016, f"{v:.4f}",
                ha="center", fontsize=7.6, rotation=0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("51 类 Macro-F1（验证集）", fontsize=10)
    ax.set_title("图4  基线与各改进实验的 51 类 Macro-F1 对比", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#B0B0B0", label="基线"),
                       Patch(color="#4F81BD", label="分辨率/骨干改进"),
                       Patch(color="#9BBB59", label="单模型最优"),
                       Patch(color="#C0504D", label="集成/负结果")],
              fontsize=8.4, frameon=False, ncol=4, loc="lower right")
    save(fig, "fig4_实验对比柱状图.png")


# ══════════════════════════════════════════════════════════════
# 图5 验证集错误构成（饼图）
# ══════════════════════════════════════════════════════════════
def fig5_pie_errors() -> None:
    s = DATA["t6_stats"]
    labels = ["未知→已知（漏出）", "已知→unknown（误拒）", "已知→他类（混淆）"]
    vals = [s["n_unknown_leak"], s["n_known_wrong_unknown"], s["n_confusion"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    total = sum(vals)
    ax.pie(vals, labels=[f"{a}\n{b} 张（{b/total:.0%}）" for a, b in zip(labels, vals)],
           colors=["#C0504D", "#4F81BD", "#9BBB59"], startangle=110,
           wedgeprops={"edgecolor": "white", "linewidth": 1.6},
           textprops={"fontsize": 10})
    ax.set_title(f"图5  验证集错误类型构成（共 {total} 张错误）", fontsize=12,
                 fontweight="bold", pad=14)
    ax.text(0, -1.42, f"验证集 {s['n_val']:,} 张，错误率 "
                      f"{total/s['n_val']:.2%}", ha="center", fontsize=9, color="#555")
    save(fig, "fig5_验证集错误构成饼图.png")


# ══════════════════════════════════════════════════════════════
# 图6 系统总体架构（流程结构图）
# ══════════════════════════════════════════════════════════════
def _box(ax, x, y, w, h, text, ec, fc="white", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", zorder=4, linespacing=1.45,
            fontweight="bold" if bold else "normal")


def _arrow(ax, p1, p2, color="#555", style="-|>", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def fig6_arch() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    _box(ax, 5, 9.35, 5.6, 0.85, "浏览器前端\n图片上传 · 自然语言输入 · 流式聊天 · 工具调用记录",
         C_MAIN, "#EAF0F7", fs=9.5, bold=True)
    _box(ax, 5, 7.9, 5.6, 0.85,
         "Flask 服务  web/app.py\n页面路由 · 会话接口 · SSE 流式对话 · 图片落盘",
         "#4F81BD", "#F2F7FC", fs=9.5)
    _box(ax, 5, 6.45, 5.6, 0.85,
         "智能体循环  agent/loop.py\n系统提示词 · DeepSeek-V4-Flash · 工具调度 · 结果回传",
         "#9BBB59", "#F4F8EC", fs=9.5)
    _box(ax, 2.35, 4.75, 3.9, 0.95,
         "视觉工具\nclassify_car\n自训练集成模型",
         C_ACC, "#FBF0F0", fs=9)
    _box(ax, 7.65, 4.75, 3.9, 0.95,
         "业务工具（11 个）\n档案 / 车库 / 配置\n联网搜索 / 网页抓取",
         "#8064A2", "#F5F2F8", fs=9)
    _box(ax, 2.35, 2.9, 3.9, 0.95,
         "视觉模型\noutputs/checkpoints\n三模型集成 + τ=0.2",
         C_ACC, "white", fs=8.6)
    _box(ax, 7.65, 2.9, 3.9, 0.95,
         "存储\nSQLite 车库 · JSON 知识库\nJSONL 会话日志",
         "#8064A2", "white", fs=8.6)
    _box(ax, 5, 1.15, 8.4, 0.85,
         "DeepSeek-V4-Flash（deepseek-v4-flash）——只处理文本与工具调用决策，图片不进入大模型",
         "#F79646", "#FDF6EF", fs=9)

    _arrow(ax, (5, 8.92), (5, 8.36))
    _arrow(ax, (5, 7.47), (5, 6.90))
    _arrow(ax, (3.6, 6.02), (2.35, 5.25))
    _arrow(ax, (6.4, 6.02), (7.65, 5.25))
    _arrow(ax, (2.35, 4.27), (2.35, 3.40))
    _arrow(ax, (7.65, 4.27), (7.65, 3.40))
    _arrow(ax, (3.6, 2.9), (4.6, 1.6), color="#F79646", style="<|-|>")
    _arrow(ax, (6.4, 2.9), (5.4, 1.6), color="#F79646", style="<|-|>")
    ax.text(5, 0.45, "图6  系统总体架构与模块关系", ha="center", fontsize=12,
            fontweight="bold")
    save(fig, "fig6_系统总体架构图.png")


# ══════════════════════════════════════════════════════════════
# 图7 智能体工具调用流程（流程结构图）
# ══════════════════════════════════════════════════════════════
def fig7_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    steps = [
        (1.35, 9.1, "用户输入\n上传车照 +「这车多少钱？\n合适就记下来，再看看新改款」", "#4F81BD", "#F2F7FC"),
        (1.35, 7.35, "① classify_car\n自训练集成模型识别\n→ 0632 奥迪_奥迪A6  conf 91.5%", C_ACC, "#FBF0F0"),
        (1.35, 5.6, "② query_car_info\n查本组车型档案\n→ 奥迪｜轿车", "#9BBB59", "#F4F8EC"),
        (1.35, 3.85, "③ web_search / browse_page\n联网查最新价格与改款\n→ 汽车之家 10 个档次指导价", "#8064A2", "#F5F2F8"),
        (1.35, 2.1, "④ add_car_record\n写入个人看车档案\n→ 记录 #110 与写入确认", "#F79646", "#FDF6EF"),
        (6.4, 2.1, "模型综合工具结果\n生成自然语言回复\n（并说明已写入档案）", C_MAIN, "#EAF0F7"),
    ]
    for x, y, txt, ec, fc in steps:
        _box(ax, x, y, 2.5 if x < 3 else 3.4, 1.15, txt, ec, fc, fs=8.6)

    for i in range(len(steps) - 1):
        x1, y1 = steps[i][0], steps[i][1]
        x2, y2 = steps[i + 1][0], steps[i + 1][1]
        if abs(x1 - x2) < 0.1:
            _arrow(ax, (x1, y1 - 0.58), (x2, y2 + 0.58))
    _arrow(ax, (2.62, 2.1), (4.68, 2.1), color=C_MAIN, lw=1.6)

    # 右侧说明
    ax.text(6.4, 9.1, "调用链要点", ha="center", fontsize=10.5, fontweight="bold")
    notes = [
        "· 共 4 个工具，满足「连续调用 ≥3 个」要求",
        "· 上一步的返回值即下一步的入参：",
        "   ① 的 car_id → ② 的 car_id 参数",
        "   ② 的车型名 → ③ 的 query",
        "   ①③ 的结论 → ④ 的 car_id / trim",
        "· 视觉识别由工具真实调用模型完成，",
        "   不是后台预先算好再拼进提示词",
        "· 每一步的输入/输出/耗时都写入",
        "   agent/data/tool_calls.jsonl，可逐条复核",
    ]
    for i, t in enumerate(notes):
        ax.text(5.15, 8.4 - i * 0.52, t, ha="left", fontsize=8.6, color="#333")
    ax.text(5.0, 0.55, "图7  一次连续任务的智能体工具调用流程", ha="center",
            fontsize=12, fontweight="bold")
    save(fig, "fig7_工具调用流程图.png")


def main() -> None:
    print("生成图表：")
    fig1_tree()
    fig2_pie_group()
    fig3_bar_samples()
    fig4_bar_experiments()
    fig5_pie_errors()
    fig6_arch()
    fig7_flow()
    print(f"\n[输出目录] {OUT}")


if __name__ == "__main__":
    main()
