# -*- coding: utf-8 -*-
"""任务一 - 汇总所有实验，生成对比表格与图表（供实训报告直接引用）

输出：
    outputs/reports/experiment_comparison.md / .csv   各实验关键指标对比
    outputs/reports/training_curves.png               训练曲线（loss / 准确率 / Macro-F1）
    outputs/reports/confusion_matrix_<run>.png        混淆矩阵热力图
    outputs/reports/per_class_f1_<run>.png            逐类 F1 条形图

用法：
    python src/make_reports.py                       # 汇总全部已完成的实验
    python src/make_reports.py --runs baseline_resnet18_160 improved_resnet50_224
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.utils import get_logger  # noqa: E402


def load_runs(runs: list[str] | None) -> list[dict]:
    out = []
    for d in sorted(paths.RUN_DIR.iterdir()):
        if not d.is_dir() or not (d / "summary.json").exists():
            continue
        if runs and d.name not in runs:
            continue
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        s["_dir"] = d
        report = d / "val_report.json"
        if report.exists():
            s["val_report"] = json.loads(report.read_text(encoding="utf-8"))
        sweep = d / "threshold_sweep.json"
        if sweep.exists():
            s["threshold"] = json.loads(sweep.read_text(encoding="utf-8"))
        calib = d / "calibration.json"
        if calib.exists():
            s["calibration"] = json.loads(calib.read_text(encoding="utf-8"))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--skip-smoke", action="store_true", default=True)
    args = ap.parse_args()

    logger = get_logger("reports")
    runs = load_runs(args.runs)
    runs = [r for r in runs if not r["run_name"].startswith(("smoke", "probe"))]
    if not runs:
        raise SystemExit("没有找到已完成的实验（outputs/runs/*/summary.json）")

    # ---------------- 对比表 ----------------
    rows = []
    for r in runs:
        hist = r["history"]
        best_epoch = r["best"]["epoch"]
        best_row = next((h for h in hist if h["epoch"] == best_epoch), hist[-1])
        vr = r.get("val_report", {})
        cal = r.get("calibration", {})
        best_joint = cal.get("best_joint", {})
        rows.append(
            {
                "实验": r["run_name"],
                "骨干": r["config"]["arch"],
                "输入": r["config"]["image_size"],
                "epochs": r["config"]["epochs"],
                "增强": r["config"].get("train_transform", ""),
                "类别权重": r["config"].get("class_weight", "none"),
                "EMA": r["config"].get("ema_decay", 0) or 0,
                "MixUp/CutMix": (r["config"].get("mix_prob", 0) or 0),
                "TTA": r["config"].get("tta_hflip", False),
                "最佳epoch": best_epoch,
                "未标定Macro-F1": round(vr.get("macro_f1", r["best"]["macro_f1"]), 4),
                "最优τ": best_joint.get("tau", ""),
                "拒识阈值t": best_joint.get("threshold", ""),
                "标定后Macro-F1": best_joint.get("macro_f1", ""),
                "Accuracy": round(vr.get("accuracy", best_row["val_acc"]), 4),
                "unknown检出率": round(vr.get("unknown_detection_rate", float("nan")), 4),
                "已知类误拒率": round(vr.get("known_false_reject_rate", float("nan")), 4),
                "训练集": r["train_size"],
                "验证集": r["val_size"],
                "用时(min)": round(r["total_time_s"] / 60, 1),
            }
        )

    csv_path = paths.REPORT_DIR / "experiment_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md: list[str] = []
    md.append("# 任务一 实验结果对比\n")
    md.append(f"（数据：train {rows[0]['训练集']} 张 / val {rows[0]['验证集']} 张；"
              f"评分口径：51 类等权 Macro-F1）\n")
    md.append("## 1. 实验总表\n")
    cols = list(rows[0].keys())
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md.append("")

    md.append("## 2. 逐轮训练过程\n")
    for r in runs:
        md.append(f"### {r['run_name']}\n")
        md.append("| epoch | train_loss | train_acc | val_loss | val_acc | val_MacroF1 | unknown检出 | 已知误拒 | 累计用时(s) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for h in r["history"]:
            md.append(
                f"| {h['epoch']} | {h['train_loss']} | {h['train_acc']} | {h['val_loss']} | "
                f"{h['val_acc']} | **{h['val_macro_f1']}** | {h['unknown_detection_rate']} | "
                f"{h['known_false_reject_rate']} | {h['elapsed_s']} |"
            )
        md.append("")
        md.append(f"配置：`{json.dumps(r['config'], ensure_ascii=False)}`\n")

    md.append("## 3. 决策标定结果（由验证集 Macro-F1 选出 τ 与拒识阈值）\n")
    any_calib = False
    for r in runs:
        cal = r.get("calibration")
        if not cal:
            continue
        any_calib = True
        base = cal["no_calibration"]
        tau_only = cal["best_tau_only"]
        joint = cal["best_joint"]
        md.append(f"### {r['run_name']}\n")
        md.append("| 配置 | τ | 拒识阈值 t | Macro-F1 | Accuracy | unknown 检出率 | 已知类误拒率 |")
        md.append("|---|---|---|---|---|---|---|")
        md.append(f"| 未标定 | 0.00 | 0.00 | {base['macro_f1']:.4f} | {base['accuracy']:.4f} | "
                  f"{base['unknown_detection_rate']:.4f} | {base['known_false_reject_rate']:.4f} |")
        md.append(f"| 仅先验校正 | {tau_only['tau']:+.2f} | 0.00 | {tau_only['macro_f1']:.4f} | "
                  f"{tau_only['accuracy']:.4f} | {tau_only['unknown_detection_rate']:.4f} | "
                  f"{tau_only['known_false_reject_rate']:.4f} |")
        md.append(f"| **联合最优** | **{joint['tau']:+.2f}** | **{joint['threshold']:.2f}** | "
                  f"**{joint['macro_f1']:.4f}** | {joint['accuracy']:.4f} | "
                  f"{joint['unknown_detection_rate']:.4f} | {joint['known_false_reject_rate']:.4f} |")
        md.append("")
        md.append("| τ 扫描（t=0） | Macro-F1 | unknown 检出率 | 已知类误拒率 |")
        md.append("|---|---|---|---|")
        for row in cal["sweep"]:
            if row["threshold"] != 0.0:
                continue
            mark = "  ←" if abs(row["tau"] - tau_only["tau"]) < 1e-9 else ""
            md.append(f"| {row['tau']:+.2f}{mark} | {row['macro_f1']:.4f} | "
                      f"{row['unknown_detection_rate']:.4f} | {row['known_false_reject_rate']:.4f} |")
        md.append("")
    if not any_calib:
        md.append("（尚未运行 `src/calibrate.py`）\n")

    md.append("## 4. F1 最低的 10 个类别（最终模型）\n")
    final = max(runs, key=lambda r: r.get("val_report", {}).get("macro_f1", 0))
    vr = final.get("val_report", {})
    if vr.get("per_class"):
        md.append(f"模型：**{final['run_name']}**\n")
        md.append("| 索引 | 类别 | Precision | Recall | F1 | 验证样本 |")
        md.append("|---|---|---|---|---|---|")
        for row in sorted(vr["per_class"], key=lambda x: x["f1"])[:10]:
            md.append(f"| {row['index']} | {row['label']} | {row['precision']:.3f} | "
                      f"{row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
        md.append("")
        md.append("## 5. F1 最高的 10 个类别\n")
        md.append("| 索引 | 类别 | Precision | Recall | F1 | 验证样本 |")
        md.append("|---|---|---|---|---|---|")
        for row in sorted(vr["per_class"], key=lambda x: -x["f1"])[:10]:
            md.append(f"| {row['index']} | {row['label']} | {row['precision']:.3f} | "
                      f"{row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
        md.append("")

    (paths.REPORT_DIR / "experiment_comparison.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("[写出] %s", csv_path)
    logger.info("[写出] %s", paths.REPORT_DIR / "experiment_comparison.md")

    # ---------------- 训练曲线 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for r in runs:
        hist = r["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], marker="o", ms=3, label=r["run_name"])
        axes[0].plot(ep, [h["val_loss"] for h in hist], marker="s", ms=3, ls="--",
                     label=f"{r['run_name']} (val)")
        axes[1].plot(ep, [h["train_acc"] for h in hist], marker="o", ms=3, label=r["run_name"])
        axes[1].plot(ep, [h["val_acc"] for h in hist], marker="s", ms=3, ls="--",
                     label=f"{r['run_name']} (val)")
        axes[2].plot(ep, [h["val_macro_f1"] for h in hist], marker="o", ms=3, label=r["run_name"])
    axes[0].set_title("Loss"); axes[1].set_title("Accuracy"); axes[2].set_title("验证集 Macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.REPORT_DIR / "training_curves.png", dpi=130)
    plt.close(fig)
    logger.info("[写出] %s", paths.REPORT_DIR / "training_curves.png")

    # ---------------- 混淆矩阵 / 逐类 F1 ----------------
    for r in runs:
        d = r["_dir"]
        cm_path = d / "eval_confusion_matrix.csv"
        if not cm_path.exists():
            continue
        with open(cm_path, encoding="utf-8") as fp:
            reader = csv.reader(fp)
            header = next(reader)[1:]
            cm = np.array([[float(x) for x in row[1:]] for row in reader])
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(np.log1p(cm), cmap="viridis")
        ax.set_xticks(range(len(header)))
        ax.set_yticks(range(len(header)))
        ax.set_xticklabels(header, rotation=90, fontsize=5)
        ax.set_yticklabels(header, fontsize=5)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title(f"混淆矩阵（log1p 计数）— {r['run_name']}\n"
                     f"验证集 Macro-F1 = {r.get('val_report', {}).get('macro_f1', 0):.4f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(paths.REPORT_DIR / f"confusion_matrix_{r['run_name']}.png", dpi=130)
        plt.close(fig)

        vr = r.get("val_report", {})
        if vr.get("per_class"):
            pc = sorted(vr["per_class"], key=lambda x: x["f1"])
            fig, ax = plt.subplots(figsize=(13, 10))
            colors = ["#d62728" if v["f1"] < 0.5 else "#ff7f0e" if v["f1"] < 0.75 else "#2ca02c"
                      for v in pc]
            ax.barh([f"{v['index']:02d} {v['label']}" for v in pc], [v["f1"] for v in pc],
                    color=colors)
            ax.axvline(vr["macro_f1"], color="black", ls="--",
                       label=f"Macro-F1 = {vr['macro_f1']:.4f}")
            ax.set_xlim(0, 1); ax.set_xlabel("F1"); ax.legend()
            ax.set_title(f"逐类 F1（升序）— {r['run_name']}")
            ax.tick_params(axis="y", labelsize=6)
            fig.tight_layout()
            fig.savefig(paths.REPORT_DIR / f"per_class_f1_{r['run_name']}.png", dpi=130)
            plt.close(fig)
        logger.info("[写出] %s 的图表", r["run_name"])

    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
