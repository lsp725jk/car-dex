# -*- coding: utf-8 -*-
"""任务一 - 收尾：在全部实验完成后自动选出最佳模型配置并重生成提交物

流程：
  1. 对每个候选模型（单模型 or 集成）在验证集上做一次前向传播（可叠加水平翻转 TTA），
     缓存 logits；
  2. 对每个候选做 (tau, threshold) 二维网格扫描，取验证集 Macro-F1 最高者；
  3. 比较所有候选，选出全局最佳配置；
  4. 用最佳配置重新生成 predictions.csv（对 data/val，或 --input-dir 指定的目录）；
  5. 写出 outputs/reports/final_model_choice.json 与 final_model_choice.md。

集成方式：几何平均（对各成员"先验校正后的 log-softmax"取平均），
可选的成员权重由 --weights 指定。

用法：
    python src/finalize.py --candidates improved_resnet18_224 improved_resnet50_224 \
                           --ensembles improved_resnet18_224+improved_resnet50_224
    python src/finalize.py --auto          # 自动使用 outputs/checkpoints 下所有已完成实验
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.metrics import compute_metrics, top_confusions  # noqa: E402
from cardex.utils import (autocast_dtype, configure_threads, get_device,  # noqa: E402
                          get_logger, save_json)

TAU_GRID = [-1.0, -0.75, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0,
            0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
THR_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def training_log_prior(n_classes: int) -> np.ndarray:
    stats = json.loads(paths.DATA_STATS_JSON.read_text(encoding="utf-8"))
    cm = paths.load_class_map()
    n_c = [stats["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(stats["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    return np.log(t / t.sum())


def _ckpt_fingerprint(ckpt_path: Path) -> str:
    """给 checkpoint 算一个"指纹"，用来判断缓存还能不能用。

    ⚠️ 缓存键里**必须**含这个指纹。原来只按 run 名索引，
    结果**重新训练之后跑 finalize 会静默复用旧 logits** ——
    指标、排名、predictions.csv 全是旧模型的，而且看不出任何异常。
    （实测：GPU 重训完 C1/C2 后 finalize 报出的数字和 CPU 版一字不差，才发现。）
    """
    if not ckpt_path.exists():
        return "missing"
    st = ckpt_path.stat()
    h = hashlib.md5()
    h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:10]


def member_logits(run: str, split: str, batch_size: int, tta: bool, amp: bool,
                  cache_dir: Path, logger, device: str = "auto",
                  tta_mode: str | None = None) -> np.ndarray:
    """返回某成员的 log-softmax 矩阵 (N, 51)，带磁盘缓存。

    缓存文件名里带上 checkpoint 指纹 + TTA 档位：
    换了权重、或者改了 TTA 设置，就会自动算一份新的，不会拿旧结果糊弄。

    `tta_mode` 直接走**生产推理代码**（`CarClassifier.tta_logits`），
    这样 finalize 的比选口径和 Web 系统 / predictions.csv **逐字节一致**。
    自己再实现一遍 TTA 是踩过的坑：一度因为两组视角的权重算法不同，
    生产代码只复现出 0.9695（实验是 0.9815）。
    """
    ckpt_path = paths.CKPT_DIR / run / "best.pt"
    fp = _ckpt_fingerprint(ckpt_path)
    if tta_mode is None:
        tta_mode = "hflip" if tta else "none"
    mode_tag = {"none": "notta", "hflip": "tta", "hflip+multicrop": "ttamc"}[tta_mode]
    cache = cache_dir / f"logits_{run}_{split}_{mode_tag}_{fp}.npy"
    if cache.exists():
        logger.info("  复用缓存 %s", cache.name)
        return np.load(cache)

    dev = get_device(device)
    from cardex.inference import CarClassifier
    from PIL import Image
    clf = CarClassifier(ckpt_path, tta_mode=tta_mode, amp=amp, device=str(dev))
    ckpt = clf.ckpt
    size = int(ckpt["image_size"])
    ds = CarDataset(paths.DATA_DIR / split)
    logger.info("  前向传播 %s（%s @%d, TTA=%s, %d 张, %s）",
                run, ckpt["arch"], size, tta_mode, len(ds.samples), dev.type)

    # 直接喂 PIL 图给 clf.tta_logits —— 多裁剪需要原始图片，
    # 不能先用 eval transform 变成张量再送进去。
    # 这条路径就是 tools/reproduce_production.py 验证过、能逐字节复现的那条。
    out_all = []
    for i in range(0, len(ds.samples), batch_size):
        chunk = ds.samples[i : i + batch_size]
        imgs = []
        for p, _ in chunk:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        out_all.append(clf.tta_logits(imgs).float().cpu().numpy())
    arr = np.concatenate(out_all)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr

    def fwd(x):
        with torch.autocast(dev.type, dtype=autocast_dtype(dev), enabled=amp):
            out = model(x)
        return out.float()

    out_all = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(dev, non_blocking=True)
            logits = fwd(images)
            if tta:
                logits = (torch.log_softmax(logits, dim=1)
                          + torch.log_softmax(fwd(torch.flip(images, dims=[3])), dim=1)) / 2
            out_all.append(torch.log_softmax(logits, dim=1).float().cpu().numpy())
    arr = np.concatenate(out_all)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--ensembles", nargs="*", default=[],
                    help="形如 runA+runB 的集成候选")
    ap.add_argument("--auto", action="store_true", help="自动收集所有已完成的实验")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta-hflip", action="store_true", default=True)
    ap.add_argument("--no-tta", dest="tta_hflip", action="store_false")
    ap.add_argument("--tta-mode", default=None,
                    choices=["none", "hflip", "hflip+multicrop"],
                    help="TTA 档位；不传则由 --tta-hflip/--no-tta 推导。"
                         "hflip+multicrop = 中心+五裁剪共 10 视角，实测 +0.0025")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--output", default=str(paths.PROJECT_ROOT / "predictions.csv"))
    ap.add_argument("--top-k", type=int, default=3, help="保留前 K 个候选写完整报告")
    args = ap.parse_args()

    configure_threads(args.threads)
    logger = get_logger("finalize", paths.REPORT_DIR / "finalize.log")
    cache_dir = paths.RUN_DIR / "finalize_cache"
    # TTA 档位：显式 --tta-mode 优先，否则由布尔开关推导（向后兼容）
    tta_mode = args.tta_mode or ("hflip" if args.tta_hflip else "none")
    mode_tag = {"none": "notta", "hflip": "tta", "hflip+multicrop": "ttamc"}[tta_mode]
    logger.info("TTA 档位：%s", tta_mode)

    candidates = list(args.candidates)
    ensembles = list(args.ensembles)
    if args.auto or not candidates:
        done = []
        for d in sorted(paths.CKPT_DIR.iterdir()):
            if (d / "best.pt").exists() and (paths.RUN_DIR / d.name / "summary.json").exists():
                done.append(d.name)
        # 过滤掉明显未训练完的（验证 Macro-F1 过低的中间态）
        keep = []
        for name in done:
            s = json.loads((paths.RUN_DIR / name / "summary.json").read_text(encoding="utf-8"))
            if s.get("best", {}).get("macro_f1", 0) > 0.5:
                keep.append(name)
        candidates = candidates or keep
        if len(keep) >= 2 and not ensembles:
            ensembles = ["+".join(keep)]
    if not candidates and not ensembles:
        raise SystemExit("没有可用的候选模型")

    logger.info("候选单模型：%s", candidates)
    logger.info("候选集成  ：%s", ensembles)

    # ---------------- 真实标签 ----------------
    # 注意：图片顺序必须与 CarDataset / DataLoader 的遍历顺序**完全一致**。
    # 不能对文件名做全局排序——unknown/ 下的图片文件名沿用的是其“来源类别 ID”
    # （例如 0022_0022.jpg），全局排序会把这些文件插到 0022 类中间，
    # 而 CarDataset 是把 unknown/ 目录排在最后的，两者错位会导致指标全部错乱。
    cm = paths.load_class_map()
    ref_ds = CarDataset(paths.DATA_DIR / args.split,
                        transform=build_transform("eval", 224, {}))
    names_sorted = [Path(p).name for p, _ in ref_ds.samples]
    y_true = np.array([lab for _, lab in ref_ds.samples])
    label_names = paths.label_names()
    log_prior = training_log_prior(51)
    labels = list(range(51))
    unk = 50

    # ---------------- 逐候选扫描 ----------------
    results: list[dict] = []
    todos = [(c, [c]) for c in candidates] + [(e, e.split("+")) for e in ensembles]
    for tag, members in todos:
        t0 = time.perf_counter()
        logger.info("[候选] %s", tag)
        logprobs = []
        for run in members:
            lp = member_logits(run, args.split, args.batch_size, args.tta_hflip, True,
                               cache_dir, logger, tta_mode=tta_mode)
            logprobs.append(lp)
        best = None
        sweep_rows = []
        for tau in TAU_GRID:
            adj = [(lp - tau * log_prior[None, :]) for lp in logprobs]
            geo = np.mean(adj, axis=0)
            geo = geo - geo.max(axis=1, keepdims=True)
            probs = np.exp(geo)
            probs /= probs.sum(axis=1, keepdims=True)
            conf = probs.max(axis=1)
            base_pred = probs.argmax(axis=1)
            for thr in THR_GRID:
                pred = base_pred.copy()
                if thr > 0:
                    pred[conf < thr] = unk
                row = {
                    "tau": tau,
                    "threshold": thr,
                    "macro_f1": round(float(f1_score(y_true, pred, labels=labels,
                                                     average="macro", zero_division=0)), 5),
                    "accuracy": round(float((pred == y_true).mean()), 5),
                    "unknown_detection_rate": round(float((pred[y_true == unk] == unk).mean()), 5),
                    "known_false_reject_rate": round(float((pred[y_true != unk] == unk).mean()), 5),
                }
                sweep_rows.append(row)
                if best is None or row["macro_f1"] > best["macro_f1"]:
                    best = row
        raw = next(r for r in sweep_rows if r["tau"] == 0.0 and r["threshold"] == 0.0)
        results.append({
            "tag": tag,
            "members": members,
            "raw": raw,
            "best": best,
            "seconds": round(time.perf_counter() - t0, 1),
        })
        logger.info("  未标定 Macro-F1=%.4f  →  最优 (tau=%+.2f, t=%.2f) Macro-F1=%.4f  "
                    "unk检出=%.3f 已知误拒=%.3f  [%.0fs]",
                    raw["macro_f1"], best["tau"], best["threshold"], best["macro_f1"],
                    best["unknown_detection_rate"], best["known_false_reject_rate"],
                    time.perf_counter() - t0)

    results.sort(key=lambda r: -r["best"]["macro_f1"])
    for r in results:
        logger.info("排名 %-46s 标定后 %.4f   未标定 %.4f", r["tag"],
                    r["best"]["macro_f1"], r["raw"]["macro_f1"])

    champion = results[0]
    logger.info("=" * 80)
    logger.info("★ 最佳配置：%s  tau=%+.2f  threshold=%.2f  验证集 Macro-F1=%.4f",
                champion["tag"], champion["best"]["tau"], champion["best"]["threshold"],
                champion["best"]["macro_f1"])
    logger.info("=" * 80)

    summary = {
        "split": args.split,
        "tta_hflip": args.tta_hflip,
        "n_val": int(len(y_true)),
        "candidates": results,
        "champion": champion,
        "note": "冠军配置由验证集 Macro-F1 选出，未使用任何测试图片。",
    }
    save_json(summary, paths.REPORT_DIR / "final_model_choice.json")

    # ---------------- 冠军配置的详细报告 ----------------
    members = champion["members"]
    # 缓存文件名要带 checkpoint 指纹 + TTA 档位，否则重训/改 TTA 后会读到旧结果
    logprobs = [np.load(cache_dir / (
                    f"logits_{r}_{args.split}_{mode_tag}_"
                    f"{_ckpt_fingerprint(paths.CKPT_DIR / r / 'best.pt')}.npy"))
                for r in members]
    tau, thr = champion["best"]["tau"], champion["best"]["threshold"]
    geo = np.mean([lp - tau * log_prior[None, :] for lp in logprobs], axis=0)
    geo = geo - geo.max(axis=1, keepdims=True)
    probs = np.exp(geo)
    probs /= probs.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    if thr > 0:
        pred[conf < thr] = unk
    metrics = compute_metrics(y_true, pred, label_names)
    (paths.REPORT_DIR / f"final_classification_report_{args.split}.txt").write_text(
        metrics["report_text"], encoding="utf-8")
    cm_arr = np.array(metrics["confusion_matrix"])
    with open(paths.REPORT_DIR / f"final_confusion_matrix_{args.split}.csv", "w",
              encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["true\\pred"] + label_names)
        for i, n in enumerate(label_names):
            w.writerow([n] + cm_arr[i].tolist())
    with open(paths.REPORT_DIR / "final_top_confusions.csv", "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["true", "pred", "count", "true_total", "rate"])
        w.writeheader()
        w.writerows(top_confusions(cm_arr, label_names, top_k=30))

    # ---------------- 重新生成 predictions.csv ----------------
    out = Path(args.output)
    with open(out, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image_id", "predicted_label", "confidence"])
        for name, p, c in zip(names_sorted, pred, conf):
            lab = "unknown" if p == unk else cm["index_to_id"][str(int(p))]
            w.writerow([name, lab, round(float(c), 6)])
    Path(str(out.with_suffix(".meta.json"))).write_text(json.dumps({
        "champion": champion["tag"],
        "members": members,
        "prior_adjust_tau": tau,
        "unknown_threshold": thr,
        "tta_hflip": args.tta_hflip,
        "val_macro_f1": champion["best"]["macro_f1"],
        "calibration_source": "outputs/reports/final_model_choice.json（由验证集 Macro-F1 选出）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[写出] %s（%d 条记录）", out, len(names_sorted))

    # ---------------- Markdown 报告 ----------------
    lines = ["# 任务一 最终模型选择", ""]
    lines.append(f"数据：`data/{args.split}`（{len(y_true)} 张）；TTA：{'开启' if args.tta_hflip else '关闭'}")
    lines.append("")
    lines.append("## 候选对比（全部由验证集 Macro-F1 选出 τ 与拒识阈值）")
    lines.append("")
    lines.append("| 排名 | 候选（单模型 / 集成） | 未标定 Macro-F1 | τ | 阈值 t | **标定后 Macro-F1** | Accuracy | unknown 检出率 | 已知类误拒率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        b = r["best"]
        mark = " ★" if i == 1 else ""
        lines.append(f"| {i}{mark} | `{r['tag']}` | {r['raw']['macro_f1']:.4f} | {b['tau']:+.2f} | "
                     f"{b['threshold']:.2f} | **{b['macro_f1']:.4f}** | {b['accuracy']:.4f} | "
                     f"{b['unknown_detection_rate']:.4f} | {b['known_false_reject_rate']:.4f} |")
    lines.append("")
    lines.append(f"## 冠军配置\n\n`{champion['tag']}`，τ = {tau:+.2f}，拒识阈值 t = {thr:.2f}")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 51 类 Macro-F1 | **{metrics['macro_f1']:.4f}** |")
    lines.append(f"| Accuracy | {metrics['accuracy']:.4f} |")
    lines.append(f"| unknown 检出率 | {metrics['unknown_detection_rate']:.4f} |")
    lines.append(f"| 已知类误拒率 | {metrics['known_false_reject_rate']:.4f} |")
    lines.append("")
    lines.append("### F1 最低的 10 个类别")
    lines.append("")
    lines.append("| 索引 | 类别 | Precision | Recall | F1 | 验证样本 |")
    lines.append("|---|---|---|---|---|---|")
    for row in sorted(metrics["per_class"], key=lambda x: x["f1"])[:10]:
        lines.append(f"| {row['index']} | {row['label']} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    lines.append("")
    (paths.REPORT_DIR / "final_model_choice.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("[写出] %s / final_model_choice.md",
                paths.REPORT_DIR / "final_model_choice.json")
    print(json.dumps({"champion": champion["tag"], "macro_f1": metrics["macro_f1"],
                      "tau": tau, "threshold": thr}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
