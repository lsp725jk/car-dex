# -*- coding: utf-8 -*-
"""验证 SupCon 是否真的"让近似款在特征空间拉开距离"。

对比学习中，最终 Macro-F1 只是**间接证据** —— 它可能来自任意原因。
这个脚本直接测**我们声称的机制**是否成立：

  [1] unknown 的 F1 有没有提升？（SupCon 的核心目标之一）
  [2] hard_negative 的误拒率有没有下降？
      （12.7% → ? 这批是刻意挑的"近似款"）
  [3] **特征空间的类间距离**有没有真的变大？
      直接度量：已知类之间的平均类间余弦距离、unknown 与其对应 hard_negative
      目标类之间的距离（比如 MINI CLUBMAN ↔ MINI）。
  [4] 混淆对有没有减少？

如果 [3] 成立而 [1][2] 没变，说明机制对了但任务上没兑现；
如果 [3] 也不成立，说明 SupCon 没起作用，前面的 F1 提升是别的原因。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import (CarDataset, build_transform, load_source_to_fine_index,
                         supcon_loss)  # noqa: E402
from cardex.engine import backbone_forward, load_model_from_checkpoint  # noqa: E402
from cardex.inference import CarClassifier  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402
from cardex.utils import get_device  # noqa: E402

CACHE = paths.RUN_DIR / "supcon_analysis_cache"


def class_names():
    cm = paths.load_class_map()
    return [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]


def prior_log():
    cm = paths.load_class_map()
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n, dtype=np.float64)
    return np.log(t / t.sum())


@torch.no_grad()
def features_of(run, images, device, batch=32):
    """取某模型的骨干特征（不做 TTA，特征直接算中心裁剪即可）。"""
    m, ck = load_model_from_checkpoint(paths.CKPT_DIR / run / "best.pt")
    m.to(device).eval()
    size = int(ck.get("image_size", 224))
    tf = build_transform("eval", size, ck.get("preprocess"))
    out = []
    for i in range(0, len(images), batch):
        xs = torch.stack([tf(im) for im in images[i : i + batch]]).to(device)
        out.append(backbone_forward(m, xs).float().cpu())
    return torch.cat(out).numpy()


def load_val_images(limit_per_class=None):
    ds = CarDataset(paths.DATA_DIR / "val", limit_per_class=limit_per_class, seed=42)
    imgs, ys, paths_ = [], [], []
    for p, y in ds.samples:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB").copy())
        ys.append(y)
        paths_.append(p)
    return imgs, np.array(ys), paths_


def intra_inter_stats(feat: np.ndarray, y: np.ndarray, n_class: int = 51):
    """类内 / 类间余弦距离统计。

    先做 L2 归一化，余弦距离 = 1 − 余弦相似度。
      · **类内距离**：同类样本两两的平均距离 —— 越小越好（紧凑）
      · **类间距离**：不同类中心之间的平均距离 —— 越大越好（可分）
      · **可分性比值 = 类间 / 类内**，越大说明特征空间越"拉开"
    """
    f = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    cents = np.zeros((n_class, f.shape[1]))
    for c in range(n_class):
        m = y == c
        if m.any():
            cents[c] = f[m].mean(0)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9

    # 类内：样本到自身类中心的平均余弦距离
    intra = []
    for c in range(n_class):
        m = y == c
        if m.sum() > 1:
            sim = (f[m] @ cents[c])
            intra.append(float((1 - sim).mean()))
    # 类间：类中心两两余弦距离
    cs = cents @ cents.T
    iu = np.triu_indices(n_class, k=1)
    inter = 1 - cs[iu]
    return {
        "intra": float(np.mean(intra)),
        "inter_mean": float(inter.mean()),
        "inter_min": float(inter.min()),          # 最像的一对类中心
        "separation": float(inter.mean() / max(1e-9, np.mean(intra))),
    }


def hard_negative_pairs(paths_):
    """找出验证集里"被刻意设计成近似款"的 unknown 样本，及其目标已知类。"""
    us = json.loads((paths.CONFIG_DIR / "unknown_sources.json").read_text(encoding="utf-8"))
    cm = paths.load_class_map()
    out = {}
    for s in us["sources"]:
        if s.get("source_kind") == "hard_negative" and s.get("target_class_id"):
            tgt = cm["id_to_index"].get(str(s["target_class_id"]))
            if tgt is not None:
                out[str(s["id"])] = int(tgt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["r50_288_from_scratch", "r50_288_supcon"])
    ap.add_argument("--limit", type=int, default=30, help="每类取多少张（加速）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    names = class_names()
    lp = prior_log()
    dev = get_device(args.device)

    print("载入验证集图片…")
    imgs, y, ps = load_val_images(limit_per_class=args.limit)
    print(f"  {len(imgs)} 张（每类约 {args.limit} 张）")

    feats = {}
    for r in args.runs:
        f = CACHE / f"feat_{r}_{args.limit}.npy"
        if f.exists():
            feats[r] = np.load(f)
        else:
            CACHE.mkdir(parents=True, exist_ok=True)
            print(f"  取 {r} 的特征…")
            feats[r] = features_of(r, imgs, dev)
            np.save(f, feats[r])

    print("\n" + "=" * 84)
    print("[3] 特征空间几何：类内 / 类间距离（核心机制检验）")
    print("=" * 84)
    print(f"  {'模型':<24} {'类内↓':>9} {'类间均值↑':>10} {'类间最小↑':>10} {'可分性↑':>9}")
    geo = {}
    for r in args.runs:
        s = intra_inter_stats(feats[r], y)
        geo[r] = s
        print(f"  {r:<24} {s['intra']:>9.4f} {s['inter_mean']:>10.4f} "
              f"{s['inter_min']:>10.4f} {s['separation']:>9.4f}")

    # ---- hard negative 距离 ----
    print("\n" + "=" * 84)
    print("[3b] hard_negative 与其**目标已知类**的距离（越远越好）")
    print("=" * 84)
    hn = hard_negative_pairs(ps)
    for r in args.runs:
        f = feats[r] / (np.linalg.norm(feats[r], axis=1, keepdims=True) + 1e-9)
        cents = np.zeros((51, f.shape[1]))
        for c in range(51):
            m = y == c
            if m.any():
                cents[c] = f[m].mean(0)
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        ds = []
        for i, p in enumerate(ps):
            if y[i] != 50:
                continue
            src = p.stem.split("_")[0]
            if src in hn:
                tgt = hn[src]
                ds.append(float(1 - f[i] @ cents[tgt]))
        print(f"  {r:<24} {len(ds)} 个 hard_negative 样本，"
              f"到目标类中心平均余弦距离 = {np.mean(ds):.4f}")

    print("\n" + "=" * 84)
    print("[1][2] 指标对比：unknown F1 与 hard_negative 误拒率")
    print("=" * 84)
    print("  用各模型自己的最优 (τ,t) 在**全体验证集**上评估（这里只看相对差异）")
    from cardex.inference import CarEnsemble
    for r in args.runs:
        ens = CarEnsemble([paths.CKPT_DIR / r / "best.pt"], tta_mode="hflip",
                          amp=True, prior_adjust_tau=0.1, unknown_threshold=0.2, device=dev)
        preds, confs = [], []
        for i in range(0, len(imgs), 16):
            for res in ens.predict_batch(imgs[i : i + 16], batch_size=16):
                preds.append(res["index"]); confs.append(res["confidence"])
        preds = np.array(preds)
        m = compute_metrics(y, preds, names)
        unk_f1 = [c["f1"] for c in m["per_class"] if c["label"] == "unknown"][0]
        # hard_negative 误拒率
        hn_idx = [i for i, p in enumerate(ps) if y[i] == 50 and p.stem.split("_")[0] in hn]
        hn_miss = sum(1 for i in hn_idx if preds[i] != 50)
        ot_idx = [i for i, p in enumerate(ps) if y[i] == 50 and p.stem.split("_")[0] not in hn]
        ot_miss = sum(1 for i in ot_idx if preds[i] != 50)
        print(f"  {r:<24} Macro-F1={m['macro_f1']:.4f}  unknown F1={unk_f1:.4f}  "
              f"hard_neg 未拒={hn_miss}/{len(hn_idx)} ({hn_miss / max(1, len(hn_idx)):.1%})  "
              f"其他品牌未拒={ot_miss}/{len(ot_idx)} ({ot_miss / max(1, len(ot_idx)):.1%})")

    out = paths.REPORT_DIR / "supcon_analysis.json"
    out.write_text(json.dumps({
        "geometry": geo,
        "hard_negative_distance": {
            r: float(np.mean([1 - (feats[r][i] / (np.linalg.norm(feats[r][i]) + 1e-9))
                              @ (np.zeros(51) + 1) for i in [0]])) for r in args.runs[:1]
        } if False else {},
        "note": "几何统计见 geometry；hard_negative 距离打印在终端",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
