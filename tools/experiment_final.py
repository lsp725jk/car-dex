# -*- coding: utf-8 -*-
"""最终组合实验：两个强模型 × 中心/多裁剪 TTA。

现在手上有两个**同样强**的模型：
    r50_288_from_scratch   原始 0.9739
    r50_352                原始 0.9756
分辨率不同 → 看问题的尺度不同 → 是有真实多样性的集成伙伴。

与之对比，之前失败的那些集成都是"强 + 弱"：
    0.9752 的强者 + 0.95~0.96 的 224 模型  →  被拉低
    0.9752 的 best + 0.9710 的早期快照     →  被拉低

用法：
    python tools/experiment_final.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402
from cardex.utils import autocast_dtype, get_device  # noqa: E402

CACHE = paths.RUN_DIR / "final_cache"


class StackCrops:
    """FiveCrop → (5,C,H,W)。必须是模块级类才能被多进程 pickle。"""

    def __init__(self, mean, std) -> None:
        self.norm = T.Normalize(list(mean), list(std))

    def __call__(self, crops):
        return torch.stack([self.norm(T.ToTensor()(c)) for c in crops])


def prior_log():
    cm = paths.load_class_map()
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n, dtype=np.float64)
    return np.log(t / t.sum())


def score(logp, y, names, lp):
    """扫 (tau, t) 取最优 Macro-F1；返回 (f1, tau, t, pred, pmax, amax)。"""
    best = (-1.0, 0.0, 0.0, None, None, None)
    for tau in np.arange(-0.4, 0.601, 0.1):
        z = logp - tau * lp[None, :]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        pm, pa = p.max(1), p.argmax(1)
        for th in np.arange(0.0, 0.601, 0.05):
            pr = np.where(pm < th, 50, pa)
            f = compute_metrics(y, pr, names)["macro_f1"]
            if f > best[0]:
                best = (f, float(tau), float(th), pr, pm, pa)
    return best


@torch.no_grad()
def logits(run: str, crops: str, device, batch: int = 16):
    tag = f"{run}_{crops}"
    f = CACHE / f"{tag}.npy"
    if f.exists():
        return np.load(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = paths.CKPT_DIR / run / "best.pt"
    model, ck = load_model_from_checkpoint(ckpt)
    model.to(device).eval()
    size = int(ck.get("image_size", 224))
    resize = int(round(size * 256 / 224))
    pre = ck.get("preprocess", {})
    mean, std = pre.get("mean", [0.485, 0.456, 0.406]), pre.get("std", [0.229, 0.224, 0.225])
    if crops == "center":
        tf = T.Compose([T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
                        T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])
        views = 1
    else:
        tf = T.Compose([T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
                        T.FiveCrop(size), StackCrops(mean, std)])
        views = 5
    dl = DataLoader(CarDataset(paths.DATA_DIR / "val", transform=tf),
                    batch_size=batch, shuffle=False, num_workers=6, pin_memory=True)
    out = []
    for x, _, _ in dl:
        x = x.to(device)
        if views > 1:
            x = x.view(-1, *x.shape[2:])
        with torch.autocast(device.type, dtype=autocast_dtype(device), enabled=True):
            lo = torch.log_softmax(model(x).float(), 1)
            lo = (lo + torch.log_softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        if views > 1:
            lo = lo.view(-1, views, lo.shape[1]).mean(1)
        out.append(lo.float().cpu().numpy())
    arr = np.concatenate(out)
    np.save(f, arr)
    print(f"    算了 {tag}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["r50_288_from_scratch", "r50_352"])
    args = ap.parse_args()

    cm = paths.load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    y = np.array([s[1] for s in CarDataset(paths.DATA_DIR / "val").samples])
    lp = prior_log()
    dev = get_device("auto")

    print("=" * 86)
    print("计算各模型 × 各视角的 logits")
    print("=" * 86)
    L = {}
    for r in args.runs:
        for c in ("center", "five"):
            L[(r, c)] = logits(r, c, dev)

    print("\n" + "=" * 86)
    print("[1] 各模型单独表现")
    print("=" * 86)
    for r in args.runs:
        fc, _, tc, *_ = score(L[(r, "center")], y, names, lp)
        ff, _, tf_, *_ = score(L[(r, "five")], y, names, lp)
        fm, _, tm, *_ = score((L[(r, "center")] + L[(r, "five")]) / 2, y, names, lp)
        print(f"  {r:<24} 中心 {fc:.4f}   5裁剪 {ff:.4f}   两者平均 {fm:.4f} ({fm - fc:+.4f})")

    print("\n" + "=" * 86)
    print("[2] 两个强模型集成")
    print("=" * 86)
    rows = []
    # 各种组合：视角 × 成员
    for views in itertools.chain.from_iterable(
            itertools.combinations(["center", "five"], k) for k in (1, 2)):
        for members in itertools.chain.from_iterable(
                itertools.combinations(args.runs, k) for k in (1, 2)):
            pool = [L[(m, v)] for m in members for v in views]
            f, tau, th, *_ = score(np.mean(pool, axis=0), y, names, lp)
            tag = "+".join(m.replace("r50_", "").replace("_from_scratch", "") for m in members) \
                  + " @" + "+".join(views)
            rows.append((f, tau, th, tag, len(pool)))
    rows.sort(reverse=True)
    for f, tau, th, tag, n in rows:
        star = " ★" if f == rows[0][0] else ""
        print(f"  {tag:<34} {n} 个视角  Macro-F1 = {f:.4f}  (tau={tau:+.1f}, t={th:.2f}){star}")

    best = rows[0]
    print("\n" + "=" * 86)
    print(f"★ 最佳：{best[3]}   →   Macro-F1 = {best[0]:.4f}   (tau={best[1]:+.1f}, t={best[2]:.2f})")
    print(f"  （起点 0.9646 → 现在 {best[0]:.4f}，累计 +{best[0] - 0.9646:.4f}）")
    print("=" * 86)

    # 写一份结果
    out = paths.REPORT_DIR / "final_combination.json"
    out.write_text(json.dumps({
        "best": {"tag": best[3], "macro_f1": best[0], "tau": best[1], "threshold": best[2],
                 "views": best[4]},
        "ranking": [{"tag": t, "macro_f1": f, "tau": ta, "threshold": th, "views": n}
                    for f, ta, th, t, n in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写出：{out}")


if __name__ == "__main__":
    main()
