# -*- coding: utf-8 -*-
"""多尺度 TTA 实验：同一个模型在多个分辨率下各跑一遍，logits 取平均。

思路：车型识别是细粒度任务，
  · 小分辨率看整体轮廓（不容易被局部干扰）
  · 大分辨率看细节（进气格栅、大灯、轮毂）
两者互补 —— 这正是集成思想的"单模型版"。

代价只有推理时间（训练成本为零），所以是先试的手段。

用法：
    python tools/experiment_scale.py --runs improved_resnet50_224 --scales 224 288 320
    python tools/experiment_scale.py --champion --scales 224 256 288
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402
from cardex.utils import autocast_dtype, get_device  # noqa: E402

CACHE = paths.RUN_DIR / "scale_cache"


def logits_at(run: str, scale: int, device: torch.device, tta: bool = True) -> np.ndarray:
    """在指定分辨率下算一遍验证集的 log-softmax，带磁盘缓存。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    ck = paths.CKPT_DIR / run / "best.pt"
    st = ck.stat()
    fp = hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]
    tag = f"{run}@{scale}{'tta' if tta else ''}_{fp}"
    f = CACHE / f"{tag}.npy"
    if f.exists():
        return np.load(f)

    model, ckpt = load_model_from_checkpoint(ck)
    model.to(device).eval()
    ds = CarDataset(paths.DATA_DIR / "val",
                    transform=build_transform("eval", scale, ckpt.get("preprocess")))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=autocast_dtype(device), enabled=True):
                lg = model(images).float()
                if tta:
                    lg2 = model(torch.flip(images, dims=[3])).float()
                    lg = (torch.log_softmax(lg, 1) + torch.log_softmax(lg2, 1)) / 2
            out.append(torch.log_softmax(lg, 1).cpu().numpy())
    arr = np.concatenate(out)
    np.save(f, arr)
    print(f"    {tag}  {arr.shape}  {time.perf_counter() - t0:.0f}s")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--champion", action="store_true", help="用 final_model_choice.json 里的冠军成员")
    ap.add_argument("--scales", nargs="*", type=int, default=[224, 256, 288])
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    runs = list(args.runs)
    if args.champion:
        ch = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
        runs += ch["members"]
    runs = list(dict.fromkeys(runs))
    if not runs:
        raise SystemExit("请用 --runs 或 --champion 指定模型")

    cm = paths.load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    from cardex.data import CarDataset as _DS
    y_true = np.array([s[1] for s in _DS(paths.DATA_DIR / "val").samples])

    device = get_device("auto")
    print(f"设备 {device}   模型 {runs}   尺度 {args.scales}")

    # 每个模型 × 每个尺度
    table: dict[str, dict[int, np.ndarray]] = {}
    for r in runs:
        table[r] = {}
        print(f"  {r}:")
        for s in args.scales:
            table[r][s] = logits_at(r, s, device, tta=not args.no_tta)

    # 先验
    stats = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n_c = [stats["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n_c.append(stats["per_class"]["train"]["unknown"]["n"])
    t = np.array(n_c, dtype=np.float64)
    log_prior = np.log(t / t.sum())

    def score(logp: np.ndarray) -> tuple[float, float, float]:
        """扫 (tau, t) 取最优 Macro-F1。"""
        best = (-1.0, 0.0, 0.0)
        for tau in np.arange(-0.4, 0.601, 0.1):
            z = logp - tau * log_prior[None, :]
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            pmax, pred0 = p.max(axis=1), p.argmax(axis=1)
            for thr in np.arange(0.0, 0.61, 0.05):
                pred = np.where(pmax < thr, 50, pred0)
                f1 = compute_metrics(y_true, pred, names)["macro_f1"]
                if f1 > best[0]:
                    best = (f1, float(tau), float(thr))
        return best

    print(f"\n{'=' * 84}\n单尺度基准\n{'=' * 84}")
    single = {}
    for r in runs:
        for s in args.scales:
            f1, tau, thr = score(table[r][s])
            single[(r, s)] = (f1, tau, thr)
            print(f"  {r} @{s:<4} Macro-F1 = {f1:.4f}  (tau={tau:+.1f}, t={thr:.2f})")

    print(f"\n{'=' * 84}\n单模型多尺度融合\n{'=' * 84}")
    ms = {}
    for r in runs:
        combos = [tuple(args.scales)] + [c for k in range(2, len(args.scales) + 1)
                                         for c in __import__("itertools").combinations(args.scales, k)]
        for c in combos:
            lp = np.mean([table[r][s] for s in c], axis=0)
            f1, tau, thr = score(lp)
            ms[(r, c)] = (f1, tau, thr)
            print(f"  {r} @{'+'.join(map(str, c)):<14} Macro-F1 = {f1:.4f}  (tau={tau:+.1f}, t={thr:.2f})")

    print(f"\n{'=' * 84}\n多模型 × 多尺度{'=' * 84}")
    best_all = (-1.0, None, 0.0, 0.0)
    for c in __import__("itertools").combinations(args.scales, min(2, len(args.scales))):
        lp = np.mean([np.mean([table[r][s] for s in c], axis=0) for r in runs], axis=0)
        f1, tau, thr = score(lp)
        print(f"  全部模型 @{'+'.join(map(str, c)):<14} Macro-F1 = {f1:.4f}  (tau={tau:+.1f}, t={thr:.2f})")
        if f1 > best_all[0]:
            best_all = (f1, f"全部模型@{'+'.join(map(str, c))}", tau, thr)

    lp = np.mean([np.mean([table[r][s] for s in args.scales], axis=0) for r in runs], axis=0)
    f1, tau, thr = score(lp)
    print(f"  全部模型 @{'全部尺度':<13} Macro-F1 = {f1:.4f}  (tau={tau:+.1f}, t={thr:.2f})")
    if f1 > best_all[0]:
        best_all = (f1, "全部模型@全尺度", tau, thr)

    print(f"\n{'=' * 84}")
    print(f"最佳：{best_all[1]}  →  Macro-F1 = {best_all[0]:.4f}  (tau={best_all[2]:+.1f}, t={best_all[3]:.2f})")
    print(f"{'=' * 84}")


if __name__ == "__main__":
    main()
