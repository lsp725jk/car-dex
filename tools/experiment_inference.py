# -*- coding: utf-8 -*-
"""推理侧手段实验（零训练成本）：快照集成 + 多裁剪 TTA。

三种手段的区别很重要，别混为一谈：

  · **多尺度 TTA**（已试过，无效）：把图片缩放到训练时没见过的尺寸再推理。
    实测 224 训的模型拿到 288 用，性能是**掉的**（0.9597→0.9509）——
    训练/测试分辨率不匹配，模型没见过这个尺度。

  · **多裁剪 TTA**（本篇）：**保持训练时的分辨率**，只是裁剪位置不同
    （中心 + 四角）。模型见过的尺度不变，只是"看画面的哪一块"变了。
    细粒度任务里这通常有效，因为区分车型的关键部位（格栅、大灯）
    可能不在正中间。

  · **快照集成**（本篇）：同一次余弦退火训练里保存的多个 epoch 权重
    （epoch_12 / 15 / 18 / last）。余弦退火会让模型在训练后期**反复经过
    几个不同的极小值**，把这些快照的预测平均起来，等于"免费"得到一个小集成。
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
import torchvision.transforms as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402
from cardex.utils import autocast_dtype, get_device  # noqa: E402

CACHE = paths.RUN_DIR / "infer_cache"


def _prior_log():
    cm = paths.load_class_map()
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n, dtype=np.float64)
    return np.log(t / t.sum())


class StackCrops:
    """把 FiveCrop 出来的 5 个 PIL 图各自转张量并归一化，再堆成 (5,C,H,W)。

    ⚠️ 必须写成**模块级类**，不能用 `T.Lambda(lambda ...)`：
    Windows 上 DataLoader 多进程用 spawn，要把 transform **pickle** 给子进程，
    而 lambda 是局部对象，序列化时会报
    `AttributeError: Can't get local object 'logits_of.<locals>.<lambda>'`。
    """

    def __init__(self, mean, std) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, crops):
        out = []
        for c in crops:
            t = T.ToTensor()(c)
            out.append(T.Normalize(self.mean, self.std)(t))
        return torch.stack(out)


def _score(logp, y, names, log_prior):
    best = (-1.0, 0.0, 0.0)
    for tau in np.arange(-0.4, 0.601, 0.1):
        z = logp - tau * log_prior[None, :]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        pm, pa = p.max(1), p.argmax(1)
        for th in np.arange(0.0, 0.601, 0.05):
            pr = np.where(pm < th, 50, pa)
            f = compute_metrics(y, pr, names)["macro_f1"]
            if f > best[0]:
                best = (f, float(tau), float(th))
    return best


@torch.no_grad()
def logits_of(ckpt: Path, device, crops: str, scale: int | None, batch: int = 24):
    """算一组 log-softmax。crops='center' 或 'five'。"""
    model, ck = load_model_from_checkpoint(ckpt)
    model.to(device).eval()
    size = int(scale or ck.get("image_size", 224))
    resize = int(round(size * 256 / 224))
    pre = ck.get("preprocess", {})
    mean, std = pre.get("mean", [0.485, 0.456, 0.406]), pre.get("std", [0.229, 0.224, 0.225])

    if crops == "center":
        base_tf = T.Compose([
            T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])
        views = 1
    else:  # five
        base_tf = T.Compose([
            T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            T.FiveCrop(size),
            StackCrops(mean, std),
        ])
        views = 5

    ds = CarDataset(paths.DATA_DIR / "val", transform=base_tf)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=6, pin_memory=True)
    acc = []
    for x, _, _ in dl:
        x = x.to(device)                       # center: (B,C,H,W)  five: (B,5,C,H,W)
        if views > 1:
            x = x.view(-1, *x.shape[2:])       # 展成 (B*5,C,H,W)
        with torch.autocast(device.type, dtype=autocast_dtype(device), enabled=True):
            lp = torch.log_softmax(model(x).float(), 1)
            lp = (lp + torch.log_softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        if views > 1:
            lp = lp.view(-1, views, lp.shape[1]).mean(1)
        acc.append(lp.float().cpu().numpy())
    return np.concatenate(acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r50_288_from_scratch")
    ap.add_argument("--snapshots", nargs="*", default=["epoch_12", "epoch_15", "epoch_18", "last"])
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cm = paths.load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    from cardex.data import CarDataset as _D
    y = np.array([s[1] for s in _D(paths.DATA_DIR / "val").samples])
    log_prior = _prior_log()
    dev = get_device("auto")
    ckdir = paths.CKPT_DIR / args.run

    def get(name: str, crops: str, scale=None):
        tag = f"{args.run}_{name}_{crops}{'' if scale is None else f'_{scale}'}"
        f = CACHE / f"{tag}.npy"
        if f.exists():
            return np.load(f)
        p = ckdir / f"{name}.pt"
        if not p.exists():
            return None
        a = logits_of(p, dev, crops, scale)
        np.save(f, a)
        print(f"    算了 {tag}")
        return a

    print("=" * 84)
    print(f"[基准] {args.run} / best.pt / 中心裁剪")
    print("=" * 84)
    base = get("best", "center")
    f1, tau, th = _score(base, y, names, log_prior)
    print(f"  Macro-F1 = {f1:.4f}   (tau={tau:+.1f}, t={th:.2f})")

    print("\n" + "=" * 84)
    print("[1] 多裁剪 TTA（5 裁剪 + 翻转，保持训练分辨率）")
    print("=" * 84)
    five = get("best", "five")
    if five is not None:
        f1f, tauf, thf = _score(five, y, names, log_prior)
        print(f"  5 裁剪     Macro-F1 = {f1f:.4f}   (tau={tauf:+.1f}, t={thf:.2f})   {f1f - f1:+.4f}")
        mix = (base + five) / 2
        f1m, taum, thm = _score(mix, y, names, log_prior)
        print(f"  中心+5裁剪  Macro-F1 = {f1m:.4f}   (tau={taum:+.1f}, t={thm:.2f})   {f1m - f1:+.4f}")

    print("\n" + "=" * 84)
    print("[2] 快照集成（同一次余弦训练的多个 epoch）")
    print("=" * 84)
    snaps = []
    for s in args.snapshots:
        a = get(s, "center")
        if a is None:
            print(f"  {s}: 不存在，跳过")
            continue
        fs, _, _ = _score(a, y, names, log_prior)
        print(f"  {s:<12} 单独 Macro-F1 = {fs:.4f}")
        snaps.append((s, a))
    if len(snaps) >= 2:
        for k in (2, 3, 4):
            if k > len(snaps):
                break
            sub = [a for _, a in snaps[-k:]]
            lp = np.mean(sub, axis=0)
            fk, tk, hk = _score(lp, y, names, log_prior)
            print(f"  最近 {k} 个快照平均 Macro-F1 = {fk:.4f}   (tau={tk:+.1f}, t={hk:.2f})   {fk - f1:+.4f}")
        lp = np.mean([a for _, a in snaps], axis=0)
        fa, ta, ha = _score(lp, y, names, log_prior)
        print(f"  全部快照平均     Macro-F1 = {fa:.4f}   (tau={ta:+.1f}, t={ha:.2f})   {fa - f1:+.4f}")

    print("\n" + "=" * 84)
    print("[3] 最佳组合：best + 快照 + 多裁剪")
    print("=" * 84)
    if snaps and five is not None:
        pool = [base, five] + [a for _, a in snaps]
        lp = np.mean(pool, axis=0)
        fx, tx, hx = _score(lp, y, names, log_prior)
        print(f"  全部平均 ({len(pool)} 个视角) Macro-F1 = {fx:.4f}   (tau={tx:+.1f}, t={hx:.2f})   {fx - f1:+.4f}")


if __name__ == "__main__":
    main()
