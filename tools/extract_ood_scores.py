# -*- coding: utf-8 -*-
"""绕开 unknown 头：用**事后 OOD 打分**（原始 logits 的 energy / 特征距离）重新判别。

## 动机

unknown 头只在 **98 个来源类**上训练过，却要判别 **777 个类** —— 它本身就是瓶颈。
事后方法不依赖这个头，理论上泛化更好：
  · Energy        `T·logsumexp(z_已知50类)`  —— 需要**原始 logits**（log-softmax 会把能量归一化掉）
  · kNN / 原型余弦  到 50 个类原型的最大余弦相似度
  · Mahalanobis   到最近类均值的马氏距离（共享协方差）
  · MSP           最大已知类 softmax（已知比 unknown 头略差，作为对照）

## 为什么必须重新前向

现成的 `result.npz` 存的是 **log-softmax**（行 logsumexp=0），energy 的尺度信息已丢失。
所以本脚本重新跑一遍前向，同时抓原始 logits 和倒数第二层特征。

## 用法

    python tools/extract_ood_scores.py --tta hflip --out outputs/reports/ood_features
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cardex import paths  # noqa: E402

DS = ROOT / "数据集"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def build_honest_val(n_known: int, n_unknown: int, seed: int):
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = [cm["index_to_id"][str(i)] for i in range(50)]
    used = {s["id"] for s in json.loads(
        (ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))["sources"]}
    info = json.loads((DS / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
    all_ids = {str(x["id"]).zfill(4) for x in info}
    honest = sorted(all_ids - set(targets) - used)

    rng = random.Random(seed)
    val = DS / "val"

    def files(c):
        d = val / c
        return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []

    paths_l, labels = [], []
    per = max(1, n_known // 50)
    for i, c in enumerate(targets):
        fs = files(c)
        if fs:
            for p in rng.sample(fs, min(per, len(fs))):
                paths_l.append(p); labels.append(i)
    picks, per_cls = [], 4
    while len(picks) < n_unknown and per_cls <= 60:
        picks = []
        for c in honest:
            fs = files(c)
            if fs:
                picks.extend(rng.sample(fs, min(per_cls, len(fs))))
        if len(picks) >= n_unknown:
            break
        per_cls += 4
    rng.shuffle(picks)
    picks = picks[:n_unknown]
    paths_l += picks
    labels += [50] * len(picks)
    return paths_l, labels


def build_train_known(per_class: int, seed: int):
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = [cm["index_to_id"][str(i)] for i in range(50)]
    rng = random.Random(seed)
    tr = ROOT / "data" / "train"
    pl, yl = [], []
    for i, c in enumerate(targets):
        d = tr / c
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d.is_dir() else []
        if fs:
            for p in rng.sample(fs, min(per_class, len(fs))):
                pl.append(p); yl.append(i)
    return pl, yl


@torch.no_grad()
def extract(ens, files: list[Path], batch: int, workers: int, tag: str):
    """返回 raw_logits (N,51) 与 per-member 特征 (N, M, 512)。"""
    logits_all = []
    n_m = len(ens.members)
    feats_by_member: list[list[np.ndarray]] = [[] for _ in range(n_m)]
    t0 = time.perf_counter()
    bufs: list[torch.Tensor] = []
    handles = []
    for m in ens.members:
        def mk(buf):
            def hook(mod, inp):
                buf.append(inp[0].detach().float())
            return hook
        handles.append(m.model.fc.register_forward_pre_hook(mk(bufs)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for s in range(0, len(files), batch):
            chunk = files[s: s + batch]
            imgs = list(pool.map(load_image, chunk))
            wsum = float(sum(ens.weights))
            acc = None
            bufs.clear()
            for member, w in zip(ens.members, ens.weights):
                lg = member.tta_logits(imgs, return_raw=True)
                if ens.log_prior is not None:
                    lg = lg - ens.prior_adjust_tau * ens.log_prior.to(lg.device)[None, :]
                term = lg * (w / wsum)
                acc = term if acc is None else acc + term
            logits_all.append(acc.float().cpu().numpy())
            # 特征：每个成员 hook 触发 per 次（hflip → 2 次），组内取平均
            # ⚠️ 最后一批的样本数可能小于 batch，所以必须**逐块 append 后 concatenate**，
            #    不能把所有块 stack 起来（形状不一致会报错）。
            if len(bufs) % n_m:
                raise RuntimeError(f"hook 触发次数 {len(bufs)} 不是成员数 {n_m} 的整数倍")
            per = len(bufs) // n_m
            for j in range(n_m):
                vs = bufs[j * per: (j + 1) * per]
                feats_by_member[j].append(torch.stack(vs).mean(0).cpu().numpy())
            done = min(s + batch, len(files))
            if done % (batch * 8) == 0 or done == len(files):
                el = time.perf_counter() - t0
                print(f"    [{tag}] {done}/{len(files)}  {el:.0f}s", flush=True)
    for h in handles:
        h.remove()
    L = np.concatenate(logits_all).astype(np.float32)
    F = np.stack([np.concatenate(feats_by_member[j]) for j in range(n_m)],
                 axis=1).astype(np.float32)          # (N, M, 512)
    return L, F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", default="hflip")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "reports" / "ood_features"))
    ap.add_argument("--n-known", type=int, default=1250)
    ap.add_argument("--n-unknown", type=int, default=1250)
    ap.add_argument("--train-per-class", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    ckpts = [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
    from cardex.inference import CarEnsemble
    ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                      prior_adjust_tau=float(cfg["best"]["tau"]),
                      unknown_threshold=0.0, device="auto")
    print(f"冠军 {cfg['tag']}  TTA={args.tta}  设备 {ens.device}\n")

    # ── 验证集 ──
    vp, vy = build_honest_val(args.n_known, args.n_unknown, seed=20250922)
    print(f"诚实 val：{len(vp)} 张（已知 {sum(1 for x in vy if x != 50)} / 未知 {sum(1 for x in vy if x == 50)}）")
    Lv, Fv = extract(ens, vp, args.batch, args.workers, "val")
    np.savez_compressed(out / "honest_val.npz", logits=Lv, feats=Fv,
                        labels=np.array(vy, dtype=np.int32),
                        paths=np.array([p.as_posix() for p in vp]))
    print(f"  → logits {Lv.shape}  feats {Fv.shape}")

    # ── 训练集已知类特征（做原型 / 协方差）──
    tp, ty = build_train_known(args.train_per_class, seed=7)
    print(f"\n训练集已知类：{len(tp)} 张（{args.train_per_class}/类）")
    _, Ft = extract(ens, tp, args.batch, args.workers, "train")
    np.savez_compressed(out / "train_known.npz", feats=Ft,
                        labels=np.array(ty, dtype=np.int32))
    print(f"  → feats {Ft.shape}")

    print(f"\n[写出] {out}\\honest_val.npz  train_known.npz")


if __name__ == "__main__":
    main()
