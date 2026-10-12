# -*- coding: utf-8 -*-
"""诚实 OOD 验证集 + 决策规则对比（**不重训模型**）。

## 为什么需要它

现状验证集 `data/val` 的 unknown 来自**训练时见过**的 98 个来源类 →
unknown_recall 虚高到 95.75%，而测试集只有 78.74%。
用这样的验证集去标定阈值，必然标歪。

本脚本改用模型**从未见过任何一张图**的 679 个类（教师 827 类 − 50 目标类 − 98 已用来源类）
的 **val 划分**构造 unknown。这对新旧模型都是诚实的，能复现测试集上的难度。

## 它回答什么

在诚实 OOD 上对比几种决策规则，看谁能超过"全局单一 θ"：
  1. 全局 θ（现状）：margin = z_unknown − max(z_known) > θ
  2. 逐类 θ_c：每个已知类有自己的偏置 —— 针对"漏出集中在少数类"这一现象
  3. 置信度阈值：max(z_known) < τ 就拒识（不用 unknown 列）
  4. 组合：margin 与置信度都过低才拒识

## 用法

    python tools/eval_decision_rules.py --tta hflip+multicrop --n-known 1500 --n-unknown 1500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402

DS = ROOT / "数据集"
CACHE = ROOT / "outputs" / "reports" / "ood_val_logits"


def sample_val_set(n_known: int, n_unknown: int, seed: int = 20250922) -> tuple[list, list]:
    """返回 (paths, labels)；label 为 0..49 或 50(unknown)。"""
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = [cm["index_to_id"][str(i)] for i in range(50)]
    used = {s["id"] for s in json.loads(
        (ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))["sources"]}
    info = json.loads((DS / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
    all_ids = {str(x["id"]).zfill(4) for x in info}
    honest = sorted(all_ids - set(targets) - used)
    print(f"  诚实 unknown 候选类：{len(honest)} 个（模型从未见过）")

    rng = random.Random(seed)
    val = DS / "val"
    IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def files(c):
        d = val / c
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG)

    paths, labels = [], []
    # 已知：每类均匀取
    per = max(1, n_known // 50)
    for i, c in enumerate(targets):
        fs = files(c)
        if fs:
            for p in rng.sample(fs, min(per, len(fs))):
                paths.append(p); labels.append(i)
    nk = len(paths)

    # 诚实 unknown：**尽量铺开到很多类别**（每类少取几张），保证来源多样性
    # 先用每类 4 张铺满所有 honest 类，不够再回填
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
    n_src = len({p.parent.name for p in picks})
    print(f"  已知 {nk} 张；诚实 unknown {len(picks)} 张，来自 {n_src} 个不同来源类")
    paths += picks
    labels += [50] * len(picks)
    return paths, labels


def load_image(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", default="hflip+multicrop")
    ap.add_argument("--n-known", type=int, default=1500)
    ap.add_argument("--n-unknown", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="忽略缓存重新推理")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{args.tta.replace('+', '_')}_k{args.n_known}_u{args.n_unknown}"
    npz_path = CACHE / f"{tag}.npz"

    if npz_path.exists() and not args.force:
        print(f"[缓存] 直接读 {npz_path}")
        z = np.load(npz_path, allow_pickle=False)
        L, y = z["logits"], z["labels"]
        z.close()
    else:
        print("构造诚实 OOD 验证集 ...")
        paths_l, labels = sample_val_set(args.n_known, args.n_unknown)
        uniform = [p.as_posix() for p in paths_l]
        print(f"  合计 {len(uniform)} 张")

        cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json")
                         .read_text(encoding="utf-8"))["champion"]
        ckpts = [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
        from cardex.inference import CarEnsemble
        ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                          prior_adjust_tau=float(cfg["best"]["tau"]),
                          unknown_threshold=float(cfg["best"]["threshold"]), device="auto")
        print(f"  冠军 {cfg['tag']}  TTA={args.tta}  设备 {ens.device}")

        from concurrent.futures import ThreadPoolExecutor
        out, t0 = [], time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for s in range(0, len(paths_l), args.batch):
                chunk = paths_l[s: s + args.batch]
                imgs = list(pool.map(load_image, chunk))
                out.append(ens.score_logits(imgs, batch_size=args.batch).float().cpu().numpy())
                done = min(s + args.batch, len(paths_l))
                if done % (args.batch * 8) == 0 or done == len(paths_l):
                    el = time.perf_counter() - t0
                    print(f"    {done}/{len(paths_l)}  {el:.0f}s  "
                          f"剩余约 {el / done * (len(paths_l) - done):.0f}s", flush=True)
        L = np.concatenate(out).astype(np.float32)
        y = np.array(labels, dtype=np.int32)
        np.savez_compressed(npz_path, logits=L, labels=y,
                            paths=np.array(uniform))
        print(f"[写出] {npz_path}  用时 {(time.perf_counter() - t0) / 60:.1f} min")

    # ══════════ 决策规则对比 ══════════
    is_unk = y == 50
    unk_z = L[:, 50]
    known_z = L[:, :50]
    known_max = known_z.max(1)
    margin = unk_z - known_max
    print(f"\n=== 诚实 OOD 验证集：{len(y)} 张，unknown {is_unk.sum()} 张（{is_unk.mean():.1%}）===")

    # 参考：旧验证集（泄漏）上的表现
    print("\n[参考] 旧的 data/val（unknown 来自训练过的 98 类，偏乐观）")
    print("       unknown_recall ≈ 95.75%（论文记录），而测试集只有 78.74%")

    print("\n[规则 1] 全局 θ：margin > θ 判 unknown")
    best = None
    for th in np.arange(-6.0, 1.01, 0.2):
        pred = margin > th
        tp = (pred & ~is_unk).sum(); fp = (pred & is_unk).sum()
        fn = (~pred & ~is_unk).sum()
        # 只看 unknown 拒识能力：UR=全测试集判 unknown 比例
        ur = pred.mean()
        if best is None or abs(ur - is_unk.mean()) < abs(best[1] - is_unk.mean()):
            best = (th, ur, tp, fp, fn)
    for th in np.arange(-4.0, 0.41, 0.2):
        pred = margin > th
        rec = (pred & is_unk).sum() / is_unk.sum()
        rej = (pred & ~is_unk).sum() / (~is_unk).sum()
        print(f"    θ={th:+.1f}  判未知 {pred.mean():6.2%}  unknown_recall {rec:6.2%}  "
              f"known_rejection {rej:6.2%}")

    # ── 规则 2：逐类 θ_c ──
    print("\n[规则 2] 逐类偏置：每个已知类有自己的 θ_c")
    # 思想：统计每个已知类"抢走"了多少诚实 unknown 的最高分，
    # 抢得多的类说明容易吸引漏出 → 给它更严的偏置（提高它的门槛）。
    attract = np.zeros(50)
    unk_rows = np.where(is_unk)[0]
    argk = known_z[unk_rows].argmax(1)
    for c in argk:
        attract[c] += 1
    print("    漏出吸引 Top10 类（诚实 unknown 中被该类抢走的张数）：")
    order = np.argsort(-attract)
    id_of = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))["index_to_id"]
    for c in order[:10]:
        print(f"      idx {c:2d} ({id_of[str(c)]})  抢走 {int(attract[c]):4d} 张"
              f"  占该类已知样本 {(attract[c] / max(1, (~is_unk & (y == c)).sum())):.1f}×")

    # 逐类偏置：θ_c = θ_global + λ · (标准化后的吸引量)
    kn_rows = np.where(~is_unk)[0]
    for lam in (0.0, 0.5, 1.0, 1.5, 2.0):
        a = attract / max(1, attract.mean())
        th_c = -5.6 + lam * (a - 1.0)          # 吸引多的类阈值更高（更难被判已知）
        pred = margin > th_c[known_z.argmax(1)]
        rec = (pred & is_unk).sum() / is_unk.sum()
        rej = (pred & ~is_unk).sum() / (~is_unk).sum()
        print(f"    λ={lam:.1f}  判未知 {pred.mean():6.2%}  unknown_recall {rec:6.2%}  "
              f"known_rejection {rej:6.2%}")

    # ── 规则 3：纯置信度阈值 ──
    print("\n[规则 3] 纯置信度：max(z_known) < τ 判 unknown（不使用 unknown 列）")
    for tau in np.arange(-6.0, -0.9, 0.5):
        pred = known_max < tau
        rec = (pred & is_unk).sum() / is_unk.sum()
        rej = (pred & ~is_unk).sum() / (~is_unk).sum()
        print(f"    τ={tau:+.1f}  判未知 {pred.mean():6.2%}  unknown_recall {rec:6.2%}  "
              f"known_rejection {rej:6.2%}")


if __name__ == "__main__":
    main()
