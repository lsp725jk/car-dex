# -*- coding: utf-8 -*-
"""构造「扩充 unknown 来源类」的训练/验证集（**硬链接，不占磁盘**）。

## 为什么

实测：unknown 头只在 98 个来源类上训练过（占可用 777 类的 12.6%），
在见过的来源类上 AUC 0.9966，在没见过的类上只有 0.9773 —— 严重过拟合。

测试集的 unknown 来自那 777 个类。训练覆盖 98 个 = 12.6%；
扩到 677 个 = **87.1%**，直接命中绝大部分测试分布。

## 三个数据集

    data_ood/train/            50 已知类 + unknown(98 原有 + 579 新增来源类)
    data_ood/val_honest/       50 已知类 + unknown(**100 个完全没参与训练**的来源类)
    data_ood/val_leaky/        50 已知类 + unknown(训练用过的来源类)  ← 上界参考

诚实 val 给下界、泄漏 val 给上界，测试集真实表现落在两者之间（按来源覆盖率加权）。

## 硬链接

同盘用 os.link，秒级完成且不占空间；跨盘自动退化为复制。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "数据集"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link(src: Path, dst: Path) -> str:
    """硬链接；失败则复制。返回 'link' / 'copy' / 'skip'。"""
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_ood"))
    ap.add_argument("--n-new-sources", type=int, default=579)
    ap.add_argument("--n-honest-sources", type=int, default=100)
    ap.add_argument("--per-source", type=int, default=12,
                    help="每个**新增**来源类取多少张训练图")
    ap.add_argument("--val-per-source", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20250923)
    args = ap.parse_args()

    out = Path(args.out)
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    targets = [cm["index_to_id"][str(i)] for i in range(50)]
    used = {s["id"] for s in json.loads(
        (ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))["sources"]}
    info = json.loads((DS / "cls_info" / "class_info.json").read_text(encoding="utf-8"))
    pool = sorted({str(x["id"]).zfill(4) for x in info} - set(targets) - used)
    print(f"目标类 {len(targets)}；已用来源类 {len(used)}；未使用候选 {len(pool)}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    honest = sorted(pool[: args.n_honest_sources])
    new_src = sorted(pool[args.n_honest_sources: args.n_honest_sources + args.n_new_sources])
    train_src = sorted(set(used) | set(new_src))
    print(f"划出：诚实 val 来源类 {len(honest)}；新增训练来源类 {len(new_src)}")
    print(f"训练来源类合计 {len(train_src)}（{len(train_src) / (len(pool) + len(used)):.1%} 覆盖）\n")

    plan = json.loads((ROOT / "configs" / "unknown_sources.json")
                      .read_text(encoding="utf-8"))
    stats: Counter = Counter()

    def do(src: Path, dst: Path) -> None:
        stats[link(src, dst)] += 1

    # ── 已知类：train 与 val 全量 ──
    for cid in targets:
        for p in files(ROOT / "data" / "train" / cid):
            do(p, out / "train" / cid / p.name)
        for p in files(ROOT / "data" / "val" / cid):
            do(p, out / "val_honest" / cid / p.name)
            do(p, out / "val_leaky" / cid / p.name)

    # ── unknown / train：原有 98 类（保留原采样）+ 新增来源类 ──
    for p in files(ROOT / "data" / "train" / "unknown"):
        do(p, out / "train" / "unknown" / p.name)
    for cid in new_src:
        fs = files(DS / "train" / cid)
        for p in rng.sample(fs, min(args.per_source, len(fs))):
            do(p, out / "train" / "unknown" / p.name)

    # ── unknown / val_honest：100 个从未参与训练的来源类 ──
    for cid in honest:
        fs = files(DS / "val" / cid)
        for p in rng.sample(fs, min(args.val_per_source, len(fs))):
            do(p, out / "val_honest" / "unknown" / p.name)

    # ── unknown / val_leaky：训练用过来源类的 val 图（上界参考，仅用于诊断）──
    for cid in train_src:
        fs = files(DS / "val" / cid)
        for p in rng.sample(fs, min(args.val_per_source, len(fs))):
            do(p, out / "val_leaky" / "unknown" / p.name)

    print(f"硬链接 {stats['link']} 个，复制 {stats['copy']} 个，跳过 {stats['skip']} 个\n")
    for split in ("train", "val_honest", "val_leaky"):
        base = out / split
        n_known = sum(len(files(base / c)) for c in targets)
        n_unk = len(files(base / "unknown"))
        tot = n_known + n_unk
        print(f"  {split:<12} 已知 {n_known:>6}  unknown {n_unk:>6}  "
              f"合计 {tot:>6}  unknown 占比 {n_unk / max(1, tot):.1%}")

    meta = {"n_targets": len(targets), "used_sources": sorted(used),
            "new_train_sources": new_src, "honest_val_sources": honest,
            "train_sources_all": train_src, "per_source": args.per_source,
            "coverage_of_777": len(train_src) / 777.0, "seed": args.seed}
    (out / "split.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"\n[写出] {out / 'split.json'}")


if __name__ == "__main__":
    main()
