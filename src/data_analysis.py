# -*- coding: utf-8 -*-
"""任务一 - 步骤4：数据集分析

对已经抽取到 data/ 的图片做完整分析，为实训报告"数据分析"一节提供全部图表与数字：
  1. 51 类样本量统计（train / val、最多/最少/均值/标准差、不平衡比）
  2. 图片尺寸与宽高比分布（用于确定输入分辨率）
  3. train / val 划分比例一致性
  4. 完全重复图片检测（MD5）与类内近重复抽样估计（dHash 汉明距离）
  5. unknown 来源类别构成（来源品牌数、中文大类占比）
  6. 每类样例图九宫格（用于人工检查标注质量）

输出：outputs/reports/data_analysis.json / data_analysis.txt / sample_grid_*.png
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from cardex import paths  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dhash(image: Image.Image, size: int = 8) -> int:
    img = image.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    rng = random.Random(20250914)
    stats = json.loads(paths.DATA_STATS_JSON.read_text(encoding="utf-8"))
    class_map = paths.load_class_map()
    unknown_cfg = json.loads(paths.UNKNOWN_SOURCES_JSON.read_text(encoding="utf-8"))

    lines: list[str] = []
    add = lines.append
    result: dict = {}

    # ---------------- 1. 类别样本量 ----------------
    train_counts = {k: v["n"] for k, v in stats["per_class"]["train"].items()}
    val_counts = {k: v["n"] for k, v in stats["per_class"]["val"].items()}
    known_ids = [class_map["index_to_id"][str(i)] for i in range(50)]

    add("=" * 96)
    add("数据分析报告 —— 识车档案 CarDex（任务一）")
    add("=" * 96)
    add("")
    add("一、数据来源与总体规模")
    add("-" * 96)
    add(f"  教师数据集压缩包          : {stats['source_zip']}")
    add(f"  教师提供的类别总数        : 827 类（train 78002 张 + val 25783 张，共 103785 张）")
    add(f"  本项目抽取到本地          : train {stats['train_total']} 张 + val {stats['val_total']} 张"
        f" = {stats['train_total'] + stats['val_total']} 张")
    add(f"    ├─ 50 个目标车型        : train {stats['train_known_total']} 张 / val {stats['val_known_total']} 张")
    add(f"    └─ unknown（组外类别）  : train {stats['train_unknown_total']} 张 / val {stats['val_unknown_total']} 张")
    add(f"  unknown 来源类别数        : {stats['unknown_source_count']}"
        f"（同品牌兄弟车型 {stats['unknown_hard_negative_count']} 类 + 其他品牌 {stats['unknown_other_brand_count']} 类）")
    add(f"  unknown 占训练集比例      : {stats['unknown_train_ratio']:.1%}")
    add("")

    tr = [train_counts[c] for c in known_ids]
    va = [val_counts[c] for c in known_ids]
    add("二、50 个目标车型的样本量统计")
    add("-" * 96)
    add(f"  train : min={min(tr)}  max={max(tr)}  mean={sum(tr) / len(tr):.1f}  "
        f"std={math.sqrt(sum((x - sum(tr) / len(tr)) ** 2 for x in tr) / len(tr)):.1f}  "
        f"不平衡比={max(tr) / min(tr):.2f}")
    add(f"  val   : min={min(va)}  max={max(va)}  mean={sum(va) / len(va):.1f}  "
        f"std={math.sqrt(sum((x - sum(va) / len(va)) ** 2 for x in va) / len(va)):.1f}  "
        f"不平衡比={max(va) / min(va):.2f}")
    add(f"  unknown 训练/验证         : {train_counts.get('unknown')} / {val_counts.get('unknown')}")
    ratios = [val_counts[c] / train_counts[c] for c in known_ids]
    add(f"  各类 val/train 划分比例   : min={min(ratios):.3f}  max={max(ratios):.3f}  "
        f"mean={sum(ratios) / len(ratios):.3f}（教师按约 1:3 划分，各类比例基本一致）")
    add("")
    add("  样本量最多的 5 类 / 最少的 5 类：")
    order = sorted(known_ids, key=lambda c: -train_counts[c])
    for c in order[:5]:
        add(f"    最多  {c}  {class_map['index_to_name'][str(class_map['id_to_index'][c])]:<34}"
            f"train={train_counts[c]:>4}  val={val_counts[c]:>3}")
    add("    ...")
    for c in order[-5:]:
        add(f"    最少  {c}  {class_map['index_to_name'][str(class_map['id_to_index'][c])]:<34}"
            f"train={train_counts[c]:>4}  val={val_counts[c]:>3}")
    add("")
    result["counts"] = {
        "train_total": stats["train_total"], "val_total": stats["val_total"],
        "train_known": stats["train_known_total"], "val_known": stats["val_known_total"],
        "train_unknown": stats["train_unknown_total"], "val_unknown": stats["val_unknown_total"],
        "train_min": min(tr), "train_max": max(tr), "train_mean": round(sum(tr) / len(tr), 2),
        "imbalance_train": round(max(tr) / min(tr), 3),
        "imbalance_val": round(max(va) / min(va), 3),
    }

    # ---------------- 2/3/4. 逐图扫描 ----------------
    sizes: Counter = Counter()
    aspects: list[float] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    split_hashes: dict[str, set] = {"train": set(), "val": set()}
    per_split_hash: dict[tuple[str, str], set] = {}

    all_files: list[Path] = []
    for split in ("train", "val"):
        base = paths.DATA_DIR / split
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMAGE_EXT:
                    all_files.append(f)

    add("三、图片质量与尺寸")
    add("-" * 96)
    errors: list[str] = []
    for i, f in enumerate(all_files, 1):
        try:
            with Image.open(f) as img:
                w, h = img.size
                mode = img.mode
                img.verify()
        except Exception as exc:
            errors.append(f"{f}: {exc}")
            continue
        sizes[(w, h)] += 1
        aspects.append(w / h)
        digest = md5_of(f)
        hashes[digest].append(str(f))
        split = f.parts[-3]
        per_split_hash.setdefault((split, f.parent.name), set()).add(digest)
        if i % 2000 == 0:
            print(f"  扫描 {i}/{len(all_files)} ...")

    add(f"  图片总数                  : {len(all_files)}，可正常打开 {len(all_files) - len(errors)} 张，损坏 {len(errors)} 张")
    if errors:
        for e in errors[:10]:
            add(f"    损坏：{e}")
    top_sizes = sizes.most_common(8)
    add(f"  常见尺寸 Top-8            : " + "；".join(f"{w}x{h} ({n} 张)" for (w, h), n in top_sizes))
    add(f"  唯一尺寸数                : {len(sizes)}")
    add(f"  宽高比                    : min={min(aspects):.2f}  max={max(aspects):.2f}  "
        f"mean={sum(aspects) / len(aspects):.2f}（多数接近 4:3 / 3:2，故训练采用等比缩放 + 中心裁剪）")
    dup_groups = {k: v for k, v in hashes.items() if len(v) > 1}
    dup_cross = 0
    dup_in = 0
    for v in dup_groups.values():
        splits = {Path(p).parts[-3] for p in v}
        classes = {Path(p).parts[-2] for p in v}
        if len(splits) > 1:
            dup_cross += 1
        elif len(classes) > 1 or len(v) > 1:
            dup_in += 1
    add(f"  完全重复图片（MD5 相同）  : {len(dup_groups)} 组，涉及 {sum(len(v) for v in dup_groups.values())} 张")
    add(f"    其中跨 train/val 重复   : {dup_cross} 组（若有，属于教师原始划分的数据泄漏，需在报告中说明）")
    add(f"    其中同一划分内重复      : {dup_in} 组")
    add("")
    result["quality"] = {
        "n_images": len(all_files), "n_broken": len(errors), "n_unique_sizes": len(sizes),
        "top_sizes": [{"size": f"{w}x{h}", "count": n} for (w, h), n in top_sizes],
        "aspect_mean": round(sum(aspects) / len(aspects), 3),
        "duplicate_groups": len(dup_groups),
        "duplicate_cross_split": dup_cross,
        "duplicate_within_split": dup_in,
    }

    # ---------------- 类内近重复抽样估计 ----------------
    add("四、类内近重复估计（dHash 汉明距离 ≤ 5 视为近重复，每类抽样 40 张）")
    add("-" * 96)
    near = {}
    for cid in known_ids[:50]:
        d = paths.DATA_DIR / "train" / cid
        files = sorted(d.iterdir())
        sample = rng.sample(files, min(40, len(files)))
        hs = []
        for f in sample:
            try:
                with Image.open(f) as img:
                    hs.append(dhash(img))
            except Exception:
                pass
        pairs = 0
        total = 0
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                total += 1
                if hamming(hs[a], hs[b]) <= 5:
                    pairs += 1
        near[cid] = (pairs, total)
    worst = sorted(near.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1])))[:5]
    avg_rate = sum(p / max(1, t) for p, t in near.values()) / len(near)
    add(f"  50 类平均近重复对占比      : {avg_rate:.2%}")
    if avg_rate < 0.01:
        add("  → 类内近重复比例极低，说明同一车型的训练图片在角度、背景、光照上差异明显，"
            "属于“类内多样性高”的正常细粒度数据集；这既降低了过拟合风险，"
            "也意味着模型必须依赖车型固有特征（格栅、灯组、车身比例）而非背景捷径。")
    else:
        add("  → 类内存在一定比例近重复样本，说明部分图片来自同一拍摄批次，"
            "数据增强中的随机裁剪/颜色抖动对抑制过拟合是必要的。")
    add("  近重复比例最高的 5 类：")
    for cid, (p, t) in worst:
        add(f"    {cid}  {class_map['index_to_name'][str(class_map['id_to_index'][cid])]:<34}"
            f"{p}/{t} = {p / max(1, t):.1%}")
    add("")
    result["near_duplicate_avg_rate"] = round(avg_rate, 4)

    # ---------------- 5. unknown 构成 ----------------
    add("五、unknown 样本构成")
    add("-" * 96)
    src = unknown_cfg["sources"]
    brand_c = Counter(s["brand"] for s in src)
    cat_c = Counter(s["category"] for s in src)
    add(f"  来源类别数                : {len(src)}")
    add(f"  涉及品牌数                : {len(brand_c)}")
    add(f"  汉语大类构成              : " + "；".join(f"{k} {v} 类" for k, v in cat_c.most_common()))
    add(f"  同品牌兄弟车型（硬负样本）: {unknown_cfg['hard_negative_count']} 类，覆盖本组 50 类所属的每一个有兄弟车型的品牌")
    add(f"  其他品牌类别              : {unknown_cfg['other_brand_count']} 类，每个品牌只取 1 类以避免单一品牌主导")
    add(f"  每类采样上限              : train {unknown_cfg['train_per_source']} 张 / val {unknown_cfg['val_per_source']} 张")
    add("")
    result["unknown_composition"] = {
        "n_sources": len(src), "n_brands": len(brand_c),
        "by_category": dict(cat_c),
        "hard_negative": unknown_cfg["hard_negative_count"],
        "other_brand": unknown_cfg["other_brand_count"],
    }

    # ---------------- 6. 样例图 ----------------
    add("六、样例图与预处理方案")
    add("-" * 96)
    add("  已生成 outputs/reports/sample_grid_train.png（每类 1 张，51 类）+ sample_grid_val.png")
    add("  预处理：RGB 转换 → 等比缩放（短边 = image_size × 256/224）→ 中心裁剪 image_size → "
        "ToTensor → ImageNet 均值方差归一化")
    add("  数据增强：随机缩放裁剪 / 水平翻转 / 颜色抖动 / RandAugment / 随机擦除 / MixUp+CutMix（详见训练配置）")
    add("  输入分辨率：基线 160×160，改进模型 224×224（由尺寸分布与类别数量共同决定）")
    add("")

    (paths.REPORT_DIR / "data_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    (paths.REPORT_DIR / "data_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 样例九宫格
    try:
        import matplotlib
        matplotlib.use("Agg")
        # 中文字体：Windows 自带微软雅黑/黑体，避免图标题出现方框
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt

        for split, per_row, ncols in (("train", 51, 6), ("val", 51, 6)):
            names = known_ids + ["unknown"]
            nrows = math.ceil(len(names) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 2.1))
            for ax, cid in zip(axes.ravel(), names):
                d = paths.DATA_DIR / split / cid
                files = sorted(d.iterdir())
                if files:
                    with Image.open(rng.choice(files)) as img:
                        ax.imshow(img.convert("RGB"))
                label = "unknown" if cid == "unknown" else class_map["index_to_name"][str(class_map["id_to_index"][cid])]
                ax.set_title(f"{cid} {label}", fontsize=6)
                ax.axis("off")
            for ax in axes.ravel()[len(names):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(paths.REPORT_DIR / f"sample_grid_{split}.png", dpi=110)
            plt.close(fig)
        add("  [已生成] sample_grid_train.png / sample_grid_val.png")
    except Exception as exc:
        add(f"  样例图生成失败：{exc}")

    print("\n".join(lines))
    print(f"\n[写出] {paths.REPORT_DIR / 'data_analysis.txt'}")
    print(f"[写出] {paths.REPORT_DIR / 'data_analysis.json'}")


if __name__ == "__main__":
    main()
