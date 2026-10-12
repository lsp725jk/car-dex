# -*- coding: utf-8 -*-
"""任务一 - 提交前自查清单自动校验

对应任务书"九、提交前自查清单"中可由程序验证的部分：
  1. classes.txt 文件名、编码（UTF-8 无 BOM）、行数（恰好 50）、ID 格式（四位数字）、
     唯一性、升序、且全部属于教师提供的 827 类；不含 unknown
  2. class_map.json 与 classes.txt 完全一致（索引 0~49 顺序、索引 50 = unknown）
  3. data/train 与 data/val 下 50 个车型 ID 目录 + unknown 目录齐备，
     且 train / val 图片文件名无交集（无泄漏）
  4. unknown 来源类别与本组 50 类无交集（未把所选车型标成 unknown）
  5. predictions.csv 表头、编码、image_id 唯一性、标签合法性与 confidence 取值范围
  6. 模型 checkpoint 中的类别映射与 classes.txt 一致

用法：
    python src/verify_submission.py
    python src/verify_submission.py --predictions predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from cardex import paths  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Checker:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warned: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        (self.passed if ok else self.failed).append(msg)
        print(("  [OK]   " if ok else "  [FAIL] ") + msg)
        return ok

    def warn(self, msg: str) -> None:
        self.warned.append(msg)
        print("  [WARN] " + msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(paths.PROJECT_ROOT / "predictions.csv"))
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    c = Checker()
    print("=" * 86)
    print("任务一 提交前自查")
    print("=" * 86)

    # ---------------- 1. classes.txt ----------------
    print("\n[1] classes.txt")
    p = paths.CLASSES_TXT
    c.check(p.exists(), f"文件存在：{p}")
    raw = p.read_bytes()
    c.check(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 编码且不带 BOM")
    try:
        text = raw.decode("utf-8")
        c.check(True, "可用 UTF-8 解码")
    except UnicodeDecodeError as exc:
        c.check(False, f"UTF-8 解码失败：{exc}")
        text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    c.check(len(lines) == 50, f"恰好 50 行（实际 {len(lines)}）")
    c.check(all(len(x) == 4 and x.isdigit() for x in lines), "每行都是四位数字 ID（保留前导零）")
    c.check(len(set(lines)) == len(lines), "50 个 ID 不重复")
    c.check(lines == sorted(lines), "按类别 ID 升序排列")
    c.check("unknown" not in text.lower(), "不含 unknown（unknown 是模型第 51 类，不写入本文件）")

    info_ids = {row["id"] for row in json.loads(paths.CLASS_INFO_JSON.read_text(encoding="utf-8"))}
    outside = [x for x in lines if x not in info_ids]
    c.check(not outside, f"全部属于教师提供的 827 类（越界 {len(outside)} 个）")

    # ---------------- 2. class_map.json ----------------
    print("\n[2] 类别映射 configs/class_map.json")
    cm = paths.load_class_map()
    n = int(cm["num_classes"])
    c.check(n == 51, f"输出类别数 = 51（实际 {n}）")
    idx_ids = [cm["index_to_id"][str(i)] for i in range(50)]
    c.check(idx_ids == lines, "索引 0~49 的顺序与 classes.txt 行顺序完全一致")
    c.check(cm["unknown_index"] == 50 and cm["unknown_label"] == "unknown",
            "索引 50 固定映射为小写 unknown")
    names = [cm["index_to_name"][str(i)] for i in range(50)]
    name_map = {r["id"]: r["name_from_new"] for r in json.loads(paths.CLASS_INFO_JSON.read_text(encoding="utf-8"))}
    bad = [i for i in range(50) if names[i] != name_map[idx_ids[i]]]
    c.check(not bad, "车型名称全部取自 class_info.json 的 name_from_new 字段")

    # ---------------- 3. 数据目录 ----------------
    print("\n[3] 数据目录 data/train、data/val")
    train_dir, val_dir = paths.DATA_DIR / "train", paths.DATA_DIR / "val"
    for split, d in (("train", train_dir), ("val", val_dir)):
        if not c.check(d.exists(), f"{d} 存在"):
            continue
        dirs = {x.name for x in d.iterdir() if x.is_dir()}
        missing = [i for i in lines if i not in dirs]
        c.check(not missing, f"{split}/ 下 50 个车型 ID 目录齐备（缺 {len(missing)}）")
        c.check("unknown" in dirs, f"{split}/unknown/ 存在")
        c.check(not (dirs - set(lines) - {"unknown", "__pycache__"}),
                f"{split}/ 下无多余类别目录")

    if train_dir.exists() and val_dir.exists():
        train_names = {f.name for f in train_dir.rglob("*") if f.suffix.lower() in IMAGE_EXT}
        val_names = {f.name for f in val_dir.rglob("*") if f.suffix.lower() in IMAGE_EXT}
        overlap = train_names & val_names
        c.check(not overlap, f"train / val 图片文件名无交集（重叠 {len(overlap)}）")
        c.check(len(train_names) > 0 and len(val_names) > 0,
                f"train {len(train_names)} 张 / val {len(val_names)} 张")

    # ---------------- 4. unknown 来源 ----------------
    print("\n[4] unknown 样本来源")
    us = json.loads(paths.UNKNOWN_SOURCES_JSON.read_text(encoding="utf-8"))
    src_ids = {s["id"] for s in us["sources"]}
    c.check(not (src_ids & set(lines)), "unknown 来源类别与本组 50 类无交集")
    c.check(all(s["id"] in info_ids for s in us["sources"]), "unknown 来源类别均为教师提供的类别")
    c.check(us["hard_negative_count"] > 0, f"包含同品牌兄弟车型硬负样本（{us['hard_negative_count']} 类）")
    brands = {s["brand"] for s in us["sources"]}
    c.check(len(brands) >= 40, f"unknown 来源覆盖品牌数 = {len(brands)}（避免单一品牌主导）")
    cats = {s["category"] for s in us["sources"]}
    c.check(len(cats) >= 5, f"unknown 来源覆盖车身大类数 = {len(cats)}：{'、'.join(sorted(cats))}")

    # ---------------- 5. predictions.csv ----------------
    print("\n[5] predictions.csv")
    pv = Path(args.predictions)
    if not pv.exists():
        c.warn(f"predictions.csv 尚未生成（{pv}）—— 训练完成后用 src/predict_batch.py 生成")
    else:
        praw = pv.read_bytes()
        c.check(not praw.startswith(b"\xef\xbb\xbf"), "UTF-8 编码且不带 BOM")
        with open(pv, encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            header = next(reader)
            rows = list(reader)
        c.check(header == ["image_id", "predicted_label", "confidence"],
                f"表头固定为 image_id,predicted_label,confidence（实际 {header}）")
        c.check(len(rows) > 0, f"记录数 {len(rows)}")
        ids = [r[0] for r in rows]
        c.check(len(set(ids)) == len(ids), "image_id 无重复（每张图片恰好一条记录）")
        c.check(all(i.lower().endswith(tuple(IMAGE_EXT)) for i in ids),
                "image_id 均为包含扩展名的图片文件名")
        labels = set(lines) | {"unknown"}
        bad_lab = [r for r in rows if r[1] not in labels]
        c.check(not bad_lab, f"predicted_label 均为四位车型 ID 或小写 unknown（非法 {len(bad_lab)}）")
        bad_conf = []
        for r in rows:
            try:
                v = float(r[2])
                if not (0.0 <= v <= 1.0):
                    bad_conf.append(r)
            except ValueError:
                bad_conf.append(r)
        c.check(not bad_conf, f"confidence 均为 0~1 的有效数值（越界 {len(bad_conf)}）")
        unk_n = sum(1 for r in rows if r[1] == "unknown")
        print(f"         预测为 unknown：{unk_n} / {len(rows)} ({unk_n / len(rows):.1%})")

    # ---------------- 6. checkpoint ----------------
    print("\n[6] 模型 checkpoint")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None:
        cands = sorted(paths.CKPT_DIR.glob("*/best.pt"), key=lambda x: -x.stat().st_mtime)
        ckpt_path = cands[0] if cands else None
    if ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        c.check(True, f"加载 checkpoint：{ckpt_path}")
        c.check(ck["num_classes"] == 51, f"checkpoint 输出类别数 = {ck['num_classes']}")
        ck_ids = [ck["class_map"]["index_to_id"][str(i)] for i in range(50)]
        c.check(ck_ids == lines, "checkpoint 内的类别映射与 classes.txt 一致")
        c.check(ck["label_names"][-1] == "unknown", "checkpoint 第 51 类标签为 unknown")
        pre = ck.get("preprocess", {})
        c.check(bool(pre), f"已保存预处理参数：{json.dumps(pre, ensure_ascii=False)}")
        c.check("train_config" in ck, "已保存训练配置")
        print(f"         骨干={ck['arch']}  输入尺寸={ck['image_size']}  "
              f"验证 Macro-F1={ck.get('val_macro_f1')}")
    else:
        c.warn("未找到 checkpoint（模型文件不随作业提交，可忽略）")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 86)
    print(f"通过 {len(c.passed)} 项，失败 {len(c.failed)} 项，警告 {len(c.warned)} 项")
    if c.failed:
        print("\n未通过项：")
        for f in c.failed:
            print("  - " + f)
    print("=" * 86)
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
