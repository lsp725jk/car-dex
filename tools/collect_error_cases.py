# -*- coding: utf-8 -*-
"""从 predictions.csv 与验证集真值里挑出典型错误样本，生成报告表 6 的行。

覆盖模板要求的三类：
  · 已知车型被判 unknown（误拒）
  · 未知车型被判已知类（漏出）
  · 已知车型被判成另一个已知类（类间混淆）
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "val"
PRED = ROOT / "predictions.csv"
OUT = ROOT / "outputs" / "reports" / "report_tables_data.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    id2name = {cm["index_to_id"][str(i)]: cm["index_to_name"][str(i)] for i in range(50)}

    # 真值：目录名即类别，unknown 目录为 unknown
    truth: dict[str, str] = {}
    for d in VAL.iterdir():
        if not d.is_dir():
            continue
        label = d.name
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXT:
                truth[p.name] = label

    rows = []
    with PRED.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = r["image_id"]
            t = truth.get(iid)
            if t is None:
                continue
            rows.append({"image_id": iid, "truth": t, "pred": r["predicted_label"],
                         "conf": float(r["confidence"])})

    known_wrong_unknown, unk_leak, confusion = [], [], []
    for r in rows:
        t, p = r["truth"], r["pred"]
        if t != "unknown" and p == "unknown":
            known_wrong_unknown.append(r)
        elif t == "unknown" and p != "unknown":
            unk_leak.append(r)
        elif t != "unknown" and p != "unknown" and t != p:
            confusion.append(r)

    def label(x):
        return "unknown" if x == "unknown" else f"{x} {id2name.get(x, '')}"

    # 每类挑 confidence 最高（最自信的错误）与最低各若干
    def pick(lst, n=6):
        lst = sorted(lst, key=lambda r: -r["conf"])
        return lst[:n]

    picked = []
    for r in pick(unk_leak, 5):
        picked.append({**r, "kind": "未知→已知（漏出）",
                       "line": f"{label(r['truth'])} → {label(r['pred'])}",
                       "why": "组外车型与某已知类造型接近，unknown 头未压制住该类得分"})
    for r in pick(known_wrong_unknown, 3):
        picked.append({**r, "kind": "已知→unknown（误拒）",
                       "line": f"{label(r['truth'])} → unknown",
                       "why": "目标车型样本受遮挡/角度罕见，unknown 头得分超过该类"})
    for r in pick(confusion, 2):
        picked.append({**r, "kind": "已知→他类（混淆）",
                       "line": f"{label(r['truth'])} → {label(r['pred'])}",
                       "why": "同品牌/同车身形态的两款车型易混"})

    stats = {
        "n_val": len(rows),
        "n_known_wrong_unknown": len(known_wrong_unknown),
        "n_unknown_leak": len(unk_leak),
        "n_confusion": len(confusion),
    }
    # 混淆对统计
    pairs = defaultdict(int)
    for r in confusion:
        pairs[(r["truth"], r["pred"])] += 1

    d = json.loads(OUT.read_text(encoding="utf-8"))
    d["t6_rows"] = picked
    d["t6_stats"] = stats
    d["t6_confusion_pairs"] = [
        {"truth": a, "truth_name": id2name.get(a, ""), "pred": b,
         "pred_name": id2name.get(b, ""), "n": n}
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]]
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"验证集 {stats['n_val']} 张")
    print(f"  未知→已知（漏出）  {stats['n_unknown_leak']}")
    print(f"  已知→unknown（误拒）{stats['n_known_wrong_unknown']}")
    print(f"  已知→他类（混淆）  {stats['n_confusion']}")
    print("\n典型错误样本：")
    for r in picked:
        print(f"  [{r['kind']}] {r['image_id']:<18} {r['line']:<34} conf={r['conf']:.4f}")
    print("\n最易混类别对：")
    for c in d["t6_confusion_pairs"][:5]:
        print(f"  {c['truth']} {c['truth_name']} → {c['pred']} {c['pred_name']}  ×{c['n']}")
    print(f"\n[写出] {OUT}")


if __name__ == "__main__":
    main()
