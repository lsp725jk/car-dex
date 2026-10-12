# -*- coding: utf-8 -*-
"""核查提交物里的 50 类是否与 class_map / 训练集目录一致，并摸清数据集真实规模。"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
print("=== class_map.json ===")
for k, v in cm.items():
    if isinstance(v, dict):
        print(f"  {k}: dict({len(v)})  前3 {list(v.items())[:3]}")
    elif isinstance(v, list):
        print(f"  {k}: list({len(v)})  前3 {v[:3]}")
    else:
        print(f"  {k}: {v!r}")

i2id = cm.get("index_to_id", {})
ids = [i2id[str(i)] for i in range(len(i2id))]
print(f"\nindex_to_id 共 {len(ids)} 项  升序={ids == sorted(ids)}")

sub = (ROOT / "student_submit" / "classes.txt").read_text(encoding="utf-8").split()
print(f"classes.txt 共 {len(sub)} 行")
print(f"  与 index_to_id 前 50 完全一致：{sub == ids[:50]}")

# 训练集目录
tr = ROOT / "data" / "train"
dirs = sorted(d.name for d in tr.iterdir() if d.is_dir())
print(f"\n=== data/train ===\n  子目录 {len(dirs)} 个: {dirs[:6]} ... {dirs[-4:]}")
cnt = {d: len(list((tr / d).glob("*"))) for d in dirs}
print(f"  每类图片数：{min(cnt.values())} ~ {max(cnt.values())}  合计 {sum(cnt.values())}")
known_dirs = [d for d in dirs if d != "unknown"]
print(f"  已知类目录 {len(known_dirs)} 个；是否含 unknown：{'unknown' in dirs}")
print(f"  训练集目录名 == classes.txt：{sorted(known_dirs) == sorted(sub)}")
if sorted(known_dirs) != sorted(sub):
    print(f"    仅在目录: {sorted(set(known_dirs) - set(sub))}")
    print(f"    仅在 classes.txt: {sorted(set(sub) - set(known_dirs))}")
print(f"  unknown 目录图片数：{cnt.get('unknown', 0)}")

# 验证集
va = ROOT / "data" / "val"
vdirs = sorted(d.name for d in va.iterdir() if d.is_dir())
vcnt = {d: len(list((va / d).glob("*"))) for d in vdirs}
tot = sum(vcnt.values())
unk = vcnt.get("unknown", 0)
print(f"\n=== data/val ===\n  目录 {len(vdirs)} 个  合计 {tot} 张")
print(f"  已知 {tot - unk} 张（{(tot - unk) / tot:.1%}）  未知 {unk} 张（{unk / tot:.1%}）")
