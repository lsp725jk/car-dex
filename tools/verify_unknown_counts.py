# -*- coding: utf-8 -*-
"""核查 unknown_sources.json 里的 train/val 字段究竟是「可用量」还是「实际采样量」。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
t3 = json.loads((ROOT / "outputs" / "reports" / "report_tables_data.json")
                .read_text(encoding="utf-8"))["t3"]
us = json.loads((ROOT / "configs" / "unknown_sources.json").read_text(encoding="utf-8"))
stats = json.loads((ROOT / "configs" / "data_stats.json").read_text(encoding="utf-8"))

print("unknown_sources.json 顶层：")
for k in ("train_per_source", "val_per_source", "source_count",
          "hard_negative_count", "other_brand_count"):
    print(f"   {k} = {us.get(k)}")

print("\n前 3 条来源记录：")
for s in us["sources"][:3]:
    print("  ", {k: v for k, v in s.items() if k in ("id", "name", "train", "val")})

tot_tr = sum(s.get("train", 0) for s in us["sources"])
tot_va = sum(s.get("val", 0) for s in us["sources"])
print(f"\n  各来源 train 求和 = {tot_tr}")
print(f"  各来源 val   求和 = {tot_va}")

pc = stats["per_class"]
print(f"\ndata_stats.json 里 unknown 实际落盘："
      f"train={pc['train']['unknown']['n']}  val={pc['val']['unknown']['n']}")
print(f"  98 类 × train_per_source({us['train_per_source']}) = "
      f"{98 * us['train_per_source']}")
print(f"  98 类 × val_per_source({us['val_per_source']})   = "
      f"{98 * us['val_per_source']}")

print("\n结论：")
if tot_tr != pc["train"]["unknown"]["n"]:
    print("  ⚠️ unknown_sources.json 的 train/val 是**可用量**，不是实际采样量！")
    print(f"     实际采样 = {pc['train']['unknown']['n']} / {pc['val']['unknown']['n']}")
    print("     → 报告表3 与图1 都必须改用实际采样量")
else:
    print("  两者一致，无需修改")
