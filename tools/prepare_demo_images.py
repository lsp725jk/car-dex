# -*- coding: utf-8 -*-
"""为智能体演示挑选测试图，并复制到 Web 上传目录，供页面内 fetch 后用 DataTransfer 构造 File。

选图原则：
  · 一张**已知类**的清晰图（走通完整业务链：识别→查档案→联网→写入）
  · 一张**未见过来源类**的图（验证 unknown 路径不编造车型）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "val"
UP = ROOT / "data" / "uploads"
UP.mkdir(parents=True, exist_ok=True)

cm = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
name_of = {cm["index_to_id"][str(i)]: cm["index_to_name"][str(i)] for i in range(50)}

picks = []


def first_img(d: Path):
    if not d.is_dir():
        return None
    fs = sorted(p for p in d.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return fs[0] if fs else None


# ① 已知类：宝马5系（0632，样本量最大的类）
for cid in ("0632", "0975", "0950"):
    p = first_img(VAL / cid)
    if p:
        dst = UP / f"demo_known_{cid}{p.suffix.lower()}"
        shutil.copy2(p, dst)
        picks.append({"kind": "known", "id": cid, "name": name_of.get(cid, ""),
                      "src": str(p.relative_to(ROOT)), "dst": dst.name})
        break

# ② 已知类：另一个品牌，用于多轮/对比
p = first_img(VAL / "0975")
if p:
    dst = UP / f"demo_known_0975{p.suffix.lower()}"
    shutil.copy2(p, dst)
    picks.append({"kind": "known", "id": "0975", "name": name_of.get("0975", ""),
                  "src": str(p.relative_to(ROOT)), "dst": dst.name})

# ③ unknown：来自训练未覆盖的来源类（0711/0702 是 98 个来源类之一，
#    但这里用诚实 val 的类别更贴近真实测试集）
honest_dir = None
for cid in ("0604", "0702", "0711"):
    d = VAL / "unknown"
    if d.is_dir():
        cands = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                       and p.name.startswith(cid))
        if cands:
            honest_dir = cands[0]
            break
if honest_dir:
    dst = UP / f"demo_unknown_{honest_dir.stem}{honest_dir.suffix.lower()}"
    shutil.copy2(honest_dir, dst)
    picks.append({"kind": "unknown", "id": honest_dir.stem.split("_")[0],
                  "name": "(组外车型)", "src": str(honest_dir.relative_to(ROOT)),
                  "dst": dst.name})

# ④ 损坏图片：用于异常路径演示
bad = UP / "demo_broken.jpg"
bad.write_bytes(b"this is not a real jpeg")
picks.append({"kind": "broken", "id": "-", "name": "(损坏文件)",
              "src": "(构造)", "dst": bad.name})

out = ROOT / "outputs" / "report" / "demo_images.json"
out.write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"上传目录：{UP}")
for p in picks:
    size = (UP / p["dst"]).stat().st_size
    print(f"  [{p['kind']:<8}] {p['dst']:<34} {size/1024:7.1f} KB  "
          f"来源 {p['src']}")
print(f"\n[写出] {out}")
