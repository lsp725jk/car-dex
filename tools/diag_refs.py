# -*- coding: utf-8 -*-
"""诊断参考文献区的重复条目与正文缺失的引用标注。"""
import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
doc = Document(str(ROOT / "outputs" / "report" / "识车档案CarDex_实训报告.docx"))

paras = list(doc.paragraphs)
start = 0
for i, p in enumerate(paras):
    if p.text.strip().startswith("参考文献"):
        start = i
        break

print(f"参考文献标题在第 {start} 段；之后共 {len(paras)-start} 段\n")
print("=== 编号出现次数 ===")
from collections import Counter
nums = []
for i in range(start, len(paras)):
    t = paras[i].text.strip()
    m = re.match(r"^\[(\d+)\]", t)
    if m:
        nums.append((int(m.group(1)), i, t[:60]))
c = Counter(n for n, _, _ in nums)
dup = {k: v for k, v in c.items() if v > 1}
print(f"  条目总数 {len(nums)}，唯一编号 {len(c)}，重复编号 {dup}\n")

print("=== 重复编号的段落 ===")
seen = set()
for n, i, t in nums:
    if n in dup:
        tag = "★重复" if n in seen else "  首次"
        seen.add(n)
        print(f"  {tag} 段#{i}  [{n}] {t}")

print("\n=== 正文引用标注检查 ===")
body = "".join(p.text for p in paras[:start])
marks = sorted({int(m) for m in re.findall(r"\[(\d+)\]", body)})
print(f"  正文标注：{marks}")
print(f"  列表编号：{sorted(c)}")
print(f"  列表有但正文没标：{[k for k in sorted(c) if k not in marks]}")
