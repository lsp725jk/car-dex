# -*- coding: utf-8 -*-
"""列出字号/缩进不符合「正文五号 + 首行缩进2字符」的段落，判断是否属于误报。"""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "outputs" / "report" / "识车档案CarDex_实训报告_new.docx"

doc = Document(str(DOC))
n = 0
for i, p in enumerate(doc.paragraphs):
    if p.style.name != "Normal" or len(p.text.strip()) < 30:
        continue
    sz = None
    for r in p.runs:
        if r.text.strip():
            sz = r.font.size.pt if r.font.size else None
            break
    ind = p.paragraph_format.first_line_indent
    ind_pt = Emu(ind).pt if ind is not None else None
    bad_sz = sz is not None and abs(sz - 10.5) > 0.6
    bad_ind = ind_pt is None or abs(ind_pt - 21.0) > 6
    if bad_sz or bad_ind:
        n += 1
        print(f"[{n}] 段#{i}  字号={sz}  首行缩进={ind_pt}  "
              f"{'字号不符 ' if bad_sz else ''}{'缩进不符' if bad_ind else ''}")
        print(f"     {p.text.strip()[:88]}")
print(f"\n合计 {n} 段")
