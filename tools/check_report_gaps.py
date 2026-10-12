# -*- coding: utf-8 -*-
"""扫描生成的报告，找出残留的模板提示语与空表格单元格。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "outputs" / "report" / "识车档案CarDex_实训报告.docx"
PLACEHOLDER = re.compile(r"【[^】]{2,}】")


def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def main() -> None:
    doc = Document(str(DOC))
    gaps: list[str] = []
    for blk in iter_blocks(doc):
        if isinstance(blk, Paragraph):
            for m in PLACEHOLDER.finditer(blk.text):
                gaps.append(f"[段落] {m.group(0)}  ← …{blk.text[max(0, m.start()-30):m.end()+20]}…")
        else:
            for ri, row in enumerate(blk.rows):
                for ci, cell in enumerate(row.cells):
                    for m in PLACEHOLDER.finditer(cell.text):
                        gaps.append(f"[表格 r{ri}c{ci}] {m.group(0)}")

    print(f"文档：{DOC.name}")
    print(f"残留提示语 {len(gaps)} 处")
    for g in gaps[:40]:
        print("  " + g)

    # 表格空单元格统计（排除表头）
    print("\n各表空单元格：")
    for i, t in enumerate(doc.tables, 1):
        empty = 0
        for ri, row in enumerate(t.rows):
            for cell in row.cells:
                if not cell.text.strip():
                    empty += 1
        print(f"  T{i:02d}  {len(t.rows)}×{len(t.columns)}  空单元格 {empty}")


if __name__ == "__main__":
    main()
