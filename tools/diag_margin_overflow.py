# -*- coding: utf-8 -*-
"""诊断：为什么渲染时正文会超出右边界。检查节属性与正文段落的缩进设置。"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu

ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    "模板": Path(r"C:\Users\y\Downloads\实训报告模板_视觉智能体综合实训.docx"),
    "产物": ROOT / "outputs" / "report" / "识车档案CarDex_实训报告.docx",
}


def emu2cm(v):
    return f"{Emu(int(v)).cm:.2f}cm" if v is not None else "-"


def main() -> None:
    for tag, p in DOCS.items():
        print("=" * 78)
        print(f"{tag}：{p.name}")
        print("=" * 78)
        doc = Document(str(p))
        for i, sec in enumerate(doc.sections):
            pw, lw, rw = sec.page_width, sec.left_margin, sec.right_margin
            usable = (pw - lw - rw) if (pw and lw is not None and rw is not None) else None
            print(f"  节{i}: 纸宽 {emu2cm(pw)}  左 {emu2cm(lw)}  右 {emu2cm(rw)}"
                  f"  → 可用宽 {emu2cm(usable)}")
            sp = sec._sectPr
            cols = sp.find(qn("w:cols"))
            if cols is not None:
                print(f"         cols: {cols.attrib}")
        # 抽 3 个正文段落看缩进
        shown = 0
        for para in doc.paragraphs:
            if para.style.name != "Normal" or len(para.text.strip()) < 40:
                continue
            pPr = para._p.find(qn("w:pPr"))
            ind = pPr.find(qn("w:ind")) if pPr is not None else None
            jc = pPr.find(qn("w:jc")) if pPr is not None else None
            print(f"  [Normal] ind={dict(ind.attrib) if ind is not None else None}"
                  f" jc={jc.attrib if jc is not None else None}")
            print(f"           「{para.text.strip()[:50]}…」")
            shown += 1
            if shown >= 3:
                break
        # Normal 样式定义
        for st in doc.styles:
            if st.name == "Normal":
                el = st.element
                ind = el.find(qn("w:pPr") + "/" + qn("w:ind")) if el.find(qn("w:pPr")) is not None else None
                print(f"  Normal 样式 ind = {dict(ind.attrib) if ind is not None else None}")
        print()


if __name__ == "__main__":
    main()
