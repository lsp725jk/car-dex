# -*- coding: utf-8 -*-
"""查 docGrid / wordWrap 等可能造成右侧溢出的设置。"""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

DOCS = [("模板", Path(r"C:\Users\y\Downloads\实训报告模板_视觉智能体综合实训.docx")),
        ("产物", Path(r"E:\Hanako\OH-WorkSpace\car-kb-agent\outputs\report"
                     r"\识车档案CarDex_实训报告.docx"))]

for tag, p in DOCS:
    d = Document(str(p))
    print("=" * 70)
    print(tag)
    for i, sec in enumerate(d.sections):
        sp = sec._sectPr
        dg = sp.find(qn("w:docGrid"))
        if dg is None:
            print(f"  节{i} docGrid: 无")
        else:
            print(f"  节{i} docGrid: {etree.tostring(dg, encoding='unicode').strip()}")
    # 样式级
    for st in d.styles:
        el = st.element
        pPr = el.find(qn("w:pPr"))
        if pPr is None:
            continue
        for t in ("w:wordWrap", "w:overflowPunct"):
            e = pPr.find(qn(t))
            if e is not None:
                print(f"  样式 {st.name}: {t} = {e.attrib}")
    # 段落级
    cnt = {}
    for para in d.paragraphs:
        pPr = para._p.find(qn("w:pPr"))
        if pPr is None:
            continue
        for t in ("w:wordWrap", "w:overflowPunct", "w:textAlignment"):
            if pPr.find(qn(t)) is not None:
                cnt[t] = cnt.get(t, 0) + 1
    print(f"  段落级设置统计: {cnt or '无'}")
