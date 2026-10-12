# -*- coding: utf-8 -*-
"""列出模板里「不带【】但属于填写指导」的段落，用于在产物中清理。

判据：在模板中出现、非【】包裹、且属于指导/要求性文字（而非标题）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"C:\Users\y\Downloads\实训报告模板_视觉智能体综合实训.docx")
OUT = ROOT / "outputs" / "report" / "识车档案CarDex_实训报告.docx"


def iter_paras(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)


def body_range(doc):
    """定位正文（第 3 个 Heading 1「1 项目概述」到「参考文献」之间）。"""
    paras = list(iter_paras(doc))
    start = end = None
    for i, p in enumerate(paras):
        t = p.text.strip()
        if start is None and t == "1 项目概述":
            start = i
        if t.startswith("参考文献"):
            end = i
            break
    return paras, start, end


def main() -> None:
    tpl = Document(str(TEMPLATE))
    tp, ts, te = body_range(tpl)
    out = Document(str(OUT))
    op, os_, oe = body_range(out)

    print(f"模板正文段落 {ts}..{te}（共 {te-ts}）；产物正文段落 {os_}..{oe}（共 {oe-os_}）\n")
    print("模板中「非【】的指导性段落」：")
    n = 0
    for i in range(ts, te):
        t = tp[i].text.strip()
        if not t or t.startswith("【"):
            continue
        if tp[i].style.name.startswith("Heading"):
            continue
        if len(t) < 12:
            continue
        n += 1
        print(f"  [模板#{i}] {t[:96]}")
    print(f"\n合计 {n} 段")

    print("\n产物中仍然存在的同文段落（需要清理的候选）：")
    tset = {tp[i].text.strip() for i in range(ts, te)
            if tp[i].text.strip() and not tp[i].text.strip().startswith("【")
            and not tp[i].style.name.startswith("Heading")}
    m = 0
    for i in range(os_, oe):
        t = op[i].text.strip()
        if t in tset:
            m += 1
            print(f"  [产物#{i}] {t[:96]}")
    print(f"\n合计 {m} 段残留")


if __name__ == "__main__":
    main()
