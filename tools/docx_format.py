# -*- coding: utf-8 -*-
"""按模板附录的排版要求规范化报告格式。

逐条对照模板「附录 → 排版要求」：
  （1）页面：A4 纵向，上2 下2 左2.5 右2 cm
  （2）标题1/2/3：黑体（**不加粗**）+ Times New Roman；四号/小四/五号；
       段前段后 6 磅、单倍行距
  （3）正文：宋体 + Times New Roman 五号；两端对齐；左右缩进 0；
       首行缩进 2 字符；段前段后 0；**行距固定值 20 磅**
  图：单倍行距、居中、左右缩进 0、**上下环绕**
"""
from __future__ import annotations

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


def _set_ea(run, cjk: str) -> None:
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rf)
    rf.set(qn("w:eastAsia"), cjk)
    run.font.name = "Times New Roman"


# ── 图片转「上下环绕」 ──
def inline_to_top_bottom(shape) -> bool:
    """把内嵌图（wp:inline）改成上下环绕的浮动图（wp:anchor + wrapTopAndBottom）。

    python-docx 只支持内嵌图，浮动与环绕必须自己改 XML。
    CT_Anchor 的子元素顺序是固定的（schema 强制）：
        simplePos, positionH, positionV, extent, effectExtent,
        wrapTopAndBottom, docPr, cNvGraphicFramePr, graphic
    插错顺序 Word 会直接报「文件已损坏」。
    """
    inline = shape._inline
    if inline is None:
        return False
    parent = inline.getparent()
    tag_anchor = qn("wp:anchor")

    inline.tag = tag_anchor
    for k, v in (("distT", "0"), ("distB", "0"), ("distL", "0"), ("distR", "0"),
                 ("simplePos", "0"), ("relativeHeight", "251658240"),
                 ("behindDoc", "0"), ("locked", "0"),
                 ("layoutInCell", "1"), ("allowOverlap", "1")):
        inline.set(k, v)

    def mk(tag, **attrs):
        # ⚠️ 只对**标签名**做命名空间限定；属性名（x / y / relativeFrom）
        #    是普通名字，不能过 qn()，否则 qn 会因缺少冒号而抛 ValueError。
        return parent.makeelement(qn(tag), dict(attrs))

    simple = mk("wp:simplePos", x="0", y="0")
    ph = mk("wp:positionH", relativeFrom="column")
    ph_align = mk("wp:align")
    ph_align.text = "center"
    ph.append(ph_align)
    pv = mk("wp:positionV", relativeFrom="paragraph")
    pv_off = mk("wp:posOffset")
    pv_off.text = "0"
    pv.append(pv_off)
    wrap = mk("wp:wrapTopAndBottom")

    extent = inline.find(qn("wp:extent"))
    effect = inline.find(qn("wp:effectExtent"))
    docPr = inline.find(qn("wp:docPr"))
    if extent is None or docPr is None:
        return False

    for el in (simple, ph, pv):
        extent.addprevious(el)
    if effect is not None:
        effect.addnext(wrap)
    else:
        extent.addnext(wrap)
    return True


# ── 主规范化 ──
def normalize(doc, *, page_margins=True, headings=True, body=True,
              figures=True, wrap=True) -> dict:
    stat = {"margins": 0, "headings": 0, "body": 0, "figs": 0, "wrapped": 0}

    # （1）页面
    if page_margins:
        for sec in doc.sections:
            sec.top_margin = Cm(2.0)
            sec.bottom_margin = Cm(2.0)
            sec.left_margin = Cm(2.5)
            sec.right_margin = Cm(2.0)
            stat["margins"] += 1

    # （2）标题
    if headings:
        for p in doc.paragraphs:
            name = p.style.name
            if name not in ("Heading 1", "Heading 2", "Heading 3"):
                continue
            if not p.text.strip():
                continue
            for r in p.runs:
                _set_ea(r, "黑体")
                r.font.bold = False
            pf = p.paragraph_format
            pf.space_before = Pt(6)
            pf.space_after = Pt(6)
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            stat["headings"] += 1

    # （3）正文
    if body:
        import re
        for p in doc.paragraphs:
            if p.style.name != "Normal":
                continue
            txt = p.text.strip()
            if len(txt) < 20:
                continue
            if re.match(r"^(图|表)\d+\s", txt):     # 图题/表题另行处理
                continue
            for r in p.runs:
                _set_ea(r, "宋体")
            pf = p.paragraph_format
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.left_indent = Pt(0)
            pf.right_indent = Pt(0)
            pf.first_line_indent = Pt(21)          # 2 字符 ≈ 21pt（五号）
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(20)               # 固定值 20 磅
            stat["body"] += 1

    # 图/表题：宋体小五号、居中、单倍行距、左右缩进 0
    if figures:
        import re
        for p in doc.paragraphs:
            if p.style.name != "Normal":
                continue
            if not re.match(r"^(图|表)\d+\s", p.text.strip()):
                continue
            for r in p.runs:
                _set_ea(r, "宋体")
                r.font.size = Pt(9)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf = p.paragraph_format
            pf.first_line_indent = Pt(0)
            pf.left_indent = Pt(0)
            pf.right_indent = Pt(0)
            pf.space_before = Pt(3)
            pf.space_after = Pt(3)
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            stat["figs"] += 1

    # 图片：居中 + 单倍行距 + 上下环绕
    for p in doc.paragraphs:
        if not p._p.findall(".//" + qn("w:drawing")):
            continue
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.left_indent = Pt(0)
        pf.right_indent = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE

    if wrap:
        for shape in list(doc.inline_shapes):
            if inline_to_top_bottom(shape):
                stat["wrapped"] += 1

    return stat
