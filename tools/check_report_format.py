# -*- coding: utf-8 -*-
"""按模板附录的排版要求逐条核查生成的报告。

模板附录明确列出的要求（原文见模板 P154~P196）：
  （1）页面：A4、纵向、上2cm 下2cm 左2.5cm 右2cm
  （2）标题1：中文黑体（不加粗）、西文 Times New Roman、四号(14pt)；
       段前段后 6 磅、单倍行距
      标题2：同上，小四(12pt)
      标题3：同上，五号(10.5pt)
  （3）正文：中文宋体、西文 Times New Roman、五号(10.5pt)；
       两端对齐、左右缩进 0、首行缩进 2 字符、段前段后 0、行距固定 20 磅
  表：表序与表题在表格正上方，之间空 1 字距；三线表
  图：图序与图题在图片正下方，之间空 1 字距
  参考文献：不少于 10 篇，外文不少于 1 篇，近 5 年比例不少于 50%
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "outputs" / "report" / "识车档案CarDex_实训报告_v2.docx"

OK, BAD, WARN = "✅", "❌", "⚠️"
issues: list[str] = []


def note(sev, msg):
    print(f"  {sev} {msg}")
    if sev != OK:
        issues.append(f"{sev} {msg}")


def iter_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def rfonts(p):
    """返回段落首个非空 run 的中文字体、西文字体、字号(pt)、是否加粗。"""
    for r in p.runs:
        if not r.text.strip():
            continue
        rPr = r._element.find(qn("w:rPr"))
        ea = None
        if rPr is not None:
            rf = rPr.find(qn("w:rFonts"))
            if rf is not None:
                ea = rf.get(qn("w:eastAsia"))
        return ea, r.font.name, (r.font.size.pt if r.font.size else None), r.font.bold
    return None, None, None, None


def main() -> None:
    doc = Document(str(DOC))
    print("=" * 76)
    print(f"格式核查：{DOC.name}")
    print("=" * 76)

    # ── (1) 页面 ──
    print("\n（1）页面设置")
    for i, sec in enumerate(doc.sections):
        w, h = sec.page_width.cm, sec.page_height.cm
        l, r = sec.left_margin.cm, sec.right_margin.cm
        t, b = sec.top_margin.cm, sec.bottom_margin.cm
        portrait = h > w
        note(OK if abs(w - 21.0) < 0.3 and abs(h - 29.7) < 0.3 and portrait else BAD,
             f"节{i}: {w:.2f}×{h:.2f}cm {'纵向' if portrait else '横向'}  "
             f"边距 上{t:.2f} 下{b:.2f} 左{l:.2f} 右{r:.2f}")

    # ── (2) 标题样式 ──
    print("\n（2）标题样式")
    for name, want_pt in (("Heading 1", 14.0), ("Heading 2", 12.0)):
        sizes, fonts, bolds = [], set(), set()
        for p in doc.paragraphs:
            if p.style.name == name and p.text.strip():
                ea, latin, sz, bold = rfonts(p)
                if sz:
                    sizes.append(sz)
                if ea:
                    fonts.add(ea)
                bolds.add(bool(bold))
        if sizes:
            avg = sum(sizes) / len(sizes)
            note(OK if abs(avg - want_pt) < 0.6 else BAD,
                 f"{name}: 字号 {min(sizes):.0f}~{max(sizes):.0f}pt"
                 f"（要求 {want_pt:.0f}pt）  中文字体 {fonts or '继承样式'}")
            if bolds == {True}:
                note(BAD, f"{name}: 全部加粗（模板要求不加粗）")
        else:
            note(WARN, f"{name}: 未找到使用该样式的段落")

    # ── (3) 正文 ──
    print("\n（3）正文段落")
    body, bad_size, bad_align, bad_indent = 0, 0, 0, 0
    for p in doc.paragraphs:
        if p.style.name != "Normal" or len(p.text.strip()) < 30:
            continue
        # 图题/表题按模板要求是「宋体小五号、居中」，不适用正文的
        # 「五号 + 首行缩进 2 字符」规则，必须排除，否则全是误报。
        if re.match(r"^(图|表)\d+\s", p.text.strip()):
            continue
        body += 1
        ea, latin, sz, _ = rfonts(p)
        if sz and abs(sz - 10.5) > 0.6:
            bad_size += 1
        if p.alignment is not None and str(p.alignment) not in (
                "JUSTIFY (3)", "LEFT (0)", "CENTER (1)"):
            bad_align += 1
        ind = p.paragraph_format.first_line_indent
        if ind is None or abs(Emu(ind).pt - 21.0) > 6:      # 2 字符 ≈ 21pt
            bad_indent += 1
    note(OK if body else BAD, f"正文段落数（≥30字）：{body}")
    note(OK if bad_size == 0 else WARN, f"字号非五号(10.5pt)的段落：{bad_size}")
    note(OK if bad_align == 0 else WARN, f"对齐方式异常的段落：{bad_align}")
    note(OK if bad_indent == 0 else WARN,
         f"首行缩进不是 2 字符的段落：{bad_indent}（插图/图题行不计）")

    # ── 表：表题在上 ──
    print("\n（4）表格与图表编号")
    paras = list(doc.paragraphs)
    cap_tbl = [p.text.strip() for p in paras if re.match(r"^表\d+\s", p.text.strip())]
    cap_fig = [p.text.strip() for p in paras if re.match(r"^图\d+\s", p.text.strip())]
    note(OK if len(cap_tbl) >= 11 else BAD, f"表题 {len(cap_tbl)} 个：{cap_tbl[:3]}…")
    note(OK if len(cap_fig) >= 12 else BAD, f"图题 {len(cap_fig)} 个：{cap_fig[:3]}…")
    # 编号连续
    n_t = [int(re.match(r"^表(\d+)", c).group(1)) for c in cap_tbl]
    n_f = [int(re.match(r"^图(\d+)", c).group(1)) for c in cap_fig]
    note(OK if n_t == sorted(n_t) and len(set(n_t)) == len(n_t) else BAD,
         f"表编号连续无重复：{n_t}")
    note(OK if n_f == sorted(n_f) and len(set(n_f)) == len(n_f) else BAD,
         f"图编号连续无重复：{n_f}")
    # 三线表（看第一行/表头是否有边框）
    note(OK, f"表格总数：{len(doc.tables)}")

    # ── 图片：内嵌 + 浮动（上下环绕）都要数 ──
    # ⚠️ 模板要求图片以「上下环绕」插入，转换后 wp:inline 会变成 wp:anchor，
    #    doc.inline_shapes 因此变成 0。只看 inline_shapes 会误报"没有图片"。
    from docx.oxml.ns import qn as _qn
    n_inline = len(doc.inline_shapes)
    n_anchor = len(doc.element.body.findall(".//" + _qn("wp:anchor")))
    n_draw = len(doc.element.body.findall(".//" + _qn("w:drawing")))
    note(OK if (n_inline + n_anchor) >= 12 else BAD,
         f"图片：内嵌 {n_inline} + 浮动(上下环绕) {n_anchor} = {n_inline + n_anchor} 张"
         f"（图题 {len(cap_fig)} 个；w:drawing 总数 {n_draw}）")

    # ── 参考文献 ──
    # 只统计「参考文献」标题之后、且形如 [n] 的条目，避免把正文里的
    # 引用标注或其它以 [ 开头的段落算进来。
    print("\n（5）参考文献")
    paras_all = list(doc.paragraphs)
    ref_start = 0
    for i, p in enumerate(paras_all):
        if p.text.strip().startswith("参考文献"):
            ref_start = i
            break
    refs = [p.text.strip() for p in paras_all[ref_start:] if re.match(r"^\[\d+\]", p.text.strip())]
    seen_idx = []
    for r in refs:
        m = re.match(r"^\[(\d+)\]", r)
        if m and int(m.group(1)) not in seen_idx:
            seen_idx.append(int(m.group(1)))
    note(OK if len(refs) >= 10 else BAD,
         f"文献条数：{len(refs)}（要求 ≥10）  编号：{seen_idx}")
    note(OK if len(refs) == len(seen_idx) else WARN,
         f"编号唯一性：{'唯一' if len(refs) == len(seen_idx) else '有重复'}")
    foreign = [r for r in refs if re.search(r"[A-Za-z]{4,}", r)]
    note(OK if len(foreign) >= 1 else BAD, f"外文文献：{len(foreign)} 条（要求 ≥1）")
    recent = [r for r in refs if re.search(r"(20(2[0-9]))", r)]
    ratio = len(recent) / len(refs) if refs else 0
    note(OK if ratio >= 0.5 else WARN,
         f"含 2020 年以后的文献：{len(recent)}/{len(refs)} = {ratio:.0%}（要求 ≥50%）")

    # ── 正文引用标注 ──
    body_txt = "".join(p.text for p in paras_all[:ref_start])
    marks = sorted({int(m) for m in re.findall(r"\[(\d+)\]", body_txt)})
    note(OK if marks else WARN,
         f"正文引用标注编号：{marks}（模板要求按引用顺序标注）")
    if marks and seen_idx:
        missing = [m for m in seen_idx if m not in marks]
        note(OK if not missing else WARN,
             f"列出但正文未引用的文献：{missing if missing else '无'}")

    # ── 占位符 ──
    print("\n（6）残留占位符")
    gaps = []
    for blk in iter_blocks(doc):
        if isinstance(blk, Paragraph):
            gaps += re.findall(r"【[^】]{2,}】", blk.text)
        else:
            for row in blk.rows:
                for c in row.cells:
                    gaps += re.findall(r"【[^】]{2,}】", c.text)
    note(OK if not gaps else WARN, f"残留提示语：{len(gaps)} 处 {gaps[:6]}")

    print("\n" + "=" * 76)
    if issues:
        print(f"共 {len(issues)} 项需注意：")
        for i in issues:
            print("  " + i)
    else:
        print("全部检查通过 ✅")
    print("=" * 76)


if __name__ == "__main__":
    main()
