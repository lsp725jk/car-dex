# -*- coding: utf-8 -*-
"""检查 esc() 不转义单引号这件事，在前端有没有实际风险。

esc() 只转义 & < > "。如果某处把用户数据插进**单引号包裹**的属性/字符串里，
就可以用 ' 闭合再注入。这里找这种位置。
"""
from __future__ import annotations

import re
from pathlib import Path

src = Path("web/static/app.js").read_text(encoding="utf-8").splitlines()

# 1) attr='...${...}...' （单引号属性里有插值）
pat_attr = re.compile(r"""\b[\w-]+='[^']*\$\{""")
# 2) 模板串里单引号包裹且含插值：'...${...}...'
pat_str = re.compile(r"""'[^'\n]*\$\{[^}]*\}[^'\n]*'""")

hits = []
for i, l in enumerate(src):
    s = l.strip()
    if s.startswith("//"):
        continue
    for pat, kind in ((pat_attr, "单引号属性"), (pat_str, "单引号字符串")):
        for m in pat.finditer(l):
            hits.append((i + 1, kind, m.group(0)[:80]))

print(f"疑似单引号内插值：{len(hits)} 处\n")
for ln, kind, text in hits:
    print(f"  L{ln:<5} [{kind}] {text}")

print()
print("=" * 74)
print("esc() 的定义（确认转义集）")
print("=" * 74)
for i, l in enumerate(src):
    if "const esc" in l:
        for j in range(i, min(i + 3, len(src))):
            print(f"  L{j + 1}: {src[j].strip()}")
        break

print()
print("=" * 74)
print("结论判据")
print("=" * 74)
print("  如果上面『单引号属性』为空 → esc() 不转义 ' 在当前代码里**没有可利用点**")
print("  但仍是隐患：将来有人写 attr='${esc(x)}' 就会破口")
