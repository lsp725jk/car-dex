# -*- coding: utf-8 -*-
"""从源码里提取 12 个工具的定义（名称/描述/参数/必填）。"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "reports" / "report_tables_data.json"

# 从每个工具模块的源码里抽 ToolDefinition(...) 的字段
FIELD_RE = {
    "name": re.compile(r'name\s*=\s*"([^"]+)"'),
    "desc": re.compile(r'description\s*=\s*\(?\s*"((?:[^"\\]|\\.)*)"', re.S),
}
MODULE_OF = {
    "vision.py": "视觉", "profile.py": "知识库", "garage.py": "个人车库",
    "trims.py": "配置参数", "browser.py": "联网抓取", "websearch.py": "联网搜索",
}


def main() -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    tools = []
    tdir = ROOT / "agent" / "tools"
    for f in sorted(tdir.glob("*.py")):
        if f.name in ("__init__.py", "registry.py"):
            continue
        src = f.read_text(encoding="utf-8", errors="ignore")
        # 按 ToolDefinition( 出现位置切块
        for m in re.finditer(r"ToolDefinition\s*\(", src):
            start = m.end()
            depth = 1
            i = start
            while i < len(src) and depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            block = src[start:i]
            nm = FIELD_RE["name"].search(block)
            if not nm:
                continue
            dm = FIELD_RE["desc"].search(block)
            desc = ""
            if dm:
                desc = dm.group(1).replace("\\n", " ").replace('" "', "")
                desc = re.sub(r"\s+", " ", desc).strip()
            pnames = re.findall(r'"(\w+)"\s*:\s*\{', block)
            req = re.findall(r'required\s*=\s*\[([^\]]*)\]', block)
            required = re.findall(r'"(\w+)"', req[0]) if req else []
            tools.append({"module": MODULE_OF.get(f.name, f.name), "file": f.name,
                          "name": nm.group(1), "desc": desc[:500],
                          "params": pnames, "required": required})
    # 去重（同名保留首个）
    seen, uniq = set(), []
    for t in tools:
        if t["name"] in seen:
            continue
        seen.add(t["name"])
        uniq.append(t)
    d["t7_tools"] = uniq
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"提取到 {len(uniq)} 个工具：")
    for t in uniq:
        print(f"  [{t['module']:<6}] {t['name']:<22} params={t['params']} req={t['required']}")
        print(f"           {t['desc'][:110]}")


if __name__ == "__main__":
    main()
