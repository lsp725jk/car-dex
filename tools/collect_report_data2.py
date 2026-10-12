# -*- coding: utf-8 -*-
"""补齐报告数据：真实训练环境 + 工具注册表全清单 + 关键报告片段。"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "report_tables_data.json"
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))

    # ── 真实训练环境（本脚本用带 torch 的解释器跑）──
    env = {}
    import torch
    import torchvision
    env["torch"] = torch.__version__
    env["torchvision"] = torchvision.__version__
    env["cuda"] = torch.version.cuda
    env["cudnn"] = torch.backends.cudnn.version()
    env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    env["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1) if props else None
    env["python"] = sys.version.split()[0]
    env["os"] = f"{platform.system()} {platform.release()}"
    env["os_build"] = platform.version()
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        env["ram_gb"] = "?"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        env["cpu"] = out.splitlines()[0] if out else "?"
        env["cpu_cores"] = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        env["cpu"] = "?"
    d["t4_env"] = env

    # ── 工具注册表全清单 ──
    tools = []
    try:
        from agent.tools import registry as R  # type: ignore
        spec = None
        for attr in ("TOOLS", "TOOL_SPECS", "SPECS", "REGISTRY"):
            if hasattr(R, attr):
                spec = getattr(R, attr)
                break
        if isinstance(spec, dict):
            items = list(spec.values())
        elif isinstance(spec, (list, tuple)):
            items = list(spec)
        else:
            items = []
        for t in items:
            if isinstance(t, dict):
                fn = t.get("function", t)
                tools.append({
                    "name": fn.get("name"),
                    "desc": (fn.get("description") or "")[:400],
                    "params": list(((fn.get("parameters") or {}).get("properties") or {}).keys()),
                    "required": ((fn.get("parameters") or {}).get("required") or []),
                })
    except Exception as e:
        d["tools_import_error"] = f"{type(e).__name__}: {e}"
    d["t7_tools"] = tools
    print(f"工具数：{len(tools)}")

    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[写出]", OUT)
    for k, v in env.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
