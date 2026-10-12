# -*- coding: utf-8 -*-
"""汇总实训报告 12 张表所需的真实数据，输出 JSON 供生成脚本使用。

只读，不改动任何项目文件。
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "report_tables_data.json"


def jload(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def main() -> None:
    data: dict = {}

    # ── 表2：50 类映射与样本统计 ──
    cm = jload(ROOT / "configs" / "class_map.json") or {}
    stats = jload(ROOT / "configs" / "data_stats.json") or {}
    per = stats.get("per_class", {})
    rows2 = []
    for i in range(50):
        cid = cm["index_to_id"][str(i)]
        rows2.append({
            "index": i, "id": cid, "name": cm["index_to_name"][str(i)],
            "group": cm["index_to_group"][str(i)],
            "train": per.get("train", {}).get(cid, {}).get("n", 0),
            "val": per.get("val", {}).get(cid, {}).get("n", 0),
        })
    unk_tr = per.get("train", {}).get("unknown", {}).get("n", 0)
    unk_va = per.get("val", {}).get("unknown", {}).get("n", 0)
    data["t2"] = {"rows": rows2, "unknown": {"train": unk_tr, "val": unk_va},
                  "total_train": sum(r["train"] for r in rows2) + unk_tr,
                  "total_val": sum(r["val"] for r in rows2) + unk_va}
    g = Counter(r["group"] for r in rows2)
    data["t2"]["group_dist"] = dict(g)
    data["t2"]["brand_count"] = len({r["name"].split("_")[0] for r in rows2})

    # ── 表3：unknown 来源与采样 ──
    us = jload(ROOT / "configs" / "unknown_sources.json") or {}
    data["t3"] = {
        "note": us.get("note", ""),
        "train_per_source": us.get("train_per_source"),
        "val_per_source": us.get("val_per_source"),
        "source_count": len(us.get("sources", [])),
        "hard_negative_count": us.get("hard_negative_count"),
        "other_brand_count": us.get("other_brand_count"),
        "rows": [{"id": s["id"], "name": s["name"], "brand": s.get("brand", ""),
                  "category": s.get("category", ""), "kind": s.get("source_kind", ""),
                  "target": s.get("target_class_id") or "",
                  "train": s.get("train", 0), "val": s.get("val", 0)}
                 for s in us.get("sources", [])],
    }
    cats = Counter(s.get("category", "") for s in us.get("sources", []))
    brands = {s.get("brand", "") for s in us.get("sources", [])}
    data["t3"]["category_dist"] = dict(cats)
    data["t3"]["brand_count"] = len(brands)

    # ── 表4：训练环境 ──
    env = {}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无"
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:
        env["torch_error"] = str(e)
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except Exception:
        pass
    env["python"] = sys.version.split()[0]
    env["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    env["machine"] = platform.machine()
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        env["ram_gb"] = "?"
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                             text=True, timeout=15).stdout
        env["cpu"] = next((l.strip() for l in out.splitlines() if l.strip()
                           and "Name" not in l), "?")
    except Exception:
        env["cpu"] = "?"
    env["train_cfg_r50_288"] = jload(ROOT / "configs" / "train_r50_288.json")
    env["train_cfg_r50_416"] = jload(ROOT / "configs" / "train_r50_416.json")
    data["t4"] = env

    # ── 表5：基线与改进实验（来自各 run 的 summary.json）──
    runs = []
    rdir = ROOT / "outputs" / "runs"
    if rdir.is_dir():
        for d in sorted(rdir.iterdir()):
            s = jload(d / "summary.json")
            if not s:
                continue
            runs.append({
                "run": d.name,
                "best_macro_f1": (s.get("best") or {}).get("macro_f1"),
                "best_epoch": (s.get("best") or {}).get("epoch"),
                "minutes": round(s.get("total_time_s", 0) / 60, 1) if s.get("total_time_s") else None,
            })
    data["t5_runs"] = runs

    # 集成/最终选择
    fmc = jload(ROOT / "outputs" / "reports" / "final_model_choice.json") or {}
    data["t5_champion"] = fmc.get("champion")
    data["t5_candidates"] = [
        {"tag": c["tag"], "members": len(c["members"]),
         "raw_f1": c["raw"]["macro_f1"], "best_f1": c["best"]["macro_f1"],
         "tau": c["best"]["tau"], "threshold": c["best"]["threshold"]}
        for c in fmc.get("candidates", [])
    ]
    data["t5_generalization"] = jload(ROOT / "outputs" / "reports" / "generalization_check.json")

    # ── 表6：典型错误样本 ──
    data["t6"] = jload(ROOT / "outputs" / "reports" / "worst_cases.json")

    # ── 第 4 章：评估口径 ──
    data["eval"] = {
        "predictions_score": jload(ROOT / "outputs" / "reports" / "predictions_score.json"),
        "class_report": None,
    }
    cr = ROOT / "outputs" / "reports" / "predictions_classification_report_val.txt"
    if cr.exists():
        data["eval"]["class_report"] = cr.read_text(encoding="utf-8")[:6000]

    # ── 表7：工具定义 ──
    tools_json = ROOT / "outputs" / "reports" / "tools_schema.json"
    reg = ROOT / "agent" / "tools" / "registry.py"
    data["t7"] = {"schema_json_exists": tools_json.exists(), "registry": str(reg)}

    # ── 表8：工程模块位置 ──
    mods = []
    for pat, name in (("src/cardex/*.py", "视觉与训练核心"),
                      ("src/*.py", "任务一脚本"),
                      ("agent/*.py", "智能体核心"),
                      ("agent/tools/*.py", "工具实现"),
                      ("web/**/*", "Web 前后端"),
                      ("configs/*.json", "配置"),
                      ("tools/*.py", "评测与提交工具")):
        base = ROOT
        files = sorted(base.glob(pat))
        files = [f for f in files if f.is_file()]
        loc = 0
        for f in files:
            try:
                loc += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                pass
        mods.append({"name": name, "pattern": pat, "files": len(files), "loc": loc})
    data["t8_modules"] = mods

    # ── 表9/10：测试与修复 ──
    data["tests"] = {}
    for f in (ROOT / "docs").glob("10_真实API与端到端测试报告.md"):
        data["tests"]["e2e_doc"] = f.read_text(encoding="utf-8")[:20000]

    # ── agent 工具调用统计（真实记录）──
    tcl = ROOT / "agent" / "data" / "tool_calls.jsonl"
    if tcl.exists():
        names = Counter()
        n = 0
        try:
            with tcl.open(encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    n += 1
                    try:
                        rec = json.loads(line)
                        nm = rec.get("name") or rec.get("tool") or rec.get("tool_name")
                        if nm:
                            names[nm] += 1
                    except Exception:
                        pass
        except Exception as e:
            data["tool_calls_error"] = str(e)
        data["tool_calls"] = {"total_lines": n, "by_name": dict(names.most_common())}

    # ── 会话统计 ──
    sdir = ROOT / "agent" / "data" / "sessions"
    if sdir.is_dir():
        data["sessions"] = {"count": len(list(sdir.glob("*.jsonl")))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {OUT}")
    print(f"  表2 类别 {len(rows2)} 行，unknown train={unk_tr} val={unk_va}")
    print(f"  表3 unknown 来源 {data['t3']['source_count']} 条")
    print(f"  表5 run 记录 {len(runs)} 条")
    print(f"  工具调用记录 {data.get('tool_calls', {}).get('total_lines', 0)} 行")
    print(f"  会话数 {data.get('sessions', {}).get('count')}")


if __name__ == "__main__":
    main()
