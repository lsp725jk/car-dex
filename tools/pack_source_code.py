# -*- coding: utf-8 -*-
"""整理需要提交的源码：生成清单 Markdown + 打包 zip。

包含规则（只提交「跑起来需要的源码与交付文件」）：
  ✅ 源代码    src/ agent/*.py agent/tools/*.py web/ configs/ tools/ scripts/
  ✅ 交付文件  classes.txt requirements.txt start.bat start.ps1 README.md .gitignore
  ✅ 运行数据  agent/data 下的知识库与阈值 JSON（不含会话日志与调用流水）
  ✅ 文档      docs/*.md 与本报告
  ❌ 排除      模型权重（outputs/checkpoints）、数据集（数据集/ data/ data_ood/）、
               会话日志（agent/data/sessions*）、调用流水（tool_calls.jsonl）、
               渲染预览（outputs/report/preview-*）、__pycache__、torchinductor_*

用法：
    python tools/pack_source_code.py            # 生成清单 + 打包
    python tools/pack_source_code.py --dry-run  # 只列文件不打包
"""
from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "23_源码清单与提交说明.md"
OUT_ZIP = ROOT / "outputs" / "report" / "识车档案CarDex_源码包.zip"

# ── 目录级说明（写进清单）──
DIR_DESC = {
    "src": "任务一全部脚本：数据准备、选题、训练、评估、收尾与各类校验",
    "src/cardex": "视觉与训练核心库（唯一实现，训练/推理/工具共用）",
    "agent": "智能体核心：配置、LLM 客户端、调用循环、提示词、会话、错误码",
    "agent/tools": "12 个工具的注册表与实现",
    "web": "Web 前后端：Flask 服务、页面模板与静态资源",
    "configs": "全部配置：类别映射、unknown 来源、各类训练配置与统计",
    "tools": "工程工具：格式核验、提交打包、评测分析、报告生成",
    "scripts": "辅助脚本",
    "docs": "项目文档（Markdown 源）",
    "agent/data": "运行期数据：车型知识库、档次配置与置信度阈值",
}

EXCLUDE_DIRS = {
    "outputs", "data", "data_ood", "数据集", "__pycache__",
    "torchinductor_y", "student_submit", "student_submit_ens4",
    "student_submit_ft", "sessions", "sessions_archive", ".git",
    "preview-v1", "preview-template", "checkpoints",
}
EXCLUDE_FILES = {"tool_calls.jsonl", "garage.db", "predictions.csv",
                 ".env.local", ".env", "runtime_config.json", "sessions_index.json"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log", ".png", ".jpg", ".jpeg", ".pt", ".pth"}

# 安全兜底：绝不允许进入提交包的内容。打包前逐个扫描，命中即中止。
# 起因：第一版规则漏掉了 agent/.env.local，而它含有真实 DEEPSEEK_API_KEY。
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{16,}", "疑似 API Key 明文"),
    (r"DEEPSEEK_API_KEY\s*=\s*[\"']?sk-[A-Za-z0-9_\-]{16,}", "DEEPSEEK_API_KEY 被直接赋真值"),
    (r"AKIA[0-9A-Z]{16}", "疑似 AWS Access Key"),
]
# 占位符白名单：模板/示例文件里出现的「假 Key」不该被误报。
# 例如 agent/.env.example 里的 sk-REPLACE_WITH_YOUR_OWN_KEY 是**应当提交**的示例。
PLACEHOLDER_HINTS = ("REPLACE", "YOUR", "EXAMPLE", "PLACEHOLDER", "XXX",
                     "你的", "****", "YOUR_KEY", "CHANGEME")
FORBIDDEN_NAMES = {".env.local", ".env"}

# 顶层单文件
ROOT_FILES = ["classes.txt", "requirements.txt", "start.bat", "start.ps1",
              "README.md", ".gitignore"]


def first_docline(p: Path) -> str:
    """取 .py 文件的模块 docstring 首句（说明由代码自己提供，不靠手写）。"""
    if p.suffix != ".py":
        return ""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        doc = ast.get_docstring(tree)
        if not doc:
            return ""
        for line in doc.splitlines():
            line = line.strip()
            if line:
                return line[:88]
    except Exception:
        pass
    return ""


def collect() -> tuple[list[Path], list[Path]]:
    """返回 (要打包的文件, 被排除的大件清单)。"""
    keep: list[Path] = []
    skipped_big: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = set(rel.parts[:-1])
        if parts & EXCLUDE_DIRS:
            # 只在顶层被排除的目录里记一笔，避免列出几万个文件
            if rel.parts[0] in {"outputs", "data", "data_ood", "数据集"}:
                continue
            continue
        if p.name in EXCLUDE_FILES or p.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        keep.append(rel)
    return keep, skipped_big


def build_markdown(files: list[Path]) -> str:
    groups: dict[str, list[Path]] = {}
    for f in files:
        top = f.parts[0] if len(f.parts) > 1 else "(根目录)"
        if top in ("src", "agent", "web", "configs", "tools", "docs", "scripts"):
            key = "/".join(f.parts[:2]) if len(f.parts) > 2 and f.parts[1] in (
                "cardex", "tools", "data", "static", "templates") else top
        else:
            key = "(根目录)"
        groups.setdefault(key, []).append(f)

    order = ["(根目录)", "src", "src/cardex", "agent", "agent/tools", "agent/data",
             "web", "configs", "tools", "scripts", "docs"]
    lines: list[str] = []
    A = lines.append
    A("# 23 · 源码清单与提交说明")
    A("")
    A("> 本文档列出需要提交的源码文件与各自作用，并说明哪些内容**不需要**提交。")
    A("> 由 `tools/pack_source_code.py` 自动生成，每个 Python 文件的说明取自它自己的模块 docstring。")
    A("")
    A("## 1 提交包内容")
    A("")
    A(f"合计 **{len(files)} 个文件**。目录职责：")
    A("")
    A("| 目录 | 职责 |")
    A("| --- | --- |")
    for k in order:
        if k in groups and k in DIR_DESC:
            A(f"| `{k}/` | {DIR_DESC[k]} |")
    A("")

    A("## 2 文件明细")
    A("")
    for k in order:
        if k not in groups:
            continue
        A(f"### {k}/")
        A("")
        if k in DIR_DESC:
            A(DIR_DESC[k] + "。")
            A("")
        A("| 文件 | 作用 |")
        A("| --- | --- |")
        for f in sorted(groups[k], key=lambda x: str(x)):
            desc = first_docline(ROOT / f)
            if not desc:
                desc = {
                    ".json": "配置 / 数据",
                    ".md": "文档",
                    ".txt": "文本",
                    ".bat": "启动脚本（Windows 批处理）",
                    ".ps1": "启动脚本（PowerShell）",
                    ".html": "页面模板",
                    ".css": "样式表",
                    ".js": "前端脚本",
                    ".db": "SQLite 数据库",
                }.get(f.suffix.lower(), "—")
            name = f.name if k == "(根目录)" else "/".join(f.parts[1:]) if len(f.parts) > 2 else f.name
            A(f"| `{name}` | {desc} |")
        A("")

    A("## 3 明确不提交的内容")
    A("")
    A("| 内容 | 位置 | 不提交的原因 |")
    A("| --- | --- | --- |")
    A("| 模型权重与中期 checkpoint | `outputs/checkpoints/`（约 1.5 GB） | "
      "报告模板明确说明「模型权重、中期 checkpoint 和最终模型保存在本地，用于运行系统；"
      "无需提交，也不要打包进项目源代码」 |")
    A("| 教师数据集 | `数据集/`（约 6.2 GB） | 教师提供，不应二次分发 |")
    A("| 抽取后的训练/验证数据 | `data/`（约 1.6 GB） | 可由 `src/prepare_data.py` 从数据集重新生成 |")
    A("| 扩充来源类数据集 | `data_ood/` | 硬链接构成，可由 `tools/build_ood_dataset.py` 重新生成 |")
    A("| 会话日志与工具调用流水 | `agent/data/sessions*/`、`tool_calls.jsonl`（约 16 MB） | "
      "运行期产物，含用户对话，不属于源码 |")
    A("| **本地密钥文件** | **`agent/.env.local`** | **含真实 DEEPSEEK_API_KEY，"
      "任何情况下都不提交**（`.gitignore` 已忽略；打包脚本还会在打包前自动扫描并拦截） |")
    A("| 个人车库数据库 | `agent/data/garage.db` | 运行期数据，首次运行会自动建表 |")
    A("| 渲染预览图 | `outputs/report/preview-*/` | 检查排版用的临时产物 |")
    A("| Python 缓存 | `__pycache__/`、`*.pyc` | 可重新生成 |")
    A("")
    A("## 4 收到源码后如何跑起来")
    A("")
    A("```")
    A("1. 安装依赖：pip install -r requirements.txt")
    A("2. 准备模型权重：把训练好的 best.pt 放到 outputs/checkpoints/<run_name>/ 下，")
    A("   集成成员由 outputs/reports/final_model_choice.json 指定")
    A("3. 配置密钥：设置环境变量 DEEPSEEK_API_KEY，或复制 agent/.env.example")
    A("   为 agent/.env.local 并填入（该文件已被 .gitignore 忽略）")
    A("4. 启动：双击 start.bat（会自动完成环境自检并给出修复建议）")
    A("```")
    A("")
    A("若只想离线验证智能体骨架而不消耗 API 额度：`python tools/verify_agent.py`"
      "（默认离线模式，124 项断言）。")
    A("")
    return "\n".join(lines)


def scan_secrets(files: list[Path]) -> list[str]:
    """打包前扫描：任何含明文密钥的文件都不允许进入提交包。"""
    import re
    hits: list[str] = []
    for f in files:
        if f.name in FORBIDDEN_NAMES:
            hits.append(f"{f}  ← 文件名本身即禁止提交")
            continue
        if f.suffix.lower() not in {".py", ".json", ".md", ".txt", ".ps1", ".bat",
                                    ".js", ".html", ".css", ".example", ".cfg", ".ini",
                                    ".yml", ".yaml", ".toml", ""}:
            continue
        try:
            txt = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, why in SECRET_PATTERNS:
            m = re.search(pat, txt)
            if not m:
                continue
            hit = m.group(0).upper()
            if any(h in hit for h in PLACEHOLDER_HINTS):
                continue          # 是示例占位符，放行
            hits.append(f"{f}  ← {why}")
            break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--zip", default=str(OUT_ZIP))
    args = ap.parse_args()

    files, _ = collect()
    total_mb = sum((ROOT / f).stat().st_size for f in files) / 1024**2
    print(f"待提交文件 {len(files)} 个，合计 {total_mb:.1f} MB")

    leaks = scan_secrets(files)
    if leaks:
        print("\n[中止] 提交包中发现疑似密钥，已停止打包：")
        for h in leaks:
            print("   " + h)
        print("\n请把这些文件加入 EXCLUDE_FILES 或从包中移除后重试。")
        raise SystemExit(2)
    print("密钥扫描：通过（未发现明文密钥）")

    md = build_markdown(files)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[写出] {OUT_MD}")

    if args.dry_run:
        for f in files[:60]:
            print("   ", f)
        return

    zpath = Path(args.zip)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(ROOT / f, arcname=str(Path("识车档案CarDex") / f).replace("\\", "/"))
        # 报告本体也一并放进源码包（优先用最新生成的那份）
        rep_dir = ROOT / "outputs" / "report"
        cands = sorted(rep_dir.glob("识车档案CarDex_实训报告*.docx"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            z.write(cands[0], arcname="识车档案CarDex/实训报告_识车档案CarDex.docx")
        # 报告用图与截图（模板要求图表可追溯，一并提交便于核对）
        for sub in ("figures", "screenshots"):
            d = rep_dir / sub
            if d.is_dir():
                for f in sorted(d.glob("*")):
                    if f.is_file():
                        z.write(f, arcname=f"识车档案CarDex/报告图表/{sub}/{f.name}")
    size = zpath.stat().st_size / 1024**2
    with zipfile.ZipFile(zpath) as z:
        n = len(z.namelist())
        bad = z.testzip()
    print(f"[写出] {zpath}  ({n} 个条目，{size:.1f} MB，CRC {'全部通过' if bad is None else bad})")


if __name__ == "__main__":
    main()
