# -*- coding: utf-8 -*-
"""启动前环境自检。

在启动 Web 系统 / 命令行之前，把「能不能跑起来」这件事一次性问清楚，
避免出现"页面打开了、一上传图片就报 torch 没装"这类问题。

用法：
    python tools/selfcheck.py                 # 快速自检（依赖 + 文件 + 配置）
    python tools/selfcheck.py --api           # 额外做一次真实 API 连通性测试
    python tools/selfcheck.py --vision        # 额外做一次真实视觉推理冒烟测试
    python tools/selfcheck.py --all           # 全量（--api --vision）
    python tools/selfcheck.py --json          # 以 JSON 输出（供启动脚本解析）

退出码：0 = 全部通过（可以有 WARN）；1 = 存在 FAIL（阻断启动）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {OK: "[OK]  ", WARN: "[WARN]", FAIL: "[FAIL]"}


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, group: str, name: str, detail: str = "") -> None:
        self.items.append({"level": level, "group": group, "name": name, "detail": detail})

    def ok(self, g: str, n: str, d: str = "") -> None:
        self.add(OK, g, n, d)

    def warn(self, g: str, n: str, d: str = "") -> None:
        self.add(WARN, g, n, d)

    def fail(self, g: str, n: str, d: str = "") -> None:
        self.add(FAIL, g, n, d)

    @property
    def n_fail(self) -> int:
        return sum(1 for i in self.items if i["level"] == FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for i in self.items if i["level"] == WARN)

    @property
    def n_pass(self) -> int:
        return sum(1 for i in self.items if i["level"] == OK)


# --------------------------------------------------------------------------- #
# 1. Python 与依赖
# --------------------------------------------------------------------------- #
REQUIRED = [
    ("torch", "深度学习框架，视觉模型推理必需"),
    ("torchvision", "ResNet 权重与图像预处理必需"),
    ("flask", "任务三 Web 系统必需"),
    ("requests", "调用 DeepSeek 接口必需"),
    ("PIL", "图片读取与校验必需"),
    ("numpy", "数值计算必需"),
]
OPTIONAL = [
    ("pandas", "数据分析脚本用（任务一可选）"),
    ("sklearn", "Macro-F1 / 混淆矩阵用（任务一可选）"),
    ("matplotlib", "训练曲线绘图用（任务一可选）"),
]


def check_python(rep: Report) -> None:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) >= (3, 10):
        rep.ok("Python", "解释器版本", s)
    else:
        rep.fail("Python", "解释器版本", f"{s} —— 需要 3.10 及以上")


def _version(mod, dist_name: str) -> str:
    """优先用 importlib.metadata（__version__ 在新版本 Flask 等库中已废弃）。"""
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:                                # noqa: BLE001
        return getattr(mod, "__version__", "?")


def check_packages(rep: Report) -> None:
    for name, why in REQUIRED:
        try:
            m = importlib.import_module(name)
            ver = _version(m, name)
            line = f"{name} {ver}"
            if name == "torch":
                line += f"  (CUDA 可用: {getattr(m.cuda, 'is_available', lambda: False)()})"
            rep.ok("依赖", line, why)
        except Exception as e:                       # noqa: BLE001
            rep.fail("依赖", f"{name} 未安装", f"{why} —— {type(e).__name__}；"
                                             f"请执行 pip install -r requirements.txt")
    for name, why in OPTIONAL:
        try:
            m = importlib.import_module(name)
            rep.ok("依赖(可选)", f"{name} {getattr(m, '__version__', '?')}", why)
        except Exception:                            # noqa: BLE001
            rep.warn("依赖(可选)", f"{name} 未安装", f"{why} —— 不影响系统运行")


# --------------------------------------------------------------------------- #
# 2. API Key
# --------------------------------------------------------------------------- #
def check_api_key(rep: Report, cfg) -> None:
    env_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    local = ROOT / "agent" / ".env.local"
    if env_key:
        rep.ok("API Key", "来源：环境变量 DEEPSEEK_API_KEY", cfg.masked_key())
    elif local.exists():
        rep.ok("API Key", "来源：agent/.env.local", cfg.masked_key())
    else:
        rep.warn("API Key", "未配置",
                 "将进入**离线替身模式**：工具链可跑通，但文字由脚本化替身生成，不代表模型效果。"
                 "配置方法见 agent/.env.example")
        return

    if not cfg.api_key_configured:
        rep.fail("API Key", "读取到的 Key 为空", "请检查 agent/.env.local 的 DEEPSEEK_API_KEY= 一行")

    # .env.local 必须仍被 gitignore 覆盖
    gi = ROOT / ".gitignore"
    if gi.exists() and "agent/.env.local" in gi.read_text(encoding="utf-8", errors="ignore"):
        rep.ok("API Key", "密钥文件已在 .gitignore 中", "agent/.env.local 不会随作业提交")
    else:
        rep.warn("API Key", ".gitignore 未覆盖 agent/.env.local",
                 "请补上该规则，避免密钥被提交")

    if cfg.model:
        rep.ok("API Key", f"接口模型名 {cfg.model}", f"endpoint: {cfg.api_url}")


def check_network_layers(rep: Report, cfg) -> None:
    """把「能不能连上 DeepSeek」拆成 DNS → TCP → TLS → HTTP 四层，逐层定位。

    报错时用户看到的往往是一坨 urllib3 堆栈（Max retries exceeded / NameResolutionError…），
    看不出到底是哪一层坏了、该改什么。这里按网络栈自下而上逐层实测，
    坏在哪一层就停在哪一层，并直接给出对应的处理办法。
    """
    import socket
    import ssl

    g = "网络分层诊断"
    host = "api.deepseek.com"
    port = 443

    # ---- 第 1 层：DNS ----
    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos if i[0] == socket.AF_INET})
        ms = (time.perf_counter() - t0) * 1000
        rep.ok(g, "① DNS 解析", f"{host} → {', '.join(ips) or '(仅 IPv6)'} · {ms:.0f} ms")
    except Exception as e:                           # noqa: BLE001
        rep.fail(g, "① DNS 解析失败", f"{type(e).__name__}: {e}")
        rep.add(WARN, g, "后续层级已跳过", "第 1 层就断了，问题就是它")
        _dns_advice(rep, g)
        return

    # ---- 第 2 层：TCP 443 ----
    # ⚠️ 必须**逐个 IP** 测。api.deepseek.com 挂在 CDN 后面，一次解析会返回多个 A 记录，
    #    实测就出现过「3 个 IP 里 1 个黑洞、另外 2 个 6ms」的情况。
    #    只测第一个 IP 会给出误导性的结论（要么假 FAIL，要么漏报）。
    ok_ips: list[tuple[str, float]] = []
    dead_ips: list[tuple[str, str]] = []
    for ip in ips:
        t0 = time.perf_counter()
        try:
            with socket.create_connection((ip, port), timeout=5):
                ok_ips.append((ip, (time.perf_counter() - t0) * 1000))
        except Exception as e:                       # noqa: BLE001
            dead_ips.append((ip, type(e).__name__))

    if ok_ips and not dead_ips:
        rep.ok(g, "② TCP 443 连接",
               f"{len(ok_ips)}/{len(ips)} 个 IP 全部可达 · "
               + "、".join(f"{ip} {ms:.0f}ms" for ip, ms in ok_ips))
    elif ok_ips:
        rep.warn(g, f"② TCP 443：{len(dead_ips)}/{len(ips)} 个 IP 不可达",
                 "可达：" + "、".join(f"{ip} {ms:.0f}ms" for ip, ms in ok_ips)
                 + "；不可达：" + "、".join(f"{ip}({err})" for ip, err in dead_ips)
                 + "\n          → 不可达的是 CDN 的某个边缘节点，不影响使用："
                   "解析结果里它通常排在最后，请求会直接命中可达的那个；"
                   "万一先落到它身上，本项目会对连接超时自动重试并换一个 IP。"
                   "若频繁感知到卡顿，换网络或稍后再试即可（节点会轮换）。")
    else:
        rep.fail(g, "② TCP 443 全部不可达", "、".join(f"{ip}({err})" for ip, err in dead_ips))
        rep.add(WARN, g, "处理办法",
                "端口被防火墙 / 安全软件 / 代理拦截，或当前网络禁止访问 443。"
                "换网络试试；公司或校园网请确认是否需要配置 http_proxy / https_proxy。")
        return

    # ---- 第 3 层：TLS ----
    t0 = time.perf_counter()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=8) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as ss:
                ver = ss.version()
        ms = (time.perf_counter() - t0) * 1000
        rep.ok(g, "③ TLS 握手", f"{ver} · {ms:.0f} ms")
    except Exception as e:                           # noqa: BLE001
        rep.fail(g, "③ TLS 握手失败", f"{type(e).__name__}: {e}")
        rep.add(WARN, g, "处理办法",
                "证书校验或 TLS 协商失败：检查系统时间是否正确、是否有杀毒/抓包软件插入了自签名证书。")
        return

    # ---- 第 4 层：HTTP ----
    # 不带 Key 请求根路径，预期就是 401/403 —— 只要能拿到**任何** HTTP 状态码，
    # 就说明 DNS、TCP、TLS 三层都通了，问题不在网络上。
    try:
        import requests
        r = requests.get("https://api.deepseek.com", timeout=(8, 15))
        rep.ok(g, "④ HTTPS 可达",
               f"GET https://api.deepseek.com → HTTP {r.status_code}"
               + ("（未带 Key，401/403 属正常，说明网络链路完好）" if r.status_code in (401, 403) else ""))
    except Exception as e:                           # noqa: BLE001
        rep.fail(g, "④ HTTPS 请求失败", f"{type(e).__name__}: {str(e)[:120]}")


def _dns_advice(rep: Report, g: str) -> None:
    rep.add(WARN, g, "处理办法",
            "这是本机 DNS 的问题，不是代码或 Key 的问题，通常几秒后自愈。"
            "① 换网络或重连 Wi-Fi ② 关闭/切换 VPN ③ 刷新缓存：ipconfig /flushdns "
            "④ 手动改 DNS 为 223.5.5.5（阿里）或 119.29.29.29（腾讯）"
            "⑤ 查看 hosts 文件是否被写入过 api.deepseek.com 的静态映射。")


def check_api_live(rep: Report, cfg) -> None:
    group = "API 连通性"
    if not cfg.api_key_configured:
        rep.warn(group, "跳过", "未配置 Key")
        return
    try:
        from agent.llm import DeepSeekClient
    except Exception as e:                           # noqa: BLE001
        rep.fail(group, "无法导入 agent.llm", f"{type(e).__name__}: {e}")
        return

    cli = DeepSeekClient(cfg)
    t0 = time.perf_counter()
    try:
        r = cli.complete([{"role": "user", "content": "只回复两个字：在线"}],
                         tools=None)
    except Exception as e:                           # noqa: BLE001
        msg = getattr(e, "message", str(e))
        code = getattr(e, "code", type(e).__name__)
        info = getattr(e, "info", {}) or {}
        hint = info.get("hint", "")
        rep.fail(group, f"调用 /chat/completions 失败（{code}）", msg + (f"\n          → {hint}" if hint else ""))
        # 网络类错误：自动往下钻一层，直接指出坏在 DNS / TCP / TLS 的哪一环
        if code.startswith("NETWORK_") or code == "API_TIMEOUT":
            rep.add(WARN, group, "触发分层诊断", "下面逐层定位具体是哪里断了")
            check_network_layers(rep, cfg)
        return
    ms = (time.perf_counter() - t0) * 1000
    extra = f" · 自动重试 {r.retries} 次后成功" if getattr(r, "retries", 0) else ""
    rep.ok(group, "接口返回正常",
           f"HTTP 200 · 模型 {r.model} · {ms:.0f} ms · "
           f"tokens {r.usage.get('total_tokens', '?')} · 回复 {r.content.strip()[:20]!r}{extra}")


# --------------------------------------------------------------------------- #
# 3. 文件与配置
# --------------------------------------------------------------------------- #
def check_files(rep: Report, cfg) -> None:
    g = "交付文件"
    checks = [
        (ROOT / "classes.txt", "classes.txt（类别定义，提交必需）"),
        (ROOT / "predictions.csv", "predictions.csv（预测结果，提交必需）"),
        (ROOT / "configs" / "class_info.json", "configs/class_info.json"),
        (ROOT / "configs" / "class_map.json", "configs/class_map.json"),
        (ROOT / "configs" / "data_stats.json", "configs/data_stats.json"),
        (ROOT / "agent" / "data" / "car_profiles.json", "agent/data/car_profiles.json（车型资料库）"),
        (ROOT / "agent" / "data" / "confidence_threshold.json", "agent/data/confidence_threshold.json（阈值推导）"),
    ]
    for p, label in checks:
        if p.exists():
            rep.ok(g, label, f"{p.stat().st_size:,} B")
        else:
            rep.fail(g, f"缺少 {label}", str(p))

    # classes.txt 的三个硬性约束
    ct = ROOT / "classes.txt"
    if ct.exists():
        raw = ct.read_bytes()
        lines = [x for x in ct.read_text(encoding="utf-8").splitlines() if x.strip()]
        problems = []
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append("存在 UTF-8 BOM")
        if len(lines) != 50:
            problems.append(f"行数 {len(lines)} != 50")
        if not all(len(x) == 4 and x.isdigit() for x in lines):
            problems.append("存在非 4 位数字行")
        if [int(x) for x in lines] != sorted(int(x) for x in lines):
            problems.append("未按升序排列")
        if "unknown" in lines:
            problems.append("不应包含 unknown")
        if problems:
            rep.fail(g, "classes.txt 不符合任务书要求", "；".join(problems))
        else:
            rep.ok(g, "classes.txt 校验通过", "50 行 · 4 位数字 · 升序 · 无 BOM · 不含 unknown")

    # predictions.csv 表头
    pc = ROOT / "predictions.csv"
    if pc.exists():
        head = pc.read_text(encoding="utf-8").splitlines()
        want = "image_id,predicted_label,confidence"
        if head and head[0].strip() == want:
            rep.ok(g, "predictions.csv 表头正确", f"{len(head) - 1} 行数据")
        else:
            rep.fail(g, "predictions.csv 表头不符", f"实际：{head[0] if head else '(空)'}；要求：{want}")


def check_vision(rep: Report, cfg) -> None:
    g = "视觉模型"
    found = []
    for p in cfg.checkpoint_paths():
        if p.exists():
            mb = p.stat().st_size / 1024 / 1024
            rep.ok(g, f"{p.parent.name}/best.pt", f"{mb:.1f} MB")
            found.append(p)
        else:
            rep.fail(g, f"缺少权重 {p.relative_to(ROOT)}", "请先完成任务一训练，或修改 agent/config.py")
    if len(found) == 2:
        rep.ok(g, "C1+C2 集成配置完整", "improved_resnet18_224 + improved_resnet50_224（任务一最终方案）")
    elif len(found) == 1:
        rep.warn(g, "只有 1 个权重", "将退化为单模型推理，精度低于集成方案")


def check_runtime(rep: Report, cfg) -> None:
    g = "运行时"
    try:
        import torch
        rep.ok(g, f"torch 线程数 {torch.get_num_threads()}",
               "本项目 CPU 训练实测 12 线程最优（16 线程反而慢 17%）")
    except Exception:                                # noqa: BLE001
        pass

    for d, label in [(cfg.data_dir, "agent/data（会话与档案）"),
                     (cfg.upload_dir, "data/uploads（上传图片）")]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            rep.ok(g, f"目录可写：{label}", str(d))
        except Exception as e:                       # noqa: BLE001
            rep.fail(g, f"目录不可写：{label}", f"{type(e).__name__}: {e}")

    # 端口占用（默认 5000）
    try:
        import socket
        s = socket.socket()
        s.settimeout(0.4)
        busy = s.connect_ex(("127.0.0.1", 5000)) == 0
        s.close()
        if busy:
            rep.warn(g, "端口 5000 已被占用",
                     "可能有另一个实例在跑；启动脚本会自动换到 5001")
        else:
            rep.ok(g, "端口 5000 空闲", "Web 默认端口可用")
    except Exception:                                # noqa: BLE001
        pass

    total, used, free = shutil.disk_usage(str(ROOT))
    rep.ok(g, "磁盘剩余空间", f"{free / 1024**3:.1f} GB 可用 / {total / 1024**3:.1f} GB")


def check_vision_live(rep: Report, cfg) -> None:
    g = "视觉推理实测"
    # 找一张真实图片：优先 data/val 下的第一张
    candidate = None
    val = ROOT / "data" / "val"
    if val.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            hits = sorted(val.rglob(ext))
            if hits:
                candidate = hits[0]
                break
    if candidate is None:
        ups = ROOT / "data" / "uploads"
        if ups.exists():
            hits = sorted(ups.glob("*.jpg")) + sorted(ups.glob("*.png"))
            candidate = hits[0] if hits else None
    if candidate is None:
        rep.warn(g, "跳过", "data/val 与 data/uploads 下都没有可用图片")
        return

    try:
        from agent.tools.vision import get_classifier, model_label
        clf = get_classifier(cfg)
        t0 = time.perf_counter()
        res = clf.predict_path(candidate, top_k=3)
        ms = (time.perf_counter() - t0) * 1000
        rep.ok(g, "真实前向推理成功",
               f"{model_label(cfg)} · {candidate.name} → {res['label_name']} "
               f"({float(res['confidence']):.2%}, unknown={res['is_unknown']}) · {ms:.0f} ms")
        # 第二次应为缓存路径（模型已加载）
        t1 = time.perf_counter()
        clf.predict_path(candidate, top_k=3)
        rep.ok(g, "模型常驻复用", f"第二次调用 {(time.perf_counter() - t1) * 1000:.0f} ms（含加载时会更久）")
    except Exception as e:                           # noqa: BLE001
        rep.fail(g, "真实推理失败", f"{type(e).__name__}: {e}")


def check_agent(rep: Report, cfg) -> None:
    g = "智能体装配"
    try:
        from agent.loop import Agent
        from agent.session import SessionStore
        a = Agent(cfg, store=SessionStore(cfg))
        tools = a.describe_tools()
    except Exception as e:                           # noqa: BLE001
        rep.fail(g, "无法装配 Agent", f"{type(e).__name__}: {e}")
        return
    names = [t["name"] for t in tools]
    vision = [t for t in tools if t.get("requires_image")]
    rep.ok(g, f"工具注册 {len(names)} 个", "、".join(names))
    if vision:
        rep.ok(g, f"视觉工具 {len(vision)} 个", "、".join(t["name"] for t in vision))
    else:
        rep.fail(g, "没有注册视觉工具", "任务书 5.2 要求至少 1 个真实加载模型的视觉工具")
    if len(names) >= 3:
        rep.ok(g, f"非视觉工具 {len(names) - len(vision)} 个", "满足任务书「≥2 个其他工具」")
    else:
        rep.fail(g, "工具数不足", "任务书要求 ≥1 视觉 + ≥2 其他")
    rep.ok(g, f"循环预算 max_rounds={cfg.max_rounds} / max_tool_calls={cfg.max_tool_calls}",
           f"模式：{'离线替身' if cfg.offline else '真实 DeepSeek API'}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="识车档案 CarDex 启动前自检")
    ap.add_argument("--api", action="store_true", help="额外做真实 API 连通性测试")
    ap.add_argument("--vision", action="store_true", help="额外做真实视觉推理测试")
    ap.add_argument("--net", action="store_true",
                    help="额外做网络分层诊断（DNS → TCP 443 → TLS → HTTPS）")
    ap.add_argument("--all", action="store_true", help="等价于 --api --vision --net")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()
    if args.all:
        args.api = args.vision = args.net = True

    rep = Report()
    try:
        from agent.config import load_config
        cfg = load_config(offline=False if args.api else None)
    except Exception as e:                           # noqa: BLE001
        print(f"[FAIL] 无法导入 agent.config：{type(e).__name__}: {e}")
        return 1

    check_python(rep)
    check_packages(rep)
    check_api_key(rep, cfg)
    check_files(rep, cfg)
    check_vision(rep, cfg)
    check_runtime(rep, cfg)
    check_agent(rep, cfg)
    if args.api:
        check_api_live(rep, cfg)
    if args.net:
        check_network_layers(rep, cfg)
    if args.vision:
        check_vision_live(rep, cfg)

    if args.json:
        print(json.dumps({"pass": rep.n_pass, "warn": rep.n_warn, "fail": rep.n_fail,
                          "items": rep.items}, ensure_ascii=False, indent=2))
        return 1 if rep.n_fail else 0

    print("=" * 78)
    print("识车档案 CarDex —— 启动前环境自检")
    print("=" * 78)
    last_group = None
    for i in rep.items:
        if i["group"] != last_group:
            print(f"\n【{i['group']}】")
            last_group = i["group"]
        print(f"  {_MARK[i['level']]} {i['name']}")
        if i["detail"]:
            print(f"          {i['detail']}")
    print("\n" + "-" * 78)
    print(f"通过 {rep.n_pass} 项 · 警告 {rep.n_warn} 项 · 失败 {rep.n_fail} 项")
    if rep.n_fail:
        print("存在阻断项（FAIL），请先按上面的提示修复再启动。")
    elif rep.n_warn:
        print("可以启动。警告项不阻断运行。")
    else:
        print("环境就绪，可以启动。")
    print("-" * 78)
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
