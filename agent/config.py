# -*- coding: utf-8 -*-
"""智能体配置。

**API Key 的获取顺序**（任务书要求"通过环境变量或本地配置管理，不写入公开代码、报告或截图"）：

    1. 环境变量 `DEEPSEEK_API_KEY`
    2. 本地配置文件 `agent/.env.local`（已在 .gitignore 中忽略，不随作业提交）
    3. 都没有 → 进入「离线模式」（用脚本化的假模型驱动同一套循环，便于无 Key 时验证流程）

本项目**从不接收、不打印、不落盘**用户的 API Key：
日志里只出现「已配置 / 未配置」，请求头不打印。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"
DATA_DIR = AGENT_DIR / "data"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# 任务书规定：接口模型名固定为 deepseek-v4-flash
MODEL_NAME = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/chat/completions"

LOCAL_ENV_FILE = AGENT_DIR / ".env.local"

# 界面上「设置」改的东西落在这里（**不含密钥**）：
# 密钥永远只进环境变量或 agent/.env.local，其余可调参数进这个 JSON。
RUNTIME_CONFIG_FILE = DATA_DIR / "runtime_config.json"

# 允许在界面上修改、且会被持久化的字段白名单（防止前端塞进乱七八糟的键）
RUNTIME_EDITABLE: dict[str, type] = {
    # ---- API 配置 ----
    "model": str,
    "api_url": str,
    "connect_timeout": float,
    "read_timeout": float,
    "max_retries": int,
    "retry_backoff": float,
    # ---- 模型配置 ----
    "temperature": float,
    "max_tokens": int,
    "max_rounds": int,
    "max_tool_calls": int,
    "top_k": int,
    # ---- 决策阈值 ----
    "unknown_threshold": float,
    "display_confidence_threshold": float,
    "prior_adjust_tau": float,
    # ---- 联网搜索 ----
    "web_search_provider": str,
}


def _coerce(field: str, value):
    """把界面传来的值转成该字段应有的类型；转不了就返回 None（表示忽略这一项）。"""
    want = RUNTIME_EDITABLE[field]
    try:
        if want is str:
            v = str(value).strip()
            return v or None
        if want is bool:
            return bool(value)
        if want is int:
            return int(float(value))
        return float(value)
    except (TypeError, ValueError):
        return None


def load_runtime_config() -> dict:
    """读取界面上保存过的运行配置（不含密钥）。"""
    if not RUNTIME_CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for k, v in data.items():
        if k in RUNTIME_EDITABLE:
            got = _coerce(k, v)
            if got is not None:
                out[k] = got
    return out


def save_runtime_config(updates: dict, reset: list[str] | None = None) -> dict:
    """把界面传来的改动合并进运行配置并落盘，返回**完整**的新配置。

    只接受白名单里的键；类型不对的项忽略而不是报错——
    界面上一个输错的数字不该让整份配置失效。

    `reset` 里的键会被**删除**，用于"用户把某项改回了默认值"的情况：
    如果只是把默认值又写一遍，旧的覆盖值会留在文件里，看起来改了其实没改。
    所以只保留真正的**差量**，代码里的默认值后续仍能生效。
    """
    current = load_runtime_config()
    for k, v in (updates or {}).items():
        if k not in RUNTIME_EDITABLE:
            continue
        got = _coerce(k, v)
        if got is not None:
            current[k] = got
    for k in (reset or []):
        current.pop(k, None)
    RUNTIME_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_FILE.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def _load_local_env() -> dict[str, str]:
    """读取 agent/.env.local（KEY=VALUE 每行一条，# 开头为注释）。"""
    out: dict[str, str] = {}
    if not LOCAL_ENV_FILE.exists():
        return out
    for line in LOCAL_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class Config:
    # ---- 模型 ----
    model: str = MODEL_NAME
    api_url: str = API_URL
    api_key: str = ""
    temperature: float = 0.2
    # 输出预算不能太小：模型在生成**很大的工具参数**时会撞上限。
    # 实测让它把轩逸整张配置表（10 个档次 × 几十项参数）一次性塞进 save_car_trims，
    # 2048 个 token 直接不够 —— finish_reason=length，整轮失败。
    # 现在默认 4096，撞上了还会自动放大到 max_tokens_ceiling 再试一次。
    max_tokens: int = 4096
    max_tokens_ceiling: int = 8192
    connect_timeout: float = 10.0
    read_timeout: float = 90.0

    # ---- 瞬时网络故障的自动重试 ----
    # 家用宽带 / 移动网络的 DNS 会偶发解析失败（Errno 11001 / getaddrinfo failed），
    # 一秒钟就恢复。没有重试的话，一次抖动就会让用户白打一遍字。
    # 只重试「网络类错误 + 429 + 5xx」，参数错误（400/401/402/403）绝不重试。
    max_retries: int = 3            # 额外重试次数（总共最多请求 1 + max_retries 次）
    retry_backoff: float = 0.8      # 首次退避秒数，之后翻倍，并叠加 ±25% 抖动

    # ---- 循环预算 ----
    # 这两个值要按"最重的任务"来定：联网多轮（搜索 → 打开若干页面 → 换词再搜 → 作答）
    # 实测一次会用到 10~20 次工具调用。定小了，模型会在还没得出结论时就撞上限、交白卷
    # （实测 8 轮 / 16 次时：模型搜了 6 次、开了 10 个页面，最后 MAX_ROUNDS_EXCEEDED，回复 0 字）。
    # 循环本身还有两道保护：临近上限时注入 <budget_warning> 要求立即作答；
    # 完全相同的联网调用直接复用上次结果，不重复执行。
    max_rounds: int = 12           # 一次用户请求内最多几轮「模型 + 工具」
    max_tool_calls: int = 24       # 一次用户请求内最多执行几次工具
    max_parallel_tool_calls: int = 4

    # ---- 视觉模型 ----
    vision_checkpoints: list[str] = field(default_factory=lambda: [
        "outputs/checkpoints/r50_416/best.pt",
        "outputs/checkpoints/r50_288_supcon/best.pt",
        "outputs/checkpoints/wide_r50_2_288/best.pt",
    ])
    vision_tta: bool = True
    # TTA 档位："none" / "hflip" / "hflip+multicrop"
    #   hflip+multicrop = 中心裁剪 + 五裁剪，各配水平翻转，共 10 个视角。
    #   实测比只用 hflip 高 +0.0025，代价是推理时间约 5 倍（几十毫秒 → 一百多毫秒）。
    #   ⚠️ 换档位后**必须重新标定 τ 和拒识阈值**：多视角平均会把概率分布压平，
    #      峰值置信度明显下降（实测同一张图 0.979 → 0.629），沿用旧阈值会大量误拒。
    vision_tta_mode: str = "hflip+multicrop"
    # 推理设备："auto"（有 CUDA 就用 GPU）/ "cuda" / "cuda:0" / "cpu"
    vision_device: str = "auto"
    # 决策标定：由验证集扫描得到（src/calibrate.py / src/finalize.py）
    prior_adjust_tau: float = 0.2
    unknown_threshold: float = 0.0
    # 低置信度"提示"阈值：预测仍是具体车型，但低于它时智能体应展示 Top-K / 请用户确认
    # 由 tools/derive_confidence_threshold.py 在验证集上做选择性预测分析得到
    display_confidence_threshold: float = 0.70
    top_k: int = 5

    # ---- 路径 ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    upload_dir: Path = UPLOAD_DIR
    profiles_file: Path = DATA_DIR / "car_profiles.json"
    threshold_file: Path = DATA_DIR / "confidence_threshold.json"
    garage_db: Path = DATA_DIR / "garage.db"
    tool_log_file: Path = DATA_DIR / "tool_calls.jsonl"

    # ---- 离线替身 ----
    offline: bool = False

    # ---- 联网搜索 ----
    # bing（默认，无需 Key，实测本机唯一可用的通道）/ duckduckgo / tavily（需 TAVILY_API_KEY）
    web_search_provider: str = "bing"

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)

    def checkpoint_paths(self) -> list[Path]:
        return [self.project_root / p for p in self.vision_checkpoints]

    def masked_key(self) -> str:
        """用于日志/界面展示——永不返回完整 Key。"""
        if not self.api_key:
            return "(未配置)"
        k = self.api_key
        return f"{k[:4]}…{k[-4:]}（长度 {len(k)}）" if len(k) > 10 else "(已配置)"


def save_api_key(key: str) -> None:
    """把 API Key 写进 agent/.env.local（该文件已被 .gitignore 忽略）。

    **只写这一个文件**：界面上的设置绝不把 Key 写进 runtime_config.json，
    也绝不写进任何代码、报告、日志或截图。
    """
    key = (key or "").strip()
    if not key:
        return
    kept: list[str] = []
    if LOCAL_ENV_FILE.exists():
        for line in LOCAL_ENV_FILE.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*DEEPSEEK_API_KEY\s*=", line):
                continue                             # 丢掉旧值
            kept.append(line)
    else:
        kept = ["# 识车档案 CarDex —— 本地密钥配置（已被 .gitignore 忽略，不随作业提交）",
                "# 真实 Key 只允许出现在本文件或环境变量中，绝不写进代码 / 报告 / 截图。", ""]
    kept.append(f"DEEPSEEK_API_KEY={key}")
    # 不带 BOM：读取端会把 BOM 当成密钥首字符
    LOCAL_ENV_FILE.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.environ["DEEPSEEK_API_KEY"] = key


def load_config(offline: bool | None = None) -> Config:
    """按优先级装载配置。

    顺序（后面的覆盖前面的）：
        1. `Config` 里的代码默认值
        2. `agent/data/confidence_threshold.json`（由验证集扫描得到）
        3. `agent/data/runtime_config.json`（用户在界面「设置」里改的，优先级最高）
    密钥单独走：环境变量 `DEEPSEEK_API_KEY` ＞ `agent/.env.local`。
    """
    local = _load_local_env()
    key = (os.getenv("DEEPSEEK_API_KEY") or local.get("DEEPSEEK_API_KEY") or "").strip()

    cfg = Config(api_key=key)
    if cfg.threshold_file.exists():
        try:
            d = json.loads(cfg.threshold_file.read_text(encoding="utf-8"))
            cfg.display_confidence_threshold = float(d.get("recommended_display_threshold", 0.70))
        except Exception:
            pass

    # 界面上的设置覆盖一切（它代表用户此刻的明确选择）
    for k, v in load_runtime_config().items():
        setattr(cfg, k, v)

    # 离线模式：显式指定，或没有 Key 时自动进入
    cfg.offline = bool(offline) if offline is not None else not cfg.api_key_configured
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return cfg
