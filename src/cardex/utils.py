# -*- coding: utf-8 -*-
"""随机种子、线程、日志等通用工具。"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 设备选择
# --------------------------------------------------------------------------- #
def get_device(prefer: str | None = None) -> torch.device:
    """决定这次训练/推理跑在哪块设备上。

    顺序：
      1. 显式指定（配置里的 `device` 字段、或推理构造函数的参数）；
      2. 有可用的 CUDA 设备就用 `cuda`；
      3. 否则退回 `cpu`。

    ⚠️ 注意"有 N 卡"≠"能用 CUDA"：如果装的是 **CPU-only 构建**的 PyTorch
    （`torch.version.cuda is None`），`torch.cuda.is_available()` 会一直是 False。
    这种情况要用 CUDA 版重装：
        pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 \\
            --index-url https://download.pytorch.org/whl/cu126
    （必须带 `+cu126` 后缀 —— 只写 `==2.13.0` 的话 pip 会认为已装的 `2.13.0+cpu`
    已经满足要求而不去换，因为 PEP 440 规定公共版本号匹配时忽略本地版本段。）
    """
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """给日志用的一句话设备描述。"""
    if device.type == "cuda":
        idx = device.index or 0
        try:
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            return f"cuda:{idx}（{name}，{total:.1f} GB）"
        except Exception:                            # noqa: BLE001
            return f"cuda:{idx}"
    return "cpu"


def autocast_dtype(device: torch.device) -> torch.dtype:
    """混合精度用什么类型。

    · CUDA：`float16` —— 现代 N 卡（图灵及以后）有张量核心，fp16 才是快的那个；
    · CPU：`bfloat16` —— CPU 上 fp16 几乎没有加速，bf16 有 AVX512-BF16 支持时约 1.5×。
    """
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def configure_threads(n: int | None = None) -> int:
    """CPU 训练时把线程数拉满（默认取逻辑核数）。"""
    n = n or (os.cpu_count() or 8)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 4))
    except RuntimeError:
        pass  # 已经初始化过则忽略
    return n


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        target = str(Path(log_file).resolve())
        already = any(
            isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == target
            for h in logger.handlers
        )
        if not already:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def reset(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def fmt(self) -> str:
        s = self.elapsed
        if s < 60:
            return f"{s:.1f}s"
        if s < 3600:
            return f"{s / 60:.1f}min"
        return f"{s / 3600:.2f}h"


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
