# -*- coding: utf-8 -*-
"""视觉分类工具：`classify_car`

任务书 5.2 节要求：
    「视觉分类工具  至少 1 个  真实加载并调用本组训练的模型，返回类别和 confidence」

任务书 5.1 节要求：
    「视觉识别必须通过智能体工具调用执行。仅由后台预先识别图片，
      再把识别结果拼接到 Prompt 中让大模型回答，不能满足视觉工具要求。」

因此本工具：
  · 由**模型自己决定调用**（不是我们在外部先算好再塞进提示词）；
  · 调用时**真实加载 checkpoint 并前向推理**（复用任务一的 `CarEnsemble`）；
  · 返回 car_id / car_name / confidence / Top-K / is_unknown；
  · 按任务书 5.5 节给出「下一步该怎么办」的结构化建议。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..errors import Code, ToolError
from .registry import ToolContext, ToolDefinition

# 懒加载的模型单例：模型加载需数秒，全局只做一次
_LOCK = threading.Lock()
_CLASSIFIER: Any = None
_CLASSIFIER_KEY: tuple | None = None


def get_classifier(cfg) -> Any:
    """构造（或复用）视觉分类器。

    优先使用 C1+C2 集成（任务一最终交付配置），任一成员缺失时退化为可用的单模型。
    """
    global _CLASSIFIER, _CLASSIFIER_KEY

    existing = [p for p in cfg.checkpoint_paths() if p.exists()]
    if not existing:
        raise ToolError(
            Code.MODEL_NOT_LOADED,
            "找不到任何可用的视觉模型权重",
            hint=f"期望的路径：{[str(p) for p in cfg.checkpoint_paths()]}；"
                 f"请先完成任务一的训练，或修改 agent/config.py 的 vision_checkpoints",
        )

    key = (tuple(str(p) for p in existing), cfg.vision_tta, cfg.prior_adjust_tau,
           cfg.unknown_threshold, str(getattr(cfg, "vision_device", "auto")),
           str(getattr(cfg, "vision_tta_mode", "hflip")))
    with _LOCK:
        if _CLASSIFIER is not None and _CLASSIFIER_KEY == key:
            return _CLASSIFIER
        try:
            from cardex.inference import CarClassifier, CarEnsemble
        except Exception as e:            # noqa: BLE001
            raise ToolError(Code.MODEL_NOT_LOADED, f"导入推理模块失败：{e}",
                            hint="请确认 src/ 在 sys.path 中，且 torch/torchvision 已安装") from e

        dev = str(getattr(cfg, "vision_device", "auto"))
        tta_mode = str(getattr(cfg, "vision_tta_mode", None)
                       or ("hflip" if cfg.vision_tta else "none"))
        try:
            if len(existing) > 1:
                clf = CarEnsemble(
                    existing, tta_mode=tta_mode,
                    prior_adjust_tau=cfg.prior_adjust_tau,
                    unknown_threshold=cfg.unknown_threshold,
                    device=dev,
                )
            else:
                clf = CarClassifier(
                    existing[0], tta_mode=tta_mode,
                    prior_adjust_tau=cfg.prior_adjust_tau,
                    unknown_threshold=cfg.unknown_threshold,
                    device=dev,
                )
        except Exception as e:            # noqa: BLE001
            raise ToolError(Code.MODEL_NOT_LOADED, f"加载视觉模型失败：{e}") from e

        _CLASSIFIER = clf
        _CLASSIFIER_KEY = key
        return clf


def model_label(cfg) -> str:
    names = []
    for p in cfg.checkpoint_paths():
        if p.exists():
            names.append(p.parent.name)
    return "+".join(names) if names else "(无可用模型)"


def _decide(is_unknown: bool, confidence: float, cfg) -> dict:
    """把任务书 5.5 节的低置信度 / unknown 处理规则编码成结构化建议。

    这样模型不必自己"想起来"该怎么做，工具结果里已经写明了该走哪条分支。
    """
    if is_unknown:
        return {
            "action": "report_unknown",
            "guidance": (
                "模型判定该车不属于本组支持的 50 款车型。"
                "请如实告知用户「未识别为本组支持的车型」，"
                "**不要**据此编造具体车型，也不要继续按某个确定车型执行业务操作。"
                "可以展示 top_candidates 供参考，并邀请用户手动指定车型或换一张更清晰的照片。"
            ),
        }
    if confidence < cfg.display_confidence_threshold:
        return {
            "action": "confirm_with_user",
            "guidance": (
                f"模型给出了具体车型但置信度偏低（{confidence:.2f} "
                f"< 提示阈值 {cfg.display_confidence_threshold:.2f}）。"
                "请展示 top_candidates 的前几名，请用户确认是哪一款，"
                "或提示重新拍摄（正侧面、光线充足、车辆占画面主体）。"
                "**在用户确认之前不要把它当作确定结论写入档案。**"
            ),
        }
    return {
        "action": "accept",
        "guidance": "置信度充足，可作为识别结果继续后续业务操作（查询档案 / 写入看车记录）。",
    }


def _handler(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()

    image_path = args.get("image_path") or ctx.image_path
    if not image_path:
        raise ToolError(
            Code.DEPENDENCY_MISSING,
            "没有可用的图片",
            hint="用户还没有上传图片。请先请用户上传一张车辆照片，再调用本工具。",
        )

    p = Path(image_path)
    if not p.is_absolute():
        p = cfg.project_root / p
    if not p.exists():
        raise ToolError(Code.IMAGE_NOT_FOUND, f"图片不存在：{p}",
                        hint="请确认图片已上传成功，或重新上传")

    clf = get_classifier(cfg)
    ctx.check_abort()

    try:
        # 多要一位：下面会把 unknown 那一行滤掉，所以按 top_k+1 取，
        # 保证返回给模型的永远是 top_k 个**真实车型**候选。
        res = clf.predict_path(p, top_k=int(args.get("top_k") or cfg.top_k) + 1)
    except ToolError:
        raise
    except Exception as e:                # noqa: BLE001
        raise ToolError(Code.IMAGE_UNREADABLE, f"图片读取或推理失败：{type(e).__name__}: {e}",
                        hint="请确认文件是有效的图片格式（jpg/png），或重新上传") from e

    conf = float(res["confidence"])
    is_unknown = bool(res["is_unknown"])
    want_k = int(args.get("top_k") or cfg.top_k)
    # unknown 不是一个"候选车型"，它是"以上都不是"这个结论本身，
    # 因此不放进 top_candidates —— 低置信度时让用户确认的是具体车型。
    top = [
        {
            "car_id": t["label_id"],
            "car_name": t["label_name"],
            "confidence": round(float(t["probability"]), 6),
        }
        for t in res.get("top_k", [])
        if t.get("label_id") != "unknown"
    ][:want_k]

    return {
        "ok": True,
        "car_id": res["label_id"],
        "car_name": res["label_name"],
        "confidence": round(conf, 6),
        "is_unknown": is_unknown,
        "low_confidence": (not is_unknown) and conf < cfg.display_confidence_threshold,
        "top_candidates": top,
        "image": {"path": str(p), "file_name": p.name},
        "model": {
            "name": model_label(cfg),
            "tta_hflip": cfg.vision_tta,
            "prior_adjust_tau": cfg.prior_adjust_tau,
            "unknown_threshold": cfg.unknown_threshold,
            "display_confidence_threshold": cfg.display_confidence_threshold,
        },
        "decision": _decide(is_unknown, conf, cfg),
        "note": (
            "confidence 是 softmax 最大概率，且已应用先验校正（τ）与拒识阈值（t），"
            "两者均由验证集扫描确定；阈值推导见 outputs/reports/final_model_choice.json "
            "与 agent/data/confidence_threshold.json"
        ),
    }


VISION_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="classify_car",
        description=(
            "识别图片中的车型。真实加载并调用本组自训练的视觉分类模型，"
            "返回 50 款目标车型之一的四位类别 ID、车型名称与置信度；"
            "若判定不属于这 50 款，则返回 car_id=\"unknown\"。"
            "当用户上传了车辆照片并要求识别、或后续操作需要知道车型时调用本工具。"
            "本工具只接受文本参数，图片由用户在 Web 界面上传，会话中已绑定最近一次上传的图片。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "可选。要识别的图片路径；省略时使用当前会话最近一次上传的图片。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "可选。返回前 K 个候选车型，默认 5。",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler,
        timeout_s=120.0,
        requires_image=True,
        tags=("vision", "core"),
    ),
]
