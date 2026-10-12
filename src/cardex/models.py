# -*- coding: utf-8 -*-
"""模型工厂。

仅使用 torchvision 官方 **ImageNet 预训练权重**（本地缓存 resnet18/resnet50/wide_resnet50_2），
不使用任何现成的汽车分类模型或第三方在线识别接口 —— 符合任务书 4.2 节要求。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED = {
    # 权重枚举必须与本地已缓存的权重文件名一致，否则会尝试联网下载：
    #   resnet18-f37072fd.pth         <= ResNet18_Weights.IMAGENET1K_V1
    #   resnet50-0676ba61.pth         <= ResNet50_Weights.IMAGENET1K_V1   (注意不是 V2)
    #   wide_resnet50_2-9ba9bcbe.pth  <= Wide_ResNet50_2_Weights.IMAGENET1K_V2
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
}


def build_model(
    arch: str = "resnet18",
    num_classes: int = 51,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    if arch not in SUPPORTED:
        raise ValueError(f"不支持的骨干网络：{arch}，可选 {list(SUPPORTED)}")
    factory, weights = SUPPORTED[arch]
    try:
        model = factory(weights=weights if pretrained else None)
    except Exception as exc:  # 无网络时回退到本地缓存失败
        raise RuntimeError(
            f"加载 {arch} 预训练权重失败（{exc}）。请确认本地缓存存在："
            r"%USERPROFILE%\.cache\torch\hub\checkpoints"
        ) from exc

    # ⚠️ 按**结构**判断分类头，不要按架构名的前缀。
    # 原来写的是 `arch.startswith("resnet")`，结果 `wide_resnet50_2` 不以 "resnet" 开头，
    # 直接被判成"不支持的架构"报错。按属性判断更稳、也不会随新架构名失效。
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        # ResNet 系列（resnet18/50、wide_resnet50_2…）：
        # 分类头是单个 Linear。wide 版只是中间通道翻倍，fc.in_features 仍是 2048。
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        else:
            model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"{arch} 的分类头结构不认识，无法替换为 {num_classes} 类输出")

    model.num_classes = num_classes  # type: ignore[attr-defined]
    return model


def param_groups(model: nn.Module, base_lr: float, head_lr_mult: float = 10.0,
                 include_frozen: bool = False) -> list[dict]:
    """骨干与分类头使用不同学习率（分类头是新初始化的）。

    `include_frozen=True` 会把**冻结的参数也放进优化器**。
    这是"渐进解冻"的正确做法：优化器一开始就持有全部参数，
    之后只需切换 `requires_grad` 即可，不用动优化器本身 ——
    重建 param_groups 会丢掉动量／二阶矩，是个看不见的性能损失。
    没有梯度的参数优化器会自己跳过，不产生额外开销。
    """
    head_names = head_param_names()
    head, backbone = [], []
    for name, param in model.named_parameters():
        if not include_frozen and not param.requires_grad:
            continue
        (head if name.startswith(head_names) else backbone).append(param)
    groups: list[dict] = []
    if backbone:
        groups.append({"params": backbone, "lr": base_lr})
    if head:
        groups.append({"params": head, "lr": base_lr * head_lr_mult})
    return groups


# --------------------------------------------------------------------------- #
# 微调支持
# --------------------------------------------------------------------------- #
# 分类头的参数名前缀。ResNet 是 `fc.`，EfficientNet 是 `classifier.`。
HEAD_PREFIXES: tuple[str, ...] = ("fc.", "classifier.")

# ResNet 的层序（从浅到深），用于"部分冻结"与逐层递减学习率
RESNET_STAGES: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def head_param_names() -> tuple[str, ...]:
    return HEAD_PREFIXES


def load_weights_into(
    model: nn.Module,
    checkpoint: Path | str,
    *,
    allow_head_mismatch: bool = True,
    verbose: bool = True,
) -> dict:
    """把已有 checkpoint 的权重装进一个新模型 —— **微调的起点**。

    和 `build_model(pretrained=True)` 的区别：
      · ImageNet 权重是通用特征，标签是 1000 类 ImageNet 类别；
      · 本函数装的是**我们自己训练好的**权重，标签就是本项目的 51 类车型。

    `allow_head_mismatch=True` 时会**跳过分类头**只装骨干 ——
    这样"换一批类别重训"或"加新类别"时也能复用已学到的主体特征，
    而分类头按新的类别数重新初始化。

    返回一个统计字典，说明装了多少、跳过了多少。
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    src_state = ckpt.get("model_state") or ckpt
    if not isinstance(src_state, dict):
        raise ValueError(f"checkpoint 结构不认识：{type(src_state)}")

    tgt_state = model.state_dict()
    head = head_param_names()
    loaded, skipped_shape, skipped_head, missing = [], [], [], []

    for k, v in src_state.items():
        if k not in tgt_state:
            missing.append(k)
            continue
        if k.startswith(head) and allow_head_mismatch and tgt_state[k].shape != v.shape:
            skipped_head.append(k)
            continue
        if tgt_state[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        tgt_state[k] = v
        loaded.append(k)

    model.load_state_dict(tgt_state, strict=False)
    info = {
        "checkpoint": str(checkpoint),
        "source_arch": ckpt.get("arch"),
        "source_num_classes": ckpt.get("num_classes"),
        "source_epoch": ckpt.get("epoch"),
        "source_val_macro_f1": ckpt.get("val_macro_f1"),
        "loaded": len(loaded),
        "skipped_head": skipped_head,        # 分类头形状不符 → 重新初始化
        "skipped_shape": skipped_shape,      # 其他形状不符（换骨干时会有）
        "missing": missing,
    }
    if verbose:
        print(f"[微调] 载入权重 {checkpoint}")
        print(f"       来源：{info['source_arch']} / {info['source_num_classes']} 类"
              f" / epoch {info['source_epoch']}"
              + (f" / val Macro-F1 {info['source_val_macro_f1']:.4f}"
                 if isinstance(info["source_val_macro_f1"], (int, float)) else ""))
        print(f"       装载 {info['loaded']} 个参数张量"
              + (f"，跳过分类头 {len(skipped_head)} 个（类别数不同，将重新初始化）"
                 if skipped_head else ""))
    return info


def freeze_by_mode(model: nn.Module, mode: str) -> list[str]:
    """按微调模式设置 `requires_grad`，返回被冻结的参数名列表。

    几种典型模式：

      `full`     全量微调 —— 所有层都训。效果最好，也最容易过拟合小数据。
      `head`     只训分类头 —— 骨干完全冻结。适合**小数据快速适配**，
                 几百张图、几分钟就能收敛；也适合作为"线性探针"看特征质量。
      `partial`  只训后半段 —— 冻结前面的层（低级纹理/边缘），
                 训后面的层（高级语义）。是 head 与 full 的折中，
                 在数据量中等时通常最划算。
      `none`     不冻结（等同 full，保留给"从零训"的语义）。
    """
    mode = (mode or "none").lower()
    head = head_param_names()
    frozen: list[str] = []

    def set_frozen(name: str, param: nn.Parameter, freeze: bool) -> None:
        param.requires_grad = not freeze
        if freeze:
            frozen.append(name)

    if mode in ("none", "full"):
        for name, p in model.named_parameters():
            p.requires_grad = True
        return frozen

    if mode == "head":
        for name, p in model.named_parameters():
            set_frozen(name, p, freeze=not name.startswith(head))
        return frozen

    if mode == "partial":
        # ResNet 的层序：conv1/bn1/layer1/layer2 冻结，layer3/layer4/fc 训练。
        # 这是个经验分界：浅层学的是通用边缘/纹理，深层才是"车型"这种语义特征。
        train_prefixes = ("layer3", "layer4", "fc.", "classifier.")
        for name, p in model.named_parameters():
            set_frozen(name, p, freeze=not name.startswith(train_prefixes))
        return frozen

    raise ValueError(f"未知的微调模式：{mode}（可选 full / head / partial / none）")


def param_groups_llrd(
    model: nn.Module,
    base_lr: float,
    head_lr_mult: float = 1.0,
    decay: float = 0.75,
    include_frozen: bool = False,
) -> list[dict]:
    """**逐层递减学习率（LLRD, Layer-wise LR Decay）** 的参数分组。

    思路：越是浅层，学到的越是通用特征，**越不该被大幅改动**；
    越是靠近输出层，越需要适应新任务。所以按层深给递减的学习率：

        浅层      lr × decay³      （几乎不动）
        中层      lr × decay²
        深层      lr × decay
        分类头    lr × head_lr_mult（通常最高）

    这是 BERT/ViT 微调里的常用手法，CNN 微调同样适用 ——
    尤其在**小数据**上，能明显减少"灾难性遗忘"（把预训练学到的东西训没了）。
    """
    head = head_param_names()
    buckets: dict[int, list[nn.Parameter]] = {}

    def depth_of(name: str) -> int:
        """粗略的层深编号：数字越大越靠近输出。"""
        if name.startswith(head):
            return 99
        for i, key in enumerate(RESNET_STAGES):
            if name.startswith(key):
                return i
        return 0

    for name, p in model.named_parameters():
        if not include_frozen and not p.requires_grad:
            continue
        buckets.setdefault(depth_of(name), []).append(p)

    max_depth = max((d for d in buckets if d != 99), default=0)
    groups: list[dict] = []
    for depth in sorted(buckets):
        if depth == 99:
            lr = base_lr * head_lr_mult
        else:
            # 越浅越慢：深度 0 拿到的倍率最小
            lr = base_lr * (decay ** (max_depth - depth))
        groups.append({"params": buckets[depth], "lr": lr, "depth": depth})
    return groups
