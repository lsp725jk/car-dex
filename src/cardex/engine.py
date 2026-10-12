# -*- coding: utf-8 -*-
"""训练 / 验证引擎（CPU 友好）。"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import paths
from .data import (CarDataset, build_transform, mixup_cutmix_batch,
                   soft_cross_entropy, supcon_loss)
from .metrics import compute_metrics, fmt_rate
from .models import (build_model, freeze_by_mode, load_weights_into,
                     param_groups, param_groups_llrd)
from .utils import (AverageMeter, Timer, autocast_dtype, configure_threads,
                    device_label, get_device, get_logger, save_json, set_seed)


def module_device(model: nn.Module) -> torch.device:
    """模型参数现在挂在哪块设备上（EMA 的副本会跟着主模型走）。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def backbone_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """只跑到**全局平均池化**为止，返回倒数第二层的特征（ResNet50 是 2048 维）。

    对比学习必须在**特征空间**上做，而不是在分类头的输出上 ——
    因为分类头的输出维度等于类别数（51），已经被"压"成了类别证据，
    特征之间的几何关系所剩无几。倒数第二层的 2048 维特征才是模型真正学到的表示。

    这里手工展开 ResNet 的 forward（而不是注册 hook），是为了显式、好调试。
    """
    m = model
    if not hasattr(m, "layer4"):
        raise ValueError(f"{type(m).__name__} 的骨干结构不认识，无法取中间特征")
    out = m.conv1(x)
    out = m.bn1(out)
    out = m.relu(out)
    out = m.maxpool(out)
    out = m.layer1(out)
    out = m.layer2(out)
    out = m.layer3(out)
    out = m.layer4(out)
    out = m.avgpool(out)
    return torch.flatten(out, 1)


def build_proj_head(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """SupCon / SimCLR 用的**投影头**：2 层 MLP + ReLU。

    为什么要多加一个投影头，而不是直接在特征上算对比损失？（SimCLR 论文的发现）

    * 对比损失会"消耗"表示能力 —— 如果直接在 2048 维特征上施加对比约束，
      模型会为了满足对比目标而丢掉对分类有用的信息（**表示崩塌**）；
    * 加一层投影头相当于给对比损失一个"缓冲区"：**对比约束作用于投影空间，
      而分类用的还是原始特征**，两者互不干扰；
    * 推理时**投影头直接丢掉**，不增加任何成本。

    输出维度用 128（SimCLR 的推荐值），中间层用特征维度的 1/2。
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class ModelEMA:
    """权重指数滑动平均，验证/推理时使用平滑后的权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    label_names: list[str],
    tta_hflip: bool = False,
    amp: bool = False,
    device: torch.device | str | None = None,
) -> dict:
    dev = torch.device(device) if device is not None else module_device(model)
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []
    paths_seen: list[str] = []
    loss_meter = AverageMeter()
    crit = nn.CrossEntropyLoss()

    def forward(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(dev.type, dtype=autocast_dtype(dev), enabled=amp):
            out = model(x)
        return out.float()

    for images, targets, batch_paths in loader:
        images = images.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        logits = forward(images)
        if tta_hflip:
            logits_flip = forward(torch.flip(images, dims=[3]))
            logits = (torch.log_softmax(logits, dim=1) + torch.log_softmax(logits_flip, dim=1)) / 2
        loss_meter.update(crit(logits, targets).item(), images.size(0))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_true.extend(targets.tolist())
        all_pred.extend(pred.tolist())
        all_conf.extend(conf.tolist())
        if loader.dataset.return_path:  # type: ignore[attr-defined]
            paths_seen.extend(list(batch_paths))

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    result = compute_metrics(y_true, y_pred, label_names)
    result["loss"] = loss_meter.avg
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["confidence"] = np.array(all_conf)
    result["paths"] = paths_seen
    return result


def train_model(cfg: dict, logger=None, verbose: bool = True) -> dict:
    """按配置训练一个模型，返回训练结果字典并落盘。

    cfg 关键字段见 configs/train_baseline.json / configs/train_improved.json。
    """
    set_seed(cfg.get("seed", 42))
    n_threads = configure_threads(cfg.get("threads"))
    paths.ensure_dirs()

    # ---- 设备 ----
    # 配置里可以写 "auto"（默认）/ "cuda" / "cuda:1" / "cpu"。
    # 有可用的 CUDA 就用 CUDA —— 实测同一份配置在 RTX 4060 上比 CPU 快一个数量级。
    device = get_device(cfg.get("device", "auto"))
    on_cuda = device.type == "cuda"
    if on_cuda:
        # cudnn 自动选最快卷积算法；输入尺寸固定时收益明显
        torch.backends.cudnn.benchmark = True
        # TF32：Ampere 及以后的卡上，矩阵乘用 TF32 能大幅提速且精度损失很小
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:                            # noqa: BLE001
            pass

    run_name = cfg["run_name"]
    run_dir = paths.RUN_DIR / run_name
    ckpt_dir = paths.CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 无论外部是否传入 logger，都保证把本轮训练写入 outputs/runs/<run>/train.log
    logger = logger or get_logger(f"train.{run_name}")
    logger = get_logger(logger.name, run_dir / "train.log")

    cm = paths.load_class_map()
    num_classes = int(cfg.get("num_classes", cm["num_classes"]))
    label_names = paths.label_names()

    # ---------------- 数据 ----------------
    # train_dir / val_dir 可以覆盖默认的 data/train、data/val ——
    # 微调时经常要用**自己的照片目录**做领域适配，不想污染总数据集。
    image_size = int(cfg["image_size"])
    aug = cfg.get("aug", {})
    train_tf = build_transform(cfg.get("train_transform", "baseline_train"), image_size, aug)
    eval_tf = build_transform("eval", image_size, aug)

    train_root = Path(cfg["train_dir"]) if cfg.get("train_dir") else paths.DATA_DIR / "train"
    val_root = Path(cfg["val_dir"]) if cfg.get("val_dir") else paths.DATA_DIR / "val"

    # ---------------- 对比学习（SupCon）开关 ----------------
    # 目标：让"看起来很像但不是同一款"的车在特征空间里拉开距离 ——
    # 这正是 hard_negative（如 MINI CLUBMAN vs MINI）最难的地方。
    use_supcon = bool(cfg.get("supcon", False))
    supcon_weight = float(cfg.get("supcon_weight", 0.5))
    supcon_temp = float(cfg.get("supcon_temperature", 0.1))
    if use_supcon:
        # 双视图：对同一张图独立跑两遍随机增强。
        # 51 类、batch=32 时同 batch 内平均每类不足 1 张，凑不出同类样本对；
        # 有了双视图，每个样本至少有自己的另一个视角作为正样本。
        n_views = int(cfg.get("supcon_views", 2))
        if cfg.get("mix_prob"):
            logger.warning("SupCon 已开启 → 自动关闭 MixUp/CutMix："
                           "混合样本的软标签无法判断谁是正样本。")

    limit = cfg.get("limit_train_per_class")
    train_ds = CarDataset(train_root, transform=train_tf, class_map=cm,
                          limit_per_class=limit, seed=cfg.get("seed", 42),
                          n_views=n_views if use_supcon else 1,
                          with_fine_label=use_supcon)
    # 微调时可以没有独立验证集（数据太少，全拿去训了）→ 退化成用训练集评估，
    # 但要明确警告：这时候的指标**只反映拟合程度，不代表泛化能力**。
    if val_root.exists() and any(val_root.iterdir()):
        val_ds = CarDataset(val_root, transform=eval_tf, class_map=cm,
                            limit_per_class=cfg.get("limit_val_per_class"),
                            seed=cfg.get("seed", 42))
        val_is_train = False
    else:
        val_ds = train_ds
        val_is_train = True
    batch_size = int(cfg.get("batch_size", 32))
    # num_workers 的默认值取决于设备：
    #   GPU 训练 —— 必须开多进程，否则数据增强（RandAugment/ColorJitter/RandomErasing）
    #              会把 GPU 饿死。实测（Ryzen 9 7940H，8 核 16 线程）：
    #                每步耗时   取数据        前向+反向
    #   nw=0        373 ms    295 ms(79%)    62 ms
    #   nw=4        131 ms     76 ms(58%)    45 ms
    #   nw=6         48 ms/批（每 epoch 数据侧约 11s，比 nw=4 快一倍多）
    #   CPU 训练 —— 保持 0：多进程会和训练抢 CPU 核，而且某些受限环境
    #              （禁用命名管道）下多进程 DataLoader 会直接失败。
    default_workers = 6 if on_cuda else 0
    num_workers = int(cfg.get("num_workers", default_workers))
    pin = bool(cfg.get("pin_memory", on_cuda))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False,
                              pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)

    logger.info("=" * 78)
    logger.info(f"运行：{run_name}  骨干={cfg['arch']}  输入={image_size}  "
                f"batch={batch_size}  epochs={cfg['epochs']}")
    logger.info(f"设备：{device_label(device)}"
                + (f"   线程={n_threads}（CPU 侧数据预处理）" if not on_cuda else ""))
    logger.info(f"训练集 {len(train_ds)} 张 / {len(train_ds.class_dirs)} 类"
                f"（{train_root}）")
    if val_is_train:
        logger.warning("没有找到独立的验证集目录，本次验证用的就是训练集 —— "
                       "指标只反映拟合程度，**不能当作泛化能力**。")
    else:
        logger.info(f"验证集 {len(val_ds)} 张 / {len(val_ds.class_dirs)} 类（{val_root}）")
    logger.info("=" * 78)

    # ---------------- 模型 ----------------
    model = build_model(cfg["arch"], num_classes=num_classes,
                        pretrained=cfg.get("pretrained", True),
                        dropout=cfg.get("dropout", 0.0))

    # ---- 微调起点：从自己的 checkpoint 装载权重（而不是只用 ImageNet 通用特征）----
    init_from = cfg.get("init_from")
    if init_from:
        info = load_weights_into(model, paths.PROJECT_ROOT / init_from
                                 if not Path(init_from).is_absolute() else init_from)
        logger.info(f"微调起点：{Path(init_from).name}  "
                    f"（来源 {info['source_arch']} / {info['source_num_classes']} 类"
                    + (f" / val Macro-F1 {info['source_val_macro_f1']:.4f}"
                       if isinstance(info["source_val_macro_f1"], (int, float)) else "")
                    + f"；装载 {info['loaded']} 个参数张量"
                    + (f"，分类头 {len(info['skipped_head'])} 个重新初始化"
                       if info["skipped_head"] else "") + "）")

    model.to(device)                                  # 关键：把模型搬到目标设备

    # ---- SupCon 的投影头 ----
    # 只在训练时存在，推理时丢掉（不增加任何推理成本）。
    # 有它才能让对比约束作用于"投影空间"而不破坏分类用的原始特征。
    proj_head: nn.Module | None = None
    if use_supcon:
        with torch.no_grad():
            feat_dim = backbone_forward(
                model, torch.zeros(1, 3, image_size, image_size, device=device)).shape[1]
        proj_head = build_proj_head(feat_dim,
                                    hidden=int(cfg.get("supcon_hidden", feat_dim // 2)),
                                    out_dim=int(cfg.get("supcon_dim", 128))).to(device)
        logger.info(f"对比学习已启用：SupCon 权重={supcon_weight} 温度={supcon_temp} "
                    f"视图数={n_views} 特征维度={feat_dim} 投影维度="
                    f"{cfg.get('supcon_dim', 128)} fine 类别数={train_ds.fine_num_classes}")

    opt_name = cfg.get("optimizer", "sgd").lower()
    lr = float(cfg.get("lr", 0.01))
    wd = float(cfg.get("weight_decay", 1e-4))

    # ---- 优化器：**先**用全部参数建组，**再**应用冻结 ----
    # 顺序很重要。这样"渐进解冻"只需要切换 requires_grad，
    # 不必在训练中途重建优化器（重建会丢掉动量/二阶矩）。
    # 冻结的参数没有梯度，优化器会自动跳过，不产生任何开销。
    # SupCon 的投影头参数必须一并加进来，否则它根本不更新、对比损失毫无作用。
    # 投影头给**更高的学习率**（×5）：它是随机初始化的新模块，
    # 而骨干是预训练的 —— 两者"成熟度"差得远（和分类头要单独给学习率是同一个道理）。
    if cfg.get("llrd"):
        groups = param_groups_llrd(model, lr,
                                   head_lr_mult=cfg.get("head_lr_mult", 1.0),
                                   decay=float(cfg.get("llrd_decay", 0.75)),
                                   include_frozen=True)
    else:
        groups = param_groups(model, lr, cfg.get("head_lr_mult", 10.0),
                              include_frozen=True)
    if proj_head is not None:
        groups.append({"params": list(proj_head.parameters()),
                       "lr": lr * float(cfg.get("supcon_head_lr_mult", 5.0))})
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(groups, lr=lr, momentum=cfg.get("momentum", 0.9),
                                    weight_decay=wd, nesterov=True)

    # ---- 冻结策略（在优化器建好之后应用）----
    freeze_mode = cfg.get("freeze_mode")
    if freeze_mode is None:
        # 兼容老配置：freeze_backbone_epochs 非 0 表示"先冻骨干若干轮再解冻"
        freeze_mode = "head" if cfg.get("freeze_backbone_epochs") else "none"
    frozen = freeze_by_mode(model, freeze_mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    if frozen:
        logger.info(f"冻结策略：{freeze_mode} —— 冻结 {len(frozen)} 个参数张量，"
                    f"可训练 {trainable:,} / {total_p:,}（{trainable / total_p:.1%}）")
    if cfg.get("llrd"):
        lr_desc = "、".join(f"d{g.get('depth')}={g['lr']:.2e}"
                            for g in optimizer.param_groups if "depth" in g)
        logger.info(f"逐层递减学习率（LLRD, decay={cfg.get('llrd_decay', 0.75)}）：{lr_desc}")

    epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 0))
    sched_name = cfg.get("scheduler", "cosine")
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if sched_name == "cosine":
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
        if sched_name == "step":
            return 0.1 ** (step // max(1, total_steps // 3))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- 类别权重 / 损失 ----------------
    smoothing = float(cfg.get("label_smoothing", 0.0))
    class_weight_cfg = cfg.get("class_weight", "none")
    weight_tensor = None
    if class_weight_cfg != "none":
        counts = np.zeros(num_classes, dtype=np.float64)
        for label, n in train_ds.class_counts().items():
            counts[label] = n
        counts = np.maximum(counts, 1.0)
        if class_weight_cfg == "inverse":
            w = counts.sum() / (num_classes * counts)
        elif class_weight_cfg == "sqrt_inverse":
            inv = counts.sum() / (num_classes * counts)
            w = np.sqrt(inv)
        else:
            raise ValueError(class_weight_cfg)
        w = w / w.mean()
        if cfg.get("unknown_weight_mult"):
            w[num_classes - 1] *= float(cfg["unknown_weight_mult"])
        weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        logger.info("类别权重已启用（%s）：unknown=%.3f  min=%.3f max=%.3f",
                    class_weight_cfg, w[-1], w.min(), w.max())

    ema = ModelEMA(model, cfg["ema_decay"]) if cfg.get("ema_decay") else None
    if ema is not None:
        ema.ema.to(device)                            # EMA 副本也要在同一块设备上
    amp = bool(cfg.get("amp", False))
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(cfg.get("cutmix_alpha", 0.0))
    mix_prob = float(cfg.get("mix_prob", 0.0))
    if amp:
        logger.info(f"已启用混合精度（autocast, {autocast_dtype(device)}），损失在 float32 中计算")

    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": -1}
    timer = Timer()

    for epoch in range(1, epochs + 1):
        # 渐进解冻：前 N 轮只训分类头，之后放开骨干。
        # 小数据微调时这一步很关键 —— 一上来就全量更新会把预训练特征冲坏。
        # 因为优化器从一开始就持有全部参数（见上面 include_frozen=True），
        # 这里**只需要切换 requires_grad**，不用动优化器。
        if (cfg.get("freeze_backbone_epochs")
                and epoch == int(cfg["freeze_backbone_epochs"]) + 1):
            unfreeze_to = cfg.get("unfreeze_to", "full")
            freeze_by_mode(model, unfreeze_to)
            n_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("→ 第 %d 轮解冻骨干网络（模式 %s）：可训练参数升到 %s（%.1f%%）",
                        epoch, unfreeze_to, f"{n_now:,}", n_now / total_p * 100)

        model.train()
        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        sc_meter = AverageMeter()
        correct = total = 0
        ep_timer = Timer()
        for step, batch in enumerate(train_loader, 1):
            # 对比学习时会多返回一个 fine 标签（unknown 的源类别）
            if use_supcon and len(batch) == 4:
                images, targets, _, fine = batch
                fine = fine.to(device, non_blocking=True)
            else:
                images, targets, _ = batch[:3]
                fine = None
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 多视图：(B, V, C, H, W) → (B*V, C, H, W)。
            # SupCon 需要"同一样本的另一个视角"当正样本，
            # 因为 51 类、batch=32 时同 batch 内几乎凑不出同类样本对。
            n_views = images.dim() == 5 and images.size(1) > 1
            if n_views:
                B, V = images.shape[0], images.shape[1]
                images = images.view(B * V, *images.shape[2:])
                if fine is not None:
                    fine = fine.repeat_interleave(V)
                targets_v = targets.repeat_interleave(V)
            else:
                targets_v = targets

            # ⚠️ MixUp / CutMix 与 SupCon **互斥**：
            # 混合样本的标签是"两个类的线性组合"，而 SupCon 需要**硬标签**才能
            # 判断谁是正样本。两者同时开，SupCon 会拿到无意义的标签。
            if mix_prob > 0 and not use_supcon:
                images, soft_targets = mixup_cutmix_batch(
                    images, targets_v, num_classes,
                    mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                    prob=mix_prob, smoothing=smoothing,
                )
            else:
                soft_targets = None

            if soft_targets is None:
                off = smoothing / num_classes
                on = 1.0 - smoothing + off
                soft_targets = torch.full((images.size(0), num_classes), off, device=device)
                soft_targets.scatter_(1, targets_v.unsqueeze(1), on)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=autocast_dtype(device), enabled=amp):
                if use_supcon:
                    # 需要**中间特征**，所以不能直接调 model(x)。
                    # 前向拆成两段：骨干 → 特征；分类头 → logits。
                    feats = backbone_forward(model, images)
                    logits = model.fc(feats) if hasattr(model, "fc") else model(images)
                    proj = proj_head(feats)
                else:
                    logits = model(images)
            logits = logits.float()
            ce = soft_cross_entropy(logits, soft_targets, weight_tensor)
            if use_supcon:
                # 投影头输出的浮点精度要拉回 fp32：SupCon 里有 exp 和除法，
                # fp16 下容易溢出/下溢（余弦相似度除以 τ=0.1 会放大 10 倍）。
                sc = supcon_loss(proj.float(), fine, temperature=supcon_temp)
                loss = ce + supcon_weight * sc
                ce_meter.update(ce.item(), images.size(0))
                # 用 .item() 而不是 float()：float() 作用在带 requires_grad 的张量上
                # 会触发 UserWarning（还可能意外保留计算图）
                sc_meter.update(sc.item(), images.size(0))
            else:
                loss = ce
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            loss_meter.update(loss.item(), images.size(0))
            pred = logits.argmax(dim=1)
            correct += int((pred == targets_v).sum())
            total += images.size(0)
            if verbose and (step % cfg.get("log_interval", 50) == 0 or step == steps_per_epoch):
                extra = (f"  ce={ce_meter.avg:.4f} supcon={sc_meter.avg:.4f}"
                         if use_supcon else "")
                logger.info(
                    "  epoch %d/%d  step %d/%d  loss=%.4f  acc=%.4f  lr=%.2e%s  %s",
                    epoch, epochs, step, steps_per_epoch, loss_meter.avg,
                    correct / max(1, total), optimizer.param_groups[0]["lr"], extra,
                    ep_timer.fmt(),
                )

        eval_model = ema.ema if ema is not None else model
        val = evaluate(eval_model, val_loader, num_classes,
                       label_names, tta_hflip=cfg.get("tta_hflip", False), amp=amp,
                       device=device)
        row = {
            "epoch": epoch,
            "train_loss": round(loss_meter.avg, 5),
            "train_acc": round(correct / max(1, total), 5),
            "val_loss": round(val["loss"], 5),
            "val_acc": round(val["accuracy"], 5),
            "val_macro_f1": round(val["macro_f1"], 5),
            # 这两个指标在小数据集上可能"无定义"（没有 unknown 样本）→ 保留 None
            "unknown_detection_rate": (None if val["unknown_detection_rate"] is None
                                       else round(float(val["unknown_detection_rate"]), 5)),
            "known_false_reject_rate": (None if val["known_false_reject_rate"] is None
                                        else round(float(val["known_false_reject_rate"]), 5)),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(timer.elapsed, 1),
        }
        history.append(row)
        logger.info(
            "epoch %2d/%d  train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f "
            "MacroF1=%.4f  unk检出=%s 已知误拒=%s  [%s]",
            epoch, epochs, row["train_loss"], row["train_acc"], row["val_loss"],
            row["val_acc"], row["val_macro_f1"],
            fmt_rate(row["unknown_detection_rate"], 3),
            fmt_rate(row["known_false_reject_rate"], 3), timer.fmt(),
        )

        is_best = row["val_macro_f1"] > best["macro_f1"]
        if is_best:
            best = {"macro_f1": row["val_macro_f1"], "epoch": epoch}

        ckpt = {
            "model_state": (ema.ema if ema is not None else model).state_dict(),
            "arch": cfg["arch"],
            "num_classes": num_classes,
            "image_size": image_size,
            "class_map": cm,
            "label_names": label_names,
            "preprocess": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "eval_resize": int(round(image_size * aug.get("eval_resize_ratio", 256 / 224))),
                "eval_crop": image_size,
                "interpolation": "bilinear",
            },
            "train_config": cfg,
            "epoch": epoch,
            "val_macro_f1": row["val_macro_f1"],
            "ema": ema is not None,
        }
        # 中期 checkpoint：每轮都保留最近一个 last.pt，并按间隔保留 epoch_XX.pt
        torch.save(ckpt, ckpt_dir / "last.pt")
        interval = int(cfg.get("checkpoint_interval", 5))
        if interval and epoch % interval == 0:
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:02d}.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info("  ★ 新的最佳验证 Macro-F1 = %.4f（epoch %d）已保存 best.pt",
                        row["val_macro_f1"], epoch)

    # ---------------- 收尾 ----------------
    summary = {
        "run_name": run_name,
        "config": cfg,
        "history": history,
        "best": best,
        "final": history[-1] if history else {},
        "total_time_s": round(timer.elapsed, 1),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "num_threads": n_threads,
        "checkpoints": str(ckpt_dir),
    }
    save_json(summary, run_dir / "summary.json")

    # 用 best.pt 在验证集上做一次最终评估并落盘完整报告
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        final_eval = evaluate_checkpoint(best_path, val_loader, num_classes, label_names,
                                         tta_hflip=cfg.get("tta_hflip", False), amp=amp)
        save_json(_strip_arrays(final_eval), run_dir / "val_report.json")
        (run_dir / "val_classification_report.txt").write_text(
            final_eval["report_text"], encoding="utf-8"
        )
        np.save(run_dir / "val_confusion_matrix.npy",
                np.array(final_eval["confusion_matrix"]))
        summary["best_val_macro_f1"] = final_eval["macro_f1"]
        summary["best_val_accuracy"] = final_eval["accuracy"]
        logger.info("最佳模型验证结果：Macro-F1=%.4f  Accuracy=%.4f",
                    final_eval["macro_f1"], final_eval["accuracy"])

    logger.info("训练完成，总用时 %s。结果目录：%s", timer.fmt(), run_dir)
    return summary


def evaluate_checkpoint(ckpt_path: Path, loader: DataLoader, num_classes: int,
                        label_names: list[str], tta_hflip: bool = False,
                        amp: bool = False) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return evaluate(model, loader, num_classes, label_names, tta_hflip=tta_hflip, amp=amp)


def load_model_from_checkpoint(ckpt_path: Path, tta_hflip: bool = False):
    """供推理脚本使用：返回 (model, checkpoint)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _strip_arrays(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("y_true", "y_pred", "confidence", "paths")}
