# -*- coding: utf-8 -*-
"""微调入口 —— 在已经训练好的模型上继续学。

**微调和"从头训练"的区别**：从头训练时骨干是 ImageNet 通用特征、分类头是随机的；
微调时**起点就是本任务已经学好的权重**，我们只想让它适应新情况。

三种典型用法：

  ① 换一批数据再来（领域适配）
     你自己拍的照片、你所在城市的车、某个 4S 店的展车 —— 分布和教师数据集不一样。
     python src/finetune.py --from improved_resnet50_224 --data D:\\my_car_photos --mode partial

  ② 只要它学会"我这批新样本"（小数据快速适配）
     每个类别只有几十张图时，全量微调会过拟合。只训分类头（或 LLDR）更稳。
     python src/finetune.py --from improved_resnet50_224 --data D:\\my_photos --mode head --epochs 12

  ③ 在原有基础上继续提升（数据没变，就是再练练）
     python src/finetune.py --from improved_resnet50_224 --mode full --lr 3e-5 --epochs 6

**学习率是微调里最容易搞错的一件事**（见 `LR_RECIPE`），脚本会按模式给推荐值。

用法速查：
    python src/finetune.py --help
    python src/finetune.py --from improved_resnet18_224 --dry-run     # 只看计划不训练
    python src/finetune.py --from outputs/checkpoints/improved_resnet50_224/best.pt --mode partial
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardex import engine, paths  # noqa: E402
from cardex.models import freeze_by_mode  # noqa: E402
from cardex.utils import get_device, device_label, get_logger  # noqa: E402


# --------------------------------------------------------------------------- #
# 学习率配方 —— 微调成败的头号因素
# --------------------------------------------------------------------------- #
# 一句话原则：**改动越多的部分，学习率越小；越靠近输出的部分，学习率可以越大。**
#
# 为什么不能直接用从头训练的 lr（1e-2）？因为那会把已经学好的特征整个冲掉
# （灾难性遗忘）：模型会先"忘掉"再"重学"，小数据上根本学不回来。
LR_RECIPE = {
    "head": {
        "lr": 1e-3,
        "freeze_why": "只训分类头，骨干完全冻结 —— 特征是现成的，不必再动，也**不可能过拟合到骨干上**。",
        "lr_why": "只更新一个 51×2048 的线性层，参数量极小，用较大学习率能很快收敛（几十秒到几分钟）。",
    },
    "partial": {
        "lr": 2e-4,
        "freeze_why": "训 layer3/layer4 + 分类头。浅层学的是通用边缘/纹理，深层才是车型语义 —— 只动深层更划算。",
        "lr_why": "比从头训练小 50 倍。深层虽然要适应新数据，但也不能冲太猛，否则把已学的语义特征带偏。",
    },
    "full": {
        "lr": 5e-5,
        "freeze_why": "不冻结，所有层都参与更新。表达力最强，也最容易在**小数据上过拟合**。",
        "lr_why": "比从头训练（1e-2）小 **200 倍**。这是微调最关键的一条：学习率大了会「灾难性遗忘」——"
                  "模型先把学好的东西忘掉、再重学一遍，小数据上根本学不回来。",
    },
}


def resolve_checkpoint(spec: str) -> Path:
    """把 `--from` 的写法解析成 checkpoint 路径。

    支持：run 名（improved_resnet50_224）、带 .pt 的路径、绝对/相对路径。
    """
    p = Path(spec)
    if p.suffix == ".pt" and p.exists():
        return p
    cand = paths.CKPT_DIR / spec / "best.pt"
    if cand.exists():
        return cand
    if p.exists():
        return p
    raise SystemExit(f"找不到起点权重：{spec}\n"
                     f"  试过：{cand}\n"
                     f"  可用的 run：{[d.name for d in paths.CKPT_DIR.iterdir() if d.is_dir()]}")


def build_config(args) -> dict:
    ckpt = resolve_checkpoint(args.from_)
    meta = {}
    try:
        import torch
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:                                # noqa: BLE001
        pass

    arch = args.arch or meta.get("arch") or "resnet18"
    num_classes = int(meta.get("num_classes") or paths.load_class_map()["num_classes"])
    image_size = int(args.image_size or meta.get("image_size") or 224)

    recipe = LR_RECIPE[args.mode]
    lr = float(args.lr) if args.lr else recipe["lr"]

    run_name = args.run_name or f"ft_{Path(ckpt).parent.name}_{args.mode}_{time.strftime('%m%d')}"

    # 分类头是新初始化的（换类别数时），给它更高的学习率；
    # 用 LLRD 时倍率设成 1（LLRD 自己就会把深层/头抬高）。
    head_lr_mult = 1.0 if args.llrd else float(args.head_lr_mult)

    cfg: dict = {
        "run_name": run_name,
        "arch": arch,
        "pretrained": False,             # 不用 ImageNet 权重 —— 我们用 --from 那份
        "init_from": str(ckpt),          # ← 微调的起点
        "num_classes": num_classes,
        "image_size": image_size,
        "device": args.device,
        "train_transform": args.train_transform,
        "aug": {
            # 微调时**数据增强要收敛一些**：模型已经学得不错了，
            # 再上 RandAugment 这种强增强反而会让它"退步"。
            "crop_scale": [0.65, 1.0],
            "hflip": 0.5,
            "color_jitter": True,
            "jitter_brightness": 0.2,
            "jitter_contrast": 0.2,
            "jitter_saturation": 0.2,
            "jitter_hue": 0.05,
            "randaugment": args.strong_aug,
            "ra_num_ops": 2,
            "ra_magnitude": 5,
            "random_erasing": 0.1 if args.strong_aug else 0.0,
            "eval_resize_ratio": 1.142857,
        },
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "optimizer": args.optimizer,
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": float(args.weight_decay),
        "head_lr_mult": head_lr_mult,
        "llrd": bool(args.llrd),
        "llrd_decay": float(args.llrd_decay),
        "freeze_mode": args.mode,
        "scheduler": "cosine",
        "warmup_epochs": int(args.warmup),
        "label_smoothing": float(args.label_smoothing),
        "class_weight": "none",
        "ema_decay": float(args.ema_decay),
        # 微调时关掉 MixUp/CutMix —— 它们会拖慢收敛，而微调本来轮数就少。
        # 只在 --strong-aug 时才开。
        "mixup_alpha": 0.2 if args.strong_aug else 0.0,
        "cutmix_alpha": 0.5 if args.strong_aug else 0.0,
        "mix_prob": 0.3 if args.strong_aug else 0.0,
        "tta_hflip": True,
        "grad_clip": 5.0,
        "seed": int(args.seed),
        "threads": None,
        "amp": True,
        "pin_memory": True,
        "checkpoint_interval": max(1, int(args.epochs)),
        "log_interval": 50,
        "note": (f"微调：起点 {Path(ckpt).parent.name}，模式 {args.mode}，"
                 f"lr={lr:g}，epochs={args.epochs}"
                 + ("，LLRD" if args.llrd else "")
                 + (f"，数据目录 {args.data}" if args.data else "")),
    }
    # num_workers 只有显式传了才写进配置，否则交给引擎按设备自动决定
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)

    if args.data:
        cfg["train_dir"] = str(Path(args.data).resolve())
    if args.val_data:
        if args.val_data.lower() in ("none", "-", "off"):
            # 显式声明"没有验证集"：用训练集自身评估（引擎会打警告）
            cfg["val_dir"] = str(Path(paths.DATA_DIR / "__no_val__").resolve())
        else:
            cfg["val_dir"] = str(Path(args.val_data).resolve())
    # 没给 --val-data 时保持默认的 data/val：
    # 这样即使你用自己的照片微调，也还能顺便看到**原来那 51 类的本事有没有被弄坏**
    # （这是微调最常见的翻车方式 —— 新任务学会了，老任务忘了）。

    if args.progressive_unfreeze:
        cfg["freeze_backbone_epochs"] = int(args.progressive_unfreeze)
        cfg["unfreeze_to"] = "full"

    return cfg


def explain(cfg: dict, ckpt: Path) -> None:
    """把"这次微调到底在干什么"讲清楚 —— 微调的参数比从头训练更微妙。"""
    recipe = LR_RECIPE[cfg["freeze_mode"]]
    print("=" * 84)
    print("微调计划")
    print("=" * 84)
    print(f"  起点权重   : {ckpt}")
    print(f"  输出 run   : {cfg['run_name']}")
    print(f"  骨干 / 类别 : {cfg['arch']} / {cfg['num_classes']} 类 @ {cfg['image_size']}")
    print(f"  设备       : {device_label(get_device(cfg['device']))}")
    print()
    print(f"  冻结模式   : {cfg['freeze_mode']}")
    print(f"      {recipe['freeze_why']}")
    print(f"  学习率     : {cfg['lr']:g}"
          + ("（LLRD 逐层递减）" if cfg["llrd"] else f"（分类头 ×{cfg['head_lr_mult']:g}）"))
    print(f"      {recipe['lr_why']}")
    print(f"  轮数/批大小 : {cfg['epochs']} / {cfg['batch_size']}")
    print(f"  预热/调度   : {cfg['warmup_epochs']} epoch 预热 + 余弦退火")
    print(f"  数据增强   : {'强（RandAugment + 随机擦除 + MixUp/CutMix）' if cfg['strong_aug'] else '温和（裁剪 + 翻转 + 轻色彩抖动）'}")
    print(f"  训练数据   : {cfg.get('train_dir') or '默认 data/train'}")
    print(f"  验证数据   : {cfg.get('val_dir') or '默认 data/val'}")
    if cfg.get("freeze_backbone_epochs"):
        print(f"  渐进解冻   : 前 {cfg['freeze_backbone_epochs']} 轮只训分类头，之后放开骨干")
    print("=" * 84)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="在已训练模型上继续微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("--from", dest="from_", required=True,
                    help="起点：run 名（improved_resnet50_224）或 checkpoint 路径")
    ap.add_argument("--run-name", default=None, help="输出 run 名，默认按起点+模式自动生成")
    ap.add_argument("--mode", default="partial", choices=["head", "partial", "full"],
                    help="冻结模式：head=只训分类头 / partial=训深层 / full=全量（默认 partial）")

    ap.add_argument("--data", default=None, help="训练数据目录（默认 data/train）；用自己的照片做领域适配时指定")
    ap.add_argument("--val-data", default=None,
                    help="验证数据目录（默认 data/val）。传 none 表示没有验证集，"
                         "退化为用训练集评估（指标只能看拟合程度）")

    ap.add_argument("--epochs", type=int, default=8, help="默认 8（微调不需要很多轮）")
    ap.add_argument("--lr", type=float, default=None, help="学习率，默认按 --mode 自动选")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=5.0,
                    help="分类头学习率倍数（不用 --llrd 时生效），默认 5")
    ap.add_argument("--llrd", action="store_true",
                    help="启用逐层递减学习率（浅层几乎不动，深层/头动得多）")
    ap.add_argument("--llrd-decay", type=float, default=0.75, help="LLRD 每层的衰减倍率")
    ap.add_argument("--progressive-unfreeze", type=int, default=0,
                    help="前 N 轮只训分类头，之后解冻全量（渐进解冻）")
    ap.add_argument("--warmup", type=int, default=1, help="预热轮数")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--strong-aug", action="store_true",
                    help="启用强增强（RandAugment + 随机擦除 + MixUp/CutMix）；数据多时可用")
    ap.add_argument("--train-transform", default="moderate_train",
                    choices=["baseline_train", "moderate_train", "strong_train"])

    ap.add_argument("--arch", default=None, help="覆盖骨干（默认沿用起点的）")
    ap.add_argument("--image-size", type=int, default=None, help="覆盖输入尺寸")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader 进程数；不传 = 自动（GPU 用 6，CPU 用 0）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不真的训练")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.from_)
    cfg = build_config(args)
    cfg["strong_aug"] = args.strong_aug          # 只给 explain() 用
    explain(cfg, ckpt)
    cfg.pop("strong_aug", None)

    if args.dry_run:
        print("\n[dry-run] 配置如下（未训练）：")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    out_cfg = paths.CONFIG_DIR / f"{cfg['run_name']}.json"
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n配置已写出：{out_cfg}")

    logger = get_logger(f"finetune.{cfg['run_name']}")
    t0 = time.perf_counter()
    summary = engine.train_model(cfg, logger=logger)
    print(f"\n微调完成，用时 {(time.perf_counter() - t0) / 60:.1f} 分钟")
    print(json.dumps({k: summary[k] for k in ("run_name", "best", "total_time_s")},
                     ensure_ascii=False, indent=2))
    print(f"\n接下来可以：")
    print(f"  1) 评估：python src/evaluate.py --checkpoint outputs/checkpoints/{cfg['run_name']}/best.pt --tta-hflip")
    print(f"  2) 比选：python src/finalize.py --candidates {cfg['run_name']} improved_resnet18_224 improved_resnet50_224")
    print(f"  3) 单张试：python src/infer.py --checkpoint outputs/checkpoints/{cfg['run_name']}/best.pt --image <图片>")


if __name__ == "__main__":
    main()
