# -*- coding: utf-8 -*-
"""给训练循环分段计时，找出真正的瓶颈。

⚠️ 必须有 `if __name__ == "__main__":` 保护！
Windows 上 DataLoader 多进程用 **spawn** 启动子进程，子进程会**重新导入主模块**。
没有这个保护，子进程一导入就又去创建子进程，直接无限递归刷爆 CPU
（实测：脚本卡死 10 分钟，最后靠手动杀进程才停下来）。
`src/train.py` 等正式脚本都有这个保护，本文件是自建脚本时踩到的坑。
"""
import json
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import torch
from torch.utils.data import DataLoader

from cardex import paths
from cardex.data import CarDataset, build_transform, mixup_cutmix_batch, soft_cross_entropy
from cardex.engine import ModelEMA
from cardex.models import build_model, param_groups
from cardex.utils import autocast_dtype, get_device, set_seed

set_seed(42)
cm = paths.load_class_map()
cfg = json.load(open("configs/train_improved.json", encoding="utf-8"))
NUM_CLASSES = 51


def main() -> None:
    dev = get_device("auto")
    print(f"设备: {dev}")

    tf = build_transform("strong_train", 224, cfg["aug"])
    ds = CarDataset(paths.DATA_DIR / "train", transform=tf, class_map=cm,
                    limit_per_class=30, seed=42)      # 51*30 ≈ 1530 张，够代表真实负载
    print(f"数据集: {len(ds)} 张")

    for nw in (0, 4):
        run_one(ds, dev, nw)


def run_one(ds, dev, nw: int) -> None:
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=nw,
                        pin_memory=True, persistent_workers=nw > 0, drop_last=False)
    model = build_model(cfg["arch"], num_classes=NUM_CLASSES, pretrained=True,
                        dropout=cfg.get("dropout", 0.0)).to(dev)
    opt = torch.optim.SGD(param_groups(model, cfg["lr"], cfg["head_lr_mult"]),
                          lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=True)
    ema = ModelEMA(model, cfg["ema_decay"])
    ema.ema.to(dev)
    acc: dict[str, float] = {}

    @contextmanager
    def tick(name):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        if dev.type == "cuda":
            torch.cuda.synchronize()
        acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t)

    it = iter(loader)
    N = 15
    model.train()
    for _ in range(N):
        with tick("① 取数据"):
            try:
                images, targets, _ = next(it)
            except StopIteration:
                it = iter(loader)
                images, targets, _ = next(it)
        images = images.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        with tick("② mixup/cutmix"):
            images2, soft = mixup_cutmix_batch(images, targets, NUM_CLASSES,
                                               mixup_alpha=cfg["mixup_alpha"],
                                               cutmix_alpha=cfg["cutmix_alpha"],
                                               prob=cfg["mix_prob"],
                                               smoothing=cfg["label_smoothing"])
            if soft is None:
                off = cfg["label_smoothing"] / NUM_CLASSES
                soft = torch.full((images2.size(0), NUM_CLASSES), off, device=dev)
                soft.scatter_(1, targets.unsqueeze(1),
                              1 - cfg["label_smoothing"] + off)
        with tick("③ 前向"):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=autocast_dtype(dev), enabled=True):
                logits = model(images2)
            logits = logits.float()
        with tick("④ 反向"):
            loss = soft_cross_entropy(logits, soft, None)
            loss.backward()
        with tick("⑤ 裁剪+更新"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        with tick("⑥ EMA 更新"):
            ema.update(model)
        with tick("⑦ loss.item()"):
            _ = loss.item()

    total = sum(acc.values())
    print(f"\n=== num_workers={nw}  共 {N} 步，合计 {total:.2f}s，平均 {total / N * 1000:.0f} ms/步 ===")
    for k in sorted(acc):
        print(f"  {k:16} {acc[k] / N * 1000:7.1f} ms/步   {acc[k] / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
