# -*- coding: utf-8 -*-
"""数据集与图像预处理 / 数据增强。

目录结构：
    data/train/<四位车型ID>/   或 data/train/unknown/
    data/val/<四位车型ID>/     或 data/val/unknown/
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import paths

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 少量教学图片尾部损坏，容错读取

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_SOURCE_CACHE: dict[str, int] | None = None


def load_source_to_fine_index() -> dict[str, int]:
    """unknown 的 98 个来源类别 → fine 标签索引（从 50 开始，接在 50 个已知类后面）。

    来源清单来自 `configs/unknown_sources.json`（选类阶段就落盘了，可追溯）。
    """
    global _SOURCE_CACHE
    if _SOURCE_CACHE is not None:
        return _SOURCE_CACHE
    p = paths.CONFIG_DIR / "unknown_sources.json"
    mapping: dict[str, int] = {}
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for i, s in enumerate(d.get("sources", [])):
            mapping[str(s["id"])] = 50 + i
    _SOURCE_CACHE = mapping
    return mapping


def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1, base_temperature: float = 0.1,
                eps: float = 1e-8) -> torch.Tensor:
    """**Supervised Contrastive Loss**（Khosla et al., NeurIPS 2020）。

    ## 它在做什么

    交叉熵只要求"正确答案的 logit 最大" —— 它**不关心**错的那几类之间离得多近。
    对细粒度任务这是不够的：模型可以学会"MINI 的分数比 CLUBMAN 高一点点"就交差，
    但特征空间里两者仍然挤在一起。

    SupCon 直接在**特征空间**上施加约束：

        L = Σ_i  −1/|P(i)| Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )

    通俗讲：**同类样本的相似度要高，异类样本的相似度要低**，
    而且因为分母对所有负样本求和，**"很像但不是同类"的样本会贡献最大的梯度** ——
    这正好就是我们要的"让近似款拉开距离"。

    几个实现要点：

    * `z` 必须先 **L2 归一化**，这样点积就是余弦相似度，尺度不影响损失；
    * `temperature` 控制"关注多难的负样本"：越小越只关注最像的那几个。
      细粒度任务常用 **0.05~0.1**；
    * 用 `logsumexp` 的数值稳定写法（先减最大值），否则 exp 会溢出；
    * 没有正样本的样本（该 batch 里只有它自己**且**没有第二个视图）要**剔除**，
      否则 −1/0 会出 NaN。
    """
    if features.dim() != 2 or features.size(0) < 2:
        return features.new_zeros(())
    z = F.normalize(features, dim=1)                  # L2 归一化 → 点积 = 余弦相似度
    n = z.size(0)
    sim = (z @ z.t()) / temperature                   # (N, N)

    # 数值稳定：减去每行最大值（不影响 softmax 结果，但避免 exp 溢出成 inf）
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # 正样本掩码：同标签且不是自己
    lab = labels.view(-1, 1)
    pos = (lab == lab.t()).float()
    pos.fill_diagonal_(0)

    # log( exp(s_ii)/Σ_{a≠i} exp(s_ia) ) = s_ii − log Σ_{a≠i} exp(s_ia)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    denom = torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1)
    log_prob = sim - denom.view(-1, 1)

    n_pos = pos.sum(dim=1)
    valid = n_pos > 0                                  # 没有正样本的样本直接跳过
    if not bool(valid.any()):
        return features.new_zeros(())
    loss = -(pos * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return loss.mean() * (temperature / base_temperature)


class CarDataset(Dataset):
    """按目录组织的车型分类数据集。unknown 目录固定映射到索引 50。

    ## 两个可选能力（为对比学习准备）

    **`n_views`：一张图返回多个增强视图。**
    对比学习（SupCon / SimCLR）需要"同一个样本的两个不同视角"作为正样本对。
    对同一张 PIL 图**独立地跑两遍随机增强**，就得到两个不同视角 ——
    这一步很关键，因为 **51 类、batch=32 时平均每类不到 1 张**，
    指望"同一 batch 里有同类别的另一张"是不现实的。
    有了双视图，每个样本**至少有自己的另一个视角**作为正样本。

    **`fine_label`：unknown 样本的"来源类别"标签。**
    unknown 目录里的图片来自 98 个不同的源类别，文件名前缀就是源类别 ID
    （如 `0702_0237.jpg` → `0702` = MINI_MINI CLUBMAN）。

    ⚠️ **为什么对比学习必须用 fine_label 而不是 51 类标签**：
    如果对 51 个标签做 SupCon，损失函数会把**所有 unknown 样本在特征空间里拉到一起** ——
    那意味着让"MINI CLUBMAN"和"卡车"变得相似，是**有害的**。
    用 fine_label（50 个已知类 + 98 个 unknown 源类 = 148 类）就对了：
    同源 unknown 拉近、异源推远，而 MINI CLUBMAN 会被**推开** MINI。
    这正是"让近似款在特征空间拉开距离"的数学表达。
    """

    def __init__(
        self,
        root: Path,
        transform=None,
        class_map: dict | None = None,
        limit_per_class: int | None = None,
        seed: int = 42,
        return_path: bool = True,
        n_views: int = 1,
        with_fine_label: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.return_path = return_path
        self.n_views = max(1, int(n_views))
        self.with_fine_label = with_fine_label
        cm = class_map or paths.load_class_map()
        self.id_to_index: dict[str, int] = dict(cm["id_to_index"])
        self.unknown_index: int = int(cm["unknown_index"])

        # unknown 源类别 → fine 索引（接在 50 个已知类后面，从 50 开始）
        self.source_to_fine: dict[str, int] = {}
        if with_fine_label:
            self.source_to_fine = load_source_to_fine_index()

        rng = random.Random(seed)
        self.samples: list[tuple[Path, int]] = []
        self.fine_labels: list[int] = []
        self.class_dirs: list[str] = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label_name = class_dir.name
            if label_name == paths.UNKNOWN_LABEL:
                label = self.unknown_index
            elif label_name in self.id_to_index:
                label = self.id_to_index[label_name]
            else:
                continue
            files = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXT
            )
            if limit_per_class is not None and len(files) > limit_per_class:
                files = sorted(rng.sample(files, limit_per_class))
            self.class_dirs.append(label_name)
            for p in files:
                self.samples.append((p, label))
                self.fine_labels.append(self._fine_of(p, label))

    def _fine_of(self, path: Path, label: int) -> int:
        """算出这张图的 fine 标签。已知类就是类索引；unknown 看文件名前缀。"""
        if not self.with_fine_label:
            return label
        if label != self.unknown_index:
            return label
        stem = path.stem
        src = stem.split("_")[0] if "_" in stem else stem[:4]
        # 没在清单里的源类别 → 全部归到一个"其他"桶，避免标签空间无界
        return self.source_to_fine.get(src, 50 + len(self.source_to_fine))

    @property
    def fine_num_classes(self) -> int:
        return 50 + len(self.source_to_fine) + 1 if self.with_fine_label else 51

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception:
            # 单张图片损坏时用灰色占位，避免整轮训练崩溃；在报告中作为数据质量问题记录
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform is not None:
            if self.n_views > 1:
                # 对同一张 PIL 图跑 n_views 次随机增强 → n_views 个不同视角
                image = torch.stack([self.transform(image) for _ in range(self.n_views)])
            else:
                image = self.transform(image)
        if not self.return_path:
            return image, label
        if self.with_fine_label:
            return image, label, str(path), self.fine_labels[index]
        return image, label, str(path)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# 预处理 / 增强
# --------------------------------------------------------------------------- #
def build_transform(kind: str, image_size: int, cfg: dict | None = None):
    """kind: 'baseline_train' | 'baseline_eval' | 'strong_train' | 'eval'

    预处理参数（ImageNet 均值方差、resize 策略）会随 checkpoint 一起保存，
    保证训练、Web 推理与批量推理使用同一套参数。
    """
    cfg = cfg or {}
    resize_ratio = cfg.get("eval_resize_ratio", 256 / 224)
    eval_resize = int(round(image_size * resize_ratio))

    eval_tf = transforms.Compose(
        [
            transforms.Resize(eval_resize, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if kind.endswith("_eval") or kind == "eval":
        return eval_tf

    if kind == "baseline_train":
        scale = tuple(cfg.get("crop_scale", (0.70, 1.0)))
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=scale, ratio=(3 / 4, 4 / 3),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=cfg.get("hflip", 0.5)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if kind == "moderate_train":
        # 中等强度增强：在基线基础上加入颜色抖动与随机擦除，但不使用 RandAugment / MixUp
        scale = tuple(cfg.get("crop_scale", (0.65, 1.0)))
        ops: list = [
            transforms.RandomResizedCrop(
                image_size, scale=scale, ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=cfg.get("hflip", 0.5)),
        ]
        if cfg.get("color_jitter", True):
            ops.append(
                transforms.ColorJitter(
                    brightness=cfg.get("jitter_brightness", 0.25),
                    contrast=cfg.get("jitter_contrast", 0.25),
                    saturation=cfg.get("jitter_saturation", 0.25),
                    hue=cfg.get("jitter_hue", 0.05),
                )
            )
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if cfg.get("random_erasing", 0):
            ops.append(
                transforms.RandomErasing(
                    p=cfg["random_erasing"], scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"
                )
            )
        return transforms.Compose(ops)

    if kind == "strong_train":
        scale = tuple(cfg.get("crop_scale", (0.45, 1.0)))
        ops: list = [
            transforms.RandomResizedCrop(
                image_size, scale=scale, ratio=(3 / 4, 4 / 3),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=cfg.get("hflip", 0.5)),
        ]
        if cfg.get("color_jitter", True):
            ops.append(
                transforms.ColorJitter(
                    brightness=cfg.get("jitter_brightness", 0.35),
                    contrast=cfg.get("jitter_contrast", 0.35),
                    saturation=cfg.get("jitter_saturation", 0.35),
                    hue=cfg.get("jitter_hue", 0.08),
                )
            )
        if cfg.get("rotation", 0):
            ops.append(transforms.RandomRotation(cfg["rotation"], interpolation=InterpolationMode.BILINEAR))
        if cfg.get("randaugment", False):
            ops.append(
                transforms.RandAugment(num_ops=cfg.get("ra_num_ops", 2),
                                       magnitude=cfg.get("ra_magnitude", 7))
            )
        if cfg.get("random_grayscale", 0):
            ops.append(transforms.RandomGrayscale(p=cfg["random_grayscale"]))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if cfg.get("random_erasing", 0):
            ops.append(
                transforms.RandomErasing(
                    p=cfg["random_erasing"], scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"
                )
            )
        return transforms.Compose(ops)

    raise ValueError(f"未知的 transform 类型：{kind}")


def mixup_cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 0.0,
    smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (images, soft_targets)，soft_targets 形状 (B, num_classes)。

    ⚠️ 这里所有新建的张量都必须**显式指定 device**：
    训练时 images 已经在 GPU 上，如果这里建在 CPU 上，
    `scatter_` / 混合运算会因为"index 在 cuda、其他张量在 cpu"直接报错。
    （实测：把训练搬到 GPU 后第一个撞上的就是这个。）
    """
    device = images.device
    batch_size = images.size(0)
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    targets_1h = torch.full((batch_size, num_classes), off, dtype=images.dtype,
                            device=device)
    targets_1h.scatter_(1, targets.unsqueeze(1), on)

    if prob <= 0 or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return images, targets_1h
    if random.random() > prob:
        return images, targets_1h

    perm = torch.randperm(batch_size, device=device)
    targets_2h = targets_1h[perm]
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)

    if use_cutmix:
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        h, w = images.shape[-2:]
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        if rh > 0 and rw > 0:
            cy, cx = random.randint(0, h - rh), random.randint(0, w - rw)
            images[:, :, cy : cy + rh, cx : cx + rw] = images[perm][:, :, cy : cy + rh, cx : cx + rw]
            lam = 1 - (rh * rw) / (h * w)
    else:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        images = lam * images + (1 - lam) * images[perm]

    targets_mixed = lam * targets_1h + (1 - lam) * targets_2h
    return images, targets_mixed


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor,
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    log_prob = torch.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_prob)
    if weight is not None:
        loss = loss * weight.unsqueeze(0)
    return loss.sum(dim=1).mean()
