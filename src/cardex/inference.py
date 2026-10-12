# -*- coding: utf-8 -*-
"""推理核心：单张图片识别 + 批量识别。

被 CLI 脚本调用，后续任务二的视觉分类工具（classify_car）也直接复用本模块，
保证“训练 / 保存 / Web 推理 / 批量推理使用同一映射与同一套预处理参数”。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import paths
from .engine import load_model_from_checkpoint
from .utils import autocast_dtype, get_device


class _StackCrops:
    """把 FiveCrop 出的 5 个 PIL 图各自转张量、归一化，再堆成 (5,C,H,W)。

    写成**模块级类**而不是 `transforms.Lambda(lambda ...)`：
    lambda 是局部对象，无法被 pickle —— 一旦 DataLoader 开了多进程
    （Windows 用 spawn 会序列化 transform），就会报
    `AttributeError: Can't get local object ...`。
    """

    def __init__(self, mean, std) -> None:
        self.norm = transforms.Normalize(list(mean), list(std))

    def __call__(self, crops):
        return torch.stack([self.norm(transforms.ToTensor()(c)) for c in crops])

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CarClassifier:
    """加载自训练模型的车型分类器。

    支持两项由验证集确定的决策校准参数（见 src/calibrate.py）：
      prior_adjust_tau    先验校正强度，z'_c = z_c - tau * log(pi_c)
      unknown_threshold   拒识阈值，max softmax < threshold 时输出 unknown
    """

    def __init__(self, checkpoint: Path, tta_hflip: bool = False, device: str = "auto",
                 amp: bool = False, prior_adjust_tau: float = 0.0,
                 unknown_threshold: float = 0.0, tta_mode: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint)
        # device="auto" → 有 CUDA 就用 CUDA，否则 CPU。
        # 显式传 "cpu" / "cuda" / "cuda:1" 可以强制指定。
        self.device = get_device(device)
        self.model, self.ckpt = load_model_from_checkpoint(self.checkpoint_path)
        self.model.to(self.device).eval()
        # TTA 档位：显式给 tta_mode 优先；否则由老的 tta_hflip 布尔值推导（向后兼容）
        self.tta_mode = tta_mode or ("hflip" if tta_hflip else "none")
        self.tta_hflip = self.tta_mode != "none"
        self.amp = amp
        self.prior_adjust_tau = float(prior_adjust_tau)
        self.unknown_threshold = float(unknown_threshold)
        self.log_prior = None  # 在 label_names 就绪后计算

        pre = self.ckpt.get("preprocess", {})
        size = int(self.ckpt.get("image_size", 224))
        resize = int(pre.get("eval_resize", round(size * 256 / 224)))
        crop = int(pre.get("eval_crop", size))
        mean = pre.get("mean", [0.485, 0.456, 0.406])
        std = pre.get("std", [0.229, 0.224, 0.225])
        self.crop_size = crop
        # 中心裁剪：常规路径，单视角
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        # 多裁剪：FiveCrop 出 5 个视角（中心 + 四角），各自转张量后堆成 (5,C,H,W)。
        # 实测对细粒度任务有效（+0.0025 左右）—— 因为区分车型的关键部位
        # （进气格栅、大灯形状）不一定落在画面正中央。
        # ⚠️ 注意这和「多尺度 TTA」完全不同：多尺度是改分辨率（模型没见过，会掉点），
        #    多裁剪保持训练分辨率不变，只是换"看哪一块"。
        self.transform_five = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
                transforms.FiveCrop(crop),
                _StackCrops(mean, std),
            ]
        )
        self.label_names: list[str] = list(self.ckpt["label_names"])
        self.label_ids: list[str] = self._label_ids()
        self.unknown_index = len(self.label_names) - 1
        if self.prior_adjust_tau:
            self.log_prior = self._training_log_prior()

    def _label_ids(self) -> list[str]:
        cm = self.ckpt.get("class_map") or paths.load_class_map()
        ids = [cm["index_to_id"][str(i)] for i in range(cm["num_classes"] - 1)]
        ids.append(paths.UNKNOWN_LABEL)
        return ids

    def _training_log_prior(self) -> torch.Tensor:
        """从训练集统计中还原训练先验 pi_c = n_c / N，供先验校正使用。"""
        try:
            stats = json.loads(paths.DATA_STATS_JSON.read_text(encoding="utf-8"))
            cm = self.ckpt.get("class_map") or paths.load_class_map()
            n_c = []
            for i in range(len(self.label_names) - 1):
                cid = cm["index_to_id"][str(i)]
                n_c.append(stats["per_class"]["train"][cid]["n"])
            n_c.append(stats["per_class"]["train"]["unknown"]["n"])
        except Exception:
            n_c = [1] * len(self.label_names)
        t = torch.tensor(n_c, dtype=torch.float32)
        return torch.log(t / t.sum())

    def _decode(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (probs, 预测索引, 置信度)，已应用先验校正与拒识阈值。"""
        if self.log_prior is not None:
            logits = logits - self.prior_adjust_tau * self.log_prior.to(logits.device)[None, :]
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        if self.unknown_threshold > 0:
            idx = torch.where(
                conf < self.unknown_threshold,
                torch.full_like(idx, self.unknown_index),
                idx,
            )
        return probs, idx, conf

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=autocast_dtype(self.device),
                            enabled=self.amp):
            out = self.model(x)
        return out.float()

    @torch.no_grad()
    def tta_logits(self, images: list[Image.Image],
                   return_raw: bool = False) -> torch.Tensor:
        """对一批 PIL 图做 TTA，返回平均后的 log-softmax (B, C)。

        TTA 档位：
          · `none`            单视角，只做中心裁剪；
          · `hflip`           中心裁剪 + 水平翻转，2 个视角（原来只有这一档）；
          · `hflip+multicrop` 中心裁剪 + 五裁剪，各配翻转，共 **10 个视角**。

        多裁剪实测能给 **+0.0025 左右**：车型识别是细粒度任务，
        区分 A6 和 A6L 的关键部位（格栅、大灯、后门长度）不一定在画面正中，
        换几个裁剪位置相当于让模型"多看几眼"。

        ⚠️ 代价是**推理时间 ×5**（10 次前向 vs 2 次）。
        对 Web 交互场景，2 张图 batch 也就几百毫秒，可以接受。

        `return_raw=True` 返回**原始 logits**（不取 log_softmax）——
        Energy-based OOD 需要 logits 的**量纲**（`logsumexp(z)`），
        而 log_softmax 把它归一化掉了，信息已经丢失。
        两条分支共用同一条前向路径，保证"标定口径 == 生产口径"。
        """
        mode = self.tta_mode
        if mode == "none":
            x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
            return self._pool([self._forward(x)], return_raw)

        # 视角组 1：中心裁剪，2 个视角（原图 + 水平翻转）
        xc = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        run_c = [self._forward(xc), self._forward(torch.flip(xc, dims=[3]))]
        if mode != "hflip+multicrop" or self.crop_size <= 0:
            return self._pool(run_c, return_raw)

        # 视角组 2：五裁剪（中心 + 四角），每个位置再配翻转。
        # transform_five 返回 (5,C,H,W)，堆成 (B,5,C,H,W) 后展平成 (B*5,C,H,W)
        # 一次前向算完，比为每个视角单独调用快得多。
        xf = torch.stack([self.transform_five(im.convert("RGB")) for im in images])
        b, v = xf.shape[0], xf.shape[1]
        xf = xf.to(self.device).view(b * v, *xf.shape[2:])
        run_f = [self._forward(xf), self._forward(torch.flip(xf, dims=[3]))]

        # 两组各自先按视角平均，再对两组**等权**平均。
        # ⚠️ 这里踩过坑：一开始写成"把两组相加再除以 (2+5)"，结果中心裁剪被算成 2/7、
        # 五裁剪被算成 5/7 —— 权重和实验标定时用的 50/50 不一致，
        # 生产代码复现出来只有 0.9695（实验是 0.9815），差了 0.012。
        # **标定用的口径必须和推理时的口径逐字节一致**，否则 τ/t 全是错的。
        if return_raw:
            lc = torch.stack(run_c).mean(0)
            lfr = torch.stack(run_f).view(2, b, v, -1).mean(2).mean(0)
            return (lc + lfr) / 2
        lc = torch.stack([torch.log_softmax(t, dim=1) for t in run_c]).mean(0)
        lfr = torch.stack([torch.log_softmax(t, dim=1) for t in run_f]).view(2, b, v, -1)
        lfr = lfr.mean(2).mean(0)
        return (lc + lfr) / 2

    @staticmethod
    def _pool(run: list[torch.Tensor], return_raw: bool) -> torch.Tensor:
        if return_raw:
            return torch.stack(run).mean(0)
        return torch.stack([torch.log_softmax(t, dim=1) for t in run]).mean(0)

    @torch.no_grad()
    def predict_image(self, image: Image.Image, top_k: int = 5) -> dict:
        logits = self.tta_logits([image])
        probs, idx, conf = self._decode(logits)
        probs, idx, conf = probs[0], idx[0], conf[0]
        raw_order = torch.argsort(probs, descending=True)[:top_k]
        return {
            "label_id": self.label_ids[int(idx)],
            "label_name": self.label_names[int(idx)],
            "index": int(idx),
            "confidence": float(conf),
            "is_unknown": int(idx) == self.unknown_index,
            "top_k": [
                {
                    "label_id": self.label_ids[int(i)],
                    "label_name": self.label_names[int(i)],
                    "index": int(i),
                    "probability": float(probs[int(i)]),
                }
                for i in raw_order
            ],
        }

    @torch.no_grad()
    def predict_path(self, path: Path, top_k: int = 5) -> dict:
        path = Path(path)
        with Image.open(path) as img:
            result = self.predict_image(img, top_k=top_k)
        result["image_path"] = str(path)
        result["image_id"] = path.name
        return result

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image], batch_size: int = 32) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            logits = self.tta_logits(chunk)
            probs, idx, conf = self._decode(logits)
            for j in range(len(chunk)):
                out.append(
                    {
                        "label_id": self.label_ids[int(idx[j])],
                        "label_name": self.label_names[int(idx[j])],
                        "index": int(idx[j]),
                        "confidence": float(conf[j]),
                        "is_unknown": int(idx[j]) == self.unknown_index,
                    }
                )
        return out


def list_images(directory: Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


class CarEnsemble:
    """多模型软投票集成：对若干 checkpoint 的概率取平均。

    集成方式与 `src/finalize.py` 完全一致：
      1. 每个成员先做 log_softmax，再减去先验校正项 tau * log(pi)；
      2. 对各成员"校正后的 log 概率"做加权平均（几何平均）；
      3. 对平均结果做 softmax 得到集成概率；
      4. 若 max 概率 < 拒识阈值，则输出 unknown。

    集成成员必须使用同一份类别映射（本项目由 configs/class_map.json 唯一确定）。
    """

    def __init__(self, checkpoints: list[Path], tta_hflip: bool = False,
                 weights: list[float] | None = None, amp: bool = False,
                 prior_adjust_tau: float = 0.0,
                 unknown_threshold: float = 0.0,
                 device: str = "auto", tta_mode: str | None = None) -> None:
        if not checkpoints:
            raise ValueError("至少需要一个 checkpoint")
        # 集成会把成员模型**串行**跑（省显存），所以它们共用同一块设备
        self.device = get_device(device)
        self.tta_mode = tta_mode or ("hflip" if tta_hflip else "none")
        # 成员自身不做标定，由集成统一处理
        self.members = [CarClassifier(Path(p), amp=amp, device=str(self.device),
                                      tta_mode=self.tta_mode) for p in checkpoints]
        self.paths = [str(p) for p in checkpoints]
        base = self.members[0]
        for m in self.members[1:]:
            if m.label_ids != base.label_ids:
                raise ValueError("集成成员的类别映射不一致，无法集成")
        self.label_ids = base.label_ids
        self.label_names = base.label_names
        self.unknown_index = base.unknown_index
        self.weights = weights or [1.0] * len(self.members)
        self.prior_adjust_tau = float(prior_adjust_tau)
        self.unknown_threshold = float(unknown_threshold)
        self.log_prior = base._training_log_prior() if self.prior_adjust_tau else None

    @torch.no_grad()
    def score_logits(self, images: list[Image.Image],
                     batch_size: int = 32) -> torch.Tensor:
        """集成的**打分本体**：返回 (N, 51) 的 logits，已做权重融合与先验校正。

        这是**唯一**的打分入口 —— `predict_batch` 和"导出 logits 做提交"都走它。
        之所以抽出来：项目早期踩过"实验脚本自己实现一遍打分、结果和线上差 0.012"的坑。
        打分口径必须只有一份实现。

        返回的是 logits（softmax 之前的分数），与教师要求的 `result.npz` 一致：
        可以取负、不要求和为 1；`argmax` 即最终预测（阈值 t=0 时无额外拒识）。
        """
        wsum = float(sum(self.weights))
        acc = None
        # 每批再切成 batch_size 的小块：多裁剪模式下一次前向的实际张量数是
        # batch_size × 视角数（默认 10），不切块在大分辨率下会爆显存。
        step = max(1, batch_size)
        for s in range(0, len(images), step):
            chunk = images[s : s + step]
            acc_c = None
            for member, w in zip(self.members, self.weights):
                logits = member.tta_logits(chunk)          # 已按 tta_mode 平均好
                if self.log_prior is not None:
                    logits = logits - self.prior_adjust_tau * self.log_prior.to(logits.device)[None, :]
                term = logits * (w / wsum)
                acc_c = term if acc_c is None else acc_c + term
            acc = acc_c if acc is None else torch.cat([acc, acc_c], dim=0)
        return acc

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image], batch_size: int = 32,
                      top_k: int | None = None) -> list[dict]:
        acc = self.score_logits(images, batch_size=batch_size)
        probs = torch.softmax(acc, dim=1)
        conf, idx = probs.max(dim=1)
        if self.unknown_threshold > 0:
            idx = torch.where(conf < self.unknown_threshold,
                              torch.full_like(idx, self.unknown_index), idx)

        # Top-K 候选（按**校正后的集成概率**排序）。
        # 注意：拒识时 idx 被改写成 unknown，但 top_k 仍然列出真实候选——
        # 这正是「拒识 + 给用户几个参考项」所需要的信息。
        order = None
        if top_k:
            order = torch.argsort(probs, descending=True, dim=1)[:, : int(top_k)]

        out: list[dict] = []
        for j in range(len(images)):
            item = {
                "label_id": self.label_ids[int(idx[j])],
                "label_name": self.label_names[int(idx[j])],
                "index": int(idx[j]),
                "confidence": float(conf[j]),
                "is_unknown": int(idx[j]) == self.unknown_index,
            }
            if order is not None:
                item["top_k"] = [
                    {
                        "label_id": self.label_ids[int(i)],
                        "label_name": self.label_names[int(i)],
                        "index": int(i),
                        "probability": float(probs[j, int(i)]),
                    }
                    for i in order[j]
                ]
            out.append(item)
        return out

    @torch.no_grad()
    def predict_path(self, path: Path, top_k: int = 5) -> dict:
        path = Path(path)
        with Image.open(path) as img:
            image = img.convert("RGB").copy()
        pred = self.predict_batch([image], top_k=top_k)[0]
        pred["image_path"] = str(path)
        pred["image_id"] = path.name
        return pred
