# -*- coding: utf-8 -*-
"""泛化能力验证：确认 0.9815 不是"在验证集上刷出来的"。

## 为什么需要这个

我们在这条路上做了**大量基于验证集的选择**：
  · 扫 τ 和拒识阈值 t（这是任务书要求的，合理）
  · 比较 6 种集成组合、3 种 TTA 档位
  · 选出冠军

每一次"在验证集上挑最好的"，都会让验证集分数**偏乐观**。
教师测试集没有提供（数据集里只有 train/val），所以必须用严谨的替代方案。

## 三个检验

**[1] 校准的交叉验证（5 折）**
把验证集分 5 份。每一折：在**其余 4 份**上扫 τ/t，在**这一份**上评估。
这样得到的分数 = "用一批新数据标定、拿到另一批新数据上用"的真实水平。
如果它和 0.9815 接近，说明标定没怎么过拟合。

**[2] 逐折模型选择压力测试**
更狠一点：每一折不仅重扫 τ/t，还在 3 种 TTA 档位 + 若干集成组合里**重新挑冠军**，
再看它在留出折上的表现。这模拟"完全诚实的模型选择流程"。

**[3] 鲁棒性压力测试**
模拟真实拍照条件（JPEG 压缩、模糊、亮度变化、小角度旋转），
看性能掉多少。这是"能不能用在真实场景"的关键。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402
from cardex.inference import CarEnsemble  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402

CACHE = paths.RUN_DIR / "generalization_cache"


def prior_log():
    cm = paths.load_class_map()
    st = json.loads((paths.CONFIG_DIR / "data_stats.json").read_text(encoding="utf-8"))
    n = [st["per_class"]["train"][cm["index_to_id"][str(i)]]["n"] for i in range(50)]
    n.append(st["per_class"]["train"]["unknown"]["n"])
    t = np.array(n, dtype=np.float64)
    return np.log(t / t.sum())


def class_names():
    cm = paths.load_class_map()
    return [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]


def apply(logp, tau, thr, log_prior):
    z = logp - tau * log_prior[None, :]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pm, pa = p.max(axis=1), p.argmax(axis=1)
    return np.where(pm < thr, 50, pa)


def best_params(logp, y, names, log_prior, taus=None, thrs=None):
    """在给定数据上扫出最优 (τ, t)。"""
    taus = np.arange(-0.4, 0.601, 0.1) if taus is None else taus
    thrs = np.arange(0.0, 0.601, 0.05) if thrs is None else thrs
    best = (-1.0, 0.0, 0.0)
    for tau in taus:
        for th in thrs:
            pr = apply(logp, tau, th, log_prior)
            f = compute_metrics(y, pr, names)["macro_f1"]
            if f > best[0]:
                best = (f, float(tau), float(th))
    return best


def get_champion_logits(members, tta_mode, batch, device):
    """用生产推理代码算冠军配置在验证集上的 log-softmax。"""
    key = f"champ_{'_'.join(members)}_{tta_mode.replace('+', '')}"
    f = CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f), np.load(CACHE / f"{key}_y.npy")
    CACHE.mkdir(parents=True, exist_ok=True)

    ds = CarDataset(paths.DATA_DIR / "val")
    y = np.array([s[1] for s in ds.samples])
    ens = CarEnsemble([paths.CKPT_DIR / m / "best.pt" for m in members],
                      tta_mode=tta_mode, amp=True, prior_adjust_tau=0.0,
                      unknown_threshold=0.0, device=device)
    print(f"  用生产代码算 {len(ds.samples)} 张的 logits（{tta_mode}）…")
    out = []
    for i in range(0, len(ds.samples), batch):
        chunk = ds.samples[i : i + batch]
        imgs = []
        for p, _ in chunk:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        with torch.no_grad():
            # tta_logits 返回的是**未做先验校正**的 log-softmax
            lps = [m.tta_logits(imgs) for m in ens.members]
            out.append(torch.stack(lps).mean(0).float().cpu().numpy())
        if i % 400 == 0:
            print(f"    {i}/{len(ds.samples)}")
    logp = np.concatenate(out)
    np.save(f, logp)
    np.save(CACHE / f"{key}_y.npy", y)
    return logp, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*",
                    default=["r50_288_from_scratch", "r50_352"])
    ap.add_argument("--tta-mode", default="hflip+multicrop")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()

    names = class_names()
    lp = prior_log()

    print("=" * 84)
    print("载入冠军配置在验证集上的 logits")
    print("=" * 84)
    logp, y = get_champion_logits(args.members, args.tta_mode, args.batch, args.device)
    print(f"  {logp.shape}  验证集 {len(y)} 张")

    # ---- 基准：在整个验证集上标定 ----
    f_all, tau_all, th_all = best_params(logp, y, names, lp)
    print(f"\n基准（在整个验证集上标定，即我们报出的数字）：")
    print(f"  Macro-F1 = {f_all:.4f}   τ={tau_all:+.1f}  t={th_all:.2f}")

    # ---- [1] 5 折交叉验证 ----
    print("\n" + "=" * 84)
    print(f"[1] {args.folds} 折交叉验证：在其中 {args.folds - 1} 折上标定，在留出折上评估")
    print("=" * 84)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, args.folds)
    scores_in, scores_out = [], []
    for k, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
        _f_in, tau_k, th_k = best_params(logp[tr], y[tr], names, lp)
        f_out = compute_metrics(y[te], apply(logp[te], tau_k, th_k, lp), names)["macro_f1"]
        scores_in.append(_f_in)
        scores_out.append(f_out)
        print(f"  折 {k + 1}: 标定集 {len(tr)} 张 → τ={tau_k:+.1f} t={th_k:.2f} "
              f"（标定集内 {_f_in:.4f}）｜留出折 {len(te)} 张 → **{f_out:.4f}**")
    m_out, s_out = float(np.mean(scores_out)), float(np.std(scores_out))
    print(f"\n  留出折平均 Macro-F1 = {m_out:.4f} ± {s_out:.4f}")
    print(f"  与基准 {f_all:.4f} 的差距 = {m_out - f_all:+.4f}")

    # ---- [2] 更严格：留出折上连 TTA 档位也重选 ----
    print("\n" + "=" * 84)
    print("[2] 压力测试：把 TTA 档位/集成组合的选择也纳入折内（完全诚实的流程）")
    print("=" * 84)
    variants = {}
    for tag, members, mode in [
        ("288+352@mc", ["r50_288_from_scratch", "r50_352"], "hflip+multicrop"),
        ("352@mc", ["r50_352"], "hflip+multicrop"),
        ("288@mc", ["r50_288_from_scratch"], "hflip+multicrop"),
        ("288+352@hflip", ["r50_288_from_scratch", "r50_352"], "hflip"),
    ]:
        try:
            lv, _ = get_champion_logits(members, mode, args.batch, args.device)
            variants[tag] = lv
            fv, tv, hv = best_params(lv, y, names, lp)
            print(f"  {tag:<16} 全体标定 {fv:.4f}  (τ={tv:+.1f}, t={hv:.2f})")
        except Exception as e:            # noqa: BLE001
            print(f"  {tag:<16} 跳过（{type(e).__name__}）")

    if len(variants) >= 2:
        honest = []
        for k, te in enumerate(folds):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
            pick, pick_f = None, -1.0
            for tag, lv in variants.items():
                f_tr, tv, hv = best_params(lv[tr], y[tr], names, lp)
                if f_tr > pick_f:
                    pick, pick_f = (tag, lv, tv, hv), f_tr
            tag, lv, tv, hv = pick
            f_out = compute_metrics(y[te], apply(lv[te], tv, hv, lp), names)["macro_f1"]
            honest.append(f_out)
            print(f"  折 {k + 1}: 选中 {tag}（标定集内 {pick_f:.4f}）→ 留出折 **{f_out:.4f}**")
        print(f"\n  完全诚实流程的平均 Macro-F1 = {np.mean(honest):.4f} ± {np.std(honest):.4f}")
        print(f"  与基准 {f_all:.4f} 的差距 = {np.mean(honest) - f_all:+.4f}")

    # ---- [3] 鲁棒性压力测试 ----
    if not args.skip_robustness:
        print("\n" + "=" * 84)
        print("[3] 鲁棒性压力测试：模拟真实拍照条件")
        print("=" * 84)
        ds = CarDataset(paths.DATA_DIR / "val")
        sub = list(range(0, len(ds.samples), 4))        # 每 4 张取 1 张，约 600 张
        ys = y[sub]
        ens = CarEnsemble([paths.CKPT_DIR / m / "best.pt" for m in args.members],
                          tta_mode=args.tta_mode, amp=True,
                          prior_adjust_tau=tau_all, unknown_threshold=th_all,
                          device=args.device)

        def perturb(im: Image.Image, kind: str) -> Image.Image:
            if kind == "原图":
                return im
            if kind == "JPEG q30":
                import io
                b = io.BytesIO()
                im.save(b, "JPEG", quality=30)
                b.seek(0)
                return Image.open(b).convert("RGB")
            if kind == "高斯模糊 r2":
                return im.filter(ImageFilter.GaussianBlur(2))
            if kind == "亮度 ×0.6":
                return ImageEnhance.Brightness(im).enhance(0.6)
            if kind == "亮度 ×1.5":
                return ImageEnhance.Brightness(im).enhance(1.5)
            if kind == "低分辨率 96px":
                w, h = im.size
                s = 96 / max(w, h)
                return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            if kind == "旋转 10°":
                return im.rotate(10, expand=True, fillcolor=(128, 128, 128))
            return im

        print(f"  子集 {len(sub)} 张")
        for kind in ["原图", "JPEG q30", "高斯模糊 r2", "亮度 ×0.6", "亮度 ×1.5",
                     "低分辨率 96px", "旋转 10°"]:
            preds = []
            for i in range(0, len(sub), 8):
                chunk = [ds.samples[j] for j in sub[i : i + 8]]
                imgs = []
                for p, _ in chunk:
                    with Image.open(p) as im:
                        imgs.append(perturb(im.convert("RGB").copy(), kind))
                for r in ens.predict_batch(imgs, batch_size=8):
                    preds.append(r["index"])
            preds = np.array(preds)
            m = compute_metrics(ys, preds, names)
            drop = m["macro_f1"] - f_all
            print(f"  {kind:<16} Macro-F1 = {m['macro_f1']:.4f}   ({drop:+.4f})")

    # ---- 汇总 ----
    out = {
        "baseline": {"macro_f1": f_all, "tau": tau_all, "threshold": th_all},
        "cv": {"folds": args.folds, "mean": m_out, "std": s_out,
               "gap": m_out - f_all, "per_fold": scores_out},
    }
    p = paths.REPORT_DIR / "generalization_check.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    print(f"  验证集标定值（我们报的）      ：{f_all:.4f}")
    print(f"  交叉验证的诚实估计            ：{m_out:.4f} ± {s_out:.4f}")
    gap = m_out - f_all
    if abs(gap) < 0.004:
        print(f"  差距 {gap:+.4f} —— 很小，**标定没有明显过拟合，0.9815 可信**")
    elif gap < 0:
        print(f"  差距 {gap:+.4f} —— 新数据上会略低，这是正常的乐观偏差，应在报告里如实说明")
    else:
        print(f"  差距 {gap:+.4f}")
    print("=" * 84)
    print(f"\n结果已写出：{p}")


if __name__ == "__main__":
    main()
