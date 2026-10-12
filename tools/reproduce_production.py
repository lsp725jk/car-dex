# -*- coding: utf-8 -*-
"""用**生产推理代码**（CarEnsemble）跑一遍验证集，确认实验结论能复现。

为什么必须做这一步：实验脚本 tools/experiment_final.py 是自己实现的 TTA，
如果它和 src/cardex/inference.py 的实现有细微差别（裁剪顺序、归一化时机、
平均方式），那实验里的 0.9815 就是**实验室数字**，交付物上根本复现不出来。

这个脚本走的是和 Web 系统 / predictions.csv 完全相同的代码路径。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.data import CarDataset  # noqa: E402
from cardex.inference import CarEnsemble  # noqa: E402
from cardex.metrics import compute_metrics  # noqa: E402


def main() -> None:
    cfg = json.loads((paths.REPORT_DIR / "final_combination.json").read_text(encoding="utf-8"))["best"]
    members = ["r50_288_from_scratch", "r50_352"]
    tau, thr = float(cfg["tau"]), float(cfg["threshold"])
    print(f"复现目标：{cfg['tag']}   tau={tau}  t={thr}   期望 {cfg['macro_f1']:.4f}")

    cm = paths.load_class_map()
    names = [cm["index_to_name"][str(i)] for i in range(50)] + [cm["unknown_label"]]
    ds = CarDataset(paths.DATA_DIR / "val")
    y = np.array([s[1] for s in ds.samples])

    ens = CarEnsemble(
        [paths.CKPT_DIR / m / "best.pt" for m in members],
        tta_mode="hflip+multicrop", amp=True,
        prior_adjust_tau=tau, unknown_threshold=thr,
    )
    print(f"成员 {members}   设备 {ens.device}   TTA {ens.tta_mode}")

    from PIL import Image
    preds, confs = [], []
    B = 16
    for s in range(0, len(ds.samples), B):
        chunk = ds.samples[s : s + B]
        imgs = []
        for p, _ in chunk:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        for r in ens.predict_batch(imgs, batch_size=B):
            preds.append(r["index"])
            confs.append(r["confidence"])
        if s % 320 == 0:
            print(f"  {s}/{len(ds.samples)}")

    preds = np.array(preds)
    m = compute_metrics(y, preds, names)
    print()
    print("=" * 78)
    print(f"生产代码复现结果：Macro-F1 = {m['macro_f1']:.4f}")
    print(f"实验脚本结果    ：{cfg['macro_f1']:.4f}")
    d = m["macro_f1"] - cfg["macro_f1"]
    print(f"差异 = {d:+.4f}  →  {'✅ 一致，实验结论可用' if abs(d) < 0.003 else '❌ 不一致，需要排查'}")
    print(f"Accuracy = {m['accuracy']:.4f}   unknown 检出率 = {m['unknown_detection_rate']:.4f}"
          f"   已知类误拒率 = {m['known_false_reject_rate']:.4f}")
    print(f"总错误 {int((preds != y).sum())} / {len(y)}")
    print("=" * 78)

    out = paths.REPORT_DIR / "production_reproduce.json"
    out.write_text(json.dumps({
        "members": members, "tta_mode": ens.tta_mode, "tau": tau, "threshold": thr,
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "unknown_detection_rate": m["unknown_detection_rate"],
        "known_false_reject_rate": m["known_false_reject_rate"],
        "experiment_macro_f1": cfg["macro_f1"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
