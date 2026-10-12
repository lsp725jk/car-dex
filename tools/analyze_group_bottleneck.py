# -*- coding: utf-8 -*-
"""拆解各来源类分组的可达 F1，定位瓶颈到底在哪。"""
from pathlib import Path

import numpy as np

N_K, N_U = 1000.0, 15540.0
CACHE = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "roc_strict"


def f1curve(mk, mu):
    best = None
    for th in np.arange(-12, 8.01, 0.05):
        kr = float((mk > th).mean())
        ur = float((mu > th).mean())
        tp = N_K * (1 - kr)
        fp = N_U * (1 - ur)
        if tp + fp <= 0:
            continue
        f = 2 * tp / (tp + fp + N_K)
        if best is None or f > best[0]:
            best = (f, ur, kr, tp, fp, float(th))
    return best


for tag in ("champion_orig", "finetune_ens", "ft_plus_orig2"):
    p = CACHE / f"{tag}_hflip.npz"
    if not p.exists():
        continue
    z = np.load(p, allow_pickle=False)
    mk, mo, mn, mh = z["mk"], z["mo"], z["mn"], z["mh"]
    z.close()
    print(f"===== {tag} =====")
    groups = (("98 个精选来源类(orig)", mo), ("579 个新增来源类(new)", mn),
              ("100 个诚实类(unseen)", mh))
    for name, mu in groups:
        f, ur, kr, tp, fp, th = f1curve(mk, mu)
        print(f"  {name:<26} F1={f:.4f}  UR={ur:6.2%}  KR={kr:6.2%}  "
              f"TP={tp:5.0f}  FP={fp:5.0f}  θ={th:+.2f}")
    print()
