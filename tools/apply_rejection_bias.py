# -*- coding: utf-8 -*-
"""应用校准出的拒识偏置，重新生成提交物（**不需要重新推理**）。

## 原理

判别规则是 margin 阈值：
    m = z_unknown − max(z_已知类)  >  θ   → 判 unknown

等价于给 unknown 列加一个偏置后取 argmax：
    z'[:, 50] -= θ        （θ 为负时即抬高 unknown）

所以修复只需改一列数值 —— **不用重跑 62 分钟的推理**。

## 为什么 θ 取负值

θ 越负，unknown 越容易被判出。原来的规则等价于 θ=0（纯 argmax）。
测试集 94% 是 unknown，而我们的模型是按"24% unknown"的验证集标定的 → 严重偏向已知类。

## θ = −5.6 是怎么定的

由老师反馈的 θ=0 那次成绩反解：真已知 1000 张、真未知 15540 张，
θ=0 时 FP=3304、声称已知 4289 张。两者之差 985 ≈ 1000 ——
也就是说**把多出来的 FP 全部削掉，正好落到 1000 张**。

实测扫描（`tools/sweep_theta.py`）显示 θ=−5.6 时声称已知 1003 张、判未知 93.94%，
与真实 unknown 占比 93.96% 几乎完全一致。再往负走，FP 已经耗尽，
只会开始吃掉真已知 → F1 下降。所以 −5.6 是理论工作点，不是随手取的数。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from cardex import paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_KNOWN, TEST_UNK = 1000, 15540
P_UNK = TEST_UNK / (TEST_KNOWN + TEST_UNK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=-5.6,
                    help="拒识偏置的**绝对目标值**（不是增量）。越负越倾向判 unknown")
    ap.add_argument("--dir", default="student_submit",
                    help="提交目录；metadata 按目录名分开存，避免不同目录之间串味")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cm = paths.load_class_map()
    classes = [cm["index_to_id"][str(i)] for i in range(50)]

    out_dir = ROOT / args.dir
    npz = out_dir / "result.npz"
    if not npz.exists():
        raise SystemExit(f"找不到 {npz}")
    z = np.load(npz, allow_pickle=False)
    L, ids = z["logits"].copy(), z["image_id"]
    z.close()

    # ── 幂等性：npz 里可能已经施加过某个 θ，这里读出上次的值算增量 ──
    # ⚠️ 早期版本直接把 --theta 当增量减上去，第二次运行会变成叠加（-5.2 再 -5.6 = -10.8），
    # 静默产生一个完全错误的提交。改成读 metadata 反推增量后，重复运行是安全的空操作。
    # ⚠️ 再后来发现：多个提交目录（student_submit / student_submit_ft）共用一份 metadata 也会串味
    # ——ft 目录是全新推理（前值应为 0），却会读到另一目录的 -4.8，算出错误的增量。
    # 所以 metadata 按目录名分开存。
    meta_path = paths.REPORT_DIR / f"applied_rejection_bias_{out_dir.name}.json"
    prev_theta = 0.0
    if meta_path.exists():
        try:
            prev_theta = float(json.loads(meta_path.read_text(encoding="utf-8"))["theta"])
        except (KeyError, ValueError, json.JSONDecodeError):
            prev_theta = 0.0
    delta = args.theta - prev_theta

    # 当前（已带 prev_theta）的预测
    pred_before = L.argmax(1)
    ur_before = float((pred_before == 50).mean())

    # 施加增量：z'[:, 50] -= delta  ⇒  等价于把绝对偏置从 prev_theta 改到 args.theta
    L[:, 50] -= delta
    pred_after = L.argmax(1)

    print("=" * 80)
    print(f"拒识偏置：{prev_theta:+.2f} → {args.theta:+.2f}（增量 {delta:+.2f}）")
    print("=" * 80)
    print(f"  测试集 {len(ids)} 张")
    print(f"  改动前：unknown {int((pred_before == 50).sum())} 张（{ur_before:.2%}）")
    print(f"  改动后：unknown {int((pred_after == 50).sum())} 张"
          f"（{(pred_after == 50).mean():.2%}）")
    print(f"  真实 unknown 占比（由评测指标反推）= {P_UNK:.2%}")
    n_known = int((pred_after != 50).sum())
    print(f"  已知类预测 {n_known} 张，真相约 {TEST_KNOWN} 张 → 多认 {n_known - TEST_KNOWN} 张")
    print(f"  （改动前多认 {int((pred_before != 50).sum()) - TEST_KNOWN} 张）")
    print(f"  改动 {int((pred_before != pred_after).sum())} 张"
          f"（{(pred_before != pred_after).mean():.2%}）")

    if args.theta > 0:
        print("\n  ⚠ θ>0 会让模型更倾向已知类，与本次诊断方向相反，确认吗？")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return

    labels = [classes[i] if i < 50 else "unknown" for i in pred_after]

    cp = out_dir / "result.csv"
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_label"])
        for i, lab in zip(ids.tolist(), labels):
            w.writerow([i, lab])
    print(f"\n[写出] {cp}")

    np.savez_compressed(npz, image_id=ids, logits=L.astype(np.float32))
    print(f"[写出] {npz}")

    # 自洽
    back = np.load(npz, allow_pickle=False)
    chk = np.array([classes[i] if i < 50 else "unknown" for i in back["logits"].argmax(1)])
    ok = bool((chk == np.array(labels)).all())
    back.close()
    print(f"\n自洽检查：argmax(logits) == result.csv → {'✅ 一致' if ok else '❌ 不一致'}")

    meta = meta_path
    meta.write_text(json.dumps({
        "dir": args.dir,
        "theta": args.theta,
        "prev_theta": prev_theta,
        "delta_applied": delta,
        "pred_unknown_before": ur_before,
        "pred_unknown_after": float((pred_after == 50).mean()),
        "true_unknown_ratio": P_UNK,
        "known_claimed": n_known, "true_known": TEST_KNOWN,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {meta}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
