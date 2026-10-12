# -*- coding: utf-8 -*-
"""分析测试集预测分布，判断 74% unknown 是否合理。

三个角度：
  1. 已知类的覆盖度 —— 50 个类是不是都有被预测到（有没有整类"消失"）
  2. 置信度分布 —— unknown 是"果断拒绝"还是"拿不准"
  3. 空间分布 —— 测试集是不是按类别排序的（那样 known/unknown 会呈块状）
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
classes = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
z = np.load(ROOT / "student_submit" / "result.npz", allow_pickle=False)
L, ids = z["logits"], z["image_id"]
N = len(ids)

e = np.exp(L - L.max(1, keepdims=True))
P = e / e.sum(1, keepdims=True)
arg = L.argmax(1)
unk = arg == 50

print("=" * 76)
print(f"测试集 {N} 张：已知 {int((~unk).sum())}（{(~unk).mean():.1%}） / "
      f"unknown {int(unk.sum())}（{unk.mean():.1%}）")
print("=" * 76)

print("\n[1] 已知类覆盖度")
cnt = collections.Counter(int(i) for i in arg[~unk])
hit = len(cnt)
print(f"  被预测到的类别数：{hit} / 50")
missing = [classes[i] for i in range(50) if i not in cnt]
if missing:
    print(f"  ⚠ 从未被预测的类（{len(missing)} 个）：{missing[:10]}")
else:
    print("  ✅ 50 个类全部至少被预测到一次")
top = cnt.most_common(8)
print("  预测最多的类：")
for i, c in top:
    print(f"    {classes[i]}  {c} 张（占已知的 {c / (~unk).sum():.1%}）")
rare = sorted(cnt.items(), key=lambda x: x[1])[:5]
print("  预测最少的类：")
for i, c in rare:
    print(f"    {classes[i]}  {c} 张")

print("\n[2] 置信度（softmax 后的最大概率）")
for label, mask in (("已知预测", ~unk), ("unknown 预测", unk)):
    conf = P[mask].max(1) if mask.any() else np.array([0.0])
    print(f"  {label}：中位 {np.median(conf):.3f}  "
          f"| >0.8 的占 {(conf > 0.8).mean():.1%}  "
          f"| <0.3 的占 {(conf < 0.3).mean():.1%}")
print("  → unknown 的置信度中位数越高，说明模型越「果断地否定」，而不是含糊")

print("\n[3] 空间分布：前 8 段的 known 比例（看测试集是否按类排序）")
seg = 8
size = N // seg
rates = [float((~unk)[i * size:(i + 1) * size].mean()) for i in range(seg)]
for i, r in enumerate(rates):
    bar = "█" * int(r * 40)
    print(f"    段{i + 1} [{i * size:>6}-{(i + 1) * size:>6})  {r:5.1%}  {bar}")
spread = max(rates) - min(rates)
print(f"  段间差异 {spread:.1%} —— {'较大，可能按类别排序' if spread > 0.25 else '不大，说明大致均匀混排'}")

print("\n[4] unknown 的第二候选（模型认为「最像哪个已知类」）")
second = np.argsort(-P, axis=1)[:, 0]
second = np.where(second == 50, np.argsort(-P, axis=1)[:, 1], second)
sc = collections.Counter(int(i) for i in second[unk])
print("  unknown 样本最接近的已知类 Top-5：")
for i, c in sc.most_common(5):
    print(f"    {classes[i]}  {c} 张")

print("\n[5] 提交物规模")
print(f"  result.npz  {z['logits'].nbytes / 1024**2:.2f} MiB（未压缩）")
print(f"  列序：0..49 = classes.txt 第 1..50 行，50 = unknown  →  argmax 即预测")
