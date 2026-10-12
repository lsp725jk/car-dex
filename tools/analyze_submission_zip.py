# -*- coding: utf-8 -*-
"""对一个 student_submit.zip 做**内容剖析**（不是合规检查，是"里面到底装了什么"）。

与另外两个脚本的分工：
  · verify_submission_edubench.py   41 项 → 格式对不对
  · final_acceptance_submission.py  32 项 → 从 zip 解压后独立复核
  · 本脚本                                → 内容合不合理（分布、边际、一致性）

用法：
    python tools/analyze_submission_zip.py <zip 路径>
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", help="student_submit.zip 路径")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        raise SystemExit(f"找不到 {zp}")
    print(f"文件：{zp}")
    print(f"大小：{zp.stat().st_size:,} B ({zp.stat().st_size / 1024**2:.2f} MiB)")

    zf = zipfile.ZipFile(zp)
    names = zf.namelist()
    hr("① 包内结构")
    for n in names:
        info = zf.getinfo(n)
        print(f"  {n:<34} {info.file_size:>10,} B   压缩后 {info.compress_size:>9,} B")
    junk = [n for n in names if "__MACOSX" in n or n.endswith(".DS_Store") or "/._" in n]
    print(f"\n  macOS 垃圾文件：{len(junk)} 个 {'✅ 干净' if not junk else '❌ ' + str(junk[:5])}")
    print(f"  CRC：{'✅ 全部可读' if zf.testzip() is None else '❌ ' + str(zf.testzip())}")

    # ── classes.txt ──
    hr("② classes.txt")
    ct = zf.read("student_submit/classes.txt")
    print(f"  字节数 {len(ct)}  BOM={'有 ❌' if ct[:3] == b'\xef\xbb\xbf' else '无 ✅'}")
    lines = ct.decode("utf-8").splitlines()
    classes = [x.strip() for x in lines if x.strip()]
    print(f"  行数 {len(classes)}（空行 {len(lines) - len(classes)}）")
    print(f"  前 3：{classes[:3]}   后 3：{classes[-3:]}")
    print(f"  全部 4 位数字：{'✅' if all(len(c) == 4 and c.isdigit() for c in classes) else '❌'}")
    print(f"  严格升序：{'✅' if classes == sorted(classes) else '❌'}")
    print(f"  含 unknown：{'❌ 不该有' if 'unknown' in classes else '✅ 无'}")

    # ── result.csv ──
    hr("③ result.csv")
    raw = zf.read("student_submit/result.csv")
    print(f"  字节数 {len(raw):,}  BOM={'有 ❌' if raw[:3] == b'\xef\xbb\xbf' else '无 ✅'}")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    header, data = rows[0], rows[1:]
    print(f"  表头 {header}  {'✅ 恰好两列' if header == ['image_id', 'predicted_label'] else '❌'}")
    widths = Counter(len(r) for r in data)
    print(f"  每行列数分布 {dict(widths)}  {'✅' if set(widths) == {2} else '❌'}")
    ids = [r[0] for r in data]
    labs = [r[1] for r in data]
    print(f"  数据行 {len(data):,}")
    print(f"  image_id 唯一：{'✅' if len(set(ids)) == len(ids) else '❌'}"
          f"   6 位定长：{'✅' if all(len(x) == 6 and x.isdigit() for x in ids) else '❌'}")
    print(f"  排序：{'按文件名升序 ✅' if ids == sorted(ids) else '⚠️ 非升序（不违规，但留意）'}")
    legal = set(classes) | {"unknown"}
    bad = sorted({l for l in labs if l not in legal})
    print(f"  非法标签：{len(bad)} {'✅' if not bad else '❌ ' + str(bad[:10])}")

    # ── result.npz ──
    hr("④ result.npz")
    zz = zipfile.ZipFile(io.BytesIO(zf.read("student_submit/result.npz")))
    print(f"  内部成员：{zz.namelist()}")
    nz = np.load(io.BytesIO(zf.read("student_submit/result.npz")), allow_pickle=False)
    nid, L = nz["image_id"], nz["logits"]
    print(f"  image_id  dtype={nid.dtype}  shape={nid.shape}")
    print(f"  logits    dtype={L.dtype}   shape={L.shape}")
    print(f"  有限性：{'✅ 无 NaN/Inf' if np.isfinite(L).all() else '❌ 有 NaN/Inf'}")
    print(f"  取值范围 [{L.min():.4f}, {L.max():.4f}]   均值 {L.mean():.4f}")
    # 每一行做一个 softmax 看是否归一化（log_softmax 的口径）
    lse = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
    print(f"  行 logsumexp：均值 {lse.mean():.4f}  标准差 {lse.std():.4f}"
          f"   {'≈0 → 是 log-softmax 口径' if abs(lse.mean()) < 1e-2 else '≠0 → 已被平移过（正常，见下）'}")
    nz.close()

    # ── 一致性 ──
    hr("⑤ 三文件互相一致")
    arg = L.argmax(1)
    from_npz = [classes[i] if i < 50 else "unknown" for i in arg]
    same_lab = from_npz == labs
    print(f"  argmax(npz.logits) == csv 标签：{'✅ 逐行一致' if same_lab else f'❌ {sum(1 for a,b in zip(from_npz,labs) if a!=b)} 行不一致'}")
    print(f"  npz.image_id == csv.image_id：{'✅ 完全一致' if list(nid) == ids else '❌'}")
    print(f"  csv 行数 == npz 行数：{'✅' if len(ids) == L.shape[0] else '❌'}")

    # ── 预测分布 ──
    hr("⑥ 预测分布")
    cnt = Counter(labs)
    n_unk = cnt.get("unknown", 0)
    n_tot = len(labs)
    print(f"  判未知 {n_unk:,} 张（{n_unk / n_tot:.2%}）   判已知 {n_tot - n_unk:,} 张（{(n_tot - n_unk) / n_tot:.2%}）")
    known_cnt = {k: v for k, v in cnt.items() if k != "unknown"}
    print(f"  已知类覆盖：{len(known_cnt)} / 50 "
          f"{'✅ 全覆盖' if len(known_cnt) == 50 else '❌ 缺 ' + str(sorted(set(classes) - set(known_cnt)))}")
    vals = np.array([known_cnt.get(c, 0) for c in classes])
    print(f"  每类预测数：均值 {vals.mean():.1f}  中位 {np.median(vals):.0f}  "
          f"最小 {vals.min()}  最大 {vals.max()}")
    top = sorted(known_cnt.items(), key=lambda kv: -kv[1])[:8]
    bot = sorted(known_cnt.items(), key=lambda kv: kv[1])[:8]
    print(f"  最多的 8 个：{', '.join(f'{k}×{v}' for k, v in top)}")
    print(f"  最少的 8 个：{', '.join(f'{k}×{v}' for k, v in bot)}")

    # ── 边际分析 ──
    hr("⑦ 判决边际（模型有多「勉强」）")
    known_max = L[:, :50].max(1)
    margin = L[:, 50] - known_max       # > 0 判未知
    is_unk = arg == 50
    print(f"  判未知的 margin：均值 {margin[is_unk].mean():+.3f}  "
          f"最小 {margin[is_unk].min():+.3f}  最大 {margin[is_unk].max():+.3f}")
    if (~is_unk).any():
        print(f"  判已知的 margin：均值 {margin[~is_unk].mean():+.3f}  "
              f"最小 {margin[~is_unk].min():+.3f}  最大 {margin[~is_unk].max():+.3f}")
    # 判已知但 margin 离边界很近的 = 最可能其实是未知
    k_idx = np.where(~is_unk)[0]
    order = k_idx[np.argsort(-margin[k_idx])]
    print(f"\n  判已知里 margin 最高的 10 张（最可疑，margin 越接近边界越像未知）：")
    for i in order[:10]:
        print(f"    {ids[i]}  {labs[i]}  margin={margin[i]:+.3f}  "
              f"unk={L[i,50]:+.2f}  best={known_max[i]:+.2f}")
    # 判未知里 margin 最负的 = 最可能其实是已知
    u_idx = np.where(is_unk)[0]
    order2 = u_idx[np.argsort(margin[u_idx])]
    print(f"\n  判未知里 margin 最低的 10 张（最可能被误拒）：")
    for i in order2[:10]:
        near = classes[int(L[i, :50].argmax())]
        print(f"    {ids[i]}  margin={margin[i]:+.3f}  "
              f"unk={L[i, 50]:+.2f}  best={known_max[i]:+.2f}  最像={near}")

    out = ROOT / "outputs" / "reports" / "submission_content_analysis.json"
    out.write_text(json.dumps({
        "zip": str(zp), "size_bytes": zp.stat().st_size,
        "entries": names, "n_rows": n_tot,
        "n_unknown": n_unk, "unknown_ratio": n_unk / n_tot,
        "classes_covered": len(known_cnt),
        "per_class": {c: int(known_cnt.get(c, 0)) for c in classes},
        "logits_shape": list(L.shape), "logits_dtype": str(L.dtype),
        "logsumexp_mean": float(lse.mean()),
        "consistent": bool(same_lab and list(nid) == ids),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
