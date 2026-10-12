# -*- coding: utf-8 -*-
"""任务一 - 交付物总览核验

一次性打印任务一全部交付物的当前状态与关键指标，便于提交前快速确认。

用法：
    python src/final_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardex import paths  # noqa: E402

BOM = b"\xef\xbb\xbf"


def main() -> None:
    ok = True
    line = "=" * 78
    print(line)
    print("任务一 交付物总览核验")
    print(line)

    # ---------------- 1. classes.txt ----------------
    raw = paths.CLASSES_TXT.read_bytes()
    ids = raw.decode("utf-8").split()
    checks = {
        "恰好 50 行": len(ids) == 50,
        "每行四位数字": all(len(x) == 4 and x.isdigit() for x in ids),
        "升序": ids == sorted(ids),
        "无重复": len(set(ids)) == 50,
        "UTF-8 无 BOM": not raw.startswith(BOM),
        "不含 unknown": "unknown" not in raw.decode("utf-8").lower(),
    }
    print("\n[1] classes.txt")
    for k, v in checks.items():
        print(f"    {'[OK]  ' if v else '[FAIL]'}{k}")
        ok &= v

    # ---------------- 2. 数据目录 ----------------
    print("\n[2] 数据目录（train / val 各 51 个类别目录）")
    for split in ("train", "val"):
        d = paths.DATA_DIR / split
        dirs = sorted(p.name for p in d.iterdir() if p.is_dir())
        n_img = sum(1 for f in d.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        good = len(dirs) == 51 and "unknown" in dirs
        ok &= good
        print(f"    {'[OK]  ' if good else '[FAIL]'}{split}/: {len(dirs)} 个目录、{n_img} 张图片、"
              f"{'含' if 'unknown' in dirs else '缺'} unknown")

    # ---------------- 3. 实验结果 ----------------
    print("\n[3] 实验与模型")
    runs = []
    for d in sorted(paths.RUN_DIR.iterdir()):
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if s.get("best", {}).get("macro_f1", 0) < 0.5:
            continue
        calib = d / "calibration.json"
        best_cal = ""
        if calib.exists():
            c = json.loads(calib.read_text(encoding="utf-8"))
            best_cal = f"  标定后 {c['best_joint']['macro_f1']:.4f}"
        ck = len(list((paths.CKPT_DIR / d.name).glob("*.pt")))
        print(f"    {d.name:<28} 未标定 {s['best']['macro_f1']:.4f}{best_cal}  "
              f"checkpoint {ck} 个")
        runs.append(d.name)
    if not runs:
        ok = False
        print("    [FAIL] 没有找到任何已完成的实验")

    # ---------------- 4. 最终模型 ----------------
    print("\n[4] 最终模型选择")
    fc_path = paths.REPORT_DIR / "final_model_choice.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text(encoding="utf-8"))
        champ = fc["champion"]
        print(f"    冠军配置 : {champ['tag']}")
        print(f"    成员     : {champ['members']}")
        print(f"    决策标定 : tau={champ['best']['tau']:+.2f}  拒识阈值={champ['best']['threshold']:.2f}")
        print(f"    验证集 Macro-F1 = {champ['best']['macro_f1']:.4f}")
        print("    候选排名 :")
        for i, r in enumerate(fc["candidates"], 1):
            print(f"      {i}. {r['tag']:<46} {r['best']['macro_f1']:.4f}")
    else:
        ok = False
        print("    [FAIL] 未找到 final_model_choice.json")

    # ---------------- 5. predictions.csv ----------------
    print("\n[5] predictions.csv")
    pv = paths.PROJECT_ROOT / "predictions.csv"
    if pv.exists():
        import csv as _csv

        rows = list(_csv.DictReader(pv.open(encoding="utf-8")))
        head = pv.read_text(encoding="utf-8").splitlines()[0]
        meta = paths.PROJECT_ROOT / "predictions.meta.json"
        print(f"    表头     : {head}")
        print(f"    记录数   : {len(rows)}")
        print(f"    预测为 unknown: {sum(1 for r in rows if r['predicted_label'] == 'unknown')}"
              f" ({100 * sum(1 for r in rows if r['predicted_label'] == 'unknown') / len(rows):.1f}%)")
        good = head == "image_id,predicted_label,confidence"
        ok &= good
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            print(f"    生成配置 : {m.get('champion')}  tau={m.get('prior_adjust_tau')}  "
                  f"t={m.get('unknown_threshold')}  TTA={m.get('tta_hflip')}")
    else:
        ok = False
        print("    [FAIL] predictions.csv 不存在")

    score = paths.REPORT_DIR / "predictions_score.json"
    if score.exists():
        s = json.loads(score.read_text(encoding="utf-8"))
        print(f"    独立核算 : Macro-F1={s['macro_f1']}  Accuracy={s['accuracy']}  "
              f"unknown检出={s['unknown_detection_rate']}  已知误拒={s['known_false_reject_rate']}")

    # ---------------- 6. 文档 ----------------
    print("\n[6] 报告与记录")
    docs = [
        paths.PROJECT_ROOT / "README.md",
        paths.PROJECT_ROOT / "docs" / "任务一_数据分析与视觉模型训练.md",
        paths.REPORT_DIR / "selection_report.txt",
        paths.REPORT_DIR / "data_analysis.txt",
        paths.REPORT_DIR / "experiment_comparison.md",
        paths.REPORT_DIR / "final_model_choice.md",
    ]
    for d in docs:
        exists = d.exists()
        ok &= exists
        size = f"{d.stat().st_size / 1024:.1f} KB" if exists else "-"
        print(f"    {'[OK]  ' if exists else '[FAIL]'}{d.relative_to(paths.PROJECT_ROOT)}  ({size})")

    print("\n" + line)
    print("核验结果：" + ("全部通过 ✅" if ok else "存在问题 ❌"))
    print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
