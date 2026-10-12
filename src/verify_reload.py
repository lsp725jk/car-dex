# -*- coding: utf-8 -*-
"""任务一 - 校验"模型保存与重新加载"（任务书 4.2 节第 1 条要求）

任务书要求实现"训练、验证、模型保存和重新加载"。本脚本独立地验证这一点：

  1. 从 checkpoint 重新构建模型并加载权重（不依赖任何训练时的内存状态）；
  2. 对固定的一批验证图片做推理，把「图片名 → 预测标签 ID / 置信度」写成 JSON；
  3. 由外部脚本在**两个独立进程**中各跑一次，对比两份 JSON 是否完全一致。

用法：
    python src/verify_reload.py --checkpoint outputs/checkpoints/<run>/best.pt \
           --n 300 --out outputs/runs/<run>/reload_check_1.json
    # 换个进程再跑一次，然后：
    python src/verify_reload.py --compare a.json b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardex import paths  # noqa: E402
from cardex.data import CarDataset, build_transform  # noqa: E402
from cardex.engine import load_model_from_checkpoint  # noqa: E402
from cardex.inference import CarClassifier  # noqa: E402
from cardex.utils import configure_threads, get_logger  # noqa: E402


def run(args) -> None:
    configure_threads(args.threads)
    logger = get_logger("verify_reload")
    ckpt_path = Path(args.checkpoint)

    model, ckpt = load_model_from_checkpoint(ckpt_path)
    logger.info("已重新加载 checkpoint：%s", ckpt_path)
    logger.info("  骨干=%s  输入尺寸=%s  类别数=%s  验证 Macro-F1=%s",
                ckpt["arch"], ckpt.get("image_size"), ckpt.get("num_classes"),
                ckpt.get("val_macro_f1"))
    logger.info("  预处理参数=%s", json.dumps(ckpt.get("preprocess", {}), ensure_ascii=False))

    # 用 CarDataset 保证与训练时完全相同的样本顺序
    ds = CarDataset(paths.DATA_DIR / args.split,
                    transform=build_transform("eval", int(ckpt["image_size"]),
                                              ckpt.get("preprocess")))
    n = min(args.n, len(ds))
    records = []
    clf = CarClassifier(ckpt_path, tta_hflip=args.tta_hflip)
    step = max(1, len(ds) // n)
    picked = ds.samples[::step][:n]
    for path, _ in picked:
        res = clf.predict_path(path, top_k=3)
        records.append({
            "image_id": Path(path).name,
            "label_id": res["label_id"],
            "index": res["index"],
            "confidence": round(res["confidence"], 6),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(ckpt_path), "n": len(records),
                               "records": records}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    logger.info("已推理 %d 张，写出 %s", len(records), out)


def compare(a: Path, b: Path) -> int:
    da = json.loads(Path(a).read_text(encoding="utf-8"))
    db = json.loads(Path(b).read_text(encoding="utf-8"))
    ra, rb = da["records"], db["records"]
    print(f"A: {a}  ({len(ra)} 条)")
    print(f"B: {b}  ({len(rb)} 条)")
    if len(ra) != len(rb):
        print("[FAIL] 记录数不一致")
        return 1
    diff = [x for x, y in zip(ra, rb) if x != y]
    same = len(ra) - len(diff)
    print(f"完全一致的记录：{same}/{len(ra)}")
    if diff:
        print("[FAIL] 两次独立加载的推理结果不一致，前 5 条差异：")
        for d in diff[:5]:
            print("   ", d)
        return 1
    print("[OK] 两次独立进程重新加载 checkpoint 后的推理结果逐条一致 —— "
          "模型保存与重新加载正确、且可复现。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta-hflip", action="store_true")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(Path(args.compare[0]), Path(args.compare[1])))
    if not args.checkpoint or not args.out:
        raise SystemExit("需要 --checkpoint 与 --out（或用 --compare A B）")
    run(args)


if __name__ == "__main__":
    main()
