# -*- coding: utf-8 -*-
"""生成 EduBench 提交物（classes.txt / result.csv / result.npz）。

规范来自教师示例包里的 README.md：
  result.csv   只有两列：表头 `image_id,predicted_label`
  result.npz   np.savez，键 `image_id`（Unicode 字符串）与 `logits`（N,51 float32）
               列序 = classes.txt 的行序，第 50 列 = unknown

打分**必须**走生产推理路径（`CarEnsemble.score_logits`），
不能在脚本里另写一套 —— 项目早期就因为"实验口径 ≠ 生产口径"差过 0.012。

用法：
    python tools/make_submission.py --limit 200      # 冒烟：只跑前 200 张
    python tools/make_submission.py                  # 全量 16540 张
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.inference import CarEnsemble  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "student_submit"


def load_image(p: Path) -> Image.Image:
    """读图并转 RGB。单张失败时回退成灰图（不让一张坏图毁掉整批）。"""
    try:
        with Image.open(p) as im:
            return im.convert("RGB").copy()
    except Exception as e:                                  # noqa: BLE001
        print(f"  [警告] 读图失败 {p.name}：{type(e).__name__}，用灰图占位")
        return Image.new("RGB", (416, 416), (128, 128, 128))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test_images"))
    ap.add_argument("--batch", type=int, default=8,
                    help="每批张数；多裁剪下一次前向是 batch×10 张，别开太大")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟用）")
    ap.add_argument("--workers", type=int, default=6, help="预取图片的线程数")
    ap.add_argument("--out", default=str(OUT_DIR))
    # ---- 速度 / 决策 ----
    ap.add_argument("--tta", default="hflip", choices=["hflip", "multicrop", "none"],
                    help="TTA 档位。hflip=2 视角（默认，比多裁剪快约 5 倍）；"
                         "multicrop=10 视角（精度 +0.0025，但慢 5 倍）")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="拒识偏置：unknown 列减去 θ。测试集 unknown 占比远高于验证集时，"
                         "取负值（如 -5.2）把决策推向「多说不知道」")
    ap.add_argument("--members", default=None,
                    help="逗号分隔的 checkpoint 目录名，覆盖冠军集成的成员。"
                         "用于评测微调后的模型，如 ood_r50_416,ood_r50_288_supcon")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 类别顺序：必须与 classes.txt 逐行一致（logits 的列序依赖它）----
    classes = [l.strip() for l in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cm = paths.load_class_map()
    model_ids = [cm["index_to_id"][str(i)] for i in range(50)]
    if model_ids != classes:
        raise SystemExit("模型 index_to_id 与 classes.txt 顺序不一致，无法生成 logits")
    print(f"类别顺序校验通过：50 类，unknown 为第 51 列（索引 50）")

    # ---- 测试图片 ----
    files = sorted((p for p in Path(args.test_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
                   key=lambda p: p.stem)
    if args.limit:
        files = files[: args.limit]
    print(f"测试图片：{len(files)} 张（{files[0].stem} … {files[-1].stem}）\n")

    # ---- 模型：默认与线上冠军配置完全一致，--members 可覆盖 ----
    cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    member_names = ([m.strip() for m in args.members.split(",") if m.strip()]
                    if args.members else cfg["members"])
    ckpts = [paths.CKPT_DIR / m / "best.pt" for m in member_names]
    missing = [c for c in ckpts if not c.exists()]
    if missing:
        raise SystemExit(f"以下 checkpoint 不存在：{[str(m) for m in missing]}")
    tau = float(cfg["best"]["tau"])
    thr = float(cfg["best"]["threshold"])
    print(f"集成：{cfg['tag'] if not args.members else ' + '.join(member_names)}")
    print(f"  成员 {len(ckpts)} 个：{member_names}")
    print(f"  τ={tau}  拒识阈值 t={thr}  TTA={args.tta}"
          + (f"  拒识偏置 θ={args.theta:+.2f}" if args.theta else ""))
    t0 = time.perf_counter()
    ens = CarEnsemble(ckpts, tta_mode=args.tta, amp=True,
                      prior_adjust_tau=tau, unknown_threshold=thr, device="auto")
    print(f"  模型加载完成 {time.perf_counter() - t0:.1f}s，设备 {ens.device}\n")

    # ---- 推理（边预取边算）----
    all_logits: list[np.ndarray] = []
    ids: list[str] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s in range(0, len(files), args.batch):
            chunk_paths = files[s : s + args.batch]
            imgs = list(pool.map(load_image, chunk_paths))
            logits = ens.score_logits(imgs, batch_size=args.batch)
            all_logits.append(logits.float().cpu().numpy())
            ids.extend(p.stem for p in chunk_paths)
            done = len(ids)
            if done % (args.batch * 10) == 0 or done == len(files):
                el = time.perf_counter() - t0
                eta = el / done * (len(files) - done)
                print(f"  {done:>6}/{len(files)}  已用 {el / 60:.1f} min  "
                      f"预计还需 {eta / 60:.1f} min", flush=True)

    L = np.concatenate(all_logits).astype(np.float32)
    print(f"\n推理完成：{L.shape}  用时 {(time.perf_counter() - t0) / 60:.1f} min")
    if not np.isfinite(L).all():
        raise SystemExit("logits 里有 NaN/Inf，不能提交")

    if args.theta:
        # 拒识偏置：只在 unknown 列（第 50 列）上做平移。
        # 决策只取 argmax，所以这等价于「unknown 的分数要比最强已知类高出 θ，
        # 才判为未知」—— 正是 calibrate_rejection_v2.py 标定出的 margin 规则。
        # 平移后的 L 同时写进 npz，保证 argmax(npz.logits) == result.csv（下面自洽检查会验）。
        L = L.copy()
        L[:, 50] -= np.float32(args.theta)
        print(f"已施加拒识偏置 θ={args.theta:+.2f}（unknown 列平移）")

    # ---- 预测标签：argmax ----
    arg = L.argmax(axis=1)
    labels = [classes[i] if i < 50 else "unknown" for i in arg]
    n_unk = sum(1 for x in labels if x == "unknown")
    print(f"预测分布：未知 {n_unk} 张（{n_unk / len(labels):.1%}），"
          f"已知 {len(labels) - n_unk} 张")

    # ---- 写 result.csv（**只有两列**）----
    csv_path = out_dir / "result.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_label"])
        for i, lab in zip(ids, labels):
            w.writerow([i, lab])
    print(f"[写出] {csv_path}  ({len(ids)} 行 + 表头)")

    # ---- 写 result.npz ----
    npz_path = out_dir / "result.npz"
    np.savez_compressed(npz_path,
                        image_id=np.array(ids, dtype=f"<U{max(6, max(len(x) for x in ids))}"),
                        logits=L)
    print(f"[写出] {npz_path}  ({npz_path.stat().st_size / 1024**2:.2f} MiB)")

    # ---- 复制 classes.txt ----
    (out_dir / "classes.txt").write_bytes((ROOT / "classes.txt").read_bytes())
    print(f"[写出] {out_dir / 'classes.txt'}")

    # ---- 自洽检查：argmax(logits) 必须等于写进 csv 的标签 ----
    back = np.load(npz_path, allow_pickle=False)
    chk = np.array([classes[i] if i < 50 else "unknown"
                    for i in back["logits"].argmax(axis=1)])
    ok = bool((chk == np.array(labels)).all())
    print(f"\n自洽检查：argmax(npz.logits) == result.csv 标签 → {'✅ 一致' if ok else '❌ 不一致'}")
    if not ok:
        sys.exit(1)
    print("\n下一步：python tools/verify_submission_edubench.py  然后打包")


if __name__ == "__main__":
    main()
