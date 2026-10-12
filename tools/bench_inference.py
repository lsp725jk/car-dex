"""推理速度基准：量出 TTA 档位 / batch / 预取线程对吞吐的影响。

用途：为「推理太慢」提供**实测**依据，而不是拍脑袋估算。

跑法：
    python tools/bench_inference.py --n 96
    python tools/bench_inference.py --n 96 --modes hflip multicrop --batches 8 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cardex import paths  # noqa: E402
from cardex.inference import CarEnsemble  # noqa: E402


def load_image(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return im.convert("RGB")


def bench(ens: CarEnsemble, files: list[Path], batch: int, workers: int,
          warmup: int = 8, reps: int = 2) -> dict:
    """跑 reps 遍，取最快的一遍（避开首次 CUDA kernel 编译和缓存冷启动）。"""
    best = None
    for r in range(reps):
        t0 = time.perf_counter()
        n = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s in range(0, len(files), batch):
                chunk = files[s : s + batch]
                imgs = list(pool.map(load_image, chunk))
                _ = ens.score_logits(imgs, batch_size=batch)
                n += len(chunk)
        el = time.perf_counter() - t0
        # 首遍含 warmup，从计时里扣掉（粗略但一致）
        if r == 0:
            continue
        best = el if best is None else min(best, el)
    if best is None:
        best = el
    return {"images": n, "sec": best, "ips": n / best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test_images"))
    ap.add_argument("--n", type=int, default=96, help="基准用的图片数")
    ap.add_argument("--modes", nargs="+", default=["none", "hflip", "hflip+multicrop"])
    ap.add_argument("--batches", nargs="+", type=int, default=[8])
    ap.add_argument("--workers", nargs="+", type=int, default=[6])
    args = ap.parse_args()

    files = sorted(Path(args.test_dir).glob("*.jpg"))[: args.n]
    if not files:
        raise SystemExit(f"没找到图片：{args.test_dir}")
    print(f"基准样本：{len(files)} 张\n")

    cfg = json.loads((paths.REPORT_DIR / "final_model_choice.json").read_text(encoding="utf-8"))["champion"]
    ckpts = [paths.CKPT_DIR / m / "best.pt" for m in cfg["members"]]
    tau = float(cfg["best"]["tau"])
    thr = float(cfg["best"]["threshold"])
    print(f"冠军：{cfg['tag']}（{len(ckpts)} 个成员）\n")

    rows = []
    for mode in args.modes:
        ens = CarEnsemble(ckpts, tta_mode=mode, amp=True,
                          prior_adjust_tau=tau, unknown_threshold=thr, device="auto")
        for bs in args.batches:
            for w in args.workers:
                r = bench(ens, files, bs, w)
                # 全测试集 16540 张的推算耗时
                full_min = 16540 / r["ips"] / 60
                rows.append({"mode": mode, "batch": bs, "workers": w,
                             "ips": r["ips"], "full_min": full_min})
                print(f"  TTA={mode:<16} batch={bs:<3} workers={w}  "
                      f"{r['ips']:6.1f} 张/秒   全测试集≈{full_min:5.1f} min")
        del ens
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    print("=" * 66)
    base = next((x for x in rows if x["mode"] == "hflip+multicrop"), rows[0])
    print(f"{'档位':<18}{'batch':<7}{'张/秒':<10}{'全测试集':<12}{'相对加速'}")
    for r in rows:
        print(f"{r['mode']:<18}{r['batch']:<7}{r['ips']:<10.1f}"
              f"{r['full_min']:<12.1f}{base['ips'] / r['ips']:.1f}×")

    out = paths.REPORT_DIR / "inference_bench.json"
    out.write_text(json.dumps({"rows": rows, "n_images": len(files)},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
