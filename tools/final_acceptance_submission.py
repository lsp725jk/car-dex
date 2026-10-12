# -*- coding: utf-8 -*-
"""终验：把提交 zip 解压到临时目录，完全按教师 README 规范逐条校验。

这一步是"模拟教师端"——不看源目录，只看 zip 里的内容。
能查出一类容易漏的问题：**打包时的路径结构错了**（比如多套了一层目录、
用了绝对路径、漏了文件），而源目录里看起来一切正常。
"""
from __future__ import annotations

import csv
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ZIP = Path(r"C:\Users\y\Downloads\student_submit.zip")
ok_n, bad_n = 0, 0


def check(ok: bool, msg: str) -> None:
    global ok_n, bad_n
    if ok:
        ok_n += 1
    else:
        bad_n += 1
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {msg}")


def main() -> None:
    print("=" * 78)
    print(f"终验：{ZIP}")
    print("=" * 78)
    check(ZIP.exists() and ZIP.stat().st_size > 0,
          f"文件存在，{ZIP.stat().st_size / 1024**2:.2f} MiB")

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(ZIP) as z:
            check(z.testzip() is None, "ZIP CRC 全部通过（未损坏）")
            names = z.namelist()
            check(all(n.startswith("student_submit/") for n in names),
                  f"所有条目都在 student_submit/ 下（{len(names)} 个）")
            JUNK = ("__MACOSX", ".DS_Store")
            check(not any(j in n for n in names for j in JUNK),
                  "不含 __MACOSX / .DS_Store 等 macOS 垃圾")
            z.extractall(td)

        d = Path(td) / "student_submit"
        check(d.is_dir(), "解压后 student_submit/ 目录存在")

        # ---- classes.txt ----
        print("\n--- classes.txt ---")
        cp = d / "classes.txt"
        check(cp.exists(), "文件存在")
        raw = cp.read_bytes()
        check(len(raw) <= 4096, f"大小 {len(raw)} B ≤ 4 KiB")
        check(raw[:3] != b"\xef\xbb\xbf", "UTF-8 无 BOM")
        cl = raw.decode("utf-8").splitlines()
        check(len(cl) == 50, f"恰好 50 行（实际 {len(cl)}）")
        check(all(l.isdigit() and len(l) == 4 for l in cl), "每行 4 位数字")
        check(cl == sorted(set(cl)), "升序且无重复")
        check("unknown" not in raw.decode("utf-8").lower(), "不含 unknown")

        # ---- result.csv ----
        print("\n--- result.csv ---")
        rp = d / "result.csv"
        check(rp.exists(), "文件存在")
        raws = rp.read_bytes()
        check(raws[:3] != b"\xef\xbb\xbf", "UTF-8 无 BOM")
        rows = list(csv.reader(raws.decode("utf-8").splitlines()))
        check(rows[0] == ["image_id", "predicted_label"],
              f"表头 = ['image_id','predicted_label']（实际 {rows[0]}）")
        body = rows[1:]
        check(all(len(r) == 2 for r in body), "每行恰好 2 列（无 confidence）")
        check(all(r[0] and r[1] for r in body), "无空字段")
        legal = set(cl) | {"unknown"}
        bad = [r for r in body if r[1] not in legal]
        check(not bad, f"标签全部合法（非法 {len(bad)}）")
        check(all(r[0].isdigit() and len(r[0]) == 6 for r in body), "image_id 都是 6 位数字")
        got = [r[0] for r in body]
        check(len(got) == len(set(got)), "image_id 无重复")
        print(f"         数据行 {len(body)} 行 + 表头 1 行")

        # ---- result.npz ----
        print("\n--- result.npz ---")
        np_ = d / "result.npz"
        check(np_.exists(), "文件存在")
        check(np_.stat().st_size <= 20 * 1024 * 1024,
              f"{np_.stat().st_size / 1024**2:.2f} MiB ≤ 20 MiB")
        # ⚠️ np.load 返回的是**惰性对象**，会一直握着文件句柄。
        # Windows 下句柄没关就删不掉文件，临时目录清理会报 WinError 32。
        # 所以必须用 with（或显式 close）包起来。
        zz = np.load(np_, allow_pickle=False)
        try:
            keys = sorted(zz.files)
            check(keys == ["image_id", "logits"], f"键 = {keys}")
            ids, lg = zz["image_id"], zz["logits"]
            check(ids.dtype.kind == "U", f"image_id dtype={ids.dtype}（Unicode）")
            check(ids.dtype.itemsize // 4 <= 128, f"字符串容量 {ids.dtype.itemsize // 4} ≤ 128")
            check(lg.dtype == np.float32, f"logits dtype={lg.dtype}")
            check(lg.ndim == 2 and lg.shape[1] == 51, f"logits 形状 {lg.shape}")
            check(ids.shape[0] == lg.shape[0] == len(body),
                  f"行数三者一致：npz {ids.shape[0]} / csv {len(body)}")
            check(bool(np.isfinite(lg).all()), "logits 全有限（无 NaN/Inf）")
            check(set(ids.tolist()) == set(got), "npz 的 image_id 与 csv 完全一致")
            check(ids.nbytes + lg.nbytes <= 64 * 1024 * 1024,
                  f"数组未压缩 {((ids.nbytes + lg.nbytes) / 1024**2):.1f} MiB ≤ 64 MiB")

            # 列序 ↔ classes.txt
            arg = lg.argmax(1)
            expect = np.array([cl[i] if i < 50 else "unknown" for i in arg])
            check(bool((expect == np.array([r[1] for r in body])).all()),
                  "argmax(logits) 与 result.csv 标签逐行一致")
        finally:
            zz.close()

    print("\n" + "=" * 78)
    print(f"终验通过 {ok_n} 项，失败 {bad_n} 项")
    print("=" * 78)
    sys.exit(0 if bad_n == 0 else 1)


if __name__ == "__main__":
    main()
