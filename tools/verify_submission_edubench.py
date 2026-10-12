# -*- coding: utf-8 -*-
"""核对本组提交物是否满足教师 EduBench 规范（README.md 中的四条）。

检查项：
  classes.txt  恰好 50 行 / 4 位数字 / 升序无重复 / UTF-8 无 BOM / <=4KiB / 无 unknown
  模型索引 ↔ classes.txt 行的对应关系（logits 的列序依赖它）
  result.csv  两列 / 表头固定 / image_id 覆盖测试集且不重不漏 / 标签合法
  result.npz  image_id(N,) Unicode / logits(N,51) float32 / 有限数 / 键名唯一
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "src")

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ROOT / "classes.txt"
TEST_DIR = ROOT / "data" / "test_images"

OK, BAD, WARN = "[OK]", "[FAIL]", "[WARN]"
results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(f"  {OK if ok else BAD}   {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN} {msg}")


def check_classes() -> list[str]:
    print("\n[1] classes.txt")
    raw = CLASSES.read_bytes()
    check(len(raw) <= 4096, f"文件大小 {len(raw)} B ≤ 4 KiB")
    check(raw[:3] != b"\xef\xbb\xbf", "UTF-8 无 BOM")
    txt = raw.decode("utf-8")
    lines = txt.splitlines()
    check(len(lines) == 50, f"恰好 50 行（实际 {len(lines)}）")
    check(all(l.isdigit() and len(l) == 4 for l in lines), "每行都是 4 位数字（含前导零）")
    check(lines == sorted(set(lines)), "升序排列且无重复")
    check("unknown" not in txt.lower(), "不含 unknown")
    check(not any(l != l.strip() or not l for l in lines), "无空行/首尾空格")
    return lines


def check_index_mapping(lines: list[str]) -> None:
    print("\n[2] 模型索引 ↔ classes.txt 行序（result.npz 的列序依赖）")
    from cardex import paths
    cm = paths.load_class_map()
    ids = [cm["index_to_id"][str(i)] for i in range(50)]
    check(ids == sorted(ids), "模型 index_to_id 本身升序")
    check(ids == lines, "与 classes.txt 逐行完全一致（第 i 行=模型索引 i）")
    if ids != lines:
        for i, (a, b) in enumerate(zip(ids, lines)):
            if a != b:
                warn(f"第 {i + 1} 行不一致：模型={a}  文件={b}")
                break
    check(int(cm["unknown_index"]) == 50, f"unknown 索引 = {cm['unknown_index']}（应为 50）")


def check_csv(lines: list[str], expected_ids: list[str] | None) -> None:
    print("\n[3] result.csv")
    p = ROOT / "student_submit" / "result.csv"
    if not p.exists():
        warn("尚未生成，跳过")
        return
    raw = p.read_bytes()
    check(raw[:3] != b"\xef\xbb\xbf", "UTF-8 无 BOM")
    txt = raw.decode("utf-8")
    rows = txt.splitlines()
    check(rows[0] == "image_id,predicted_label", f"表头固定为 image_id,predicted_label（实际 {rows[0]!r}）")
    check(all(r.count(",") == 1 for r in rows), "每行恰好 2 列（无 confidence）")
    body = [r.split(",") for r in rows[1:]]
    check(all(len(x) == 2 and x[0] and x[1] for x in body), "无空字段")
    check(all(x[0] == x[0].strip() and x[1] == x[1].strip() for x in body), "字段无首尾空格")
    legal = set(lines) | {"unknown"}
    bad = [x for x in body if x[1] not in legal]
    check(not bad, f"标签都是 classes.txt 里的 ID 或小写 unknown（非法 {len(bad)} 条）")
    if bad:
        warn(f"示例非法标签：{bad[:3]}")
    check(all(len(x[0]) == 6 and x[0].isdigit() for x in body), "image_id 都是 6 位数字")
    if expected_ids is not None:
        got = [x[0] for x in body]
        check(len(got) == len(set(got)), "image_id 无重复")
        check(set(got) == set(expected_ids),
              f"image_id 集合与测试集完全一致（csv {len(set(got))} / 测试集 {len(set(expected_ids))}）")
        missing = set(expected_ids) - set(got)
        extra = set(got) - set(expected_ids)
        if missing:
            warn(f"缺失 {len(missing)} 张，例如 {sorted(missing)[:3]}")
        if extra:
            warn(f"多出 {len(extra)} 张，例如 {sorted(extra)[:3]}")


def check_npz(lines: list[str], expected_ids: list[str] | None) -> None:
    print("\n[4] result.npz")
    p = ROOT / "student_submit" / "result.npz"
    if not p.exists():
        warn("尚未生成，跳过")
        return
    check(p.stat().st_size <= 20 * 1024 * 1024, f"文件 {p.stat().st_size / 1024**2:.2f} MiB ≤ 20 MiB")
    z = np.load(p, allow_pickle=False)
    keys = sorted(z.files)
    check(keys == ["image_id", "logits"], f"只含 image_id 和 logits 两个数组（实际 {keys}）")
    if keys != ["image_id", "logits"]:
        return
    ids, lg = z["image_id"], z["logits"]
    check(ids.dtype.kind == "U", f"image_id 是 Unicode 字符串数组（实际 dtype={ids.dtype}）")
    check(ids.dtype.itemsize // 4 <= 128, f"单个字符串容量 {ids.dtype.itemsize // 4} ≤ 128 字符")
    check(lg.dtype == np.float32, f"logits 是 float32（实际 {lg.dtype}）")
    check(lg.ndim == 2 and lg.shape[1] == 51, f"logits 形状 (N, 51)（实际 {lg.shape}）")
    check(ids.shape[0] == lg.shape[0], f"image_id 与 logits 行数一致（{ids.shape[0]}）")
    check(bool(np.isfinite(lg).all()), "logits 全是有限数（无 NaN / Inf）")
    if expected_ids is not None:
        check(set(ids.tolist()) == set(expected_ids), "image_id 集合与测试集一致")
        dup = len(ids) - len(set(ids.tolist()))
        check(dup == 0, f"image_id 无重复（重复 {dup}）")
    # 展开后总大小
    total = ids.nbytes + lg.nbytes
    check(total <= 64 * 1024 * 1024, f"数组未压缩总大小 {total / 1024**2:.1f} MiB ≤ 64 MiB")
    # logits 与 csv 标签一致性（argmax 应等于 predicted_label）
    cp = ROOT / "student_submit" / "result.csv"
    if cp.exists():
        body = [l.split(",") for l in cp.read_text(encoding="utf-8").splitlines()[1:]]
        lab = {a: b for a, b in body}
        idx = {c: i for i, c in enumerate(lines)}
        mismatch = 0
        for i, iid in enumerate(ids.tolist()):
            want = lab.get(iid)
            if want is None:
                continue
            j = int(np.argmax(lg[i]))
            got = lines[j] if j < 50 else "unknown"
            if got != want:
                mismatch += 1
        check(mismatch == 0, f"argmax(logits) 与 predicted_label 完全一致（不一致 {mismatch} 条）")


def main() -> None:
    print("=" * 78)
    print("EduBench 提交物合规检查")
    print("=" * 78)
    lines = check_classes()
    check_index_mapping(lines)

    expected = None
    if TEST_DIR.exists():
        files = [f for f in TEST_DIR.iterdir() if f.is_file()]
        expected = sorted(f.stem for f in files)
        print(f"\n[0] 测试集：{len(files)} 张，image_id 例如 {expected[:3]} … {expected[-3:]}")
        check(all(len(x) == 6 and x.isdigit() for x in expected), "测试集文件名都是 6 位数字")
    else:
        warn("测试集目录不存在")

    check_csv(lines, expected)
    check_npz(lines, expected)

    print("\n" + "=" * 78)
    ok = sum(1 for k, _ in results if k)
    print(f"通过 {ok} 项，失败 {len(results) - ok} 项")
    for k, n in results:
        if not k:
            print(f"  - {n}")
    print("=" * 78)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
