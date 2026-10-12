# -*- coding: utf-8 -*-
"""任务一 - 步骤2：确定 50 个目标类别 + unknown 来源类别

输入：
  configs/selection.json      人工策划的 50 类清单（含选题理由）
  configs/class_info.json     教师提供的 827 类类别说明
  outputs/reports/class_inventory.csv  步骤1 生成的样本量统计

输出：
  classes.txt                     项目根目录，恰好 50 行四位类别 ID，升序，UTF-8 无 BOM
  configs/unknown_sources.json    unknown 训练/验证样本的来源类别清单
  configs/class_map.json          模型标签映射（索引 0~49 -> 车型 ID，索引 50 -> unknown）
  outputs/reports/selection_report.txt  选类报告
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_JSON = PROJECT_ROOT / "configs" / "selection.json"
CLASS_INFO = PROJECT_ROOT / "configs" / "class_info.json"
INVENTORY_CSV = PROJECT_ROOT / "outputs" / "reports" / "class_inventory.csv"
CLASSES_TXT = PROJECT_ROOT / "classes.txt"
UNKNOWN_JSON = PROJECT_ROOT / "configs" / "unknown_sources.json"
CLASS_MAP_JSON = PROJECT_ROOT / "configs" / "class_map.json"
REPORT_TXT = PROJECT_ROOT / "outputs" / "reports" / "selection_report.txt"

RANDOM_SEED = 20250914


def parse_brand(type_info: str) -> str:
    return ""


def main() -> None:
    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    info = {row["id"]: row for row in json.loads(CLASS_INFO.read_text(encoding="utf-8"))}
    with open(INVENTORY_CSV, encoding="utf-8") as fp:
        inv = {row["id"]: row for row in csv.DictReader(fp)}

    picked = selection["classes"]
    ids = [c["id"] for c in picked]

    # ---------- 校验 ----------
    errors: list[str] = []
    if len(ids) != 50:
        errors.append(f"类别数量必须为 50，当前 {len(ids)}")
    if len(set(ids)) != len(ids):
        dup = [i for i in set(ids) if ids.count(i) > 1]
        errors.append(f"存在重复类别 ID：{dup}")
    for cid in ids:
        if cid not in info:
            errors.append(f"ID {cid} 不在教师提供的 827 类中")
        elif not (len(cid) == 4 and cid.isdigit()):
            errors.append(f"ID {cid} 不是四位数字")
    if errors:
        raise SystemExit("选类校验失败：\n  - " + "\n  - ".join(errors))

    ids_sorted = sorted(ids)

    # ---------- classes.txt ----------
    # UTF-8 无 BOM，恰好 50 行，按类别 ID 升序
    CLASSES_TXT.write_text("\n".join(ids_sorted) + "\n", encoding="utf-8", newline="\n")

    raw = CLASSES_TXT.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "classes.txt 不能带 BOM"
    assert raw.decode("utf-8").count("\n") == 50, "classes.txt 必须恰好 50 行"

    # ---------- 标签映射 ----------
    class_map = {
        "note": "索引 0~49 按 classes.txt 行顺序对应 50 个目标车型；索引 50 对应 unknown。训练/保存/Web 推理/批量推理共用本映射。",
        "index_to_id": {str(i): cid for i, cid in enumerate(ids_sorted)},
        "index_to_name": {
            str(i): info[cid]["name_from_new"] for i, cid in enumerate(ids_sorted)
        },
        "index_to_group": {
            str(i): next(c["group"] for c in picked if c["id"] == cid)
            for i, cid in enumerate(ids_sorted)
        },
        "id_to_index": {cid: i for i, cid in enumerate(ids_sorted)},
        "unknown_index": 50,
        "unknown_label": "unknown",
        "num_classes": 51,
    }
    CLASS_MAP_JSON.write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- unknown 来源类别 ----------
    plan = selection["unknown_plan"]
    rng = random.Random(RANDOM_SEED)
    min_train = plan["min_train_images"]

    chosen_brands = {inv[cid]["brand"] for cid in ids_sorted}
    chosen_cats = {inv[cid]["category"] for cid in ids_sorted}
    chosen = set(ids_sorted)

    # (a) 同品牌兄弟车型 —— 硬负样本，抑制“只认品牌不认车型”
    brand_classes: dict[str, list[str]] = defaultdict(list)
    for cid, row in inv.items():
        if cid in chosen or int(row["train"]) < min_train:
            continue
        brand_classes[row["brand"]].append(cid)

    hard_negatives: list[dict] = []
    for brand in sorted(chosen_brands):
        cands = brand_classes.get(brand, [])
        if not cands:
            continue
        # 优先选与本组该品牌车型同中文大类的兄弟车型（更像的硬负样本）
        my_cats = {inv[cid]["category"] for cid in ids_sorted if inv[cid]["brand"] == brand}
        same_cat = [c for c in cands if inv[c]["category"] in my_cats]
        pool = same_cat or cands
        # 同池内按 train 样本量降序，取前 3 随机挑 1，兼顾确定性与多样性
        pool = sorted(pool, key=lambda c: -int(inv[c]["train"]))[:3]
        pick = rng.choice(pool)
        hard_negatives.append(
            {
                "id": pick,
                "name": info[pick]["name_from_new"],
                "brand": inv[pick]["brand"],
                "category": inv[pick]["category"],
                "body_type": inv[pick]["body_type"],
                "train": int(inv[pick]["train"]),
                "val": int(inv[pick]["val"]),
                "source_kind": "hard_negative",
                "target_class_id": next(c for c in ids_sorted if inv[c]["brand"] == brand),
            }
        )

    used = chosen | {h["id"] for h in hard_negatives}

    # (b) 其他品牌 —— 覆盖每个中文大类，按大类内类别数占比分配名额，每品牌至多 1 类
    need = plan["other_brand_classes"]
    cat_pool: dict[str, list[str]] = defaultdict(list)
    for cid, row in inv.items():
        if cid in used or int(row["train"]) < min_train:
            continue
        if row["brand"] in chosen_brands:
            continue
        cat_pool[row["category"]].append(cid)

    # 按各大类在“可用池”中的规模占比分配名额，保证覆盖所有大类
    cats = sorted(cat_pool, key=lambda c: -len(cat_pool[c]))
    quota = {}
    total_pool = sum(len(v) for v in cat_pool.values())
    assigned = 0
    for cat in cats:
        q = max(1, round(need * len(cat_pool[cat]) / total_pool))
        quota[cat] = q
        assigned += q
    # 名额微调到恰好 need
    order = sorted(cats, key=lambda c: -len(cat_pool[c]))
    idx = 0
    while assigned != need:
        cat = order[idx % len(order)]
        if assigned > need and quota[cat] > 1:
            quota[cat] -= 1
            assigned -= 1
        elif assigned < need:
            quota[cat] += 1
            assigned += 1
        idx += 1

    other_brand: list[dict] = []
    for cat in cats:
        pool = sorted(cat_pool[cat], key=lambda c: -int(inv[c]["train"]))
        # 品牌去重：每个品牌只取 1 类，且与已选品牌不重复
        picked_brands: set[str] = set()
        shuffled: list[str] = []
        for cid in pool:
            if inv[cid]["brand"] in picked_brands:
                continue
            picked_brands.add(inv[cid]["brand"])
            shuffled.append(cid)
        rng.shuffle(shuffled)
        for cid in shuffled[: quota[cat]]:
            other_brand.append(
                {
                    "id": cid,
                    "name": info[cid]["name_from_new"],
                    "brand": inv[cid]["brand"],
                    "category": inv[cid]["category"],
                    "body_type": inv[cid]["body_type"],
                    "train": int(inv[cid]["train"]),
                    "val": int(inv[cid]["val"]),
                    "source_kind": "other_brand",
                    "target_class_id": None,
                }
            )

    sources = hard_negatives + other_brand
    # 同 ID 去重（理论上不会发生）
    seen: set[str] = set()
    sources = [s for s in sources if not (s["id"] in seen or seen.add(s["id"]))]

    unknown_cfg = {
        "note": "unknown 样本来源。train/unknown 取这些类别的训练图片，val/unknown 取这些类别的验证图片，"
        "训练与验证样本不重复，且完全不使用测试图片。",
        "train_per_source": plan["train_per_source"],
        "val_per_source": plan["val_per_source"],
        "seed": RANDOM_SEED,
        "hard_negative_count": len(hard_negatives),
        "other_brand_count": len(other_brand),
        "source_count": len(sources),
        "sources": sources,
    }
    UNKNOWN_JSON.write_text(
        json.dumps(unknown_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 报告 ----------
    group_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for cid in ids_sorted:
        g = next(c["group"] for c in picked if c["id"] == cid)
        group_stat[g][0] += 1
        group_stat[g][1] += int(inv[cid]["train"])
        group_stat[g][2] += int(inv[cid]["val"])

    brand_count = defaultdict(int)
    for cid in ids_sorted:
        brand_count[inv[cid]["brand"]] += 1

    lines: list[str] = []
    add = lines.append
    add("=" * 92)
    add(f"{selection['project']['name']} — 类别选择报告")
    add(f"选题方向：{selection['project']['theme']}")
    add("=" * 92)
    add("")
    add("选类原则：")
    for p in selection["selection_principles"]:
        add(f"  · {p}")
    add("")
    add("-" * 92)
    add(f"目标类别：{len(ids_sorted)} 类   涉及品牌：{len(brand_count)} 个")
    add("-" * 92)
    add(f"{'车型ID':<8}{'索引':<6}{'车型名称':<34}{'品牌':<12}{'大类':<8}{'train':>6}{'val':>5}")
    add("-" * 92)
    for i, cid in enumerate(ids_sorted):
        row = inv[cid]
        add(
            f"{cid:<8}{i:<6}{row['name_from_new']:<34}{row['brand']:<12}"
            f"{row['category']:<8}{row['train']:>6}{row['val']:>5}"
        )
    add("-" * 92)
    add(f"{'合计':<48}{'':<12}{'':<8}{sum(int(inv[c]['train']) for c in ids_sorted):>6}"
        f"{sum(int(inv[c]['val']) for c in ids_sorted):>5}")
    add("")
    add("车身大类分布：")
    for g, (n, tr, va) in sorted(group_stat.items(), key=lambda kv: -kv[1][0]):
        add(f"  {g:<6} {n:2d} 类   train {tr:5d}   val {va:4d}")
    add("")
    add("品牌分布（同一品牌出现 2 次表示该品牌同时提供两类不同车身形态）：")
    for brand, n in sorted(brand_count.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "  ← 同品牌 2 类" if n > 1 else ""
        add(f"  {brand:<16}{n}{mark}")
    add("")
    add("=" * 92)
    add("unknown 样本来源类别")
    add("=" * 92)
    add(f"来源类别总数：{len(sources)}  "
        f"（同品牌兄弟车型 {len(hard_negatives)} 类 + 其他品牌 {len(other_brand)} 类）")
    add(f"采样：每类 train 最多 {plan['train_per_source']} 张、val 最多 {plan['val_per_source']} 张")
    add("")
    add("[A] 同品牌兄弟车型（硬负样本，防止模型只靠品牌判断）")
    add(f"{'车型ID':<8}{'车型名称':<34}{'品牌':<12}{'大类':<8}{'train':>6}{'val':>5}  {'对应本组类别':<12}")
    add("-" * 92)
    for s in sorted(hard_negatives, key=lambda x: x["target_class_id"]):
        add(f"{s['id']:<8}{s['name']:<34}{s['brand']:<12}{s['category']:<8}"
            f"{s['train']:>6}{s['val']:>5}  {s['target_class_id']:<12}")
    add("")
    add("[B] 其他品牌类别（覆盖组外车型分布）")
    add(f"{'车型ID':<8}{'车型名称':<34}{'品牌':<12}{'大类':<8}{'train':>6}{'val':>5}")
    add("-" * 92)
    for s in sorted(other_brand, key=lambda x: (x["category"], -x["train"])):
        add(f"{s['id']:<8}{s['name']:<34}{s['brand']:<12}{s['category']:<8}"
            f"{s['train']:>6}{s['val']:>5}")
    add("")
    est_train = len(sources) * plan["train_per_source"]
    est_val = len(sources) * plan["val_per_source"]
    known_train = sum(int(inv[c]["train"]) for c in ids_sorted)
    known_val = sum(int(inv[c]["val"]) for c in ids_sorted)
    add("-" * 92)
    add(f"预计 unknown 训练样本 ≈ {est_train}（上限），验证样本 ≈ {est_val}（上限）")
    add(f"已知 50 类训练样本 = {known_train}，验证样本 = {known_val}")
    add(f"unknown 训练占比 ≈ {est_train / (est_train + known_train):.1%}")
    add("-" * 92)

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n[写出] {CLASSES_TXT}  (50 行, {len(raw)} bytes, 无 BOM)")
    print(f"[写出] {CLASS_MAP_JSON}")
    print(f"[写出] {UNKNOWN_JSON}  (来源类别 {len(sources)} 个)")
    print(f"[写出] {REPORT_TXT}")


if __name__ == "__main__":
    main()
