# -*- coding: utf-8 -*-
"""车型档次（配置版本）知识库：`save_car_trims` / `compare_car_trims`

**要解决的问题**：教师数据集只提供「类别 ID + 车型名」，没有档次、价格、参数。
所以「轩逸多少钱」这类问题，档案库原本答不了 —— 只能靠联网查。
但联网查完就丢了，下次还得重查。

本模块把**联网读到的档次信息沉淀进档案库**，形成可复用、可检索的本地知识：

    浏览网页（browse_page） → 抽取档次/价格/参数 → save_car_trims 落库
    → 以后 query_car_info / compare_car_trims 直接就能答，不用再联网

**关键设计：反编造**

档次和价格是最容易被大模型"顺手编一个"的东西（而且编得还挺像）。
三道防线：

  1. `source_url` 是**必填**的 —— 不写清数据从哪来，就不许入库；
  2. 校验这个 URL 是不是**本次会话真的用 browse_page 打开过**（`verified` 标记）；
     没打开过也允许存（用户可能自己贴的链接），但会标 `verified: false` 并在提示词里
     要求模型如实说明"未经核实"；
  3. 每条档次都带 `source` 与 `fetched_at`，界面上直接展示来源和抓取时间。

存储：`agent/data/car_trims.json`。
**故意与 `car_profiles.json` 分开** —— 后者是 `tools/build_car_profiles.py` 从数据集
重新生成的，档次是**运行时积累的知识**，不能被重新生成覆盖掉。
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..errors import Code, ToolError
from .profile import load_profiles, resolve_car
from .registry import ToolContext, ToolDefinition

_LOCK = threading.Lock()
SCHEMA_VERSION = 1

MAX_TRIMS = 30
MAX_SPEC_KEYS = 40


# --------------------------------------------------------------------------- #
# 存取
# --------------------------------------------------------------------------- #
def trims_path(cfg) -> Path:
    return Path(cfg.data_dir) / "car_trims.json"


def _empty() -> dict:
    return {"schema": SCHEMA_VERSION,
            "note": "车型档次（配置版本）知识库，由联网抓取沉淀而来，每条都带来源与抓取时间。",
            "updated_at": "",
            "cars": {}}


def load_trims(cfg) -> dict:
    p = trims_path(cfg)
    if not p.exists():
        return _empty()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(d, dict) or "cars" not in d:
        return _empty()
    d.setdefault("cars", {})
    return d


def save_trims(cfg, data: dict) -> None:
    p = trims_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_car_trims(cfg, car_id: str) -> dict | None:
    return load_trims(cfg)["cars"].get(car_id)


def dedupe_car(cfg, car_id: str) -> tuple[int, int]:
    """对某一款车已有的档次做一次归并清理，返回 (清理前, 清理后)。

    归并规则是逐步收紧的，早期存进去的数据可能带着按旧规则没能合并的重复项。
    这个函数用来"补一次账"，也可以在发现重复时手动调用。
    """
    with _LOCK:
        data = load_trims(cfg)
        entry = data["cars"].get(car_id)
        if not entry:
            return 0, 0
        before = len(entry.get("trims", []))
        kept: list[dict] = []
        for t in entry.get("trims", []):
            # 先把这个档次自己的名字洗干净：早期版本可能把模型加的括号注解
            # （`科技（指导价11.49万款）`）当正式名存了下来
            clean = _strip_annotation(t.get("name", ""))
            if clean and clean != t.get("name"):
                t["name"] = clean
                t["trim_id"] = _trim_id(car_id, clean)
            hit = next((k for k in kept if _same_trim(
                k["name"], t["name"], k.get("guide_price_wan"), t.get("guide_price_wan"))), None)
            if hit is None:
                kept.append(t)
                continue
            hit["name"] = _better_name(hit["name"], t["name"])
            hit["trim_id"] = _trim_id(car_id, hit["name"])
            hit["specs"] = {**(hit.get("specs") or {}), **(t.get("specs") or {})}
            hit["highlights"] = list(dict.fromkeys(
                (hit.get("highlights") or []) + (t.get("highlights") or [])))
            for f in ("guide_price_wan", "dealer_price_wan", "price_text", "year"):
                if not hit.get(f) and t.get(f):
                    hit[f] = t[f]
        entry["trims"] = kept
        entry["trims"].sort(key=lambda x: (x.get("guide_price_wan") is None,
                                           x.get("guide_price_wan") or 0, x.get("name", "")))
        data["cars"][car_id] = entry
        save_trims(cfg, data)
        return before, len(kept)


# --------------------------------------------------------------------------- #
# 规范化
# --------------------------------------------------------------------------- #
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")
# 表示"元"的符号/单位，和表示"万"的
_YUAN_MARKERS = ("¥", "￥", "元", "rmb", "cny")
_WAN_MARKERS = ("万", "w")


def _to_wan(v: Any) -> float | None:
    """把「7.99」「7.99万」「¥79,900」「12.5 万元」这类写法统一成**万元**数字。

    模型从网页抄下来的价格格式五花八门，这里做一次宽容归一；
    归不出来的返回 None（宁可留空，也不要瞎猜一个数）。

    注意 `¥79,900` 不能当成 79900 万 —— 带「元」标记且不带「万」的要除以 10000。
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        val = float(v)
    else:
        s = str(v).strip().replace(",", "").replace("，", "")
        if not s:
            return None
        m = _PRICE_RE.search(s)
        if not m:
            return None
        val = float(m.group(1))
        low = s.lower()
        has_yuan = any(k in low for k in _YUAN_MARKERS)
        has_wan = any(k in low for k in _WAN_MARKERS)
        if has_yuan and not has_wan:
            val = val / 10000.0
        elif not has_yuan and not has_wan and val >= 10000:
            # 没写单位但数字很大 → 几乎肯定是"元"（一辆车不可能卖几千万元）
            val = val / 10000.0
    if val <= 0 or val > 100000:          # 明显不合理的值直接丢弃
        return None
    return round(val, 4)


def _norm_specs(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for i, (k, v) in enumerate(raw.items()):
        if i >= MAX_SPEC_KEYS:
            break
        key = str(k).strip()[:40]
        if not key:
            continue
        if isinstance(v, bool):
            val = "是" if v else "否"
        elif v is None:
            continue
        elif isinstance(v, (list, tuple)):
            val = "、".join(str(x).strip() for x in v if str(x).strip())
        else:
            val = str(v).strip()
        if val:
            out[key] = val[:200]
    return out


def _trim_id(car_id: str, name: str) -> str:
    import hashlib
    return "t-" + hashlib.md5(f"{car_id}|{_canon_name(name)}".encode("utf-8")).hexdigest()[:10]


def _canon_name(name: str) -> str:
    """把档次名归一化，用于判断"这两个名字是不是同一档"。

    实测问题：同一个档次的名称在不同次抓取里会有细微差别 ——
    `全电驱大屏` vs `全电驱大屏版`、`超Ultra` vs `超豪华Ultra`。
    只按字符串精确匹配合并的话，档案库会越用越多重复项。

    还要剥掉**模型自己加的括号注释**：它偶尔会给重名档次加个说明，
    比如 `2026款 1.6L CVT 科技（指导价11.49万款）` —— 那是对同一档的解释性写法，
    括号里的内容不是名字的一部分。**不剥掉就永远合并不上**（长度差太多，
    子序列/子串规则都失效）。
    """
    s = re.sub(r"\s+", "", str(name or ""))
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[（(【\[][^)）】\]]*[)）】\]]", "", s)   # 去掉括号及其内容
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"(版|款)$", "", s)          # 结尾的"版/款"不影响档次身份
    return s.strip().lower()


def _is_subsequence(short: str, long: str) -> bool:
    """short 是否是 long 的子序列（按顺序出现即可，中间可以插字）。

    `2023款超混电驱超ultra` 是 `2023款超混电驱超豪华ultra` 的子序列（中间插了"豪华"），
    但 `…舒适` **不是** `…舒享` 的子序列 —— 这正是我们要区分的：
    **缩写是删字，异档是换字。**
    """
    it = iter(long)
    return all(ch in it for ch in short)


def _has_annotation(name: str) -> bool:
    return bool(re.search(r"[（(【\[]", str(name or "")))


def _strip_annotation(name: str) -> str:
    """去掉名字里的括号注解，保留其余部分。

    `2026款 1.6L CVT 科技（指导价11.49万款）` → `2026款 1.6L CVT 科技`
    """
    s = str(name or "")
    s = re.sub(r"[（(【\[][^)）】\]]*[)）】\]]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _better_name(a: str, b: str) -> str:
    """合并两个名字时保留哪个。

    **不能只比长度** —— 模型偶尔会自己加括号注解
    （`2026款 1.6L CVT 科技（指导价11.49万款）`），它比正式名长，
    按长度比会把注解名留下来，越用越乱。所以：先去掉带注解的那个，
    再在剩下的里选**归一化后更长**的（信息更全，如 `超豪华Ultra` > `超Ultra`）。
    """
    if _has_annotation(a) != _has_annotation(b):
        return b if _has_annotation(a) else a
    return a if len(_canon_name(a)) >= len(_canon_name(b)) else b


def _same_trim(a: str, b: str, price_a: float | None = None,
               price_b: float | None = None) -> bool:
    """两个档次名是否指同一档。

    ⚠️ 这里**不能用字符相似度**。中文档次名恰恰只差一个字：
    `舒适版` / `舒享版`、`豪华版` / `尊贵版` —— 相似度高达 0.93，
    按相似度合并会把**不同的档次错误地并成一个**（实测：16 个档次被并成 10 个，丢真数据）。

    判据按可靠性从高到低：

    1. **价格**。名字会骗人，价格不会：`科技版 10.49万` 与 `舒享科技版 10.99万`
       名字看着像缩写关系，但**价格不同就是两款车**。
       反之 `超Ultra 17.49万` 与 `超豪华Ultra 17.49万` 同价，就是同一个档次的两种写法。
    2. **缩写关系**（删字，不是换字）：
       · `全电驱大屏` ⊂ `全电驱大屏版`（连续子串）→ 同一档；
       · `超Ultra` 是 `超豪华Ultra` 的子序列（中间插了"豪华"）→ 同一档；
       · `舒适版` 不是 `舒享版` 的子序列 → 不同档次。
    3. 长度护栏：差别超过 6 个字的不算缩写，
       避免「1.6L CVT」这种过短的名字把「1.6L CVT 豪华科技版」吞掉。

    **原则：拿不准就不合并。** 多一条重复记录只是有点烦，
    错误合并却会真的丢数据。
    """
    # ---- 1. 价格优先：两边都有价且不一样 → 铁定是不同档次 ----
    if price_a and price_b and abs(float(price_a) - float(price_b)) > 0.005:
        return False

    na, nb = _canon_name(a), _canon_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    diff = len(long) - len(short)
    if len(short) < 4 or diff > 6:
        return False
    if short in long:                       # 连续子串：全电驱大屏 / 全电驱大屏版
        return True
    # 子序列：缩写时中间被插了字（超Ultra / 超豪华Ultra）。
    # 差别必须更小、且**价格不能矛盾**（上面已排除有价且不同的情况），才认。
    return diff <= 4 and _is_subsequence(short, long)


def _norm_trim(car_id: str, raw: dict, source: dict) -> dict:
    name = str(raw.get("name") or "").strip()[:80]
    if not name:
        raise ToolError(Code.INVALID_ARGUMENTS, "每个档次都必须有 name（档次名称）",
                        hint='例如 "2026款 1.6L CVT 舒适版"')
    trim = {
        "trim_id": _trim_id(car_id, name),
        "name": name,
        "year": (str(raw.get("year")).strip()[:12] if raw.get("year") else ""),
        "guide_price_wan": _to_wan(raw.get("guide_price_wan")),      # 厂商指导价
        "dealer_price_wan": _to_wan(raw.get("dealer_price_wan")),    # 经销商参考价
        "price_text": (str(raw.get("price_text")).strip()[:60] if raw.get("price_text") else ""),
        "specs": _norm_specs(raw.get("specs")),
        "highlights": [str(x).strip()[:120] for x in (raw.get("highlights") or [])
                       if str(x).strip()][:20],
        "source": dict(source),
    }
    return trim


# --------------------------------------------------------------------------- #
# 来源校验：这个 URL 真的在本会话打开过吗
# --------------------------------------------------------------------------- #
def _was_visited(ctx: ToolContext, url: str) -> bool:
    """扫描本会话的事件日志，看有没有用 browse_page 真打开过这个 URL。

    这是反编造的第二道防线：模型不能凭空给一个 URL 就说"我查到了"。

    比对要**宽容但不能放水**：同一个页面会有很多写法 ——
    带不带 `?pvareaid=...` 查询串、`http` / `https`、`www` 与不带 www、
    结尾斜杠。这些应该算同一个页面；但**不同的路径绝不算**。
    """
    def canon(u: str) -> str:
        u = (u or "").strip().lower()
        u = re.sub(r"^https?://", "", u)
        u = re.sub(r"^www\.", "", u)
        u = re.sub(r"[?#].*$", "", u)
        return u.rstrip("/").rstrip(".html")

    want = canon(url)
    if not want:
        return False
    try:
        events = ctx.session.events(600)
    except Exception:                                # noqa: BLE001
        return False
    for ev in events:
        if ev.get("type") != "tool_result":
            continue
        p = ev.get("payload") or {}
        if p.get("tool") != "browse_page":
            continue
        res = p.get("result") or {}
        if not isinstance(res, dict):
            continue
        got = canon(res.get("url", ""))
        if not got:
            continue
        # 完全相同，或一个是另一个的前缀（页面带子路径）都认
        if got == want or got.startswith(want + "/") or want.startswith(got + "/"):
            return True
    return False


def _visited_urls(ctx: ToolContext, limit: int = 8) -> list[str]:
    """本会话用 browse_page 打开过的 URL —— 核验失败时回给模型，让它自己改正。"""
    out: list[str] = []
    try:
        for ev in ctx.session.events(600):
            if ev.get("type") != "tool_result":
                continue
            p = ev.get("payload") or {}
            if p.get("tool") != "browse_page":
                continue
            res = p.get("result") or {}
            if isinstance(res, dict) and res.get("url"):
                if res["url"] not in out:
                    out.append(res["url"])
    except Exception:                                # noqa: BLE001
        pass
    return out[-limit:]


# --------------------------------------------------------------------------- #
# save_car_trims
# --------------------------------------------------------------------------- #
def _handler_save(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()

    p = resolve_car(cfg, args.get("car_id_or_name", ""))
    car_id = p["car_id"]

    source_url = (args.get("source_url") or "").strip()
    if not source_url.startswith(("http://", "https://")):
        raise ToolError(
            Code.INVALID_ARGUMENTS,
            "必须提供 source_url（这份档次数据的来源网址）",
            hint="先把网页用 browse_page 打开、真的读到了这些数据，再把该页面的 URL 填进来。"
                 "没有来源的数据不允许写进档案库 —— 编造价格比查不到更糟。",
        )

    raw_trims = args.get("trims") or []
    if not isinstance(raw_trims, list) or not raw_trims:
        raise ToolError(Code.INVALID_ARGUMENTS, "trims 至少要有一个档次")
    if len(raw_trims) > MAX_TRIMS:
        raise ToolError(Code.INVALID_ARGUMENTS, f"一次最多写入 {MAX_TRIMS} 个档次")

    verified = _was_visited(ctx, source_url)
    source = {
        "url": source_url,
        "site": (args.get("source_site") or "").strip()[:40],
        "fetched_at": _now(),
        "verified": verified,
    }

    new_trims = [_norm_trim(car_id, t, source) for t in raw_trims if isinstance(t, dict)]
    if not new_trims:
        raise ToolError(Code.INVALID_ARGUMENTS, "trims 里没有合法的档次对象")

    replace = bool(args.get("replace"))
    note = (args.get("note") or "").strip()[:300]

    with _LOCK:
        data = load_trims(cfg)
        entry = data["cars"].get(car_id) or {
            "car_id": car_id, "car_name": p["name"], "brand": p["brand"],
            "body_group": p["body_group"], "trims": [], "sources": [], "notes": [],
        }
        existing = {} if replace else {t["name"]: t for t in entry.get("trims", [])}
        added, updated, merged_into = [], [], []
        for t in new_trims:
            # 先精确匹配，再模糊匹配 —— 跨次抓取的名字会有细微差别，
            # 不归并的话同一个档次会被反复写进来
            target = t["name"] if t["name"] in existing else None
            if target is None:
                for old_name, old_trim in existing.items():
                    if _same_trim(t["name"], old_name,
                                  t.get("guide_price_wan"),
                                  old_trim.get("guide_price_wan")):
                        target = old_name
                        break
            if target is None:
                existing[t["name"]] = t
                added.append(t["name"])
                continue

            old = existing[target]
            # 名字取更"正式"的那个（剥掉注解、信息更全的优先）
            keep_name = _better_name(target, t["name"])
            merged = dict(old)
            for k, v in t.items():
                if k == "specs":
                    merged["specs"] = {**old.get("specs", {}), **v}
                elif k == "highlights":
                    merged[k] = v or old.get("highlights", [])
                elif v not in (None, "", []):
                    merged[k] = v
            merged["name"] = keep_name
            merged["trim_id"] = _trim_id(car_id, keep_name)
            merged["source"] = t["source"]
            if keep_name != target:
                existing.pop(target, None)
                merged_into.append(f"{target} → {keep_name}")
            existing[keep_name] = merged
            updated.append(keep_name)

        entry["trims"] = list(existing.values())
        # 按指导价排序，方便比价
        entry["trims"].sort(key=lambda x: (x.get("guide_price_wan") is None,
                                           x.get("guide_price_wan") or 0,
                                           x.get("name", "")))
        srcs = [s for s in entry.get("sources", []) if s.get("url") != source_url]
        srcs.append(source)
        entry["sources"] = srcs[-12:]
        if note:
            entry.setdefault("notes", []).append({"text": note, "at": _now(), "url": source_url})
            entry["notes"] = entry["notes"][-20:]
        entry["car_name"] = p["name"]
        entry["brand"] = p["brand"]
        entry["body_group"] = p["body_group"]
        entry["updated_at"] = _now()

        data["cars"][car_id] = entry
        save_trims(cfg, data)

    summary = summarize_trims(entry)

    return {
        "ok": True,
        "car_id": car_id,
        "car_name": p["name"],
        "saved": len(new_trims),
        "added": added,
        "updated": updated,
        "merged_into": merged_into,
        "replaced": replace,
        "total_trims": len(entry["trims"]),
        "verified": verified,
        "source": source,
        "trim_summary": summary,
        "trims": entry["trims"],
        "note": ("已写入档案库。以后问这款车的价格/档次，`query_car_info` 和 "
                 "`compare_car_trims` 就能直接答，不必再联网。"
                 + ("" if verified else
                    "\n⚠️ 这个 source_url **不是本次会话用 browse_page 打开过的页面**，"
                    "已标为未核实。本会话实际打开过的是："
                    + "；".join(_visited_urls(ctx)) +
                    "。如果数据确实来自其中某一页，请用那个 URL 重新调用一次本工具覆盖；"
                    "回答用户时也请如实说明这来源未经核实。")),
    }


# --------------------------------------------------------------------------- #
# 汇总与比较
# --------------------------------------------------------------------------- #
def summarize_trims(entry: dict) -> str:
    """一句话说清这款车有哪些档次、价格跨度多少。"""
    trims = entry.get("trims") or []
    if not trims:
        return ""
    prices = [t.get("guide_price_wan") for t in trims if t.get("guide_price_wan")]
    parts = [f"共 {len(trims)} 个档次"]
    if prices:
        lo, hi = min(prices), max(prices)
        parts.append(f"指导价 {lo:g}–{hi:g} 万" if lo != hi else f"指导价 {lo:g} 万")
    return "，".join(parts)


def _diff_trims(trims: list[dict]) -> dict:
    """找出各档次之间**真正有差别**的字段。

    网页上的参数表大部分行是相同的（比如"车身结构：4门5座三厢车"），
    全列出来没有信息量。这里只保留取值不完全一致的键 —— 差异才是用户要的。
    """
    keys: list[str] = []
    for t in trims:
        for k in (t.get("specs") or {}):
            if k not in keys:
                keys.append(k)
    diff_keys = []
    for k in keys:
        vals = {(t.get("specs") or {}).get(k, "") for t in trims}
        if len(vals) > 1:
            diff_keys.append(k)
    return {
        "keys": diff_keys,
        "same_keys": [k for k in keys if k not in diff_keys],
        "table": [
            {
                "trim": t["name"],
                "guide_price_wan": t.get("guide_price_wan"),
                "dealer_price_wan": t.get("dealer_price_wan"),
                "differing": {k: (t.get("specs") or {}).get(k, "") for k in diff_keys},
                "highlights": t.get("highlights") or [],
            }
            for t in trims
        ],
        "has_highlights": any(t.get("highlights") for t in trims),
    }


def _handler_compare(args: dict, ctx: ToolContext) -> dict:
    cfg = ctx.config
    ctx.check_abort()
    p = resolve_car(cfg, args.get("car_id_or_name", ""))
    entry = get_car_trims(cfg, p["car_id"])

    if not entry or not entry.get("trims"):
        return {
            "ok": True,
            "car_id": p["car_id"],
            "car_name": p["name"],
            "has_trim_data": False,
            "message": (f"档案库里还没有 {p['name']} 的档次数据。"
                        f"可以先用 web_search 找到该车型的报价/配置页，"
                        f"再用 browse_page 打开读正文，最后用 save_car_trims 存进来。"),
            "suggest_query": f"{p['name'].split('_')[-1]} 各配置 价格 参数",
        }

    trims = entry["trims"]
    want = args.get("trim_names") or []
    if want:
        picked = [t for t in trims if any(w.strip() in t["name"] for w in want if w.strip())]
        if picked:
            trims = picked

    diff = _diff_trims(trims)
    prices = [t.get("guide_price_wan") for t in trims if t.get("guide_price_wan")]
    dealer = [t.get("dealer_price_wan") for t in trims if t.get("dealer_price_wan")]

    return {
        "ok": True,
        "car_id": p["car_id"],
        "car_name": p["name"],
        "has_trim_data": True,
        "compared": len(trims),
        "total_trims": len(entry["trims"]),
        "trim_summary": summarize_trims(entry),
        "price_range": {
            "guide": [min(prices), max(prices)] if prices else None,
            "dealer": [min(dealer), max(dealer)] if dealer else None,
        },
        "differing_specs": diff["keys"],
        "identical_specs_count": len(diff["same_keys"]),
        "table": diff["table"],
        "sources": entry.get("sources", []),
        "updated_at": entry.get("updated_at"),
        "note": ("`differing_specs` 是各档次之间**取值不同**的参数 —— 比较档次就看这些；"
                 "完全一样的参数已省略（各档次相同的部分）。"
                 "每档的 `highlights` 是该档相对低配多出来的配置，**「贵在哪」主要靠它**。"
                 "如果 differing_specs 为空且 has_highlights 为 false，说明存档时只记了"
                 "全系一致的参数 —— 请如实告诉用户「库里没记到区分性配置」，"
                 "并建议联网再抓一次配置表（重点抓各档次不同的装备项）；"
                 "**不要靠猜**补充差异。"
                 "数据来自联网抓取，来源见 sources，请一并告知用户，并提醒以官方配置表为准。"),
    }


# --------------------------------------------------------------------------- #
# 工具定义
# --------------------------------------------------------------------------- #
TRIM_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="save_car_trims",
        description=(
            "把**从网页实际读到**的车型档次信息（配置版本：名称、厂商指导价、经销商价、参数配置）"
            "写进本地档案库，之后就不用再联网查了。\n"
            "使用流程：web_search 找到报价/配置页 → browse_page 打开读正文 → 从正文里抄出档次数据 → "
            "调用本工具保存，并把那个页面的 URL 填进 source_url。\n"
            "**严禁编造**：只写正文里真实出现过的字段；没读到的字段就省略，不要猜。"
            "source_url 是必填的，没有来源的数据不允许入库。\n"
            "**分批写**：一次最多写 6~8 个档次，档次多就分几次调用（工具支持按名称合并）。"
            "`specs` 只填**有区分度**的关键项（发动机/变速箱/功率/关键配置），"
            "不要把网页上那张几十行的参数表整张抄进来 —— 那样会超出输出长度上限，"
            "而且各档次都一样的参数填了也没用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "car_id_or_name": {
                    "type": "string", "minLength": 1, "maxLength": 40,
                    "description": "四位 car_id 或车型名，如 \"0975\" / \"轩逸\"",
                },
                "source_url": {
                    "type": "string", "minLength": 8, "maxLength": 500,
                    "description": "这份数据的来源网址，必须是你刚用 browse_page 打开过的那个页面。",
                },
                "source_site": {
                    "type": "string", "maxLength": 40,
                    "description": "可选，站点名，如 \"汽车之家\"。",
                },
                "trims": {
                    "type": "array", "maxItems": 30,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string", "minLength": 1, "maxLength": 80,
                                "description": "档次名称，如 \"2026款 1.6L CVT 舒适版\"",
                            },
                            "year": {"type": "string", "maxLength": 12, "description": "可选，年款"},
                            "guide_price_wan": {
                                "type": "number", "minimum": 0, "maximum": 100000,
                                "description": "可选，厂商指导价（万元）。如页面上写 7.99万 就填 7.99",
                            },
                            "dealer_price_wan": {
                                "type": "number", "minimum": 0, "maximum": 100000,
                                "description": "可选，经销商/终端参考价（万元）",
                            },
                            "price_text": {
                                "type": "string", "maxLength": 60,
                                "description": "可选，价格原文，如 \"6.39万起\"，用于保留页面原话",
                            },
                            "specs": {
                                "type": "object",
                                "description": ("可选，该档次的参数配置键值对。"
                                                "**优先填各档次之间会不一样的项**："
                                                "天窗、座椅材质、中控屏尺寸、辅助驾驶、"
                                                "气囊数、轮毂尺寸……这些才是档次差别的来源。"
                                                "发动机/变速箱这种全系一致的项写一两个就够。"
                                                "只填页面上有的。"),
                            },
                            "highlights": {
                                "type": "array", "maxItems": 20,
                                "items": {"type": "string", "maxLength": 120},
                                "description": ("可选，该档次**相对低配多出来**的配置，"
                                                "如 [\"全景天窗\", \"真皮座椅\", \"L2 辅助驾驶\"]。"
                                                "这是回答「贵在哪」的关键，尽量填。"),
                            },
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                    "description": "档次列表，一个档次一个对象。请把页面上列出的档次尽量都填上。",
                },
                "replace": {
                    "type": "boolean",
                    "description": "可选。true = 用本次数据整体替换旧数据；默认 false = 按名称合并（推荐）。",
                },
                "note": {
                    "type": "string", "maxLength": 300,
                    "description": "可选，随数据一起记下的补充说明（例如「优惠幅度随城市浮动」）。",
                },
            },
            "required": ["car_id_or_name", "source_url", "trims"],
            "additionalProperties": False,
        },
        handler=_handler_save,
        timeout_s=20.0,
        tags=("knowledge", "trims"),
    ),
    ToolDefinition(
        name="compare_car_trims",
        description=(
            "比较一款车**各个档次之间的差别**：价格跨度、配置差异。"
            "用户问「豪华版比标准版贵多少」「这几个配置差在哪」「哪个配置更值」时调用。\n"
            "只会列出**取值不同**的参数（各档次相同的部分自动省略），所以结果很紧凑。\n"
            "如果档案库里还没有该车的档次数据，本工具会告诉你该怎么联网去查。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "car_id_or_name": {
                    "type": "string", "minLength": 1, "maxLength": 40,
                    "description": "四位 car_id 或车型名",
                },
                "trim_names": {
                    "type": "array", "maxItems": 10,
                    "items": {"type": "string", "maxLength": 80},
                    "description": "可选，只比较名字里含这些关键词的档次，如 [\"舒适版\",\"豪华版\"]；省略则比较全部。",
                },
            },
            "required": ["car_id_or_name"],
            "additionalProperties": False,
        },
        handler=_handler_compare,
        timeout_s=15.0,
        tags=("knowledge", "trims"),
    ),
]
