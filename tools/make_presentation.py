# -*- coding: utf-8 -*-
"""生成《识车档案 CarDex》演示动画（单文件 HTML）。

设计目标：**每段台词都有对应画面，且动效可操控**。
  · 15 段与《纯台词版》逐段对应，台词显示在底部字幕条
  · 每段画面不同：统计图表 / 代码 / 截图 / 架构图 / 标题卡
  · 动效：元素错峰淡入上浮、图片缓慢推近（Ken Burns）、代码自动下滚
  · 全部可控：空格 下一段 / ← → 跳转 / R 重播本段动效 /
              P 暂停动效 / F 全屏 / 1-9 直达

用法：
    python tools/make_presentation.py
    → outputs/report/demo/演示动画.html  （双击即可用浏览器打开）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "outputs" / "report" / "demo"
FIG = ROOT / "outputs" / "report" / "figures"
SHOT = ROOT / "outputs" / "report" / "screenshots"
OUT = DEMO / "演示动画.html"

# 相对路径（HTML 放在 outputs/report/demo/ 下）
def rel(p: Path) -> str:
    return "../" + p.relative_to(ROOT / "outputs" / "report").as_posix()


def load_code_slides() -> list[dict]:
    """复用 make_code_slides.py 已生成的代码片段，避免两处维护。"""
    js = (DEMO / "code_slides_data.js").read_text(encoding="utf-8")
    m = re.search(r"window\.__CODE_SLIDES\s*=\s*(\[.*\]);", js, re.S)
    return json.loads(m.group(1)) if m else []


CODE = load_code_slides()

# ── 15 段脚本 ──
# kind: title | image | images | code | shots
SLIDES = [
    dict(kind="title", dur=32, t="识车档案 CarDex",
         sub="视觉智能体综合实训 · 智能 234 · 指导教师 林俊武",
         members=[("罗烨嵘", "视觉模型训练 · Web 前端", "202312903430"),
                  ("焦超", "智能体 · 后端与测试", "202312903414")],
         bullets=["看懂车照 —— 自训练 51 类视觉模型", "查资料比价格 —— 12 个工具的智能体",
                  "记住每一次看车 —— 可检索的个人档案"],
         say="大家好，我们是智能 234 班的罗烨嵘和焦超，指导老师是林俊武。"
             "我负责视觉模型的训练与 Web 前端，焦超负责智能体与后端。"
             "我们的项目叫识车档案 CarDex——用户上传一张车照，视觉模型识别车型，"
             "再由智能体查档案、比价格、写进个人看车记录，"
             "最终形成一份可检索、可积累的对话式档案。"),

    dict(kind="image", dur=22, img=FIG / "fig1_数据集构成树状图.png",
         t="一、数据：827 个类别里怎么选出 50 款",
         say="先说数据。教师数据集一共 827 个车型类别、10 万 3 千多张图。我们从中选了两组："
             "第一组是 50 款私人用户最常看的车型，作为识别目标，覆盖轿车、SUV、MPV、跑车、微面、"
             "轻客六种车身形态，共 37 个品牌。第二组是 98 个不在 50 款里的类别，用来构造 unknown 这一类"
             "——专门让模型学会说「我不认识」，而不是硬猜一个最像的。"),

    dict(kind="images", dur=20, imgs=[FIG / "fig2_目标类车身大类分布饼图.png",
                                      FIG / "fig3_各类别样本量柱状图.png"],
         t="二、选题：两条硬标准",
         say="选题有两条硬标准。一是同品牌只取形态差异明显的两款，比如本田我们只留了缤智和奥德赛，"
             "避免近重复类别互相压制 Macro-F1；二是每一类的训练样本都在 80 张以上，样本量拉平，"
             "这样等权 Macro-F1 才不会被小类拖垮。最终 50 类的训练集 5534 张、验证集 1834 张，"
             "样本量差距控制在两倍以内。"),

    dict(kind="code", dur=20, code=0,
         t="三、视觉工具：classify_car 真实调用模型",
         say="这是 classify_car 的实现，在 agent/tools/vision.py。有一点我们特意做了保证："
             "这里没有任何捷径，它真实加载 outputs/checkpoints 下的三模型集成权重，做完整的前向推理。"
             "第 151 行的 clf.predict_path 就是真正的那一次推理调用。图片不会进入大模型，"
             "模型拿到的只有识别结果。如果权重加载失败，它会抛 MODEL_NOT_LOADED，绝不返回伪造的结果。"),

    dict(kind="image", dur=22, img=FIG / "fig4_实验对比柱状图.png",
         t="四、模型：分辨率是最大的杠杆",
         say="模型部分用 ResNet50 做迁移学习，一共跑了 15 组对照实验。最大的发现是：分辨率是这里最有效的杠杆。"
             "从 224 提到 288、352、416，每一步都有稳定收益，单模型从 0.9571 一路提到 0.9788。"
             "相比之下，单纯换更深的网络只提升不到一个百分点；而逆频率类别加权是个负结果，0.8428，"
             "反而比不加权的基线 0.8475 还低——因为我们的类别样本量差距本来就不大，加权只是放大了噪声。"),

    dict(kind="kpi", dur=22,
         t="五、结果：0.9845，并用 5 折交叉验证自证",
         kpis=[("0.9845", "验证集 51 类 Macro-F1", "#4F81BD"),
               ("0.9812 ± 0.0059", "5 折交叉验证", "#9BBB59"),
               ("−0.0003", "乐观偏差（可忽略）", "#F79646")],
         points=["集成成员：ResNet50@416 + ResNet50@288+SupCon + Wide-ResNet50-2@288",
                 "先验校正 τ = 0.2，由验证集网格搜索确定",
                 "结论：直接报告的分数不是「挑出来的」"],
         say="最终方案是三个互补模型等权集成——ResNet50@416、ResNet50@288 加监督对比学习、"
             "以及 Wide-ResNet50-2@288，再做一次先验校正。验证集上 51 类 Macro-F1 达到 0.9845。"
             "为了防止这个数字是「挑出来的」，我们又做了 5 折交叉验证，结果是 0.9812 加减 0.0059，"
             "与直接报告的只差万分之三，说明模型选择带来的乐观偏差可以忽略。"),

    dict(kind="image", dur=18, img=FIG / "fig5_验证集错误构成饼图.png",
         t="六、错误结构：瓶颈不在分类，在 unknown",
         say="错误结构很有意思。2422 张验证集上一共错了 51 张，其中模型几乎不会把一款车认成另一款"
             "——已知类之间的混淆只有 4 张。剩下 47 张错误全部集中在「已知还是未知」这条边界上："
             "25 张是组外车型被误认成已知类，22 张是已知类被误拒。"
             "这说明模型的分类能力已经足够，瓶颈在 unknown 的建模方式。"),

    dict(kind="code", dur=18, code=4,
         t="七、评分口径：51 类等权 Macro-F1",
         say="评分口径我们严格对齐教师要求：51 类等权 Macro-F1，unknown 作为第 51 类参与计算，"
             "与 50 个已知类权重完全相同。这里用的是 sklearn 的 f1_score 加 average='macro'，"
             "保证本地分数可以直接和线上评测对照，不会出现口径不一致导致的误判。"),

    dict(kind="image", dur=20, img=FIG / "fig6_系统总体架构图.png",
         t="八、系统：三个任务怎么串起来",
         say="接下来演示智能体系统。界面分三块：左边是会话列表，每次对话都独立保存，可以随时切回；"
             "中间是对话主区；右边是运行状态和看车档案面板，能看到当前用的是哪个视觉模型、拒识阈值多少、"
             "以及 12 个工具的清单。底部是图片上传和联网搜索开关。"
             "整个系统接的是真实 DeepSeek-V4-Flash 接口，不是模拟。"),

    dict(kind="shots", dur=18, img=SHOT / "02_上传图片就绪.png",
         t="九、输入：上传车照 + 一句自然语言",
         say="我们上传一张车照。大家留意这个图片预览——文件已经真实落盘到服务器，"
             "并且和当前会话绑定，750 乘 500，58 KB。然后输入一句完整的自然语言需求："
             "「这车多少钱？合适的话帮我记下来，再去网上看看有没有新改款。」"
             "这句话里其实藏了三个任务，我们看智能体怎么拆。"),

    dict(kind="shots", dur=24, img=SHOT / "03_三工具调用链完成.png",
         t="十、识别与落库：698ms → 2ms → 11ms",
         steps=[("classify_car", "698 ms", "奥迪 A6 · 0632 · 置信度 91.5%"),
                ("query_car_info", "2 ms", "奥迪 ｜ 轿车"),
                ("add_car_record", "11 ms", "已写入看车档案 · 记录 #112")],
         say="看调用记录。第一个工具 classify_car，花了 698 毫秒，识别出这是奥迪 A6，置信度 91.5%。"
             "这里说明一下，第一次调用要 20 秒，因为要加载三个模型的权重；我们提前预热过，"
             "所以现在是 698 毫秒。接着 query_car_info 查档案，2 毫秒；add_car_record 写进看车档案，"
             "11 毫秒，记录号 112。三步的输入、输出和耗时都能点开核对。"),

    dict(kind="code", dur=20, code=2,
         t="十一、智能体循环：调用链由模型自己拆",
         say="这是智能体循环的实现，在 agent/loop.py 的 run_turn。每一轮它把工具返回的结果回传给模型，"
             "由模型自己决定下一步调哪个工具，所以调用链不是我们写死的，是模型根据任务拆出来的。"
             "工具在独立线程里执行，超时和异常都被结构化捕获；每一步的输入、输出、耗时都会写进 "
             "tool_calls.jsonl，可以逐条复核。"),

    dict(kind="shots", dur=26, img=SHOT / "04_联网多工具链与结果.png",
         t="十二、联网：模型主动拒绝编造价格",
         quote="本地档案库里还没有这款车的价格，教师数据集不含价格，我不能凭记忆编一个给你。",
         say="注意这里——因为没开联网，模型的回答是：「本地档案库里还没有这款车的价格，"
             "教师数据集不含价格，我不能凭记忆编一个给你。」这是我们提示词里最重要的一条约束："
             "不确定时如实说明。打开联网后，它连续调用了 12 次工具：10 次搜索、4 次抓取汽车之家的配置页，"
             "中间用 save_car_trims 分两批存进知识库。最后拿到 2026 款奥迪 A6L 全部 10 个档次的厂商指导价，"
             "32.29 万到 55.89 万。"),

    dict(kind="shots", dur=22, img=SHOT / "05_unknown拒识路径.png",
         t="十三、unknown：给出候选，却主动否定候选",
         cands=[("蔚来 EC6", "0.37%"), ("大众 途岳", "0.32%"), ("Jeep 牧马人", "0.30%"),
                ("劳斯莱斯 库里南", "0.30%"), ("哈弗 H9", "0.25%")],
         say="再试一张不在 50 款里的车。模型判定 unknown，并且主动列出相似度最高的五个候选"
             "——蔚来 EC6、大众途岳、Jeep 牧马人等等。但紧接着它说：这几个分值都非常低、"
             "彼此差距也很小，基本等于都没匹配上，不能作为判断依据，所以也不能凭这个把它写进看车档案。"
             "给出候选却主动否定候选，这是我们希望看到的诚实行为。"),

    dict(kind="end", dur=20, t="交付物",
         items=[("实训报告", "8 章 · 11 张表 · 12 张图 · 全部数据可逐条追溯"),
                ("源码包", "4.9 MB · 220 个条目 · 不含模型权重与数据集"),
                ("提交物", "classes.txt · result.csv · result.npz · README.md")],
         tagline="让拍一张车照，成为一条结构化记录的起点。",
         say="最后是交付物。实训报告共八章、十一张表、十二张图，所有实验数据都能在项目里逐条追溯，"
             "从 outputs/runs 的训练日志到 tool_calls.jsonl 的调用记录。"
             "源码包 4.9 兆，不含模型权重和数据集，收到后按 README 装依赖即可运行。"
             "识车档案 CarDex，让拍一张车照成为一条结构化记录的起点。谢谢观看。"),
]


def build_payload() -> list[dict]:
    """把每个 slide 转成前端可直接渲染的结构。"""
    out = []
    for i, s in enumerate(SLIDES, 1):
        d = {"n": i, "dur": s["dur"], "t": s.get("t", ""), "say": s["say"],
             "kind": s["kind"]}
        for k in ("sub", "bullets", "points", "steps", "kpis", "cands",
                  "quote", "items", "tagline", "members"):
            if k in s:
                d[k] = s[k]
        if "img" in s:
            d["img"] = rel(s["img"])
        if "imgs" in s:
            d["imgs"] = [rel(p) for p in s["imgs"]]
        if "code" in s:
            ci = s["code"]
            if ci < len(CODE):
                c = CODE[ci]
                d["code"] = {"title": c["title"], "file": c["file"],
                             "start_line": c["start_line"], "html": c["html"],
                             "points": c["points"]}
        out.append(d)
    return out


HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>识车档案 CarDex · 演示动画</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; overflow:hidden; background:#0b0f16; color:#e8eef6;
              font-family:"Microsoft YaHei","Segoe UI",sans-serif; }
  #stage { position:fixed; inset:0 0 118px 0; display:flex;
           align-items:center; justify-content:center; }
  .slide { position:absolute; inset:0; display:none; padding:34px 46px 20px;
           flex-direction:column; }
  .slide.on { display:flex; }

  /* —— 入场动效：错峰淡入上浮 —— */
  .anim { opacity:0; transform:translateY(22px);
          animation:rise .62s cubic-bezier(.22,.9,.3,1) forwards; }
  @keyframes rise { to { opacity:1; transform:translateY(0); } }
  .d1{animation-delay:.05s} .d2{animation-delay:.20s} .d3{animation-delay:.35s}
  .d4{animation-delay:.50s} .d5{animation-delay:.65s} .d6{animation-delay:.80s}
  body.nofx .anim { animation:none; opacity:1; transform:none; }

  h1 { font-size:52px; letter-spacing:2px; }
  h2 { font-size:25px; font-weight:600; color:#79c0ff; margin-bottom:16px; }
  .sub { color:#8b98a9; font-size:16px; margin-top:12px; letter-spacing:1px; }
  .center { flex:1; display:flex; flex-direction:column;
            align-items:center; justify-content:center; text-align:center; }
  .members { display:flex; gap:22px; margin-top:30px; }
  .member { background:#121a26; border:1px solid #23303f; border-radius:12px;
            padding:18px 30px; min-width:250px; }
  .member .mn { font-size:23px; font-weight:700; color:#e8eef6; letter-spacing:3px; }
  .member .mr { color:#4F9BFF; font-size:15px; margin-top:9px; }
  .member .mi { color:#7d8b9c; font-size:13px; margin-top:5px;
                font-family:Consolas,monospace; }
  .bullets { list-style:none; margin-top:34px; }
  .bullets li { font-size:20px; color:#c7d3e0; margin:14px 0; }
  .bullets li b { color:#4F9BFF; margin-right:10px; }

  /* 图片：缓慢推近 */
  .imgwrap { flex:1; display:flex; align-items:center; justify-content:center;
             overflow:hidden; }
  .imgwrap img { max-width:100%; max-height:100%; border-radius:8px;
                 box-shadow:0 12px 46px rgba(0,0,0,.55); }
  .kb img { animation:kenburns 26s ease-out forwards; }
  @keyframes kenburns { from{transform:scale(1)} to{transform:scale(1.075)} }
  body.nofx .kb img { animation:none; }
  .duo { display:flex; gap:22px; flex:1; align-items:center; }
  .duo > div { flex:1; display:flex; align-items:center; justify-content:center;
               overflow:hidden; height:100%; }
  .duo img { max-width:100%; max-height:100%; border-radius:8px;
             box-shadow:0 10px 34px rgba(0,0,0,.5); }

  /* KPI */
  .kpis { display:flex; gap:26px; justify-content:center; margin-bottom:30px; }
  .kpi { background:#121a26; border:1px solid #23303f; border-radius:14px;
         padding:26px 34px; min-width:250px; text-align:center;
         box-shadow:0 10px 30px rgba(0,0,0,.4); }
  .kpi .v { font-size:40px; font-weight:700; letter-spacing:1px; }
  .kpi .l { color:#8b98a9; font-size:14px; margin-top:10px; }
  .pts { list-style:none; text-align:left; }
  .pts li { color:#c7d3e0; font-size:17px; margin:11px 0; padding-left:20px;
            position:relative; }
  .pts li:before { content:"▸"; position:absolute; left:0; color:#9BBB59; }

  /* 代码 */
  .codewrap { flex:1; display:flex; gap:18px; min-height:0; }
  .codemeta { width:270px; flex:none; background:#161b22; border:1px solid #26313d;
              border-radius:10px; padding:18px; overflow:hidden; }
  .codemeta h3 { font-size:15px; color:#79c0ff; margin-bottom:12px; line-height:1.5; }
  .codemeta .f { font:12px/1 Consolas,monospace; color:#8b98a9; margin-bottom:14px;
                 word-break:break-all; }
  .codemeta li { list-style:none; color:#adbac7; font-size:13px; margin:9px 0;
                 padding-left:15px; position:relative; line-height:1.6; }
  .codemeta li:before { content:"▸"; position:absolute; left:0; color:#3fb950; }
  .codebox { flex:1; background:#0d1117; border:1px solid #26313d; border-radius:10px;
             overflow-y:auto; overflow-x:hidden; position:relative;
             scrollbar-width:thin; scrollbar-color:#3c4756 #0d1117; }
  .codebox::-webkit-scrollbar { width:10px; }
  .codebox::-webkit-scrollbar-thumb { background:#3c4756; border-radius:5px; }
  .codebox .inner { position:relative; }
  .codebox table { border-collapse:collapse; width:100%; }
  .codebox td.ln { width:56px; text-align:right; padding-right:16px; color:#4b5563;
                   font:12.5px/1.55 Consolas,monospace; user-select:none; }
  .codebox td.src { font:12.5px/1.55 Consolas,"Cascadia Code",monospace;
                    white-space:pre; padding-right:22px; }
  .kw{color:#ff7b72} .st{color:#a5d6ff} .cm{color:#8b949e;font-style:italic}
  .nm{color:#79c0ff}

  /* 工具链步骤 */
  .steps { display:flex; flex-direction:column; gap:12px; margin-bottom:18px; }
  .step { display:flex; align-items:center; gap:16px; background:#121a26;
          border:1px solid #23303f; border-left:4px solid #4F81BD;
          border-radius:8px; padding:13px 18px; }
  .step .nm2 { font:600 15px/1 Consolas,monospace; color:#79c0ff; width:150px; }
  .step .ms { font:600 15px/1 Consolas,monospace; color:#9BBB59; width:80px; }
  .step .rs { color:#c7d3e0; font-size:15px; }

  .quote { background:#1a2230; border-left:4px solid #F79646; border-radius:8px;
           padding:16px 20px; color:#ffd9a8; font-size:17px; line-height:1.7;
           margin-bottom:16px; }
  .cands { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
  .cand { background:#121a26; border:1px solid #23303f; border-radius:8px;
          padding:9px 15px; font-size:15px; color:#c7d3e0; }
  .cand b { color:#C0504D; margin-left:9px; font-family:Consolas,monospace; }

  .cards { display:flex; gap:20px; }
  .card { flex:1; background:#121a26; border:1px solid #23303f; border-radius:12px;
          padding:24px; }
  .card h4 { color:#79c0ff; font-size:19px; margin-bottom:12px; }
  .card p { color:#adbac7; font-size:15px; line-height:1.75; }
  .tagline { text-align:center; font-size:26px; color:#4F9BFF; margin-top:36px;
             letter-spacing:2px; }

  /* —— 字幕条 —— */
  #bar { position:fixed; left:0; right:0; bottom:0; height:118px;
         background:linear-gradient(180deg,rgba(10,14,20,.9),rgba(10,14,20,.99));
         border-top:3px solid #4F81BD; padding:14px 26px 10px;
         display:flex; flex-direction:column; justify-content:center; }
  #say { font-size:19px; font-weight:600; line-height:1.55;
         max-height:62px; overflow:hidden; }
  #hint { font-size:12.5px; color:#7d8b9c; margin-top:7px; }
  #hint b { color:#9BBB59; }

  #prog { position:fixed; top:0; left:0; height:3px; background:#4F81BD;
          transition:width .35s ease; z-index:20; }
  #no { position:fixed; right:22px; top:16px; z-index:20; font:600 14px/1 monospace;
        color:#7CFC9B; background:rgba(10,14,20,.85); padding:8px 14px;
        border-radius:8px; }
  #dots { position:fixed; left:22px; top:18px; z-index:20; display:flex; gap:7px; }
  .dot { width:9px; height:9px; border-radius:50%; background:#2b3644; }
  .dot.on { background:#4F81BD; transform:scale(1.35); }
  .dot.done { background:#33506e; }
</style></head><body>
<div id="prog"></div>
<div id="dots"></div>
<div id="no"></div>
<div id="stage"></div>
<div id="bar">
  <div id="say"></div>
  <div id="hint">
    <b>空格</b> 下一段　<b>← →</b> 跳转　<b>R</b> 重播本段动效　
    <b>P</b> 暂停/恢复动画　<b>F</b> 全屏　<b>N</b> 关闭动效（录屏更稳）
  </div>
</div>
<script>
const SLIDES = __PAYLOAD__;
const stage = document.getElementById('stage');
const dots  = document.getElementById('dots');
const sayEl = document.getElementById('say');
const prog  = document.getElementById('prog');
const noEl  = document.getElementById('no');
let cur = -1, timer = null;

const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function renderSlide(s) {
  const d = ['d1','d2','d3','d4','d5','d6'];
  let h = '';
  const head = s.t ? `<h2 class="anim ${d[0]}">${esc(s.t)}</h2>` : '';
  if (s.kind === 'title') {
    h = `<div class="center">
      <h1 class="anim d1">${esc(s.t)}</h1>
      <div class="sub anim d2">${esc(s.sub||'')}</div>
      ${(s.members||[]).length ? `<div class="members anim d3">${s.members.map(m=>
        `<div class="member"><div class="mn">${esc(m[0])}</div>
         <div class="mr">${esc(m[1])}</div><div class="mi">${esc(m[2])}</div></div>`).join('')}</div>` : ''}
      <ul class="bullets anim d4">${(s.bullets||[]).map(b=>`<li>${esc(b)}</li>`).join('')}</ul>
    </div>`;
  } else if (s.kind === 'image') {
    h = head + `<div class="imgwrap kb anim d2"><img src="${s.img}" alt=""></div>`;
  } else if (s.kind === 'images') {
    h = head + `<div class="duo anim d2">` +
        (s.imgs||[]).map(p=>`<div><img src="${p}" alt=""></div>`).join('') + `</div>`;
  } else if (s.kind === 'kpi') {
    h = head + `<div class="center">
      <div class="kpis anim d2">${(s.kpis||[]).map(([v,l,c])=>
        `<div class="kpi"><div class="v" style="color:${c}">${esc(v)}</div>
         <div class="l">${esc(l)}</div></div>`).join('')}</div>
      <ul class="pts anim d3">${(s.points||[]).map(p=>`<li>${esc(p)}</li>`).join('')}</ul>
    </div>`;
  } else if (s.kind === 'code') {
    const c = s.code || {html:'', start_line:1, file:'', points:[]};
    const rows = String(c.html).split('\\n').map((l,k)=>
      `<tr><td class="ln">${(c.start_line||1)+k}</td><td class="src">${l||' '}</td></tr>`).join('');
    h = head + `<div class="codewrap anim d2">
      <div class="codemeta">
        <h3>${esc(c.title||'')}</h3>
        <div class="f">${esc(c.file||'')}</div>
        <ul>${(c.points||[]).map(p=>`<li>${esc(p)}</li>`).join('')}</ul>
      </div>
      <div class="codebox"><div class="inner"><table><tbody>${rows}</tbody></table></div></div>
    </div>`;
  } else if (s.kind === 'shots') {
    let extra = '';
    if (s.steps) {
      extra = `<div class="steps anim d3">${s.steps.map(([n,ms,r])=>
        `<div class="step"><span class="nm2">${esc(n)}</span>
         <span class="ms">${esc(ms)}</span><span class="rs">${esc(r)}</span></div>`).join('')}</div>`;
    }
    if (s.quote) extra = `<div class="quote anim d3">「${esc(s.quote)}」</div>`;
    if (s.cands) extra = `<div class="cands anim d3">${s.cands.map(([n,p])=>
        `<span class="cand">${esc(n)}<b>${esc(p)}</b></span>`).join('')}</div>`;
    h = head + extra + `<div class="imgwrap anim d4"><img src="${s.img}" alt=""></div>`;
  } else if (s.kind === 'end') {
    h = `<div class="center">
      <h2 class="anim d1">${esc(s.t)}</h2>
      <div class="cards anim d2">${(s.items||[]).map(([k,v])=>
        `<div class="card"><h4>${esc(k)}</h4><p>${esc(v)}</p></div>`).join('')}</div>
      <div class="tagline anim d3">${esc(s.tagline||'')}</div>
    </div>`;
  }
  return `<div class="slide on">${h}</div>`;
}

let autoScroll = true, manualHint = null;

function setManual(on) {
  autoScroll = !on;
  if (!manualHint) {
    manualHint = document.createElement('div');
    manualHint.style.cssText = 'position:fixed;right:22px;top:56px;z-index:30;' +
      'font:600 12px/1 "Microsoft YaHei",sans-serif;padding:6px 11px;border-radius:6px;' +
      'background:rgba(12,18,28,.9);color:#F79646;pointer-events:none';
    document.body.appendChild(manualHint);
  }
  manualHint.textContent = on ? '手动 · 按 A 恢复自动' : '';
  manualHint.style.color = on ? '#F79646' : '#7CFC9B';
}

function startCodeScroll(s) {
  const box = stage.querySelector('.codebox');
  if (!box) return;
  clearInterval(timer);
  if (!autoScroll) return;
  const secs = Math.max(6, (s.dur||20) - 4);        // 在念完前滚到底
  timer = setInterval(() => {
    if (!autoScroll) return clearInterval(timer);
    const max = Math.max(1, box.scrollHeight - box.clientHeight);
    if (box.scrollTop >= max - 1) return clearInterval(timer);
    box.scrollTop = Math.min(max, box.scrollTop + max / (secs * 60));
  }, 16);
}

/** 返回 true 表示已消费（仍在滚代码，不该翻页） */
function codeStep(ratio) {
  const box = stage.querySelector('.codebox');
  if (!box) return false;
  const max = Math.max(0, box.scrollHeight - box.clientHeight);
  if (box.scrollTop >= max - 2) return false;
  setManual(true);
  box.scrollTop = Math.min(max, box.scrollTop + Math.max(60, box.clientHeight * (ratio||0.9)));
  return true;
}

function go(i) {
  cur = Math.max(0, Math.min(SLIDES.length - 1, i));
  const s = SLIDES[cur];
  stage.innerHTML = renderSlide(s);
  sayEl.textContent = s.say;
  noEl.textContent = `${cur+1} / ${SLIDES.length}　${s.dur}s`;
  prog.style.width = ((cur+1) / SLIDES.length * 100) + '%';
  [...dots.children].forEach((d,k) => {
    d.className = 'dot' + (k === cur ? ' on' : (k < cur ? ' done' : ''));
  });
  clearInterval(timer);
  if (s.kind === 'code') setTimeout(() => startCodeScroll(s), 250);
}

SLIDES.forEach(() => dots.appendChild(Object.assign(document.createElement('i'),
  {className:'dot'})));

addEventListener('keydown', e => {
  const k = e.key;
  const ovApi = window.__DemoOverlay;
  // 代码段：方向键 / 空格 / PgUp / PgDn 先用来滚代码，滚到底才翻页
  if (ovApi && SLIDES[cur] && SLIDES[cur].kind === 'code') {
    if (k === 'ArrowDown' || k === 'PageDown') {
      if (ovApi.codeStep(0.9)) { e.preventDefault(); return; }
    } else if (k === 'ArrowUp' || k === 'PageUp') {
      const box = document.querySelector('#__ov .codebox');
      if (box && box.scrollTop > 2) {
        ovApi.setManual(true);
        box.scrollTop = Math.max(0, box.scrollTop - box.clientHeight * 0.9);
        e.preventDefault(); return;
      }
    } else if (k === ' ') {
      if (ovApi.codeStep(0.9)) { e.preventDefault(); return; }
    } else if (k.toLowerCase() === 'a') {
      ovApi.setManual(!ovApi.isAuto()); e.preventDefault(); return;
    }
  }
  if (k === ' ') { e.preventDefault(); go(cur + 1); }
  else if (k === 'ArrowRight' || k === 'PageDown') go(cur + 1);
  else if (k === 'ArrowLeft'  || k === 'PageUp')   go(cur - 1);
  else if (k.toLowerCase() === 'r') go(cur);
  else if (k.toLowerCase() === 'f') document.documentElement.requestFullscreen();
  else if (k.toLowerCase() === 'p') {
    document.body.classList.toggle('nofx');
    if (document.body.classList.contains('nofx')) clearInterval(timer);
    else if (SLIDES[cur].kind === 'code') startCodeScroll(SLIDES[cur]);
  }
  else if (k.toLowerCase() === 'n') document.body.classList.toggle('nofx');
  else if (/^[1-9]$/.test(k)) go(parseInt(k, 10) - 1);
});

// 人工操作立刻接管自动滚动
stage.addEventListener('wheel', e => {
  const b = stage.querySelector('.codebox');
  if (b && b.contains(e.target)) setManual(true);
}, { passive: true, capture: true });
stage.addEventListener('mousedown', e => {
  const b = stage.querySelector('.codebox');
  if (b && b.contains(e.target)) setManual(true);
}, true);

go(0);
</script></body></html>"""


OVERLAY_TPL = r"""
/* 识车档案 CarDex —— 演示浮层（共用模块）
   由 tools/make_presentation.py 生成，请勿手改。
   独立动画页与浏览器自动演示共用此文件，保证两边内容与观感完全一致。
   对外接口：
     window.__DemoOverlay.show(i)   显示第 i 段（0 基）
     window.__DemoOverlay.hide()    隐藏浮层
     window.__DemoOverlay.slides    全部段落数据
     window.__DemoOverlay.current   当前段号（-1 表示隐藏）        */
(function () {
  var SLIDES = __PAYLOAD__;
  if (window.__DemoOverlay) return;

  var css = document.createElement('style');
  css.textContent = `
  #__ov { position:fixed; inset:0; z-index:999998; display:none;
          background:#0b0f16; color:#e8eef6; overflow:hidden;
          font-family:"Microsoft YaHei","Segoe UI",sans-serif; }
  #__ov .slide { position:absolute; inset:0 0 118px 0; padding:30px 44px 14px;
                 display:flex; flex-direction:column; }
  #__ov .anim { opacity:0; transform:translateY(22px);
                animation:__ovrise .62s cubic-bezier(.22,.9,.3,1) forwards; }
  @keyframes __ovrise { to { opacity:1; transform:translateY(0); } }
  #__ov .d1{animation-delay:.05s} #__ov .d2{animation-delay:.20s}
  #__ov .d3{animation-delay:.35s} #__ov .d4{animation-delay:.50s}
  #__ov .d5{animation-delay:.65s} #__ov .d6{animation-delay:.80s}
  #__ov.nofx .anim { animation:none; opacity:1; transform:none; }
  #__ov h1 { font-size:50px; letter-spacing:2px; }
  #__ov h2 { font-size:24px; font-weight:600; color:#79c0ff; margin-bottom:15px; }
  #__ov .sub { color:#8b98a9; font-size:16px; margin-top:12px; letter-spacing:1px; }
  #__ov .center { flex:1; display:flex; flex-direction:column; align-items:center;
                  justify-content:center; text-align:center; }
  #__ov .members { display:flex; gap:22px; margin-top:28px; }
  #__ov .member { background:#121a26; border:1px solid #23303f; border-radius:12px;
                  padding:17px 28px; min-width:240px; }
  #__ov .member .mn { font-size:22px; font-weight:700; color:#e8eef6;
                      letter-spacing:3px; }
  #__ov .member .mr { color:#4F9BFF; font-size:15px; margin-top:9px; }
  #__ov .member .mi { color:#7d8b9c; font-size:13px; margin-top:5px;
                      font-family:Consolas,monospace; }
  #__ov .bullets { list-style:none; margin-top:32px; }
  #__ov .bullets li { font-size:20px; color:#c7d3e0; margin:13px 0; }
  #__ov .imgwrap { flex:1; display:flex; align-items:center; justify-content:center;
                   overflow:hidden; }
  #__ov .imgwrap img { max-width:100%; max-height:100%; border-radius:8px;
                       box-shadow:0 12px 46px rgba(0,0,0,.55); }
  #__ov .kb img { animation:__ovkb 26s ease-out forwards; }
  @keyframes __ovkb { from{transform:scale(1)} to{transform:scale(1.075)} }
  #__ov.nofx .kb img { animation:none; }
  #__ov .duo { display:flex; gap:22px; flex:1; align-items:center; }
  #__ov .duo > div { flex:1; display:flex; align-items:center; justify-content:center;
                     overflow:hidden; height:100%; }
  #__ov .duo img { max-width:100%; max-height:100%; border-radius:8px;
                   box-shadow:0 10px 34px rgba(0,0,0,.5); }
  #__ov .kpis { display:flex; gap:26px; justify-content:center; margin-bottom:28px; }
  #__ov .kpi { background:#121a26; border:1px solid #23303f; border-radius:14px;
               padding:24px 32px; min-width:240px; text-align:center;
               box-shadow:0 10px 30px rgba(0,0,0,.4); }
  #__ov .kpi .v { font-size:38px; font-weight:700; }
  #__ov .kpi .l { color:#8b98a9; font-size:14px; margin-top:10px; }
  #__ov .pts { list-style:none; text-align:left; }
  #__ov .pts li { color:#c7d3e0; font-size:17px; margin:10px 0; padding-left:20px;
                  position:relative; }
  #__ov .pts li:before { content:"\\25B8"; position:absolute; left:0; color:#9BBB59; }
  #__ov .codewrap { flex:1; display:flex; gap:18px; min-height:0; }
  #__ov .codemeta { width:266px; flex:none; background:#161b22;
                    border:1px solid #26313d; border-radius:10px; padding:18px;
                    overflow:hidden; }
  #__ov .codemeta h3 { font-size:15px; color:#79c0ff; margin-bottom:12px;
                       line-height:1.5; }
  #__ov .codemeta .f { font:12px/1 Consolas,monospace; color:#8b98a9;
                       margin-bottom:14px; word-break:break-all; }
  #__ov .codemeta li { list-style:none; color:#adbac7; font-size:13px; margin:9px 0;
                       padding-left:15px; position:relative; line-height:1.6; }
  #__ov .codemeta li:before { content:"\\25B8"; position:absolute; left:0; color:#3fb950; }
  #__ov .codebox { flex:1; background:#0d1117; border:1px solid #26313d;
                   border-radius:10px; overflow-y:auto; overflow-x:hidden;
                   position:relative; scrollbar-width:thin;
                   scrollbar-color:#3c4756 #0d1117; }
  #__ov .codebox::-webkit-scrollbar { width:10px; }
  #__ov .codebox::-webkit-scrollbar-thumb { background:#3c4756; border-radius:5px; }
  #__ov .codebox .inner { position:relative; }
  #__ov .codebox table { border-collapse:collapse; width:100%; }
  #__ov .codebox td.ln { width:56px; text-align:right; padding-right:16px;
                         color:#4b5563; font:12.5px/1.55 Consolas,monospace;
                         user-select:none; }
  #__ov .codebox td.src { font:12.5px/1.55 Consolas,"Cascadia Code",monospace;
                          white-space:pre; padding-right:22px; }
  #__ov .kw{color:#ff7b72} #__ov .st{color:#a5d6ff}
  #__ov .cm{color:#8b949e;font-style:italic} #__ov .nm{color:#79c0ff}
  #__ov .steps { display:flex; flex-direction:column; gap:12px; margin-bottom:16px; }
  #__ov .step { display:flex; align-items:center; gap:16px; background:#121a26;
                border:1px solid #23303f; border-left:4px solid #4F81BD;
                border-radius:8px; padding:13px 18px; }
  #__ov .step .nm2 { font:600 15px/1 Consolas,monospace; color:#79c0ff; width:150px; }
  #__ov .step .ms { font:600 15px/1 Consolas,monospace; color:#9BBB59; width:80px; }
  #__ov .step .rs { color:#c7d3e0; font-size:15px; }
  #__ov .quote { background:#1a2230; border-left:4px solid #F79646; border-radius:8px;
                 padding:16px 20px; color:#ffd9a8; font-size:17px; line-height:1.7;
                 margin-bottom:14px; }
  #__ov .cands { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
  #__ov .cand { background:#121a26; border:1px solid #23303f; border-radius:8px;
                padding:9px 15px; font-size:15px; color:#c7d3e0; }
  #__ov .cand b { color:#C0504D; margin-left:9px; font-family:Consolas,monospace; }
  #__ov .cards { display:flex; gap:20px; }
  #__ov .card { flex:1; background:#121a26; border:1px solid #23303f;
                border-radius:12px; padding:24px; }
  #__ov .card h4 { color:#79c0ff; font-size:19px; margin-bottom:12px; }
  #__ov .card p { color:#adbac7; font-size:15px; line-height:1.75; }
  #__ov .tagline { text-align:center; font-size:25px; color:#4F9BFF; margin-top:34px;
                   letter-spacing:2px; }
  #__ovbar { position:absolute; left:0; right:0; bottom:0; height:118px;
             background:linear-gradient(180deg,rgba(10,14,20,.92),rgba(10,14,20,.995));
             border-top:3px solid #4F81BD; padding:14px 26px 10px; display:flex;
             flex-direction:column; justify-content:center; }
  #__ovsay { font-size:19px; font-weight:600; line-height:1.55; max-height:62px;
             overflow:hidden; }
  #__ovno { position:absolute; right:22px; top:16px; font:600 14px/1 monospace;
            color:#7CFC9B; background:rgba(10,14,20,.85); padding:8px 14px;
            border-radius:8px; }
  #__ovdots { position:absolute; left:22px; top:18px; display:flex; gap:7px; }
  #__ovdots i { width:9px; height:9px; border-radius:50%; background:#2b3644; }
  #__ovdots i.on { background:#4F81BD; transform:scale(1.35); }
  #__ovdots i.done { background:#33506e; }
  `;
  document.head.appendChild(css);

  var ov = document.createElement('div');
  ov.id = '__ov';
  ov.innerHTML = '<div id="__ovdots"></div><div id="__ovno"></div>' +
                 '<div id="__ovstage" style="position:absolute;inset:0"></div>' +
                 '<div id="__ovbar"><div id="__ovsay"></div></div>';
  document.body.appendChild(ov);

  var stage = ov.querySelector('#__ovstage');
  var dots  = ov.querySelector('#__ovdots');
  var sayEl = ov.querySelector('#__ovsay');
  var noEl  = ov.querySelector('#__ovno');
  var timer = null;

  SLIDES.forEach(function () {
    dots.appendChild(Object.assign(document.createElement('i'), {className: ''}));
  });

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>]/g,
      function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]; });
  }

  function render(s) {
    var h = '', head = s.t ? '<h2 class="anim d1">' + esc(s.t) + '</h2>' : '';
    if (s.kind === 'title') {
      var mem = (s.members||[]).length
        ? '<div class="members anim d3">' + s.members.map(function (m) {
            return '<div class="member"><div class="mn">' + esc(m[0]) +
                   '</div><div class="mr">' + esc(m[1]) +
                   '</div><div class="mi">' + esc(m[2]) + '</div></div>';
          }).join('') + '</div>'
        : '';
      h = '<div class="center"><h1 class="anim d1">' + esc(s.t) + '</h1>' +
          '<div class="sub anim d2">' + esc(s.sub) + '</div>' + mem +
          '<ul class="bullets anim d4">' + (s.bullets||[]).map(function(b){
             return '<li>' + esc(b) + '</li>'; }).join('') + '</ul></div>';
    } else if (s.kind === 'image') {
      h = head + '<div class="imgwrap kb anim d2"><img src="' + s.img + '"></div>';
    } else if (s.kind === 'images') {
      h = head + '<div class="duo anim d2">' + (s.imgs||[]).map(function(p){
          return '<div><img src="' + p + '"></div>'; }).join('') + '</div>';
    } else if (s.kind === 'kpi') {
      h = head + '<div class="center"><div class="kpis anim d2">' +
          (s.kpis||[]).map(function(k){ return '<div class="kpi"><div class="v" style="color:' +
            k[2] + '">' + esc(k[0]) + '</div><div class="l">' + esc(k[1]) +
            '</div></div>'; }).join('') + '</div><ul class="pts anim d3">' +
          (s.points||[]).map(function(p){ return '<li>' + esc(p) + '</li>'; }).join('') +
          '</ul></div>';
    } else if (s.kind === 'code') {
      var c = s.code || {html:'', start_line:1, file:'', points:[]};
      var rows = String(c.html).split('\n').map(function(l, k){
        return '<tr><td class="ln">' + ((c.start_line||1) + k) + '</td>' +
               '<td class="src">' + (l || ' ') + '</td></tr>'; }).join('');
      h = head + '<div class="codewrap anim d2"><div class="codemeta"><h3>' +
          esc(c.title) + '</h3><div class="f">' + esc(c.file) + '</div><ul>' +
          (c.points||[]).map(function(p){ return '<li>' + esc(p) + '</li>'; }).join('') +
          '</ul></div><div class="codebox"><div class="inner"><table><tbody>' + rows +
          '</tbody></table></div></div></div>';
    } else if (s.kind === 'shots') {
      var extra = '';
      if (s.steps) extra = '<div class="steps anim d3">' + s.steps.map(function(t){
        return '<div class="step"><span class="nm2">' + esc(t[0]) +
               '</span><span class="ms">' + esc(t[1]) + '</span><span class="rs">' +
               esc(t[2]) + '</span></div>'; }).join('') + '</div>';
      if (s.quote) extra = '<div class="quote anim d3">\u300c' + esc(s.quote) + '\u300d</div>';
      if (s.cands) extra = '<div class="cands anim d3">' + s.cands.map(function(t){
        return '<span class="cand">' + esc(t[0]) + '<b>' + esc(t[1]) + '</b></span>';
        }).join('') + '</div>';
      h = head + extra + '<div class="imgwrap anim d4"><img src="' + s.img + '"></div>';
    } else if (s.kind === 'end') {
      h = '<div class="center"><h2 class="anim d1">' + esc(s.t) + '</h2>' +
          '<div class="cards anim d2">' + (s.items||[]).map(function(t){
            return '<div class="card"><h4>' + esc(t[0]) + '</h4><p>' + esc(t[1]) +
                   '</p></div>'; }).join('') + '</div><div class="tagline anim d3">' +
          esc(s.tagline) + '</div></div>';
    }
    return '<div class="slide">' + h + '</div>';
  }

  var autoScroll = true;          // 自动播放开关
  var manualHint = null;          // 「手动」角标

  function setManual(on) {
    autoScroll = !on;
    if (!manualHint) {
      manualHint = document.createElement('div');
      manualHint.style.cssText = 'position:absolute;right:14px;top:12px;z-index:5;' +
        'font:600 12px/1 "Microsoft YaHei",sans-serif;padding:6px 11px;border-radius:6px;' +
        'background:rgba(12,18,28,.9);color:#F79646;pointer-events:none';
      ov.appendChild(manualHint);
    }
    manualHint.textContent = on ? '手动 · 按 A 恢复自动' : '自动';
    manualHint.style.color = on ? '#F79646' : '#7CFC9B';
    clearTimeout(manualHint._t);
    if (!on) manualHint._t = setTimeout(function () { manualHint.textContent = ''; }, 2200);
  }

  function scrollCode(s) {
    var box = stage.querySelector('.codebox');
    if (!box) return;
    var secs = Math.max(6, (s.dur || 20) - 4);
    clearInterval(timer);
    if (!autoScroll) return;
    timer = setInterval(function () {
      if (!autoScroll || ov.style.display === 'none') return clearInterval(timer);
      var max = Math.max(1, box.scrollHeight - box.clientHeight);
      if (box.scrollTop >= max - 1) return clearInterval(timer);
      box.scrollTop = Math.min(max, box.scrollTop + max / (secs * 60));
    }, 16);
  }

  /** 供键盘/空格调用：返回 true 表示「已消费」（还在滚代码，不该翻页） */
  function codeStep(ratio) {
    var box = stage.querySelector('.codebox');
    if (!box) return false;
    var max = Math.max(0, box.scrollHeight - box.clientHeight);
    if (box.scrollTop >= max - 2) return false;         // 已到底 → 交给翻页
    setManual(true);
    box.scrollTop = Math.min(max,
      box.scrollTop + Math.max(60, box.clientHeight * (ratio || 0.9)));
    return true;
  }

  var api = {
    slides: SLIDES,
    current: -1,
    show: function (i) {
      if (i < 0 || i >= SLIDES.length) return false;
      var s = SLIDES[i];
      api.current = i;
      ov.style.display = 'block';
      stage.innerHTML = render(s);
      sayEl.textContent = s.say;
      noEl.textContent = (i + 1) + ' / ' + SLIDES.length + '\u3000' + s.dur + 's';
      Array.prototype.forEach.call(dots.children, function (d, k) {
        d.className = (k === i) ? 'on' : (k < i ? 'done' : '');
      });
      clearInterval(timer);
      if (s.kind === 'code') setTimeout(function () { scrollCode(s); }, 250);
      return true;
    },
    hide: function () {
      ov.style.display = 'none';
      api.current = -1;
      clearInterval(timer);
    },
    toggleFx: function () { ov.classList.toggle('nofx'); },
    replay: function () { if (api.current >= 0) api.show(api.current); }
  };
  // 人工操作立刻接管，避免和自动滚动打架
  ov.addEventListener('wheel', function (e) {
    if (e.target && e.target.closest && e.target.closest('.codebox')) setManual(true);
  }, { passive: true, capture: true });
  ov.addEventListener('mousedown', function (e) {
    var b = stage.querySelector('.codebox');
    if (b && b.contains(e.target)) setManual(true);
  }, true);

  api.setManual = setManual;
  api.codeStep = codeStep;
  api.isAuto = function () { return autoScroll; };

  // 浮层内部的键盘控制（独立页另有一份；两处都要有，因为
  // 浏览器自动演示加载的是本模块，拿不到独立页的监听器）。
  // 用 capture 抢在演示主循环前面处理，并用 stopPropagation 阻止它翻页。
  addEventListener('keydown', function (e) {
    if (ov.style.display === 'none') return;
    var s = SLIDES[api.current];
    if (!s || s.kind !== 'code') return;
    var k = e.key, box = stage.querySelector('.codebox');
    if (k === ' ' || k === 'ArrowDown' || k === 'PageDown') {
      if (api.codeStep(0.9)) { e.preventDefault(); e.stopPropagation(); }
    } else if (k === 'ArrowUp' || k === 'PageUp') {
      if (box && box.scrollTop > 2) {
        setManual(true);
        box.scrollTop = Math.max(0, box.scrollTop - box.clientHeight * 0.9);
        e.preventDefault(); e.stopPropagation();
      }
    } else if (k === 'Home') {
      if (box) { setManual(true); box.scrollTop = 0; e.preventDefault(); e.stopPropagation(); }
    } else if (k === 'End') {
      if (box) { setManual(true); box.scrollTop = box.scrollHeight;
                 e.preventDefault(); e.stopPropagation(); }
    } else if (k.toLowerCase() === 'a') {
      setManual(autoScroll); e.preventDefault(); e.stopPropagation();
    }
  }, true);
  window.__DemoOverlay = api;
})();
"""


def main() -> None:
    import shutil

    payload = build_payload()
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    # ── 把图片资源复制到 Web 能加载的位置 ──
    # 浏览器演示的浮层要通过 HTTP 取图，而 /uploads/ 映射的是 data/uploads/，
    # 所以把 figures 与 screenshots 复制过去。浮层模块里的路径也相应改成 /uploads/...
    assets = ROOT / "data" / "uploads" / "demo_assets"
    assets.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_dir in (FIG, SHOT):
        for f in src_dir.glob("*.png"):
            shutil.copy2(f, assets / f.name)
            copied += 1
    for s in payload:
        if "img" in s:
            s["img"] = "/uploads/demo_assets/" + Path(s["img"]).name
        if "imgs" in s:
            s["imgs"] = ["/uploads/demo_assets/" + Path(p).name for p in s["imgs"]]

    # ── 输出共用的浮层模块 ──
    # 这样「独立动画页」和「浏览器自动演示」用的是同一份内容与同一套渲染，
    # 不会出现两边不一致的问题（用户明确要求"东西要一样"）。
    overlay_js = OVERLAY_TPL.replace("__PAYLOAD__",
                                     json.dumps(payload, ensure_ascii=False))
    ov_path = DEMO / "demo_overlay.js"
    ov_path.write_text(overlay_js, encoding="utf-8")
    shutil.copy2(ov_path, ROOT / "data" / "uploads" / "demo_overlay.js")

    n_img = sum(1 for s in payload if s.get("img") or s.get("imgs"))
    n_code = sum(1 for s in payload if s.get("code"))
    total = sum(s["dur"] for s in payload)
    print(f"[写出] {OUT}")
    print(f"[写出] {ov_path}  （供浏览器自动演示加载）")
    print(f"  图片资源 {copied} 张 → {assets}")
    print(f"  {len(payload)} 段 ｜ 图片段 {n_img} ｜ 代码段 {n_code} ｜ "
          f"{OUT.stat().st_size/1024:.0f} KB")
    print(f"  合计时长 {total} 秒 = {total//60}:{total%60:02d}")


if __name__ == "__main__":
    main()
