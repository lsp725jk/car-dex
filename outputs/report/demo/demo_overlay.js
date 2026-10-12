
/* 识车档案 CarDex —— 演示浮层（共用模块）
   由 tools/make_presentation.py 生成，请勿手改。
   独立动画页与浏览器自动演示共用此文件，保证两边内容与观感完全一致。
   对外接口：
     window.__DemoOverlay.show(i)   显示第 i 段（0 基）
     window.__DemoOverlay.hide()    隐藏浮层
     window.__DemoOverlay.slides    全部段落数据
     window.__DemoOverlay.current   当前段号（-1 表示隐藏）        */
(function () {
  var SLIDES = [{"n": 1, "dur": 32, "t": "识车档案 CarDex", "say": "大家好，我们是智能 234 班的罗烨嵘和焦超，指导老师是林俊武。我负责视觉模型的训练与 Web 前端，焦超负责智能体与后端。我们的项目叫识车档案 CarDex——用户上传一张车照，视觉模型识别车型，再由智能体查档案、比价格、写进个人看车记录，最终形成一份可检索、可积累的对话式档案。", "kind": "title", "sub": "视觉智能体综合实训 · 智能 234 · 指导教师 林俊武", "bullets": ["看懂车照 —— 自训练 51 类视觉模型", "查资料比价格 —— 12 个工具的智能体", "记住每一次看车 —— 可检索的个人档案"], "members": [["罗烨嵘", "视觉模型训练 · Web 前端", "202312903430"], ["焦超", "智能体 · 后端与测试", "202312903414"]]}, {"n": 2, "dur": 22, "t": "一、数据：827 个类别里怎么选出 50 款", "say": "先说数据。教师数据集一共 827 个车型类别、10 万 3 千多张图。我们从中选了两组：第一组是 50 款私人用户最常看的车型，作为识别目标，覆盖轿车、SUV、MPV、跑车、微面、轻客六种车身形态，共 37 个品牌。第二组是 98 个不在 50 款里的类别，用来构造 unknown 这一类——专门让模型学会说「我不认识」，而不是硬猜一个最像的。", "kind": "image", "img": "/uploads/demo_assets/fig1_数据集构成树状图.png"}, {"n": 3, "dur": 20, "t": "二、选题：两条硬标准", "say": "选题有两条硬标准。一是同品牌只取形态差异明显的两款，比如本田我们只留了缤智和奥德赛，避免近重复类别互相压制 Macro-F1；二是每一类的训练样本都在 80 张以上，样本量拉平，这样等权 Macro-F1 才不会被小类拖垮。最终 50 类的训练集 5534 张、验证集 1834 张，样本量差距控制在两倍以内。", "kind": "images", "imgs": ["/uploads/demo_assets/fig2_目标类车身大类分布饼图.png", "/uploads/demo_assets/fig3_各类别样本量柱状图.png"]}, {"n": 4, "dur": 20, "t": "三、视觉工具：classify_car 真实调用模型", "say": "这是 classify_car 的实现，在 agent/tools/vision.py。有一点我们特意做了保证：这里没有任何捷径，它真实加载 outputs/checkpoints 下的三模型集成权重，做完整的前向推理。第 151 行的 clf.predict_path 就是真正的那一次推理调用。图片不会进入大模型，模型拿到的只有识别结果。如果权重加载失败，它会抛 MODEL_NOT_LOADED，绝不返回伪造的结果。", "kind": "code", "code": {"title": "① classify_car：视觉识别确实由工具真实调用模型", "file": "agent/tools/vision.py", "start_line": 126, "html": "<span class=\"kw\">def</span> _handler(args: dict, ctx: ToolContext) -&gt; dict:\n    cfg = ctx.config\n    ctx.check_abort()\n\n    image_path = args.get(<span class=\"st\">&quot;image_path&quot;</span>) <span class=\"kw\">or</span> ctx.image_path\n    <span class=\"kw\">if</span> <span class=\"kw\">not</span> image_path:\n        <span class=\"kw\">raise</span> ToolError(\n            Code.DEPENDENCY_MISSING,\n            <span class=\"st\">&quot;没有可用的图片&quot;</span>,\n            hint=<span class=\"st\">&quot;用户还没有上传图片。请先请用户上传一张车辆照片，再调用本工具。&quot;</span>,\n        )\n\n    p = Path(image_path)\n    <span class=\"kw\">if</span> <span class=\"kw\">not</span> p.is_absolute():\n        p = cfg.project_root / p\n    <span class=\"kw\">if</span> <span class=\"kw\">not</span> p.exists():\n        <span class=\"kw\">raise</span> ToolError(Code.IMAGE_NOT_FOUND, f<span class=\"st\">&quot;图片不存在：{p}&quot;</span>,\n                        hint=<span class=\"st\">&quot;请确认图片已上传成功，或重新上传&quot;</span>)\n\n    clf = get_classifier(cfg)\n    ctx.check_abort()\n\n    <span class=\"kw\">try</span>:\n        <span class=\"cm\"># 多要一位：下面会把 unknown 那一行滤掉，所以按 top_k+1 取，</span>\n        <span class=\"cm\"># 保证返回给模型的永远是 top_k 个**真实车型**候选。</span>\n        res = clf.predict_path(p, top_k=int(args.get(<span class=\"st\">&quot;top_k&quot;</span>) <span class=\"kw\">or</span> cfg.top_k) + <span class=\"nm\">1</span>)\n    <span class=\"kw\">except</span> ToolError:\n        <span class=\"kw\">raise</span>\n    <span class=\"kw\">except</span> Exception <span class=\"kw\">as</span> e:                <span class=\"cm\"># noqa: BLE001</span>\n        <span class=\"kw\">raise</span> ToolError(Code.IMAGE_UNREADABLE, f<span class=\"st\">&quot;图片读取或推理失败：{type(e).__name__}: {e}&quot;</span>,\n                        hint=<span class=\"st\">&quot;请确认文件是有效的图片格式（jpg/png），或重新上传&quot;</span>) <span class=\"kw\">from</span> e\n\n    conf = float(res[<span class=\"st\">&quot;confidence&quot;</span>])\n    is_unknown = bool(res[<span class=\"st\">&quot;is_unknown&quot;</span>])\n    want_k = int(args.get(<span class=\"st\">&quot;top_k&quot;</span>) <span class=\"kw\">or</span> cfg.top_k)\n    <span class=\"cm\"># unknown 不是一个&quot;候选车型&quot;，它是&quot;以上都不是&quot;这个结论本身，</span>\n    <span class=\"cm\"># 因此不放进 top_candidates —— 低置信度时让用户确认的是具体车型。</span>\n    top = [\n        {\n            <span class=\"st\">&quot;car_id&quot;</span>: t[<span class=\"st\">&quot;label_id&quot;</span>],\n            <span class=\"st\">&quot;car_name&quot;</span>: t[<span class=\"st\">&quot;label_name&quot;</span>],\n            <span class=\"st\">&quot;confidence&quot;</span>: round(float(t[<span class=\"st\">&quot;probability&quot;</span>]), <span class=\"nm\">6</span>),\n        }\n        <span class=\"kw\">for</span> t <span class=\"kw\">in</span> res.get(<span class=\"st\">&quot;top_k&quot;</span>, [])\n        <span class=\"kw\">if</span> t.get(<span class=\"st\">&quot;label_id&quot;</span>) != <span class=\"st\">&quot;unknown&quot;</span>\n    ][:want_k]\n\n    <span class=\"kw\">return</span> {\n        <span class=\"st\">&quot;ok&quot;</span>: <span class=\"kw\">True</span>,\n        <span class=\"st\">&quot;car_id&quot;</span>: res[<span class=\"st\">&quot;label_id&quot;</span>],\n        <span class=\"st\">&quot;car_name&quot;</span>: res[<span class=\"st\">&quot;label_name&quot;</span>],\n        <span class=\"st\">&quot;confidence&quot;</span>: round(conf, <span class=\"nm\">6</span>),\n        <span class=\"st\">&quot;is_unknown&quot;</span>: is_unknown,\n        <span class=\"st\">&quot;low_confidence&quot;</span>: (<span class=\"kw\">not</span> is_unknown) <span class=\"kw\">and</span> conf &lt; cfg.display_confidence_threshold,\n        <span class=\"st\">&quot;top_candidates&quot;</span>: top,\n        <span class=\"st\">&quot;image&quot;</span>: {<span class=\"st\">&quot;path&quot;</span>: str(p), <span class=\"st\">&quot;file_name&quot;</span>: p.name},\n        <span class=\"st\">&quot;model&quot;</span>: {\n            <span class=\"st\">&quot;name&quot;</span>: model_label(cfg),\n            <span class=\"st\">&quot;tta_hflip&quot;</span>: cfg.vision_tta,\n            <span class=\"st\">&quot;prior_adjust_tau&quot;</span>: cfg.prior_adjust_tau,\n            <span class=\"st\">&quot;unknown_threshold&quot;</span>: cfg.unknown_threshold,\n            <span class=\"st\">&quot;display_confidence_threshold&quot;</span>: cfg.display_confidence_threshold,\n        },\n        <span class=\"st\">&quot;decision&quot;</span>: _decide(is_unknown, conf, cfg),\n        <span class=\"st\">&quot;note&quot;</span>: (\n            <span class=\"st\">&quot;confidence 是 softmax 最大概率，且已应用先验校正（τ）与拒识阈值（t），&quot;</span>\n            <span class=\"st\">&quot;两者均由验证集扫描确定；阈值推导见 outputs/reports/final_model_choice.json &quot;</span>\n            <span class=\"st\">&quot;与 agent/data/confidence_threshold.json&quot;</span>\n        ),\n    }\n\n\nVISION_TOOLS: list[ToolDefinition] = [\n    ToolDefinition(\n        name=<span class=\"st\">&quot;classify_car&quot;</span>,\n        description=(\n            <span class=\"st\">&quot;识别图片中的车型。真实加载并调用本组自训练的视觉分类模型，&quot;</span>\n            <span class=\"st\">&quot;返回 50 款目标车型之一的四位类别 ID、车型名称与置信度；&quot;</span>\n            <span class=\"st\">&quot;若判定不属于这 50 款，则返回 car_id=\\&quot;</span>unknown\\<span class=\"st\">&quot;。&quot;</span>\n            <span class=\"st\">&quot;当用户上传了车辆照片并要求识别、或后续操作需要知道车型时调用本工具。&quot;</span>\n            <span class=\"st\">&quot;本工具只接受文本参数，图片由用户在 Web 界面上传，会话中已绑定最近一次上传的图片。&quot;</span>\n        ),\n        parameters={\n            <span class=\"st\">&quot;type&quot;</span>: <span class=\"st\">&quot;object&quot;</span>,\n            <span class=\"st\">&quot;properties&quot;</span>: {\n                <span class=\"st\">&quot;image_path&quot;</span>: {", "points": ["真实加载 outputs/checkpoints 下的三模型集成权重", "第 151 行 clf.predict_path() 是真正的前向推理调用", "模型不可用时抛 MODEL_NOT_LOADED，绝不返回伪造结果"]}}, {"n": 5, "dur": 22, "t": "四、模型：分辨率是最大的杠杆", "say": "模型部分用 ResNet50 做迁移学习，一共跑了 15 组对照实验。最大的发现是：分辨率是这里最有效的杠杆。从 224 提到 288、352、416，每一步都有稳定收益，单模型从 0.9571 一路提到 0.9788。相比之下，单纯换更深的网络只提升不到一个百分点；而逆频率类别加权是个负结果，0.8428，反而比不加权的基线 0.8475 还低——因为我们的类别样本量差距本来就不大，加权只是放大了噪声。", "kind": "image", "img": "/uploads/demo_assets/fig4_实验对比柱状图.png"}, {"n": 6, "dur": 22, "t": "五、结果：0.9845，并用 5 折交叉验证自证", "say": "最终方案是三个互补模型等权集成——ResNet50@416、ResNet50@288 加监督对比学习、以及 Wide-ResNet50-2@288，再做一次先验校正。验证集上 51 类 Macro-F1 达到 0.9845。为了防止这个数字是「挑出来的」，我们又做了 5 折交叉验证，结果是 0.9812 加减 0.0059，与直接报告的只差万分之三，说明模型选择带来的乐观偏差可以忽略。", "kind": "kpi", "points": ["集成成员：ResNet50@416 + ResNet50@288+SupCon + Wide-ResNet50-2@288", "先验校正 τ = 0.2，由验证集网格搜索确定", "结论：直接报告的分数不是「挑出来的」"], "kpis": [["0.9845", "验证集 51 类 Macro-F1", "#4F81BD"], ["0.9812 ± 0.0059", "5 折交叉验证", "#9BBB59"], ["−0.0003", "乐观偏差（可忽略）", "#F79646"]]}, {"n": 7, "dur": 18, "t": "六、错误结构：瓶颈不在分类，在 unknown", "say": "错误结构很有意思。2422 张验证集上一共错了 51 张，其中模型几乎不会把一款车认成另一款——已知类之间的混淆只有 4 张。剩下 47 张错误全部集中在「已知还是未知」这条边界上：25 张是组外车型被误认成已知类，22 张是已知类被误拒。这说明模型的分类能力已经足够，瓶颈在 unknown 的建模方式。", "kind": "image", "img": "/uploads/demo_assets/fig5_验证集错误构成饼图.png"}, {"n": 8, "dur": 18, "t": "七、评分口径：51 类等权 Macro-F1", "say": "评分口径我们严格对齐教师要求：51 类等权 Macro-F1，unknown 作为第 51 类参与计算，与 50 个已知类权重完全相同。这里用的是 sklearn 的 f1_score 加 average='macro'，保证本地分数可以直接和线上评测对照，不会出现口径不一致导致的误判。", "kind": "code", "code": {"title": "⑤ 评分口径：51 类 Macro-F1（含 unknown）", "file": "src/cardex/metrics.py", "start_line": 25, "html": "<span class=\"kw\">def</span> compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -&gt; dict:\n    <span class=\"st\">&quot;&quot;&quot;按任务书 4.3 节口径计算指标。\n\n    评分指标：51 类 Macro-F1（等权平均，分母为 0 的类别 F1 记 0）。\n    辅助指标：Accuracy、每类 P/R/F1、混淆矩阵、unknown 检出率、已知类误拒率。\n    &quot;&quot;&quot;</span>\n    y_true = np.asarray(y_true)\n    y_pred = np.asarray(y_pred)\n    n_classes = len(label_names)\n    unknown_index = n_classes - <span class=\"nm\">1</span>\n\n    labels = list(range(n_classes))\n    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average=<span class=\"st\">&quot;macro&quot;</span>, zero_division=<span class=\"nm\">0</span>))\n    accuracy = float(accuracy_score(y_true, y_pred))\n\n    precision, recall, f1, support = precision_recall_fscore_support(\n        y_true, y_pred, labels=labels, zero_division=<span class=\"nm\">0</span>\n    )\n    cm = confusion_matrix(y_true, y_pred, labels=labels)\n\n    <span class=\"cm\"># unknown 检出率：真实 unknown 中被正确判为 unknown 的比例。</span>\n    <span class=\"cm\"># ⚠️ 用自定义小数据集微调时，验证集里**可能一个 unknown 样本都没有** ——</span>\n    <span class=\"cm\"># 那种情况下这个指标是&quot;无定义&quot;，返回 None 让调用方显示成 `—`，</span>\n    <span class=\"cm\"># 而不是 NaN（NaN 会污染日志、也会让下游的排序/比较莫名其妙地失效）。</span>\n    unk_true = y_true == unknown_index\n    unknown_detection_rate = (\n        float((y_pred[unk_true] == unknown_index).mean()) <span class=\"kw\">if</span> unk_true.any() <span class=\"kw\">else</span> <span class=\"kw\">None</span>\n    )\n    <span class=\"cm\"># 已知类误拒率：真实已知类中被误判为 unknown 的比例</span>\n    known_true = ~unk_true\n    known_false_reject_rate = (\n        float((y_pred[known_true] == unknown_index).mean()) <span class=\"kw\">if</span> known_true.any() <span class=\"kw\">else</span> <span class=\"kw\">None</span>\n    )\n\n    per_class = []\n    <span class=\"kw\">for</span> i, name <span class=\"kw\">in</span> enumerate(label_names):\n        per_class.append(\n            {\n                <span class=\"st\">&quot;index&quot;</span>: i,\n                <span class=\"st\">&quot;label&quot;</span>: name,\n                <span class=\"st\">&quot;precision&quot;</span>: float(precision[i]),\n                <span class=\"st\">&quot;recall&quot;</span>: float(recall[i]),\n                <span class=\"st\">&quot;f1&quot;</span>: float(f1[i]),\n                <span class=\"st\">&quot;support&quot;</span>: int(support[i]),\n            }\n        )\n\n    <span class=\"kw\">return</span> {\n        <span class=\"st\">&quot;macro_f1&quot;</span>: macro_f1,\n        <span class=\"st\">&quot;accuracy&quot;</span>: accuracy,\n        <span class=\"st\">&quot;unknown_detection_rate&quot;</span>: unknown_detection_rate,\n        <span class=\"st\">&quot;known_false_reject_rate&quot;</span>: known_false_reject_rate,\n        <span class=\"st\">&quot;per_class&quot;</span>: per_class,\n        <span class=\"st\">&quot;confusion_matrix&quot;</span>: cm.tolist(),\n        <span class=\"st\">&quot;report_text&quot;</span>: classification_report(\n            y_true, y_pred, labels=labels, target_names=label_names, zero_division=<span class=\"nm\">0</span>, digits=<span class=\"nm\">3</span>\n        ),\n        <span class=\"st\">&quot;n_samples&quot;</span>: int(len(y_true)),\n    }\n\n", "points": ["51 类等权，unknown 与 50 个已知类权重完全相同", "与教师评测口径一致，本地分数可直接对照线上得分"]}}, {"n": 9, "dur": 20, "t": "八、系统：三个任务怎么串起来", "say": "接下来演示智能体系统。界面分三块：左边是会话列表，每次对话都独立保存，可以随时切回；中间是对话主区；右边是运行状态和看车档案面板，能看到当前用的是哪个视觉模型、拒识阈值多少、以及 12 个工具的清单。底部是图片上传和联网搜索开关。整个系统接的是真实 DeepSeek-V4-Flash 接口，不是模拟。", "kind": "image", "img": "/uploads/demo_assets/fig6_系统总体架构图.png"}, {"n": 10, "dur": 18, "t": "九、输入：上传车照 + 一句自然语言", "say": "我们上传一张车照。大家留意这个图片预览——文件已经真实落盘到服务器，并且和当前会话绑定，750 乘 500，58 KB。然后输入一句完整的自然语言需求：「这车多少钱？合适的话帮我记下来，再去网上看看有没有新改款。」这句话里其实藏了三个任务，我们看智能体怎么拆。", "kind": "shots", "img": "/uploads/demo_assets/02_上传图片就绪.png"}, {"n": 11, "dur": 24, "t": "十、识别与落库：698ms → 2ms → 11ms", "say": "看调用记录。第一个工具 classify_car，花了 698 毫秒，识别出这是奥迪 A6，置信度 91.5%。这里说明一下，第一次调用要 20 秒，因为要加载三个模型的权重；我们提前预热过，所以现在是 698 毫秒。接着 query_car_info 查档案，2 毫秒；add_car_record 写进看车档案，11 毫秒，记录号 112。三步的输入、输出和耗时都能点开核对。", "kind": "shots", "steps": [["classify_car", "698 ms", "奥迪 A6 · 0632 · 置信度 91.5%"], ["query_car_info", "2 ms", "奥迪 ｜ 轿车"], ["add_car_record", "11 ms", "已写入看车档案 · 记录 #112"]], "img": "/uploads/demo_assets/03_三工具调用链完成.png"}, {"n": 12, "dur": 20, "t": "十一、智能体循环：调用链由模型自己拆", "say": "这是智能体循环的实现，在 agent/loop.py 的 run_turn。每一轮它把工具返回的结果回传给模型，由模型自己决定下一步调哪个工具，所以调用链不是我们写死的，是模型根据任务拆出来的。工具在独立线程里执行，超时和异常都被结构化捕获；每一步的输入、输出、耗时都会写进 tool_calls.jsonl，可以逐条复核。", "kind": "code", "code": {"title": "③ 智能体循环：模型决策 → 执行工具 → 结果回传", "file": "agent/loop.py", "start_line": 269, "html": "    <span class=\"kw\">def</span> run_turn(self, session: Session, user_text: str,\n                 image_path: str | <span class=\"kw\">None</span> = <span class=\"kw\">None</span>,\n                 on_event: EventCallback | <span class=\"kw\">None</span> = <span class=\"kw\">None</span>) -&gt; TurnResult:\n        turn_id = f<span class=\"st\">&quot;t-{uuid.uuid4().hex[:8]}&quot;</span>\n        t0 = time.perf_counter()\n        cfg = self.config\n        records: list[ToolCallRecord] = []\n\n        <span class=\"kw\">def</span> emit(kind: str, **payload: Any) -&gt; <span class=\"kw\">None</span>:\n            <span class=\"kw\">if</span> on_event <span class=\"kw\">is</span> <span class=\"kw\">None</span>:\n                <span class=\"kw\">return</span>\n            <span class=\"kw\">try</span>:\n                on_event({<span class=\"st\">&quot;type&quot;</span>: kind, <span class=\"st\">&quot;turn_id&quot;</span>: turn_id, **payload})\n            <span class=\"kw\">except</span> Exception:\n                <span class=\"kw\">pass</span>\n\n        session.add_user(user_text, image_path)\n        session.turns += <span class=\"nm\">1</span>\n\n        <span class=\"cm\"># turn 快照：本轮生效的策略与资源（对标 Codex 的 turn_context）</span>\n        snapshot = {\n            <span class=\"st\">&quot;turn_id&quot;</span>: turn_id,\n            <span class=\"st\">&quot;index&quot;</span>: session.turns,\n            <span class=\"st\">&quot;image&quot;</span>: _name_of(image_path <span class=\"kw\">or</span> session.image_path),\n            <span class=\"st\">&quot;unknown_threshold&quot;</span>: cfg.unknown_threshold,\n            <span class=\"st\">&quot;display_confidence_threshold&quot;</span>: cfg.display_confidence_threshold,\n            <span class=\"st\">&quot;vision_model&quot;</span>: [p.parent.name <span class=\"kw\">for</span> p <span class=\"kw\">in</span> cfg.checkpoint_paths() <span class=\"kw\">if</span> p.exists()],\n            <span class=\"st\">&quot;tools&quot;</span>: self.registry.names(),\n            <span class=\"st\">&quot;offline&quot;</span>: bool(cfg.offline),\n            <span class=\"st\">&quot;max_rounds&quot;</span>: cfg.max_rounds,\n        }\n        session.append_event(<span class=\"st\">&quot;turn_context&quot;</span>, snapshot)\n        emit(<span class=\"st\">&quot;turn_start&quot;</span>, turn=snapshot, user_text=user_text)\n\n        reply, rounds, usage = <span class=\"st\">&quot;&quot;</span>, <span class=\"nm\">0</span>, {}\n        ok, err_code, err_msg = <span class=\"kw\">True</span>, <span class=\"st\">&quot;&quot;</span>, <span class=\"st\">&quot;&quot;</span>\n        err_info: dict = {}\n        self._last_rounds = <span class=\"nm\">0</span>\n\n        <span class=\"kw\">with</span> self._lock_for(session.session_id):\n            <span class=\"kw\">try</span>:\n                reply, rounds, usage = self._run_steps(session, turn_id, records, emit)\n            <span class=\"kw\">except</span> AgentError <span class=\"kw\">as</span> e:\n                ok, err_code, err_msg = <span class=\"kw\">False</span>, e.code, e.message\n                err_info = dict(e.info <span class=\"kw\">or</span> {})\n                rounds = self._last_rounds\n                session.append_event(<span class=\"st\">&quot;error&quot;</span>, {<span class=\"st\">&quot;turn_id&quot;</span>: turn_id, <span class=\"st\">&quot;code&quot;</span>: e.code,\n                                               <span class=\"st\">&quot;message&quot;</span>: e.message, <span class=\"st\">&quot;info&quot;</span>: e.info})\n                emit(<span class=\"st\">&quot;error&quot;</span>, code=e.code, message=e.message, info=e.info)\n            <span class=\"kw\">except</span> Exception <span class=\"kw\">as</span> e:                        <span class=\"cm\"># noqa: BLE001</span>\n                ok, err_code, err_msg = <span class=\"kw\">False</span>, <span class=\"st\">&quot;INTERNAL_ERROR&quot;</span>, f<span class=\"st\">&quot;{type(e).__name__}: {e}&quot;</span>\n                rounds = self._last_rounds\n                session.append_event(<span class=\"st\">&quot;error&quot;</span>, {<span class=\"st\">&quot;turn_id&quot;</span>: turn_id, <span class=\"st\">&quot;code&quot;</span>: err_code,\n                                               <span class=\"st\">&quot;message&quot;</span>: err_msg})\n                emit(<span class=\"st\">&quot;error&quot;</span>, code=err_code, message=err_msg)\n\n        elapsed = round((time.perf_counter() - t0) * <span class=\"nm\">1000</span>, <span class=\"nm\">1</span>)\n        session.append_event(<span class=\"st\">&quot;turn_end&quot;</span>, {\n            <span class=\"st\">&quot;turn_id&quot;</span>: turn_id, <span class=\"st\">&quot;turn_index&quot;</span>: session.turns, <span class=\"st\">&quot;ok&quot;</span>: ok, <span class=\"st\">&quot;reply&quot;</span>: reply,\n            <span class=\"st\">&quot;rounds&quot;</span>: rounds, <span class=\"st\">&quot;elapsed_ms&quot;</span>: elapsed,\n            <span class=\"st\">&quot;tool_calls&quot;</span>: [r.brief() <span class=\"kw\">for</span> r <span class=\"kw\">in</span> records],\n            <span class=\"st\">&quot;tools_used&quot;</span>: [r.name <span class=\"kw\">for</span> r <span class=\"kw\">in</span> records],\n        })\n        emit(<span class=\"st\">&quot;turn_end&quot;</span>, ok=ok, reply=reply, rounds=rounds, elapsed_ms=elapsed,\n             tool_calls=len(records))\n\n        <span class=\"kw\">return</span> TurnResult(\n            turn_id=turn_id, reply=reply, tool_calls=records, rounds=rounds,\n            elapsed_ms=elapsed, ok=ok, error_code=err_code, error_message=err_msg,\n            error_info=err_info, simulated=self.is_offline, usage=usage,\n        )\n\n    <span class=\"cm\"># ------------------------------------------------------------------ #</span>", "points": ["工具在独立线程执行，超时与异常都被结构化捕获", "每步的输入、输出、耗时写入 tool_calls.jsonl，可逐条复核"]}}, {"n": 13, "dur": 26, "t": "十二、联网：模型主动拒绝编造价格", "say": "注意这里——因为没开联网，模型的回答是：「本地档案库里还没有这款车的价格，教师数据集不含价格，我不能凭记忆编一个给你。」这是我们提示词里最重要的一条约束：不确定时如实说明。打开联网后，它连续调用了 12 次工具：10 次搜索、4 次抓取汽车之家的配置页，中间用 save_car_trims 分两批存进知识库。最后拿到 2026 款奥迪 A6L 全部 10 个档次的厂商指导价，32.29 万到 55.89 万。", "kind": "shots", "quote": "本地档案库里还没有这款车的价格，教师数据集不含价格，我不能凭记忆编一个给你。", "img": "/uploads/demo_assets/04_联网多工具链与结果.png"}, {"n": 14, "dur": 22, "t": "十三、unknown：给出候选，却主动否定候选", "say": "再试一张不在 50 款里的车。模型判定 unknown，并且主动列出相似度最高的五个候选——蔚来 EC6、大众途岳、Jeep 牧马人等等。但紧接着它说：这几个分值都非常低、彼此差距也很小，基本等于都没匹配上，不能作为判断依据，所以也不能凭这个把它写进看车档案。给出候选却主动否定候选，这是我们希望看到的诚实行为。", "kind": "shots", "cands": [["蔚来 EC6", "0.37%"], ["大众 途岳", "0.32%"], ["Jeep 牧马人", "0.30%"], ["劳斯莱斯 库里南", "0.30%"], ["哈弗 H9", "0.25%"]], "img": "/uploads/demo_assets/05_unknown拒识路径.png"}, {"n": 15, "dur": 20, "t": "交付物", "say": "最后是交付物。实训报告共八章、十一张表、十二张图，所有实验数据都能在项目里逐条追溯，从 outputs/runs 的训练日志到 tool_calls.jsonl 的调用记录。源码包 4.9 兆，不含模型权重和数据集，收到后按 README 装依赖即可运行。识车档案 CarDex，让拍一张车照成为一条结构化记录的起点。谢谢观看。", "kind": "end", "items": [["实训报告", "8 章 · 11 张表 · 12 张图 · 全部数据可逐条追溯"], ["源码包", "4.9 MB · 220 个条目 · 不含模型权重与数据集"], ["提交物", "classes.txt · result.csv · result.npz · README.md"]], "tagline": "让拍一张车照，成为一条结构化记录的起点。"}];
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
