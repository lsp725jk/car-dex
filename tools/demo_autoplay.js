/* ============================================================================
 * 识车档案 CarDex —— 演示视频自动播放脚本
 *
 * 用法：打开 http://127.0.0.1:5000/?demo=1&step=1
 *      （页面里的开关会自动加载本文件，不必开 F12 控制台）
 *
 * 内容与《演示动画.html》**完全一致**：同一份 15 段文稿、同一套浮层渲染
 * （共用 demo_overlay.js）。区别只在于：第 10/11/13/14 段会收起浮层，
 * 在真实的 Web 界面上跑一次智能体，用实机操作证明「真的跑起来了」。
 *
 * 控制键：空格 下一段 ｜ 空格（等待中）跳过等待 ｜ R 重播本段动效 ｜ N 关动效
 * 停止：  window.__carDexStop = true
 * ==========================================================================*/
(() => {
  'use strict';

  const CFG = {
    imageKnown:   '/uploads/demo_known_0632.jpg',
    imageUnknown: '/uploads/demo_unknown_0604_0022.jpg',
    question:     '这车多少钱？合适的话帮我记下来，再去网上看看有没有新改款',
    followUp:     '查一下它现在的价格和有没有新改款',
    askUnknown:   '这张是什么车？',
    waitSend:     120000,
    waitUpload:   10000,
    overlaySrc:   '/uploads/demo_overlay.js',
  };

  window.__carDexStop = false;

  // ── 主字幕条（只在实机操作段落显示；浮层段落用浮层自带的字幕条）──
  const bar = document.createElement('div');
  bar.id = '__demo_bar';
  bar.style.cssText = [
    'position:fixed', 'left:0', 'right:0', 'bottom:0', 'z-index:999999',
    'background:rgba(12,18,28,.93)', 'color:#fff', 'padding:14px 22px',
    'font:600 19px/1.5 "Microsoft YaHei",sans-serif', 'letter-spacing:.4px',
    'border-top:3px solid #4F81BD', 'pointer-events:none',
    'box-shadow:0 -6px 24px rgba(0,0,0,.35)',
  ].join(';');
  const line1 = document.createElement('div');
  const line2 = document.createElement('div');
  line2.style.cssText = 'font-weight:400;font-size:15px;color:#9fc4e8;margin-top:6px';
  bar.append(line1, line2);
  document.body.appendChild(bar);

  const clock = document.createElement('div');
  clock.style.cssText = [
    'position:fixed', 'right:16px', 'top:14px', 'z-index:1000000',
    'background:rgba(12,18,28,.85)', 'color:#7CFC9B', 'padding:6px 12px',
    'border-radius:6px', 'font:600 15px/1 monospace', 'pointer-events:none',
  ].join(';');
  document.body.appendChild(clock);

  const sleep = (ms) => new Promise((res) => {
    const t0 = Date.now();
    const tick = () => {
      if (window.__carDexStop) return res('stopped');
      if (Date.now() - t0 >= ms) return res('ok');
      requestAnimationFrame(tick);
    };
    tick();
  });

  let skip = false;
  addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') skip = true;
  }, true);

  const wait = async (ms) => {
    skip = false;
    const t0 = Date.now();
    while (Date.now() - t0 < ms && !skip && !window.__carDexStop) {
      await sleep(120);
      clock.textContent = '下一段 ' + Math.max(0, (ms - (Date.now() - t0)) / 1000).toFixed(1) + 's';
    }
    clock.textContent = '';
  };

  const waitSpace = async () => {
    if (!window.__DEMO_STEP) return;
    line2.textContent = '【单步模式】按空格继续下一段';
    clock.textContent = '暂停';
    await new Promise((res) => {
      const h = (e) => {
        if (e.code === 'Space') { removeEventListener('keydown', h, true); res(); }
      };
      addEventListener('keydown', h, true);
    });
    clock.textContent = '';
  };

  // ── 页面操作 ──
  const q = (s) => document.querySelector(s);
  const byText = (sel, txt) =>
    [...document.querySelectorAll(sel)].find((e) => (e.textContent || '').includes(txt));
  const waitForText = async (needle, timeout) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout && !window.__carDexStop) {
      if (document.body.innerText.includes(needle)) return true;
      await sleep(250);
    }
    return false;
  };
  const waitIdle = async (timeout) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout && !window.__carDexStop) {
      const t = document.body.innerText;
      if (!t.includes('调用中') && !t.includes('思考中')) return true;
      await sleep(300);
    }
    return false;
  };
  const setInput = (text) => {
    const el = q('#input');
    el.value = text;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const send = async () => {
    q('#btn-send').click();
    await sleep(1500);
    await waitIdle(CFG.waitSend);
  };
  const upload = async (url, name) => {
    const r = await fetch(url);
    const b = await r.blob();
    const f = new File([b], name || 'car.jpg', { type: 'image/jpeg' });
    const dt = new DataTransfer();
    dt.items.add(f);
    const input = q('#file');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return waitForText('图片已就绪', CFG.waitUpload);
  };
  const newSession = async () => {
    const b = byText('button,a,div[role=button]', '新建会话');
    if (b) { b.click(); await sleep(1200); }
  };
  const toggleWebSearch = async () => { q('#btn-websearch').click(); await sleep(1200); };
  const scrollChat = () => {
    // ⚠️ 只滚 #messages，绝不能动 window —— 实测滚 window 会把聊天区推出视野
    const m = document.getElementById('messages');
    if (m) m.scrollTop = m.scrollHeight;
    window.scrollTo(0, 0);
  };

  // ── 浮层：加载共用模块 ──
  const loadOverlay = () => new Promise((res) => {
    if (window.__DemoOverlay) return res(true);
    const s = document.createElement('script');
    s.src = CFG.overlaySrc + '?v=' + Date.now();
    s.onload = () => res(!!window.__DemoOverlay);
    s.onerror = () => res(false);
    document.body.appendChild(s);
  });

  // ── 分镜：15 段，slide = 浮层段号（0 基）；app = 实机操作 ──
  const SCRIPT = [
    { slide: 0,  dur: 20, say: '这是识车档案 CarDex，我们的视觉智能体综合实训项目……' },
    { slide: 1,  dur: 22, say: '先说数据。教师数据集一共 827 个车型类别……' },
    { slide: 2,  dur: 20, say: '选题有两条硬标准……' },
    { slide: 3,  dur: 20, say: '这是 classify_car 的实现，在 agent/tools/vision.py……' },
    { slide: 4,  dur: 22, say: '模型部分用 ResNet50 做迁移学习……' },
    { slide: 5,  dur: 22, say: '最终方案是三个互补模型等权集成……' },
    { slide: 6,  dur: 18, say: '错误结构很有意思……' },
    { slide: 7,  dur: 18, say: '评分口径我们严格对齐教师要求……' },
    { slide: 8,  dur: 20, say: '接下来演示智能体系统……' },

    // 第 10 段：收起浮层，实机上传车照 + 输入需求
    { dur: 18, say: '我们上传一张车照。大家留意这个图片预览——文件已经真实落盘到服务器，'
                 + '并且和当前会话绑定。然后输入一句完整的自然语言需求。',
      do: async () => { await upload(CFG.imageKnown, 'demo.jpg'); setInput(CFG.question); } },

    // 第 11 段：实机发送，跑出三工具链
    { dur: 24, say: '看调用记录。第一个工具 classify_car，识别出这是奥迪 A6，'
                 + '置信度 91.5%；接着 query_car_info 查档案；add_car_record 写进看车档案。',
      do: async () => { await send(); scrollChat(); } },

    { slide: 11, dur: 20, say: '这是智能体循环的实现，在 agent/loop.py 的 run_turn……' },

    // 第 13 段：实机开联网 + 12 次工具链
    { dur: 26, say: '注意这里——因为没开联网，模型的回答是「我不能凭记忆编一个给你」。'
                 + '打开联网后，它连续调用了 12 次工具，拿到全部 10 个档次的厂商指导价。',
      do: async () => { await toggleWebSearch(); setInput(CFG.followUp);
                        await send(); scrollChat(); } },

    // 第 14 段：实机组外车型，看 unknown 判定
    { dur: 22, say: '再试一张不在 50 款里的车。模型判定 unknown，并主动列出 Top-5 候选，'
                 + '紧接着说明这些候选没有参考价值，也拒绝写进档案。',
      do: async () => { await newSession(); await upload(CFG.imageUnknown, 'unknown.jpg');
                        setInput(CFG.askUnknown); await send(); scrollChat(); } },

    { slide: 14, dur: 20, say: '最后是交付物。实训报告共八章、十一张表、十二张图……' },
  ];

  // ── 主流程 ──
  const run = async () => {
    // ⚠️ 必须重置：上一轮用过 __carDexStop = true 之后不清掉，
    //    新一轮所有 wait() 会立即返回，整场 39 秒跑完且不报错。
    window.__carDexStop = false;

    const ok = await loadOverlay();
    console.log(ok ? '%c 演示浮层已加载' : '%c ⚠️ 浮层加载失败，将只跑实机段落',
                'background:#9BBB59;color:#fff');

    let total = 0;
    for (let i = 0; i < SCRIPT.length; i++) {
      const s = SCRIPT[i];
      if (window.__carDexStop) break;

      const isSlide = typeof s.slide === 'number';
      if (isSlide && ok) {
        window.__DemoOverlay.show(s.slide);
        bar.style.display = 'none';
        clock.textContent = '';
      } else {
        if (ok) window.__DemoOverlay.hide();
        bar.style.display = '';
        line1.textContent = `【${i + 1}/${SCRIPT.length}】${s.say}`;
        line2.textContent =
          `本段 ${s.dur}s ｜ 已用 ${(total / 1000).toFixed(0)}s ｜ 空格跳过等待`;
      }
      console.log(`%c 第 ${i + 1} 段 `, 'background:#9BBB59;color:#fff',
                  isSlide ? '【浮层】' : '【实机】', s.say.slice(0, 30) + '…');

      try { if (s.do) await s.do(); } catch (e) { console.warn('步骤出错：', e); }
      await wait(s.dur * 1000);
      total += s.dur * 1000;
      await waitSpace();
    }

    if (ok) window.__DemoOverlay.hide();
    bar.style.display = '';
    line1.textContent = '演示结束 ✅';
    line2.textContent = `总时长约 ${(total / 1000).toFixed(0)} 秒`;
    console.log('%c 演示结束 ', 'background:#C0504D;color:#fff;padding:4px 8px');
  };

  // 预热：首次识别要 20 秒加载权重，录前先跑一次
  window.warmup = async () => {
    console.log('预热中……首次识别约需 20 秒');
    await upload(CFG.imageKnown, 'warm.jpg');
    setInput('这是什么车？');
    await send();
    await newSession();
    console.log('%c 预热完成，可以开始录制 ', 'background:#7CFC9B;color:#000');
  };

  window.__SCRIPT = SCRIPT;
  run();
})();
