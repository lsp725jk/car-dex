# 14 · GPU 加速实战（从"有卡用不上"到 10× 提速）

> 这篇记录一件很典型的事：**机器上明明有块 RTX 4060，训练却一直跑在 CPU 上**。
> 从定位原因、改造代码、到排查性能瓶颈，全过程连同踩的坑一并写下来。
>
> 这不是"装个包就完事"——中途踩到 4 个真坑，其中两个会直接把程序搞崩。

---

## 1. 起点：有卡，但用不上

一开始的自检结论是"CPU-only"：

```
torch 版本    : 2.13.0+cpu
编译时 CUDA   : None          ← 关键
CUDA 可用     : False
设备数        : 0
```

而硬件其实是：

| 项目 | 值 |
| --- | --- |
| GPU | **NVIDIA GeForce RTX 4060 Laptop GPU** |
| 显存 | 8 GB（`nvidia-smi` 报 8188 MiB） |
| 计算能力 | **sm_89**（Ada Lovelace 架构，第 4 代张量核心） |
| 驱动 | 616.92，CUDA UMD **13.4** |
| CPU | AMD Ryzen 9 7940H，8 核 16 线程 |
| 内存 | 31.2 GB |

> ⚠️ 顺带一提：`Win32_VideoController` 报的 `AdapterRAM = 4 GB` 是**错的** ——
> 这个 WMI 字段是 32 位无符号整数，超过 4 GB 会截断。要看真实显存请用 `nvidia-smi`。

**结论**：卡没问题，是 **PyTorch 装成了 CPU-only 构建**。
`torch.cuda.is_available()` 只看编译时有没有链接 CUDA，跟驱动/卡无关。

---

## 2. 装 CUDA 版：一个 `+cu126` 的坑

第一步就踩了坑。这样装是**没用的**：

```powershell
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
# → Requirement already satisfied: torch==2.13.0 ...
```

pip 认为"已经满足了"，因为 **PEP 440 规定**：

> 如果版本说明符里写的是**公共版本号**（不含本地版本段），
> 那么候选版本的本地版本段在匹配时**必须被忽略**。

也就是说 `==2.13.0` 会匹配上已装的 `2.13.0+cpu`。必须**显式带上本地版本段**：

```powershell
pip install "torch==2.13.0+cu126" "torchvision==0.28.0+cu126" `
    --index-url https://download.pytorch.org/whl/cu126
```

**下载 2.6 GB**，装完验证：

```
torch 版本    : 2.13.0+cu126
编译时 CUDA   : 12.6
cuDNN         : 91002
CUDA 可用     : True
设备名        : NVIDIA GeForce RTX 4060 Laptop GPU
计算能力      : sm_89
```

### 怎么选 CUDA 版本

PyTorch 官方索引上各 CUDA 版本的构建情况（实测探测）：

| 索引 | 最高版本 | 说明 |
| --- | --- | --- |
| `cu124` | 2.6.0 | 太旧 |
| **`cu126`** | **2.14.0**，含 **2.13.0** | ✅ **选它**：与已装版本号完全一致，零 API 变化 |
| `cu128` | 2.11.0 | 不覆盖 2.13 |
| `cu130` | 2.14.0，含 2.13.0 | 也可以，但 cu126 更成熟 |

**选 `cu126` 的核心理由**：它的版本号 `2.13.0` 和原来**一模一样**，
所以这次替换不会引入任何 API 行为变化 —— 出问题只能是设备相关，不会是版本升级。

### 风险与回滚

装之前先备份（`outputs/env_backup/`），回滚命令：

```powershell
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

装完 pip 会报告几个依赖冲突：

```
torchaudio 2.6.0 requires torch==2.6.0, but you have torch 2.13.0+cu126 which is incompatible.
chatterbox-tts 0.1.7 requires torch==2.6.0, ...
```

这些是**用户环境里其他项目**的包，和本项目无关（本项目不用 torchaudio / TTS）。
它们原本就和 `torch 2.13.0+cpu` 冲突，不是这次造成的。

---

## 3. 代码改造：把硬编码的 CPU 拿掉

装完 CUDA 版 torch，**训练依然跑在 CPU 上** —— 因为代码里根本没有设备处理：

```python
# 改造前（src/cardex/engine.py）
with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):   # ← 写死 "cpu"
    outputs = model(images)
# 而且没有 model.to(device)，也没把数据搬到 GPU
```

### 3.1 新增设备选择工具（`src/cardex/utils.py`）

```python
def get_device(prefer: str | None = None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)          # 显式指定优先
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")               # 没有就老实回退


def autocast_dtype(device: torch.device) -> torch.dtype:
    """混合精度用什么类型 —— 这个选择很关键。"""
    return torch.float16 if device.type == "cuda" else torch.bfloat16
```

**为什么精度类型要分设备**：

* **CUDA 用 `float16`**：现代 N 卡（图灵及以后）有 fp16 张量核心，
  这才是真正快的那个；bf16 在消费级卡上经常没有对应加速。
* **CPU 用 `bfloat16`**：CPU 上 fp16 几乎没有加速，bf16 在支持 AVX512-BF16 的
  CPU 上约 1.5×。而且 bf16 的指数位和 fp32 一样宽，**不容易溢出**。

顺带还加了两个 CUDA 专属开关：

```python
torch.backends.cudnn.benchmark = True       # 输入尺寸固定时自动选最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True  # 矩阵乘用 TF32（精度损失极小）
torch.backends.cudnn.allow_tf32 = True
```

### 3.2 训练循环（`engine.py`）

```python
device = get_device(cfg.get("device", "auto"))
on_cuda = device.type == "cuda"

model.to(device)                             # ① 模型搬到设备
if ema is not None:
    ema.ema.to(device)                       # ② EMA 副本也要搬

for images, targets, _ in train_loader:
    images = images.to(device, non_blocking=True)    # ③ 数据搬到设备
    targets = targets.to(device, non_blocking=True)
    ...
    with torch.autocast(device.type, dtype=autocast_dtype(device), enabled=amp):
        logits = model(images)
```

### 3.3 推理侧

`CarClassifier` 本来就支持 `device=` 参数，只是默认写死 `"cpu"`。
改成默认 `"auto"`，并把同一处写死的 `autocast("cpu")` 也改成动态。

### 3.4 Agent 侧

`agent/config.py` 新增 `vision_device: str = "auto"`，
`agent/tools/vision.py` 把它传给分类器，并纳入模型缓存键
（换设备要重新加载模型）。

---

## 4. 踩到的 4 个坑

### 坑 1：`scatter_` 设备不匹配（会把训练打断）

```
RuntimeError: Expected all tensors to be on the same device,
but got index is on cuda:0, different from other tensors on cpu
```

**位置**：`data.py` 的 `mixup_cutmix_batch()`。

**原因**：这个函数在训练循环里被调用，此时 `images` 已经在 GPU 上了，
但函数内部**新建的张量没指定 device**，默认落在 CPU：

```python
targets_1h = torch.full((batch_size, num_classes), off, dtype=images.dtype)  # ← 漏了 device
targets_1h.scatter_(1, targets.unsqueeze(1), on)   # targets 在 cuda → 炸
```

**修复**：所有新建张量显式带上 `device=device`，`torch.randperm` 也一样。

> **教训**：从 CPU 搬到 GPU 时，最难查的不是"忘了 `model.to(device)`"，
> 而是**散落在工具函数里新建的张量**。凡是 `torch.full` / `torch.zeros` /
> `torch.randperm` / `torch.tensor` 都要问一句"该在哪个设备上"。

### 坑 2：Windows 多进程 DataLoader 无限递归

自己写性能分析脚本时踩的。脚本卡死 10 分钟，CPU 被刷满。

**原因**：Windows 上多进程用 **spawn**（不是 fork），子进程会**重新导入主模块**。
脚本没有 `if __name__ == "__main__":` 保护，子进程一导入就又去创建子进程：

```
主进程 → 6 个 worker → 每个 worker 又导入主模块 → 又创建 6 个 worker → ...
```

**修复**：加 `if __name__ == "__main__": main()`。
项目里 `src/*.py` 所有正式脚本**本来就有**这个保护（这次是自建脚本踩的）。

### 坑 3：`num_workers` 的默认值不能一刀切

原来统一是 `0`。原因是早期受限的沙箱环境禁止**命名管道**，
而 Python 多进程要用管道传数据，`num_workers>0` 会直接 `PermissionError`。

但这在 GPU 上是灾难：**数据增强全在 CPU 上做，串行跑就把 GPU 饿死了**。
改成按设备给不同默认值：

```python
default_workers = 6 if on_cuda else 0
num_workers = int(cfg.get("num_workers", default_workers))
```

### 坑 4：小数据集上多进程反而更慢

第一个冒烟测试（每类 8 张，只有 13 个 batch）用 4 个 worker，
**比单进程还慢一倍**（33s → 72s）。

**原因**：Windows 上每个 worker 都要 spawn 一个新进程并**重新 import torch**（约 3~5 秒）。
13 个 batch 根本摊不平这个启动开销，而 `val_loader` 又要再来一遍。

**结论**：`num_workers` 要按**数据量**而不是按"越多越好"来定。
真实训练集 7494 张 = 235 batch/epoch，worker 启动开销被充分摊薄，这时才划算。

---

## 5. 性能剖析：真正的瓶颈在哪

改造完之后训练能跑了，但**每个 step 2.3 秒**——
对一块 4060 来说慢得离谱。不猜，**直接给训练循环分段计时**：

```
=== num_workers=0  平均 373 ms/步 ===
  ① 取数据              295.4 ms/步    79.2%   ← 罪魁祸首
  ② mixup/cutmix          3.1 ms/步     0.8%
  ③ 前向                 32.7 ms/步     8.8%
  ④ 反向                 29.5 ms/步     7.9%
  ⑤ 裁剪+更新              7.7 ms/步     2.1%
  ⑥ EMA 更新              4.2 ms/步     1.1%
  ⑦ loss.item()          0.1 ms/步     0.0%

=== num_workers=4  平均 131 ms/步 ===
  ① 取数据               76.0 ms/步    57.9%
  ③ 前向                 21.3 ms/步    16.2%
  ④ 反向                 23.8 ms/步    18.1%
```

**读出来的结论**：

1. `num_workers=0` 时，**79% 的时间 GPU 在干等数据**。
2. 真正的 GPU 计算只要 **45 ms/步**（前向 21 + 反向 24）。
3. 开 4 个 worker 后数据侧从 295 → 76 ms，**总步时从 373 → 131 ms（2.8×）**。
4. 数据侧**仍然是瓶颈**（76 ms > 45 ms）。

继续加 worker：

| worker 数 | 每批数据准备 | 每 epoch 数据侧 |
| --- | --- | --- |
| 4 | 114 ms | 27 s |
| **6** | **48 ms** | **11 s** |

单张图的"解码 + 增强"要 **10.0 ms**（7494 张单线程需要 75 秒/epoch）——
这就是为什么要靠多进程把它摊到多个核上。

> **方法论**：性能问题**不要猜**。加 `torch.cuda.synchronize()` + 分段计时，
> 10 分钟就能定位。我先按"可能是数据加载"猜了一轮，方向对了但量级猜错了
> （以为是 330ms 对 61ms，实际是 295ms 对 45ms，而且真实训练里还有别的开销）。

---

## 6. 结果

### 6.1 训练速度：12.8 倍

| 实验 | 配置 | CPU 耗时 | **GPU 耗时** | 加速比 |
| --- | --- | --- | --- | --- |
| C1 | ResNet18 @224，18 epochs | 86.6 min | **8.7 min** | **10.0×** |
| C2 | ResNet50 @224，14 epochs | 175.2 min | **11.8 min** | **14.8×** |
| | **合计** | **261.8 min（4.4 小时）** | **20.5 min** | **12.8×** |

稳态数据：C1 约 **19 秒/epoch**（不含验证），GPU 利用率 51%、功耗 63 W；
C2 因为是 ResNet50、计算更重，GPU 利用率达到 **99%**。

* C1 里 GPU 只忙 51% —— 数据增强还没完全跟上（瓶颈在 CPU）；
* C2 里 GPU 忙到 99% —— 计算本身重到把数据侧盖过去了。

这说明**"加卡"不是万能**：模型越小，数据管道越容易成为新瓶颈。

### 6.2 精度：略有提升

| | CPU 训练 | **GPU 训练** | 变化 |
| --- | --- | --- | --- |
| C1 单模型（标定后 Macro-F1） | 0.9510 | 0.9507 | −0.0003 |
| C2 单模型 | 0.9592 | **0.9597** | +0.0005 |
| **C1+C2 集成（两人）** | **0.9604** | **0.9625** | **+0.0021** |
| **C1+C2+微调成员（三人）** | — | **0.9646** | 见 [15 · 微调实战指南](15_微调实战指南.md) 第 11 节 |

**精度没有因为换设备而下降**，集成还略好一点。
差别来自 cuDNN 的非确定性（同样是 seed=42，GPU 上的卷积算法选择和 CPU 不同），
属于随机波动 —— 结论是"**换设备只影响快慢，不影响质量**"。

> 上表最后一行的三人集成，是在 GPU 重训之后**又用原数据微调出一个 288px 的新成员**得到的
> （单独看它只有 0.9563，比父模型还差，但和另外两个成员互补）。
> 这一节讲的是 GPU 带来的速度，微调带来的精度提升见第 15 篇。

> 有意思的细节：推理侧用 GPU 跑集成模型，同一张图得到的置信度是
> **0.9618**，与 CPU 路径**完全一致**；换新权重后是 0.9672。
> float16 推理在这个任务上没有带来可见偏差。

标定参数也跟着更新了：

| | CPU 版 | GPU 版 |
| --- | --- | --- |
| 先验校正 τ | +0.05 | **+0.20** |
| 拒识阈值 t | 0.25 | 0.25 |
| unknown 检出率 | 0.929 | 0.905 |
| 已知类误拒率 | 0.034 | **0.028** |

（τ 的取值本来就该由每一版权重的验证集扫描决定，变了是正常的。）

### 6.3 推理速度也快了

| | CPU | GPU | 加速 |
| --- | --- | --- | --- |
| 集成模型加载（含 CUDA 上下文初始化） | 6.4 s | ~5.6 s | — |
| 单张推理（含 TTA） | 141~170 ms | **77~80 ms** | **≈2×** |
| 同上（模型已热） | 141 ms | **26~33 ms** | **≈5×** |
| 校准集 2422 张前向 | 数分钟 | **20~23 s** | — |

对 Web 系统来说，用户点"识别"的等待从 0.17 秒降到 0.03 秒。

---

## 6.4 顺手修掉的一个真 bug：缓存不看权重

第一次重训完跑 `finalize.py`，输出的数字和 CPU 版**一字不差**
（0.9482 / 0.9550 / 0.9587 → 0.9510 / 0.9592 / 0.9604），
而且日志里写着：

```
[候选] improved_resnet18_224
  复用缓存 logits_improved_resnet18_224_val_tta.npy      ← 用的是旧权重的结果！
```

**根因**：缓存文件名只按 `run 名 + split + 是否 TTA` 拼，**完全没看 checkpoint 有没有变**。
重训之后权重全换了，它照样读旧缓存 —— 指标、排名、`predictions.csv` 全是旧的，
**而且没有任何报错**。

**修复**：缓存键里加上 checkpoint 指纹（文件大小 + mtime 的 MD5 前 10 位）：

```python
def _ckpt_fingerprint(ckpt_path: Path) -> str:
    st = ckpt_path.stat()
    return hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:10]

cache = cache_dir / f"logits_{run}_{split}{'_tta' if tta else ''}_{fp}.npy"
```

换权重、改 TTA 设置都会自动算一份新的；顺手也把这个文件改成了设备无关。

> **教训**：**缓存键必须包含"被缓存内容的全部输入"**。
> 少一个维度，就会出现"改了东西但结果没变"这种最难查的问题 ——
> 因为它**不报错**，看起来一切正常。这次是因为我对 CPU 版的数字太熟
> （0.9604 是之前反复验证过的），一眼看出"一模一样"不对劲。

---

## 7. 配置项

改完之后，设备选择是**配置驱动**的，不用改代码：

```jsonc
// configs/train_improved.json
{
  "device": "auto",        // auto / cuda / cuda:0 / cpu
  "num_workers": 6,        // GPU 训练时必开，否则数据增强饿死 GPU
  "pin_memory": true,      // 锁页内存，加速 CPU→GPU 拷贝
  "amp": true              // 混合精度（CUDA 下自动用 fp16）
}
```

`agent/config.py` 里也加了对应的推理设备开关：

```python
vision_device: str = "auto"   # "auto" / "cuda" / "cpu"
```

**默认全是 `auto`** —— 有卡就用卡，没卡就老实回退 CPU，不需要改任何代码。

---

## 8. 温度与功耗（笔记本要注意）

| | 训练时 | 空闲 |
| --- | --- | --- |
| GPU 温度 | **88~89 °C** | 61 °C |
| 功耗 | 57~63 W | 4.8 W |
| 显存占用 | 2.4~4.2 GB | 2.2 GB（桌面） |

88~89 °C 对笔记本 GPU 属于"正常偏热"（一般 90~95 °C 才降频），
但**长时间训练建议垫高散热**。8 GB 显存跑 ResNet50@224/batch32 用 4.2 GB，
还有余量可以再调大 batch。

---

## 9. 这次改造对交付物的影响

| 项 | 变化 |
| --- | --- |
| `classes.txt` | 不变 |
| 模型权重 | **重新训练**（GPU 版），精度基本持平 |
| `predictions.csv` | 需要用新权重重新生成 |
| 代码 | `utils.py` / `engine.py` / `inference.py` / `data.py` / 配置 —— 全部改成**设备无关** |
| CPU 环境 | 依然可跑（`device: "auto"` 自动回退），旧权重复份在 `*_cpu` 目录 |

---

**相关文档**：[13 · 技术栈全解](13_技术栈全解.md)（PyTorch 训练技巧逐项讲解）·
[07 · 实验记录与结果分析](07_实验记录与结果分析.md)（5 组实验完整数据）
