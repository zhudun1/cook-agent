# AI 技术应用 / Agent 开发岗 · 手撕题与八股清单

> **用途**：传统算法题（hot100 等）之外，AI 技术应用岗面试的**手撕代码题 + 常考八股**总清单。
> **组织方式**：一道题一个小标题（`#### 1.1` 样式），可直接通过目录/锚点定位。
> **每题结构**：考察点 → 参考实现 → 关键要点 → 常见追问 → 📌 项目对应（cookhero）。
> **配合使用**：本文档是"题库索引 + 参考实现"；项目内 `cookhero/CookHero/docs/INTERVIEW_PREP.md` 是"原理深挖"，同一题的两份材料对照着背。

---

## 目录

### 第一部分 · 手撕题

- [第一类：LLM 基础编程](#第一类llm-基础编程)
  - [1.1 手撕 MultiHeadAttention（必考）](#11-手撕-multiheadattention多头自注意力必考)
  - [1.2 手撕 RoPE 旋转位置编码](#12-手撕-rope旋转位置编码高频)
  - [1.3 手撕 Sinusoidal 位置编码](#13-手撕-sinusoidal正弦余弦位置编码)
  - [1.4 手撕 LayerNorm](#14-手撕-layernorm必考)
  - [1.5 手撕 RMSNorm](#15-手撕-rmsnorm高频)
  - [1.6 手撕 Top-K / Top-P 采样](#16-手撕-top-k--top-p-nucleus-采样必考)
  - [1.7 从零实现 BPE 分词器（训练+编码）](#17-从零实现-bpe-分词器训练编码高频)
  - [1.8 手撕 SwiGLU 激活函数](#18-手撕-swiglu-激活函数中频)
- [第二类：Agent 核心模块](#第二类agent-核心模块)
  - [2.1 手撕 ReAct Agent 循环（必考）](#21-手撕-react-agent-循环必考)
  - [2.2 实现 Tool Registry 工具注册与调用系统（必考）](#22-实现-tool-registry-工具注册与调用系统必考)
  - [2.3 实现记忆系统（短期+长期）](#23-实现记忆系统短期长期高频)
  - [2.4 实现 CoT / Self-Reflection 机制](#24-实现-cot--self-reflection-机制中频)
  - [2.5 手撕工具调用失败处理（结构化错误）](#25-手撕工具调用失败处理结构化错误高频-趋势重点)
- [第三类：RAG 系统实现](#第三类rag-系统实现)
  - [3.1 实现文档切块策略](#31-实现文档切块策略必考)
  - [3.2 实现混合检索（BM25 + 向量 + 融合）](#32-实现混合检索bm25--向量--融合必考)
  - [3.3 实现 RAG 全流程（改写→检索→重排→生成）](#33-实现-rag-全流程改写检索重排生成必考)
  - [3.4 手撕暴力 KNN + 余弦相似度（补充）](#34-手撕暴力-knn--余弦相似度补充)
- [第四类：LLM 推理与部署（补充，常考）](#第四类llm-推理与部署补充)
  - [4.1 手撕 KV Cache](#41-手撕-kv-cache高频)
  - [4.2 手撕 GQA / 讲清 MHA·MQA·GQA](#42-手撕-gqa--讲清-mhamqagqa高频)
  - [4.3 八股：量化 / vLLM / 投机解码 / 连续批处理](#43-八股量化--vllm--投机解码--连续批处理)
- [第五类：深度学习与模型实现](#第五类深度学习与模型实现)
  - [5.1 手推两层神经网络反向传播](#51-手推两层神经网络反向传播高频)
  - [5.2 手撕 Adam / AdamW 优化器](#52-手撕-adam--adamw-优化器高频)
  - [5.3 手撕 LoRA 低秩适配](#53-手撕-lora-低秩适配必考)
  - [5.4 手撕 DPO 损失函数](#54-手撕-dpo-损失函数高频)
  - [5.5 手撕 Softmax 数值稳定 + 交叉熵（补充）](#55-手撕-softmax-数值稳定--交叉熵补充)

### 第二部分 · 八股清单

- [第六类：模型原理八股](#第六类模型原理八股)
- [第七类：Agent 八股](#第七类agent-八股)
- [第八类：RAG 八股](#第八类rag-八股)
- [第九类：训练与微调八股](#第九类训练与微调八股)
- [第十类：工程落地八股（趋势重点：从调 API 到写工程）](#第十类工程落地八股趋势重点从调-api-到写工程)

### 附录

- [面试趋势解读 & 复习路线](#附录-面试趋势解读--复习路线)

---

# 第一部分 · 手撕题

## 第一类：LLM 基础编程

> 面试出现频率最高的模块。考察对 Transformer 架构与 LLM 核心机制的**代码级**理解。建议全部手写一遍并能讲清每个细节。

### 1.1 手撕 MultiHeadAttention（多头自注意力）⭐必考

**考察点**：Q/K/V 线性投影 → 分头 → 缩放点积注意力 → 拼接 → 输出投影的完整链路；mask 的处理；为什么除 `sqrt(d_k)`。

**参考实现**：

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        # 1) 投影并分头：view 成 (B, T, H, d_k) 再转置为 (B, H, T, d_k)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2) 缩放点积注意力：除 sqrt(d_k) 防 softmax 饱和
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, H, T, T)
        if mask is not None:                                     # padding / causal mask
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = attn @ V                                           # (B, H, T, d_k)

        # 3) 多头拼接 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)
```

**关键要点**：
- `view + transpose` 分头，`transpose` 后必须 `contiguous()` 再 `view` 拼接。
- 缩放因子 `sqrt(d_k)`：点积大小随维度增长，不缩放会让 softmax 进入饱和区，梯度趋零。
- mask 两类：**padding mask**（对 `<pad>` 位置置 `-inf`）与 **causal mask**（下三角，防止看到未来）。
- 多头 = 不同的表示子空间，等价于"多个注意力头并行 + 拼接"。

**常见追问**：
- 为什么多头比单头好？（不同头关注不同关系：位置/句法/指代）
- 训练 vs 推理时 attention 的计算差异？（训练并行算全部位置；推理逐步 decode，靠 KV Cache）
- 手撕 causal mask：`torch.tril(torch.ones(T, T))` 转 bool 即可。
- MHA vs MQA vs GQA（见 4.2）。

**📌 项目对应**：项目走 LLM API（模型侧已实现），但 `app/llm/tokenizer.py` 的 token 预算 + `app/context/compress.py` 的窗口压缩，本质上是在应对注意力"长上下文退化"——可串起来讲。

---

### 1.2 手撕 RoPE（旋转位置编码）⭐高频

**考察点**：相对位置编码的本质（旋转矩阵只依赖位置差）、`rotate_half` 技巧、cos/sin 表预计算。

**参考实现**：

```python
import torch

def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0):
    # 频率：theta_i = base^(-2i/dim)
    i = torch.arange(0, dim, 2).float()
    theta = 1.0 / (base ** (i / dim))               # (dim/2,)
    pos = torch.arange(max_seq_len).float()          # (T,)
    angles = torch.outer(pos, theta)                 # (T, dim/2)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, T, 1, dim/2)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    return cos, sin

def rotate_half(x):
    # 把向量拆成前后两半，交叉取负拼接（等价于二维旋转的复数表示）
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # x: (B, T, H, d_k)，cos/sin: (1, T, 1, d_k/2)
    return x * cos + rotate_half(x) * sin
```

**关键要点**：
- 旋转位置编码：把位置信息编码为**旋转矩阵**作用在 Q/K 上，内积后自动只依赖**相对位置差** `m - n`。
- `rotate_half` 即旋转 90°：`(x1, x2) → (-x2, x1)`，配合 cos/sin 构成复数旋转。
- 训练长度外推：把 `base` 调大（如 10000 → 500000，即"NTK-aware"思想）可外推到更长序列。

**常见追问**：
- RoPE vs 绝对位置编码 vs ALiBi？（RoPE：相对位置、可外推；ALiBi：线性偏置、免训练外推）
- 为什么 RoPE 效果比 sinusoidal 好？（相对位置 + 可插值外推 + 不影响 attention 结构）
- 讲一下 NTK 外推 / YaRN（rope scaling）的原理。

**📌 项目对应**：项目调用 API 模型，RoPE 在模型侧；可借此讲你对"长上下文外推"的了解（context 层为什么还需要窗口+摘要，见 `app/context/compress.py`）。

---

### 1.3 手撕 Sinusoidal（正弦余弦）位置编码

**考察点**：偶 sin 奇 cos 的构造、为什么能表达相对位置（三角恒等式）、与外推的关系。

**参考实现**：

```python
import torch

def sinusoidal_pe(max_len: int, d_model: int, base: float = 10000.0):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()       # (T, 1)
    i = torch.arange(0, d_model, 2).float()                # (d_model/2,)
    theta = 1.0 / (base ** (i / d_model))                  # 不同维度的频率
    pe[:, 0::2] = torch.sin(pos * theta)                   # 偶数维 sin
    pe[:, 1::2] = torch.cos(pos * theta)                   # 奇数维 cos
    return pe                                              # 加到 token embedding 上
```

**关键要点**：
- 频率随维度递减：低维编码高频变化（区分相邻位置），高维编码低频（长距离依赖）。
- 偶 sin / 奇 cos 的用意：利用 `sin(a+b)=sin(a)cos(b)+cos(a)sin(b)`，使 `PE(pos+k)` 可表示为 `PE(pos)` 的线性函数 → 模型容易学到相对位置。
- 与 embedding **相加**（不是拼接），维度不变。

**常见追问**：
- 为什么后来被 RoPE 取代？（见 1.2）
- 可学习位置编码 vs 固定三角函数？（可学习的外推差；三角函数的频率先验更稳定）

**📌 项目对应**：无直接对应，属于模型侧知识；一般作为 RoPE 对比项讲。

---

### 1.4 手撕 LayerNorm ⭐必考

**考察点**：归一化维度、γ/β 可学习参数、与 BatchNorm 的本质区别、Pre-LN vs Post-LN。

**参考实现**：

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # 对最后一维（每个 token 的特征维）做归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

**关键要点**：
- LN 对**每个样本每个 token** 的特征维归一化；**不依赖 batch 大小**，训练/推理行为一致。
- BN 对 batch 维归一化，依赖 batch size，推理用 running stats——**与变长序列、小 batch 不兼容**，这是 Transformer 弃 BN 用 LN 的根本原因。
- `eps` 防除零；γ 初始 1、β 初始 0，防止归一化抹掉表示能力。

**常见追问**：
- 为什么 Transformer 不用 BN？（序列变长、batch 小、训练推理不一致、token 间统计无意义）
- Pre-LN vs Post-LN？（Pre-LN：残差前归一化，训练稳定无需 warmup，但表达力略弱；Post-LN：原始 Transformer，需 warmup）
- LN 在注意力前后各放一次，为什么？

**📌 项目对应**：模型侧知识；可延伸到项目 `app/llm/resilience.py` 的数值稳定考虑（如 fp 精度、JSON 修复），或单纯作为基础题背熟。

---

### 1.5 手撕 RMSNorm ⭐高频

**考察点**：去掉均值中心化、只做方差缩放、为什么在 LLM 中成为标配（Llama）。

**参考实现**：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**关键要点**：
- 与 LN 的差异：**不做均值中心化**（不减 mean、无 β），只按 RMS 缩放。
- 为什么效果不降：Transformer 里残差连接已经做了隐式的中心化，减去均值收益甚微，但省掉了 mean 的计算。
- 少一个 β 参数 + 更少计算 → 推理更快，Llama/Gemma 等主流模型标配。

**常见追问**：
- RMSNorm 的梯度与 LN 有何不同？（少了对均值的梯度项）
- 为什么 LLM 层数深了更倾向 RMSNorm？（稳定性 + 速度）

**📌 项目对应**：模型侧知识；无直接代码，作为基础题。

---

### 1.6 手撕 Top-K / Top-P（Nucleus）采样 ⭐必考

**考察点**：解码策略全链路（temperature → 截断 → softmax → 采样）、Top-P 动态截断的本质、与 greedy/beam 的区别。

**参考实现**：

```python
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1:]      # 第 k 大的值
        logits = torch.where(logits < kth,
                             torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # 累积概率超过 top_p 的 token 被剔除（保留至少一个）
        sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    return logits

def sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature                          # >1 更随机，<1 更确定
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)         # 按概率采样（不是取 argmax）
```

**关键要点**：
- temperature 缩放 logits 再 softmax：t→0 趋近 greedy，t→∞ 趋近均匀。
- **Top-K 的缺陷**：固定截断数，分布平缓时截断过多（丢合理 token）、分布尖锐时截断不足。
- **Top-P 是动态的**：按累积概率截断，分布尖锐时保留少、平缓时保留多——所以 Top-P 更常用。
- 生产常组合使用：先 Top-K 粗截断，再 Top-P 精截断。
- 采样是"从分布里抽"，不是每次取最大——这是多样性来源，也是重复/随机性的原因。

**常见追问**：
- 为什么推理用采样不用 greedy？（greedy 易重复、单调；采样更自然）
- 什么场景调低 temperature？（代码/数学/事实性任务；创作类调高）
- min-p / typical sampling 了解吗？（min-p：按相对概率阈值截断）

**📌 项目对应**：项目 `app/llm/` 通过 API 透传采样参数，未自行实现；可讲"fast/normal 分层模型 + 采样参数按任务配置"的工程考量（见 `config.yml`）。

---

### 1.7 从零实现 BPE 分词器（训练+编码）⭐高频

**考察点**：字节级 BPE 原理、merge 规则训练、编码时的最小序号合并；与 tiktoken/WordPiece/SentencePiece 的异同。

**参考实现**：

```python
from collections import defaultdict

def get_stats(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train_bpe(text: str, vocab_size: int):
    ids = list(text.encode("utf-8"))        # 初始词表 = 256 个字节（解决 OOV/多语言）
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    next_id = 256
    while next_id < vocab_size:
        stats = get_stats(ids)
        if not stats: break
        pair = max(stats, key=stats.get)    # 每次合并频率最高的相邻对（贪心）
        merges[pair] = next_id
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge(ids, pair, next_id)
        next_id += 1
    return merges, vocab

def encode(text: str, merges: dict):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # 编码时按 merge 顺序（即训练顺序）合并——优先合并更早学到的规则
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
```

**关键要点**：
- **字节级起点**：初始词表是 256 个 UTF-8 字节，任何文本都可编码，天然无 OOV。
- 训练 = 反复合并出现频率最高的相邻 token 对，直到词表大小达标（贪心、次优但有效）。
- **编码必须与训练顺序一致**：按"最早学到的规则优先"合并，否则编码结果不可复现。
- BPE vs WordPiece（按频率/按概率增益）vs SentencePiece（字符级 + 无监督分词 + 可加语言模型采样）。

**常见追问**：
- 分词对模型效果的影响？（词表大小 trade-off：太小序列长、太大 embedding 稀疏；多语言需要字节级）
- tiktoken 是什么？（OpenAI 的 BPE 实现，带正则预分词；你的项目 `app/llm/tokenizer.py` 用它做 token 计数与预算控制）

**📌 项目对应**：`app/llm/tokenizer.py` 用 tiktoken 做实时 token 计数，驱动滑动窗口与成本熔断——可讲"为什么 token 计数必须准：成本 + 延迟 + 注意力稀释三重约束"。

---

### 1.8 手撕 SwiGLU 激活函数 ⭐中频

**考察点**：门控线性单元思想、silu 定义、与标准 ReLU FFN 的差异、参数量变化。

**参考实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """Llama 系列使用的 FFN：out = w3( silu(x@w1) * (x@w2) )"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)     # 输入投影（进 silu 门）
        self.w2 = nn.Linear(d_model, d_ff)     # 门控投影
        self.w3 = nn.Linear(d_ff, d_model)     # 输出投影

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# silu(x) = x * sigmoid(x)
def silu(x):
    return x * torch.sigmoid(x)
```

**关键要点**：
- GLU（门控线性单元）：`(xW + b) ⊗ σ(xV + c)`，引入**可学习的门控**，让 FFN 能选择性地通过信息。
- SwiGLU = 用 SiLU（Swish）做门控激活：`silu(x) = x·sigmoid(x)`，平滑、无死亡 ReLU 问题。
- 相比标准 FFN（两个权重矩阵）多一个矩阵（三个权重），所以 Llama 把中间维度设为 `2/3 · 4d` 来对齐参数量。
- 主流 LLM（Llama、PaLM、Mistral）的默认 FFN 结构。

**常见追问**：
- GELU vs SiLU vs ReLU 的对比？（平滑性、梯度、负半轴行为）
- 为什么门控能提升效果？（类似"信息选择"，比固定非线性更灵活）

**📌 项目对应**：模型侧知识；无直接代码。

---

## 第二类：Agent 核心模块

> 考察对 Agent 框架的理解与工程实现能力。**必考中的必考**，且会结合项目深挖（工具失败、并发、异常处理）。

### 2.1 手撕 ReAct Agent 循环 ⭐必考

**考察点**：Thought → Action → Observation 循环、工具结果回喂、终止条件、最大迭代兜底。

**参考实现**：

```python
import json

class ReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm = llm                    # 支持 function calling 的 LLM
        self.tools = tools                # {name: tool}
        self.max_steps = max_steps

    async def run(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "按 Thought → Action → Observation 循环推理，直到问题解决。"},
            {"role": "user", "content": query},
        ]
        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]

        for _ in range(self.max_steps):
            # 1) 思考 + 行动：LLM 生成文本，可能带 tool_calls
            response = await self.llm.generate(messages, tools=tool_schemas)
            messages.append({"role": "assistant",
                             "content": response.text,
                             "tool_calls": response.tool_calls})

            # 2) 终止条件：没有工具调用 → 直接返回最终答案
            if not response.tool_calls:
                return response.text

            # 3) 执行工具 → Observation 回喂（必须保留在 messages 里！）
            for tc in response.tool_calls:
                tool = self.tools.get(tc.name)
                try:
                    result = await tool.run(**json.loads(tc.arguments))
                    content = json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    # 结构化错误：让模型自己决定恢复路径，而不是框架吞掉
                    content = json.dumps({"error": str(e), "retryable": True,
                                          "suggestion": "换个参数或换工具再试"})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": content})

        return "Reached max steps."       # 4) 迭代上限兜底，防止无限循环
```

**关键要点**：
- **LLM 无状态，工具结果就是状态**：`role="tool"` 消息必须回喂，且与 `assistant.tool_calls` 按 `tool_call_id` 严格配对（OpenAI 协议顺序）。
- 终止条件三件套：无 tool_call 即结束 / `max_steps` 上限 / 成本熔断（外部强制——自回归生成没有天然终止保证）。
- 循环的每一步都是概率采样，可能走偏 → 结构化错误（`retryable`）让模型自纠，而不是框架拍板重试。
- 工程化：事件流（SSE 逐步推送 tool_call/result）、轨迹记录（每轮落盘便于回放）、审批挂起（HITL）。

**常见追问**：
- ReAct 为什么比单次 prompt 强？（把计划-执行-验证压进同一个自回归过程，工具观察修正推理）
- 工具结果很大/超长怎么办？（截断工具返回、摘要化）
- 模型反复调用同一个失败工具怎么办？（`retryable=false` 语义 + 失败计数熔断）
- 并发工具调用怎么处理？（并行执行独立工具调用，注意顺序依赖）

**📌 项目对应**：`app/agent/agents/base.py` 的 `AgentLoop`（ReAct 循环、流式 tool_call 合并、`_append_tool_messages` 回喂）；`app/agent/tools/base.py` 的 `ToolResult`（error_code/retryable/suggestion）；`app/security/approval.py` 的审批挂起。面试可直接讲这套代码。

---

### 2.2 实现 Tool Registry（工具注册与调用系统）⭐必考

**考察点**：注册/查询/调用的抽象设计、JSON Schema 序列化、统一错误包装、权限与隔离。

**参考实现**：

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        return tool

    def register_decorator(self, name=None, description=None, schema=None):
        """@registry.tool() 装饰器式注册"""
        def deco(fn):
            return self.register(FunctionTool(name or fn.__name__, fn,
                                              description or fn.__doc__, schema))
        return deco

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        """给 LLM 看的 OpenAI function calling 格式"""
        return [t.to_openai_schema() for n, t in self._tools.items()
                if names is None or n in names]

    async def invoke(self, name: str, arguments: dict):
        tool = self._tools[name]
        # 统一走 safe_execute：超时 + 异常 → 结构化错误，不让异常穿透到循环
        return await tool.safe_execute(**arguments)
```

**关键要点**：
- 工具抽象三要素：`name`（语义清晰）、`description`（写"什么场景用"）、`parameters`（JSON Schema，字段名贴合用户语言）。
- 注册方式：类注册 / 装饰器 / 动态（MCP 服务器发现后注册）。
- **统一错误包装**（`safe_execute`）：独立超时 + `error_code/retryable/suggestion`，把错误变成模型可读的决策输入。
- 安全：工具 × 用户**权限矩阵**（白名单/黑名单/默认拒绝）、敏感工具人工审批。
- 多租户隔离：**有状态工具按会话克隆**，防止 A 用户的工具状态串到 B。

**常见追问**：
- 工具 >20 个时准确率下降怎么办？（工具路由：先向量检索 top-k 工具再注入；分层命名空间）
- 工具描述怎么写才容易被模型选中？（"什么场景用"式描述 + 与用户语言一致的字段名）
- 工具注册后发现参数写错怎么灰度？（版本化 + 按用户分流）

**📌 项目对应**：`app/agent/registry/hub.py`（AgentHub：builtin/MCP/自定义三种 provider 统一注册查询）；`app/agent/tools/base.py`（`BaseTool.to_openai_schema`、`safe_execute`、`classify_tool_error`）；`app/security/permissions.py`（工具权限矩阵）；`app/agent/tools/base.py` 的 `clone_for_session`（会话级隔离）。

---

### 2.3 实现记忆系统（短期+长期）⭐高频

**考察点**：短期记忆（上下文窗口管理：滑动窗口 + 摘要压缩）与长期记忆（跨会话持久化 + 检索注入）的分层设计；超长上下文处理。

**参考实现**：

```python
class MemorySystem:
    def __init__(self, llm, long_term_store, token_budget: int = 4000):
        self.llm = llm
        self.long_term = long_term_store    # 长期：跨会话持久化（DB/向量库）
        self.short_term = []                # 短期：当前会话 messages
        self.budget = token_budget

    async def on_turn(self, user_msg, assistant_msg):
        self.short_term += [user_msg, assistant_msg]
        # 短期记忆管理：先压缩再截断（压缩 = 有损压缩，截断 = 丢弃，能保则保）
        if count_tokens(self.short_term) > self.budget:
            summary = await self.llm.summarize(self.short_term)      # 增量摘要
            self.short_term = [{"role": "system", "content": summary}] + keep_recent(self.short_term, k=10)
        # 异步提取长期记忆（不阻塞对话）
        asyncio.create_task(self._extract_and_store(user_msg, assistant_msg))

    async def build_context(self, query) -> str:
        # 回忆注入：按 query 检索相关记忆 → 拼进 system prompt
        memories = await self.long_term.recall(query, top_k=5)
        return render_memories(memories)

    async def _extract_and_store(self, user_msg, assistant_msg):
        facts = await self.llm.extract_facts(user_msg, assistant_msg)  # LLM 抽取
        if not facts:
            facts = heuristic_extract(user_msg)                        # 启发式回退
        await self.long_term.store(facts)
```

**关键要点**：
- 短期记忆 = **上下文窗口的预算管理**：token 超预算时"先摘要压缩、再窗口截断（保末尾）"。
- 长期记忆 = **跨会话**：对话结束后异步提取（偏好/目标/限制/事实），检索注入 system prompt。
- 类型加权：限制类（restriction）记忆检索时权重最高，防止推荐踩用户红线。
- 提取可靠性：LLM 提取 + 启发式回退双保险，LLM 挂了记忆体系不崩。
- 超长上下文的三层答案：滑动窗口（硬预算）→ 摘要压缩（语义保留）→ 长期记忆（跨会话）。

**常见追问**：
- 为什么不用"无限加大上下文窗口"？（成本线性涨 + Lost in the Middle 注意力稀释）
- 摘要会丢失细节怎么办？（分层记忆：会话摘要 → 长期记忆；关键事实单独入库）
- 向量检索 vs 关键词检索做记忆回忆？（记忆条目短、语义近义词少 → 关键词+加权足够；升级路径是 embedding）
- 记忆怎么防污染/防敏感信息泄漏？（类型白名单、脱敏、用户可查看/删除）

**📌 项目对应**：`app/memory/`（`extractor.py` LLM+启发式提取、`store.py` 2-gram 关键词 + 类型加权 recall、`manager.py` 编排）；`app/context/compress.py`（双阈值压缩：消息条数 + token 预算比例，增量摘要）。

---

### 2.4 实现 CoT / Self-Reflection 机制 ⭐中频

**考察点**：提示工程中的推理增强与自我纠错；零样本/少样本 CoT、Self-Refine 循环。

**参考实现**：

```python
# 1) Zero-shot CoT：一句话触发逐步推理
def zero_shot_cot(llm, question):
    return llm(f"{question}\n\n请一步一步思考（Let's think step by step），最后给出结论。")

# 2) Few-shot CoT：带推理过程的示例（示例质量 > 数量）
def few_shot_cot(llm, question, examples):
    demo = "\n\n".join(f"问题：{e.q}\n回答：{e.reasoning_and_answer}" for e in examples)
    return llm(f"{demo}\n\n问题：{question}\n回答：")

# 3) Self-Refine（自我反思）：生成 → 批评 → 重写，循环 n 轮
def self_refine(llm, task, rounds=2):
    draft = llm(f"任务：{task}")
    for _ in range(rounds):
        feedback = llm(f"批评以下输出，指出错误与改进点：\n{draft}")
        draft = llm(f"根据反馈重写。\n原输出：{draft}\n反馈：{feedback}\n重写：")
    return draft
```

**关键要点**：
- CoT 为什么有效：把问题**分解成中间步骤**，给自回归生成更多"计算空间"，降低单步推理难度；中间推理本身就是可检查的证据。
- Zero-shot CoT 的触发词："Let's think step by step"；Few-shot CoT 的关键是**示例带推理过程**而非只给答案。
- Self-Refine 的代价：每轮 2 次 LLM 调用，成本翻倍——要控制轮数与触发时机（只在任务失败或高价值任务上启用）。
- 反思与 ReAct 的关系：ReAct 是"外部化"的反思——工具 Observation 就是事实核查；Reflexion 是"事后"反思（把失败经验写回记忆）。

**常见追问**：
- CoT 在什么任务上无效？（简单任务无增益反而增加延迟/成本；纯检索事实型任务）
- 怎么判断该不该开反思？（任务失败率、成本预算、任务价值分级）
- Self-Refine 会越改越差吗？（会——批评模型本身也可能错；需要 ground truth 或验证器兜底）

**📌 项目对应**：`app/agent/prompts/`（system prompt 中的推理要求）、`app/prompts/registry.py`（prompt 多版本灰度，可按用户分流实验 CoT 开关）、`app/security/guardrails/`（输出校验兜底）。

---

### 2.5 手撕工具调用失败处理（结构化错误）⭐高频 · 趋势重点

**考察点**：Agent 工程的"现实问题"——工具失败、重试分类、并发、超时。面试趋势明确指向这类题。

**参考实现**：

```python
def classify_error(exc: Exception) -> dict:
    """把异常映射成模型可读的结构化错误（两受众：开发者看日志、模型看决策）"""
    if isinstance(exc, TimeoutError):
        return {"error_code": "TIMEOUT", "retryable": True,
                "suggestion": "稍后重试或缩小请求范围"}
    if isinstance(exc, (RateLimitError,)):
        return {"error_code": "RATE_LIMIT", "retryable": True,
                "suggestion": "降低频率后重试"}
    if isinstance(exc, (ValidationError, PermissionError)):
        return {"error_code": "INVALID_ARGUMENT", "retryable": False,
                "suggestion": "修改参数后再调用"}
    return {"error_code": "UNKNOWN", "retryable": False,
            "suggestion": "换一种方式完成目标"}

async def safe_invoke(tool, **kwargs):
    try:
        return await asyncio.wait_for(tool.execute(**kwargs), timeout=tool.timeout)
    except Exception as e:
        return ToolResult(success=False, **classify_error(e))   # 不让异常穿透
```

**关键要点**：
- **错误处理从"代码分支"变成"模型推理"**：结构化错误 = 把控制流需要的信息编码成模型可读的 JSON。
- `retryable` 是最关键字段：模型天然会"无条件重试"，`retryable=false` 注入领域知识（"别重试，换路"）。
- 超时设计：工具独立超时（不同工具不同预算），`asyncio.wait_for` 实现。
- 重试要有**退避 + 抖动**（防雪崩）；流式只重试"首块之前"（出流后重试会重复推送）。
- 并发：独立工具并行执行；共享资源加锁/幂等键；工具执行器按会话隔离。

**常见追问**：
- 模型反复调用同一个失败工具？（失败计数 → 熔断该工具 + 强制换路）
- 工具返回内容有毒/超长？（返回内容注入检测 + 截断/摘要）
- 并发工具调用结果乱序怎么处理？（按 tool_call_id 聚合）

**📌 项目对应**：`app/agent/tools/base.py`（`classify_tool_error` + `safe_execute` + 独立超时）；`app/llm/resilience.py`（指数退避 + 抖动 + `is_retryable_error` 分类 + 模型降级链）；`app/security/cost_guard.py`（token 熔断）。

---

## 第三类：RAG 系统实现

> 考察 RAG 全链路各环节的实现能力：切块 → 索引 → 检索 → 融合 → 重排 → 生成。

### 3.1 实现文档切块策略 ⭐必考

**考察点**：固定/重叠/结构化/语义四种策略的取舍；切块与检索质量的关系。

**参考实现**：

```python
def fixed_size_chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """固定大小 + 重叠：重叠防止关键信息落在块边界被劈开"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def recursive_character_chunk(text: str, separators, size=500, overlap=50):
    """按分隔符层级（标题>段落>句子）递归切，单块超长再降级切分"""
    ...

def semantic_chunk(text: str, embed, threshold: float = 0.6) -> list[str]:
    """语义切块：句子间向量相似度低于阈值 → 切开（内容边界对齐语义）"""
    sents = split_sentences(text)
    chunks, cur = [], [sents[0]]
    for s in sents[1:]:
        if cosine(embed(cur + [s]), embed(cur)) >= threshold:
            cur.append(s)                    # 语义连贯 → 并入当前块
        else:
            chunks.append("".join(cur)); cur = [s]
    chunks.append("".join(cur))
    return chunks
```

**关键要点**：
- 切块粒度 trade-off：**太小** → 语义不全、检索上下文不足；**太大** → 语义稀释、命中后塞爆 context。
- 重叠的作用：防止关键句正好落在边界被"劈成两半"。
- **结构化切块**（按 Markdown 标题层级）优于纯固定长度：块天然带语义边界。
- **small-to-large**（小到大概回传）：用小块检索、命中后回传父文档/邻近块，兼顾精度与上下文。
- chunk 大小经验值：中文 200~500 字、英文 300~800 token，取决于 embedding 模型与下游生成。

**常见追问**：
- 切块大小怎么验证？（在评测集上对比 context_recall；不同大小 A/B）
- 结构化切块遇到"无标题长文"怎么办？（递归降级到段落/句子级）
- 语义切块的代价？（逐句 embedding O(n)，长文档慢——可缓存或抽样）

**📌 项目对应**：`app/rag/pipeline/document_processor.py`（`MarkdownHeaderTextSplitter` 按 `#`/`##` 标题结构化切块 + `parent_id` 小到大回传父文档）。

---

### 3.2 实现混合检索（BM25 + 向量 + 融合）⭐必考

**考察点**：稠密/稀疏互补性、RRF 与加权融合的数学、分数归一化问题、融合后重排。

**参考实现**：

```python
from collections import defaultdict

def rrf_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF：只依赖排名不依赖分数，天然解决"余弦 vs BM25 分数不可比"。
       score(d) = Σ 1/(k + rank_i(d))"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def weighted_fusion(dense_hits, sparse_hits, w_d=0.5, w_s=0.5):
    """加权融合：必须先对分数归一化（min-max / z-score / norm_score）"""
    merged = defaultdict(float)
    for doc_id, s in dense_hits:  merged[doc_id] += w_d * minmax_norm(s)
    for doc_id, s in sparse_hits: merged[doc_id] += w_s * minmax_norm(s)
    return sorted(merged, key=merged.get, reverse=True)
```

**关键要点**：
- 两种信号**互补**：稠密向量管语义（"清淡的菜"→清蒸鱼，对措辞鲁棒）、BM25 管词面（菜名/精确术语），缺一不可。
- **RRF 为什么只排名不看分数**：余弦相似度与 BM25 分数量纲不同、不可直接相加；RRF 用排名贡献 `1/(k+rank)`，k 通常取 60。
- 加权融合的前提是**归一化**（norm_score），否则权重无意义。
- 动态加权：按查询意图调权重（步骤/术语类偏 BM25，推荐/语义类偏向量）。
- 融合后必须 **Rerank**：交叉编码器 O(n²)，只能处理小集合——"融合召回 top_k → 重排精排 top_n"两阶段。

**常见追问**：
- RRF 的缺点？（丢弃分数信息，贡献是离散的；极端情况下召回质量差的路上升噪声文档）
- 什么时候选加权而不是 RRF？（分数可归一化、且知道两路相对质量时）
- 向量检索召回不到的本质场景？（多跳推理、否定约束、切分劈裂、query 质量差——见 INTERVIEW_PREP.md 专题二）

**📌 项目对应**：`app/rag/pipeline/retrieval.py`（Milvus 原生 dense+sparse 混合检索，`rrf`/`weighted` 两种 ranker + `intelligent_ranker_selection` 按查询类型动态加权 + score_threshold 过滤）。

---

### 3.3 实现 RAG 全流程（改写→检索→重排→生成）⭐必考

**考察点**：全链路各环节职责、Query Rewrite 的必要性、双阶段检索、生成引用。

**参考实现**：

```python
class RAGPipeline:
    def __init__(self, llm, retriever, reranker, top_k=20, top_n=5):
        ...

    async def answer(self, query: str, history=None) -> str:
        # 1) Query Rewrite：多轮指代消解（"它"→具体实体）+ 口语 → 检索式表达
        rewritten = await self.llm.rewrite_query(query, history)

        # 2) 召回（粗排，快）：混合检索 + 元数据过滤，取 top_k
        candidates = await self.retriever.hybrid_search(rewritten, top_k=self.top_k)

        # 3) 精排（重排，准）：交叉编码器只处理小集合 → top_n
        reranked = await self.reranker.rerank(rewritten, candidates, top_n=self.top_n)

        # 4) 生成：引用溯源 + 无依据不编造
        answer = await self.llm.generate(
            query, context=reranked,
            system="只依据给定上下文回答，无法回答时明确说明。引用格式[1]")
        return answer
```

**关键要点**：
- Query Rewrite 解决的三个问题：多轮指代（"它""那家店"）、口语化噪声（约束被稀释）、检索表达不匹配。
- **召回-重排双阶段**的本质：粗排用廉价双塔/BM25 快速缩小候选，精排用昂贵交叉编码器精确打分——延迟与精度的平衡。
- 生成环节的幻觉控制：system prompt 约束"只依据上下文" + **引用溯源**（[1] 标记 + 对应原文片段）。
- context 超限处理：top_n 截断 / 按分数截断 / 分块摘要。
- 评测闭环：context_precision/recall（检索质量）与 answer_correctness（生成质量）分开看。

**常见追问**：
- Query Rewrite 改写错了怎么办？（多版本改写并行检索，或 rewrite 后校验）
- Rerank 分数和检索分数可比吗？（不直接可比，rerank 单独排序）
- 检索为空/全被阈值过滤怎么办？（兜底：降阈值重试 / 告知用户"知识库无相关内容" / 走 LLM 已有知识并标注）
- 引用怎么验证？（grounding truth 评测 + 回答中的引用序号与上下文序号一致性检查）

**📌 项目对应**：`app/services/rag_service.py`（编排全流程）；`app/rag/pipeline/generation.py` 的 `rewrite_query`（查询改写）；`app/rag/rerankers/siliconflow_reranker.py`（交叉编码器重排）；`app/evaluation/runner.py`（RAGAS grounding truth 指标）。README 提到的"知识库检索 + Web 搜索"双源也是这个链路的工程化。

---

### 3.4 手撕暴力 KNN + 余弦相似度（补充）

**考察点**：向量检索的朴素实现，为讲 HNSW/IVF 索引做铺垫。

**参考实现**：

```python
import numpy as np

def cosine_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def brute_force_knn(query_vec, doc_vecs, k=10):
    """暴力扫描 O(n·d)：数据量大时慢，所以要索引（HNSW/IVF）"""
    scores = [(cosine_sim(query_vec, v), i) for i, v in enumerate(doc_vecs)]
    scores.sort(key=lambda x: -x[0])
    return [i for _, i in scores[:k]]
```

**关键要点**：
- 暴力扫描 O(n) 次相似度计算——万级可接受，百万级要索引。
- HNSW：分层小世界图，顶层枢纽快速定位 + 底层精确定位；`M/efConstruction/efSearch` 分别控制分支度/构建质量/查询质量（详见 INTERVIEW_PREP.md 专题二）。
- 文本向量通常用 COSINE（BGE 官方推荐）；归一化后 cosine 等价于内积。

**常见追问**：
- IVF vs HNSW 选哪个？（召回率优先 HNSW；内存受限/十亿级 IVF-PQ）
- 为什么稀疏向量（BM25）适合倒排索引？（天然稀疏，倒排表高效）

**📌 项目对应**：`app/rag/vector_stores/vector_store_factory.py`（Milvus，当前 AUTOINDEX；生产建议显式 HNSW + COSINE + 稀疏倒排——README/INTERVIEW_PREP 已写明这是设计债）。

---

## 第四类：LLM 推理与部署（补充）

> 用户列了三类跳到五类，中间常考的"推理与部署"在此补充。AI 应用岗经常被问 KV Cache、量化、推理框架。

### 4.1 手撕 KV Cache ⭐高频

**考察点**：decode 阶段为什么能缓存 K/V、显存占用、预分配 vs 动态增长。

**参考实现**：

```python
import torch

class KVCache:
    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 max_len: int = 4096, dtype=torch.float16):
        # 预分配整块显存，避免每次增长导致碎片与重复分配
        self.k = torch.zeros(n_layers, n_kv_heads, max_len, head_dim, dtype=dtype)
        self.v = torch.zeros(n_layers, n_kv_heads, max_len, head_dim, dtype=dtype)
        self.size = 0                      # 当前已缓存的位置

    def append(self, layer: int, new_k, new_v):
        # new_k/new_v: (n_kv_heads, new_len, head_dim)——decode 只算新 token 的 K/V
        s = new_k.shape[-2]
        self.k[layer, :, self.size:self.size + s] = new_k
        self.v[layer, :, self.size:self.size + s] = new_v
        self.size += s

    def get(self, layer: int):
        return (self.k[layer, :, :self.size], self.v[layer, :, :self.size])

    def reset(self):
        self.size = 0
```

**关键要点**：
- 推理分两阶段：**prefill**（并行算整个 prompt）与 **decode**（逐 token 生成）。
- decode 阶段第 n 个 token 的 attention 只需与**前面所有 token 的 K/V** 做点积——之前的 K/V 已被算过，缓存复用，避免每步重算前 n-1 个 token → 复杂度从 O(n²) 降到 O(n)。
- 显存占用公式：`2 × L × n_kv_heads × head_dim × seq_len × batch × bytes`（K+V 各一份）——长上下文的主要显存瓶颈。
- **GQA/MQA 减小 KV cache**（见 4.2）；PagedAttention 解决显存碎片（见 4.3）。
- 工程细节：预分配 vs 按需增长（vLLM 用 PagedAttention 块分配）；prefix caching（共享前缀复用）。

**常见追问**：
- KV cache 多大？（算一遍显存公式；举例：L=32, 8 kv heads, d=128, seq=4096, fp16 → 32×8×128×4096×2×2 ≈ 1GB/序列）
- 长上下文为什么贵？（KV cache 随 seq 线性增长，不是 attention 计算本身）
- 哪些场景能省 KV？（GQA、量化 KV cache、KV cache eviction / H2O）

**📌 项目对应**：模型侧推理知识；项目在 API 侧（成本与延迟由 KV cache 决定），可借此讲 `app/llm/resilience.py` 的降级链（fast 模型省 token）与 `app/llm/tokenizer.py` 的预算控制。

---

### 4.2 手撕 GQA / 讲清 MHA·MQA·GQA ⭐高频

**考察点**：三者结构差异、参数量与 KV cache 收益、为什么主流模型选 GQA。

**参考实现**：

```python
import math
import torch

def grouped_query_attention(Q, K, V, n_heads, n_kv_heads, mask=None):
    # Q: (B, T, n_heads, d_k)；K/V: (B, T, n_kv_heads, d_k)
    groups = n_heads // n_kv_heads
    K = K.repeat_interleave(groups, dim=-2)   # 每个 KV 头复制给组内所有 Q 头
    V = V.repeat_interleave(groups, dim=-2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V
```

**关键要点**：
- **MHA**：每头独立 K/V（`n_kv_heads = n_heads`）→ KV cache 最大、质量最好。
- **MQA**：所有 Q 头共享一组 K/V（`n_kv_heads = 1`）→ KV cache 最小、质量有损。
- **GQA**：折中——KV 头分组共享（`n_kv_heads` 介于 1 和 n_heads 之间），质量接近 MHA、显存接近 MQA。Llama 2/3 用 GQA。
- 收益点：KV cache 显存、prefill/decode 的访存带宽。

**常见追问**：
- GQA 的组数怎么定？（经验：8~32 组；越大质量越好但显存越大）
- MQA 为什么质量下降？（所有头共享同一对 K/V，表达力受限）

**📌 项目对应**：模型侧知识；无直接代码。

---

### 4.3 八股：量化 / vLLM / 投机解码 / 连续批处理

**考察点**：推理优化全景，AI 应用岗常被问"部署一个 7B/70B 模型需要什么"。

**一句话答案 + 要点**：

| 主题 | 一句话答案 | 要点 |
|---|---|---|
| **量化** | 把权重从 FP16/BF16 压到 INT8/INT4，显存降 2~4 倍，速度提升，精度轻微损失 | GPTQ（训练后量化，按列校准）、AWQ（按激活感知加权，精度更好）、QLoRA（量化 + LoRA 微调，4bit 训练）；KV cache 也可量化 |
| **vLLM / PagedAttention** | 把 KV cache 像操作系统分页一样按块分配，消除显存碎片与浪费，支持高并发 | 连续批处理（continuous batching）：一个 batch 内请求随时完成随时腾位，新请求随时加入，吞吐量数倍于静态 batching；流式输出 |
| **投机解码（Speculative Decoding）** | 小模型草稿 + 大模型验证，一次验证多 token，推理加速不损失质量 | 大模型并行验证 k 个草稿 token，接受概率匹配原分布（拒绝采样）；加速 2~3 倍 |
| **连续批处理** | 见上，vLLM 核心 | 与静态 batching 对比：不等待整批完成 |
| **前缀缓存** | 相同 prompt 前缀（system prompt）的 KV 复用，多轮/多用户共享 | 面试加分项 |

**常见追问**：
- 7B 模型 FP16 显存多少？（约 14GB 权重 + KV cache + 激活；INT4 约 3.5~4GB）
- 量化会不会崩？（LLM 对权重量化相对鲁棒，激活量化难；用 AWQ/GPTQ 校准可缓解）
- 流式 + 并发下 KV cache 怎么管理？（PagedAttention 块级分配 + 预分配池）

**📌 项目对应**：README 的"推理层未接 vLLM/continuous batching（这是真正撑吞吐的地方）"是设计债——面试可主动说"我知道下一步该上 vLLM + 量化 + prefix caching"。

---

## 第五类：深度学习与模型实现

> 部分公司要求手推/手撕训练相关代码。重点：反向传播、优化器、LoRA、对齐损失。

### 5.1 手推两层神经网络反向传播 ⭐高频

**考察点**：链式法则逐层展开、ReLU 导数掩码、形状推导。

**参考实现**：

```python
import numpy as np

def train_two_layer(X, y, hidden=16, lr=0.1, epochs=1000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * 0.01    # 小初始化：对称破坏 + 防梯度爆炸
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * 0.01
    b2 = 0.0

    for _ in range(epochs):
        # ---- 前向 ----
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)                      # ReLU
        z2 = a1 @ W2 + b2
        y_hat = z2
        loss = np.mean((y_hat - y) ** 2)            # MSE

        # ---- 反向（标量损失对各中间量求梯度，形状与变量一致）----
        dz2 = 2 * (y_hat - y) / n                   # dL/dz2  (n,1)
        dW2 = a1.T @ dz2                            # (hidden,1)
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T                            # (n,hidden)
        dz1 = da1 * (z1 > 0)                        # ReLU 导数：掩码
        dW1 = X.T @ dz1                             # (d,hidden)
        db1 = dz1.sum(axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    return W1, b1, W2, b2
```

**关键要点**：
- 反向传播 = **链式法则**：从损失一路往回，每层梯度 = 上游梯度 × 本层雅可比。
- **形状检查技巧**：`dW` 形状必须与 `W` 形状一致（`a1.T @ dz2` 的理由）；这是手推时最快的自检。
- ReLU 导数：`z > 0` 为 1，否则 0（掩码乘上去）。
- 初始化为什么要小：全零 → 对称（各神经元学到相同特征）；太大 → 深层梯度爆炸。

**常见追问**：
- 手推单样本的梯度 vs 批量？（批量 = 求和平均）
- 如果隐藏层用 sigmoid 会怎样？（梯度消失——sigmoid 导数最大 0.25，多层连乘趋零）
- 交叉熵 + softmax 的梯度为什么是 `p - y`？（优雅的抵消，见 5.5）

**📌 项目对应**：项目不训练模型（调 API），但可讲"理解训练才能讲清 LoRA/DPO/评测"，或与 `app/evaluation/` 的评测思路呼应。

---

### 5.2 手撕 Adam / AdamW ⭐高频

**考察点**：一阶矩（动量）+ 二阶矩（梯度平方 EMA）+ 偏差校正；Adam vs AdamW 的权重衰减位置。

**参考实现**：

```python
import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]   # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]   # 二阶矩
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            # AdamW 关键：权重衰减与梯度解耦（不进动量），直接乘在参数上
            p.data.mul_(1 - self.lr * self.wd)
            # 动量更新
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            # 偏差校正：t 小时 m/v 偏小，校正后无偏
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
```

**关键要点**：
- 一阶矩 = 带衰减的梯度平均（动量，稳定方向）；二阶矩 = 梯度平方的 EMA（自适应步长，小梯度大步、大梯度小步）。
- **偏差校正为什么必要**：t=1 时 `m = (1-b1)g`，比真实梯度小得多——除以 `1-b1^t` 修正。
- **Adam vs AdamW**：Adam 的 L2 正则把衰减加进梯度（与动量耦合，会干扰自适应）；AdamW 把衰减**解耦**直接乘参数，微调/大模型训练更稳，是 PyTorch 默认。
- `eps` 防除零；`addcdiv_` 即 `p -= lr * m_hat / (sqrt(v_hat) + eps)`。

**常见追问**：
- Adam 的显存开销？（额外保存 m、v 两份梯度大小张量 ≈ 2× 参数显存）
- 什么时候用 SGD？（小数据/简单任务/追求泛化；大模型几乎不用）
- AdamW vs Adam 在微调 LLM 时的差异？（解耦衰减 + 更小的有效正则干扰）

**📌 项目对应**：训练侧知识；项目是 API 调用不训练，但 LoRA/微调问题常被追问（见 5.3）。

---

### 5.3 手撕 LoRA 低秩适配 ⭐必考

**考察点**：低秩分解 ΔW = B·A、初始化策略、alpha/r 缩放、冻结与显存收益。

**参考实现**：

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)               # 冻结原权重：防灾难遗忘 + 省显存
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)  # A 用高斯/kaiming 初始化
        # B 保持全零 → 初始 ΔW = B@A = 0，训练从原权重开始，稳定
        self.scaling = alpha / r                      # r 增大时保持更新幅度不变

    def forward(self, x):
        # W' = W + (alpha/r)·B·A
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling
```

**关键要点**：
- **低秩假设**：微调时权重更新 ΔW 的有效秩很低（intrinsic dimension），可用 `ΔW = B·A` 逼近，`r << min(in, out)`。
- 初始化：A 随机、**B 全零** → 初始 ΔW=0，训练平稳，不会一上来就破坏预训练权重。
- `alpha/r` 缩放：r 变大时若不加缩放，更新幅度变大需调 lr；除以 r 保持更新量级恒定。
- 可训练参数量：`r×(in+out)`，如 4096×4096 的矩阵、r=8 → 只训 65K 参数（原 16M 的 0.4%）。
- 为什么省显存：只存可训练参数的梯度 + 优化器状态；原权重冻结不存梯度（可再量化 → QLoRA）。
- 推理时可选合并：`W' = W + ΔW` 合回原权重 → 无额外延迟。

**常见追问**：
- LoRA 与全参微调效果差距？（通常接近，r 足够时；数据少时 LoRA 更稳）
- r 怎么选？（4~64 经验；大 r 不必然更好——低秩假设上限）
- LoRA 可以只加在 attention 还是也加 FFN？（都可；attention 的 Wq/Wv 常见）
- AdaLoRA / DoRA 了解吗？（自适应秩分配 / 权重解耦幅度方向）

**📌 项目对应**：项目不微调；但面试可讲"理解 LoRA 才能理解 QLoRA 的显存账，以及为什么 RAG/Agent 场景优先选 prompt 而非微调"。

---

### 5.4 手撕 DPO 损失函数 ⭐高频

**考察点**：DPO 与 RLHF 的本质区别（无奖励模型、无 PPO 在线采样）、隐式奖励、beta 的作用。

**参考实现**：

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO：直接从偏好对（chosen > rejected）优化，隐式奖励 r = beta·log(pi/pi_ref)
    logps: 模型对该回复整个序列的对数概率（逐 token 求和）"""
    # 策略模型与参考模型的"偏好差"
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    # 隐式奖励差（Bradley-Terry 模型的对数几率）
    logits = beta * (pi_logratios - ref_logratios)
    # 最大化 P(chosen > rejected) → 最小化 -log sigmoid
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**关键要点**：
- **RLHF 流程**：SFT → 训奖励模型（RM）→ PPO 在线采样优化（复杂、不稳定、耗资源）。
- **DPO 的关键洞察**：奖励模型的最优策略有**闭式解** `r(x,y) = beta·log(pi(y|x)/pi_ref(y|x)) + const`，代入 Bradley-Terry 偏好模型后可直接得到损失——**不需要 RM、不需要 PPO**，只需策略模型 + 冻结的参考模型 + 静态偏好数据集。
- `beta` 控制与参考模型的 KL 距离：越大越贴近参考模型（越保守），越小越激进。
- 数据格式：`(prompt, chosen, rejected)` 三元组；两个回复都来自同一个模型的采样（通常 SFT 后采样）。
- 变体：IPO（去掉 log-sigmoid 的饱和问题）、KTO（无需成对数据）、SimPO（无需参考模型）。

**常见追问**：
- DPO 相比 PPO 的代价？（DPO 是离线优化，无法利用在线新数据探索；PPO 在线但复杂）
- 偏好数据怎么来？（人工标注 / AI 反馈 / 对比采样）
- 为什么 DPO 容易过拟合偏好？（数据噪声敏感；需要正则与 early stopping）

**📌 项目对应**：训练侧知识；可延伸"对齐与评测"：项目 `app/evaluation/` 用 LLM 做裁判（faithfulness 等）类似偏好判断思路。

---

### 5.5 手撕 Softmax 数值稳定 + 交叉熵（补充）

**考察点**：logsumexp trick、数值溢出、log-softmax 优雅的梯度。

**参考实现**：

```python
import torch

def stable_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)                  # 减最大值：防止 exp(大数) 溢出
    return e / e.sum(dim=-1, keepdim=True)

def cross_entropy(logits, targets):
    # log-softmax：避免中间 exp/除法的精度损失，且梯度 = p - y 最优雅
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()
```

**关键要点**：
- `exp(x)` 在 x 很大时溢出（`exp(1000) = inf`），先减 `max` 不改变分布。
- `logsumexp` 同样先减 max：`log Σ exp(x_i) = m + log Σ exp(x_i - m)`。
- softmax + cross-entropy 的梯度：`∂L/∂z = softmax(z) - onehot(y)`——指数与对数抵消，数值稳定且实现简单（这就是 PyTorch 用 `CrossEntropyLoss(logits, target)` 而非手动 softmax+CE 的原因）。

**常见追问**：
- 为什么手动 `softmax(logits)` 后再算 `log` 会精度差？（先 exp 后 log，中间大数丢失精度）
- 混合精度训练里这个 trick 为什么更重要？（fp16 动态范围小）

**📌 项目对应**：训练侧基础；`app/utils/structured_json.py` 的"坏 JSON 修复"与数值稳定是同一类工程思想（兜底 + 稳健解析）。

---

# 第二部分 · 八股清单

> 每题给"一句话答案 + 要点"，作为速查；深挖版见 `docs/INTERVIEW_PREP.md`。

## 第六类：模型原理八股

### 6.1 为什么 attention 要除以 sqrt(d_k)？
**一句话**：防止点积随维度增大导致 softmax 进入饱和区、梯度趋零。
**要点**：Q·K 是 d_k 个独立随机变量求和，方差 ≈ d_k；除 sqrt(d_k) 把方差归一为 1，softmax 输入分布稳定。

### 6.2 Transformer 整体结构？
**一句话**：Embedding + 位置编码 → N 层（多头自注意力 + 前馈网络，各有残差与归一化）→ 输出层。
**要点**：编码器双向、解码器带 causal mask；FFN 通常 4×d_model；每层 = Pre-LN/Post-LN + 残差。

### 6.3 Pre-LN vs Post-LN 的区别？
**一句话**：Pre-LN 归一化在残差之前（训练稳、无需 warmup，主流）；Post-LN 在残差之后（原论文，需 warmup）。
**要点**：Pre-LN 的梯度路径更干净（恒等捷径），深层更稳；表达力理论上略弱，实践中无差。

### 6.4 为什么 LLM 用 RMSNorm 不用 LayerNorm？
**一句话**：去均值中心化在残差结构里收益甚微，省计算、少参数、更快。
**要点**：残差连接已提供隐式中心化；RMSNorm 只需算 RMS 缩放 + 一个 weight。

### 6.5 位置编码：绝对 vs 相对 vs RoPE vs ALiBi？
**一句话**：绝对编码给每个位置一个向量（外推差）；相对编码只依赖位置差（外推好）；RoPE 用旋转矩阵实现相对位置（主流，可插值外推）；ALiBi 直接加线性偏置（免训练外推）。
**要点**：RoPE 对 Q/K 旋转、内积自动含相对位置；NTK/YaRN 缩放 base 或频率实现外推。

### 6.6 temperature / top_p / top_k 怎么调？
**一句话**：temperature 控制分布锐度，top_p 控制候选集动态大小，top_k 固定候选数。
**要点**：事实/代码任务低温（0~0.3）；创作任务高温（0.8~1.2）；top_p 通常 0.9~0.95；先 top_k 粗截再 top_p 精截。

### 6.7 LLM 为什么有幻觉？怎么缓解？
**一句话**：模型学的是"下一个 token 的概率"，不区分事实与虚构；知识过期、训练分布偏差、解码随机性都会放大。
**缓解**：RAG（检索事实）、引用溯源、限制"只依据上下文"、低温采样、self-consistency、事实性评测（hallucination 检测）。
**追问**：幻觉分"事实性幻觉"与"忠实性幻觉"（答非所问/与上下文矛盾）。

### 6.8 为什么上下文长度有上限？长上下文为什么贵？
**一句话**：三方面——attention 计算量 O(n²)、KV cache 显存随 n 线性涨、训练数据里长序列稀少。
**要点**：Lost in the Middle（模型对中间内容记忆差）；工程对策：窗口+摘要+检索（你项目 `app/context/compress.py` 的三层答案）。

### 6.9 流式生成（SSE）原理？
**一句话**：解码器逐 token 自回归生成，边生成边推送（SSE 文本流），不是等整段完成。
**要点**：首 token 延迟（TTFT）与总延迟分开看；流式中断可断点续传（游标记录）；流式下重试只能在首块之前。

### 6.10 为什么 LLM 用 GELU/SiLU 不用 ReLU？
**一句话**：平滑激活梯度更好、无死亡 ReLU；SwiGLU 加门控更灵活，是 Llama 标配。
**要点**：ReLU 在负半轴梯度为 0 → 神经元死亡；平滑激活代价是计算略多。

---

## 第七类：Agent 八股

### 7.1 模型是怎么"选"到正确工具的？
**一句话**：不是查表/检索，而是**生成**——工具描述作为 context 参与 next-token 条件概率，模型生成 `name` 与 `arguments`。
**要点**：匹配质量受工具名语义、description 质量、参数 schema 字段命名、工具总数影响（>20 个准确率明显下降 → 工具路由）；参数是模型从用户话里"抽取+补全"的（详见 INTERVIEW_PREP.md 0.1/0.2）。

### 7.2 ReAct vs Plan-and-Execute vs Reflexion？
**一句话**：ReAct 交替思考-行动（每步看结果）；Plan-and-Execute 先整体规划再分步执行（长任务更可控）；Reflexion 事后把失败经验写回记忆（自我改进）。
**要点**：ReAct 简单通用但长任务可能走偏；Plan 模式规划失败难修正；Reflexion 需要失败信号（测试/评测）。

### 7.3 Agent 为什么会"失控"？怎么兜底？
**一句话**：自回归生成没有天然终止保证，可能在错误工具间循环、无限消耗 token。
**兜底三层**：迭代上限（max_steps）→ 成本熔断（token 预算，`app/security/cost_guard.py`）→ 结构化错误（让模型自纠）+ HITL（危险操作人工审批）。
**追问**：模型"成功但没进展"怎么检测？（轨迹里工具调用无新信息 → 停滞检测）

### 7.4 多智能体（Multi-Agent）模式有哪些？什么时候用？
**一句话**：orchestrator-worker（主从分工）、debate（多模型辩论）、角色扮演（专用 subagent）。
**要点**：不是越多越好——通信成本、上下文膨胀、错误传播；单 Agent + 工具集足够时别上多智能体。你项目 `app/agent/subagents/` 是 orchestrator-worker 模式（内置 diet_planner/generic 专家 + 独立工具集）。

### 7.5 记忆有哪些类型？
**一句话**：工作记忆（当前上下文窗口）/ 短期记忆（会话内）/ 长期记忆（跨会话持久化）；认知科学分 episodic（事件）/ semantic（知识）/ procedural（技能）。
**要点**：工程实现 = 短期（窗口+摘要压缩）+ 长期（向量/关键词检索注入 system prompt）；记忆要可查看/可删除/可脱敏。

### 7.6 上下文压缩：截断 vs 摘要 vs 检索？
**一句话**：截断 = 丢弃（信息不可恢复），摘要 = 有损压缩（保留高价值事实），检索 = 按需取用（最省但每次多一跳延迟）。
**要点**：生产 = 先压缩再截断；增量摘要（旧摘要+新对话融合）避免重复叙述（你项目 `app/context/compress.py`）。

### 7.7 工具调用失败怎么让模型自愈？
**一句话**：返回结构化错误（error_code / retryable / suggestion），让模型推理恢复路径而不是框架拍板。
**要点**：`retryable=false` 注入"别重试"的领域知识；失败计数熔断；工具独立超时（详见 2.5）。

### 7.8 Agent 怎么评测？
**一句话**：三层——工具 SLO（每次调用成功率/延迟）→ 任务完成率（黄金任务集端到端）→ 轨迹质量（人工/LLM 审）。
**要点**：黄金任务集要覆盖典型+边界场景；回归门禁（baseline 对比，下降即拦截）；LLM 判定 + 启发式回退双保险（你项目 `app/evaluation/` + `scripts/run_agent_evaluation.py`）。

### 7.9 Agent 的幻觉与普通 LLM 幻觉有何不同？
**一句话**：Agent 幻觉还包括**工具幻觉**——虚构工具结果、编造工具名/参数。
**要点**：校验工具输出 schema、限制"只依据 Observation 回答"、工具结果注入检测（`app/security/injection.py` 双层检测）。

### 7.10 什么任务适合 Agent？什么不适合？
**一句话**：需要多步推理、调用外部工具、动态决策 → Agent；单轮问答、纯知识检索 → 直接 RAG/LLM 更省。
**要点**：Agent 的代价是延迟、成本、不确定性；先问"单次调用能不能解决"。

---

## 第八类：RAG 八股

### 8.1 RAG vs 长上下文 vs 微调，怎么选？
**一句话**：知识**实时/私有/可追溯** → RAG；知识**风格/格式**固化 → 微调；简单事实 → 长上下文。
**要点**：RAG 优点（更新零成本、可引用、不改变模型）；缺点（检索失败、切块损失）；微调改变模型行为但不注入新知识。

### 8.2 Naive / Advanced / Modular RAG 演进？
**一句话**：Naive（切块-检索-生成单链路）→ Advanced（query 改写、重排、混合检索、多级缓存）→ Modular（检索器/路由/融合模块化编排）。
**要点**：你的项目是 Advanced 级别（改写 + 混合 + 重排 + 双层缓存，`app/rag/`）。

### 8.3 chunk 大小怎么定？
**一句话**：太小语义不全、太大稀释——中文 200~500 字、英文 300~800 token 起步，用评测集对比 context_recall 定。
**要点**：重叠防劈裂；结构化切块（标题层级）优于纯长度；small-to-large 回传父文档。

### 8.4 混合检索的融合策略？
**一句话**：RRF（只依赖排名，`1/(k+rank)`，k≈60）或加权融合（必须先归一化分数）。
**要点**：稠密管语义、稀疏管词面，互补；动态加权按查询意图调（你项目 `retrieval.py` 的 `intelligent_ranker_selection`）。

### 8.5 为什么 Rerank 放在检索之后？
**一句话**：交叉编码器复杂度 O(n²)（query×doc），只能处理小集合；召回粗排（双塔/BM25 快）→ 精排（重排准）两阶段。
**要点**：检索 top_k（20~50）→ rerank top_n（3~5）→ 生成；rerank 分数不与检索分数直接可比。

### 8.6 Query Rewrite 解决什么问题？
**一句话**：多轮指代消解（"它"）、口语化噪声压缩、改写为检索友好表达。
**要点**：改写错误会污染检索——可多版本并行检索或校验；改写本身一次 LLM 调用（成本与延迟）。

### 8.7 RAG 评测指标有哪些？
**一句话**：检索质量（context_precision / context_recall、命中率）+ 生成质量（faithfulness / answer_relevancy / answer_correctness）。
**要点**：离线（有 ground truth，回归门禁）+ 在线（LLM 裁判）；`context_recall` 低而 precision 高 → 排序问题（调 top_k/rerank），都低 → 召回问题（换 embedding/切块）。

### 8.8 RAG 的引用与幻觉控制？
**一句话**：生成时强制"只依据上下文 + 引用 [n]"，评测时校验引用序号与上下文一致性。
**要点**：检索为空要兜底（明确告知"知识库无相关内容"）；工具返回内容注入检测（`app/security/`）。

### 8.9 向量检索 vs BM25 的本质？
**一句话**：向量 = 语义空间连续近似（对措辞鲁棒、对精确实体弱）；BM25 = 词频-逆文档频率（对词面强、对同义改写弱）。
**要点**：否定/多约束场景 embedding 上限（召回相关但不符合约束）——rerank 也未必纠得回；多跳推理两类检索都召不到（→ Agent 化/GraphRAG）。

---

## 第九类：训练与微调八股

### 9.1 SFT / RLHF / DPO 的区别与流程？
**一句话**：SFT 用标注答案监督微调；RLHF 训奖励模型 + PPO 在线优化；DPO 直接从偏好对优化（隐式奖励，免 RM/PPO）。
**要点**：RLHF 流程 SFT → RM → PPO；DPO 损失见 5.4；RLHF 稳但复杂，DPO 简单但离线。

### 9.2 LoRA / QLoRA 的原理与显存账？
**一句话**：LoRA 冻结原权重、只训低秩 ΔW=B·A（r×参数）；QLoRA 把原权重量化到 4bit + LoRA 微调。
**要点**：可训练参数量 r×(in+out)；梯度/优化器状态只对可训练参数存；推理时 ΔW 可合并回 W' 无额外延迟。

### 9.3 全参微调 vs LoRA vs 冻结？
**一句话**：全参效果上限最高但显存/成本高、易灾难遗忘；LoRA 省显存、可控、效果接近；冻结不训（只做 prompt/RAG）。
**要点**：数据少时 LoRA 更稳；r 太大不必然更好（低秩假设）。

### 9.4 学习率、batch size、梯度累积？
**一句话**：LLM 微调 lr 通常 1e-5~2e-5（比从头训练小 1~2 个数量级）；显存不够用梯度累积模拟大 batch。
**要点**：梯度累积 = 多个小 batch 梯度累加后再 step（等效大 batch，注意 BN 差异——transformer 无影响）；warmup 稳定训练。

### 9.5 混合精度（FP16/BF16）？
**一句话**：FP16 训练快省显存但动态范围小（需要 loss scaling）；BF16 动态范围与 FP32 相同（无需 scaling），是 LLM 训练首选。
**要点**：master weights 保持 FP32；数值稳定 trick（softmax 减 max）在低精度下更重要。

### 9.6 怎么判断微调过拟合？怎么缓解？
**一句话**：训练 loss 降而验证 loss 升/评测指标降 → 过拟合。
**要点**：缓解 = early stopping、数据增强/质量、正则（weight decay）、LoRA 低秩、评测集与训练集严格分离（你项目评测回归门禁的用途）。

### 9.7 蒸馏（Distillation）是什么？
**一句话**：用大模型（teacher）的输出/软标签训练小模型（student），压缩能力。
**要点**：软标签带分布信息（temperature 放大）；比直接用小数据训小模型效果好。

---

## 第十类：工程落地八股（趋势重点：从调 API 到写工程）

> 面试趋势：面试官越来越看重工程落地，会追问**工具调用失败、异常处理、并发场景**等现实问题。这组题回答时一定要结合你项目的真实实现。

### 10.1 重试、退避、抖动、熔断？
**一句话**：瞬时故障重试（指数退避 + 随机抖动防雪崩），持续故障换路径（降级链），超预算熔断（成本保护）。
**要点**：先分类能不能重试（`is_retryable_error`）；抖动 = 每次退避加随机偏移，否则所有客户端同时重试打挂下游；流式只重试首块前（`app/llm/resilience.py` 的 `async_retry` + `call_with_fallback` + 模型降级链）。

### 10.2 并发与幂等？
**一句话**：同一请求重复提交要幂等（幂等键/去重）；并发工具调用独立执行、按 tool_call_id 聚合；共享状态跨实例一致。
**要点**：审批决策"后端权威、进程内仅镜像"（你项目 StorageBackend 的返工故事，见 INTERVIEW_PREP.md B3）；成本熔断计数在 Redis 后端下跨实例一致（`app/storage/backend.py`）。

### 10.3 流式 SSE 的实现与断线重连？
**一句话**：FastAPI StreamingResponse/EventSource 逐块推送；前端断线记录游标（已消费位置），重连续传。
**要点**：流式与审批/工具事件混推（事件类型区分）；流式响应头不能压缩缓冲（`X-Accel-Buffering: no`）；你项目 `app/agent/event_stream.py` + 前端 `hooks/useAgent.ts` 断点恢复。

### 10.4 限流与成本控制？
**一句话**：令牌桶/漏桶限 QPS；token 预算（单会话/单轮）超限 degrade/refuse/warn。
**要点**：限流要在网关层（用户维度）；成本熔断要持久化（进程内会丢）；你项目 `app/security/middleware/rate_limiter.py` + `cost_guard.py`。

### 10.5 多租户隔离？
**一句话**：数据按 user_id 隔离（DB 层）；**有状态工具按会话克隆**（进程内实例隔离，防串数据）；权限按用户×工具矩阵。
**要点**：工具执行器持有 session 级副本（`clone_for_session`）；缓存/记忆按用户键隔离。

### 10.6 超时设计？
**一句话**：LLM 调用、工具调用、整体对话各自独立超时，超时 → 结构化错误回喂模型。
**要点**：不同工具不同预算（搜索 10s vs 计算 3s）；`asyncio.wait_for` 包一层；超时与重试配合（超时可重试、认证失败不可）。

### 10.7 Prompt 注入怎么防？
**一句话**：双层检测——用户输入检测 + **工具返回内容检测**（指令覆盖/提示词泄露/越狱/`<|im_start|>`），权限矩阵 + HITL 兜底。
**要点**：工具返回内容是最常被忽视的注入面；检测是正则+LLM 双路；你项目 `app/security/injection.py` + `guardrails/`（NeMo Guardrails 风格 input/output rails）。

### 10.8 可观测性怎么做？
**一句话**：traceId 全链路 + 结构化日志 + **Agent 轨迹落盘与回放**。
**要点**：traceId 用 contextvars 随协程链传播（异步局部变量）；轨迹 = 一次 turn 的决策历史（看到什么/调了什么/答了什么）；轨迹要脱敏 + 采样（你项目 `app/telemetry/` + `data/trajectories/`）。

### 10.9 JSON 解析健壮性？
**一句话**：LLM 输出 JSON 不可信——修复流水线（尾逗号/单引号/未加引号 key/截断）+ schema 校验 fail-fast。
**要点**：修复后仍失败返回结构化错误让模型重试；工具 arguments 与模型输出共用同一修复器（`app/utils/structured_json.py`）。

### 10.10 缓存策略？
**一句话**：L1 精确缓存（query 完全一致，Redis）+ L2 语义缓存（embedding 相似 > 阈值，Milvus），只缓存检索结果不缓存生成。
**要点**：缓存命中率可观测；失效策略（文档更新清缓存）；语义缓存阈值是精度 trade-off（你项目 `app/rag/cache/`）。

### 10.11 生产化优先级（如果只做三件事）？
**一句话**：架构风险（多实例状态一致）→ 质量风险（评测回归门禁）→ 合规风险（脱敏/审计/HITL）。
**要点**：先回答"为什么这三件"再说怎么做（判断力 > 清单）；你项目 `app/storage/backend.py`（Redis 多实例）、`app/evaluation/`、`app/security/` 正好对应。

---

# 附录 · 面试趋势解读 & 复习路线

## 趋势一：从"调 API"到"写工程"

- 只讲"我调了 GPT-4 API"不再有竞争力；面试官追问的是**失败怎么办、并发怎么办、多实例怎么办、成本怎么办**。
- 对应准备：把第十类工程八股每题配一个你项目的真实代码位置 + 一个真实踩坑故事（如审批跨实例一致性返工、SQLite 暴露三个上游 bug、前端白屏的契约漂移）。
- 主动暴露设计债（"当前 AUTOINDEX 应显式 HNSW"、"轨迹 100% 落盘应降采样"、"推理层未接 vLLM"）比被问倒加分。

## 趋势二：手撕题"小而深"

- 高频题（MHA、ReAct、Tool Registry、Top-K/P、LayerNorm/RMSNorm、LoRA、DPO）要求**写出关键 20 行 + 讲清每个细节**，不需要背完整库代码。
- 面试官会"换皮"考：MHA → 写 causal mask；ReAct → 加错误恢复；RAG → 加融合策略。准备时把每题常见追问背熟。

## 复习路线（建议 2~3 周）

| 周次 | 内容 | 产出 |
|---|---|---|
| 第 1 周 | 第一类全部手撕（MHA/RoPE/LN/RMSNorm/采样/BPE/SwiGLU）+ 第六类模型八股 | 每题能白板写核心代码 + 讲清原理 |
| 第 2 周 | 第二、三类（ReAct/Tool Registry/记忆/RAG 三题）+ 第七、八类八股 | 结合 `app/agent/`、`app/rag/` 源码讲实现 |
| 第 3 周 | 第四、五类（KV cache/GQA/反向传播/Adam/LoRA/DPO）+ 第九、十类八股 | 补工程故事 + 模拟面试录音 |

## 与项目内材料配合

| 材料 | 定位 |
|---|---|
| 本文档 `AI技术岗-手撕与八股清单.md` | 题库索引 + 参考实现 + 项目对应点 |
| `cookhero/CookHero/docs/INTERVIEW_PREP.md` | 项目相关题目的原理深挖（traceId 机制、混合检索边界、选型决策、踩坑复盘） |
| `cookhero/CookHero/README.md` | 项目话术总纲（七层架构、P0/P1/P2 优先级） |
| `cookhero/CookHero/docs/STORAGE_BACKEND.md` | 多实例一致性的深挖素材（审批/熔断/SLO） |
