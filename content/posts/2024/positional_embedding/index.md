---
title: "旋转位置编码"
description: "Whole hugo blog in plain text!"
date: 2024-02-01
lastmod: 2024-03-12T20:33:38+08:00
tags: ["transformer", "rope"]
draft: false
weight: 1002
toc: true
---

## 背景 {#背景}


### 主要内容 {#主要内容}

-   内容
    -   旋转位置编码的来龙去脉
    -   代码实现
-   参考：
    -   论文：2021. Enhanced Transformer with Rotary Position Embedding (<a href="#citeproc_bib_item_1">Su et al. 2022</a>)
    -   [hugging face llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)


### 回顾transformer {#回顾transformer}

-   encoder: 低阶语义向量序列转化为高阶的语义向量序列

    \\(\mathcal{T}\begin{pmatrix}
       X\_1\\\\
       X\_2\\\\
       \vdots\\\\
       X\_N
       \end{pmatrix}
       \rightarrow\begin{pmatrix}
       Y\_1\\\\
       Y\_2\\\\
       \vdots\\\\
       Y\_N
       \end{pmatrix}\\)
-   两个变换组成 \\(Y=\mathcal{T}(X)=\mathcal{F}(\mathcal{A}(X))\\)
    -   Attention \\(\mathcal{A}\\)
    -   Feedforward \\(\mathcal{F}\\)
-   详细

    \\(\begin{aligned}Q\_{i} &= X\_{i} W\_{Q} \\\\
       K\_{i} &= X\_{i} W\_{K}\\\\
       V\_{i} &= X\_{i} W\_{V}\\\\
       Y\_{i} &= \sum\_{j=1}^{N}sim(Q\_i,K\_{j}) V\_j\\\\
       sim(Q\_{i},K\_j) = &= \frac{exp(\frac{Q\_{i}K\_{j}^{T}}{\sqrt{d}})}
       {\sum\_{j=1}^N exp(\frac{Q\_iK\_j^{T}}{\sqrt{d}})}\\\\
       \end{aligned}\\)


### 为什么需要位置编码 {#为什么需要位置编码}

-   因为transformer结构本身是和距离无关的，
    \\(Y=\mathcal{T}(X)=\mathcal{F}(\mathcal{A}(X))\\)
-   高阶语义向量不仅仅是由周围token的语义向量组合表达而成
    -   还需要加上每个token所处的位置
-   下面的cls token得到的语义向量是完全一样的。
    -   &lt;cls&gt; 从 北京 到 上海 的 火车票
    -   &lt;cls&gt; 从 上海 到 北京 的 火车票
-   其他的网络结构天然有序列的位置信息 RNN/CNN


## 如何加入位置编码 {#如何加入位置编码}

\\(\mathcal{T}(X)=\mathcal{F}(\mathcal{A}(X))\\)

-   \\(\mathcal{F}\\) 是位置无关的
-   可以修改 \\(X\\) 或者 \\(\mathcal{A}\\)


### 1. 直接修改输入，加入绝对位置编码 {#1-dot-直接修改输入-加入绝对位置编码}

在\\(X\_i \rightarrow Q\_i, K\_i, V\_i\\) 之前，直接加入位置的embedding
\\(X\_i^{'}=X\_i+P\_i\\)


#### learned embedding {#learned-embedding}

-   优点简单，bert/GPT
-   外推困难，对于超过序列最大长度的位置


#### 自定义绝对位置编码 {#自定义绝对位置编码}

-   二维函数 f(position, dimension)
-   要求
    -   函数随着position,dimension增长应该是有界的
    -   足够的区分度，对position, dimension
-   例子

    \\(\begin{aligned}
       P\_{i,2t} &= sin(k/10000^{2t/d}) &&\\\\
       P\_{i,2t+1} &= cos(k/10000^{2t/d})&&\\\\
       \end{aligned}\\)


#### 问题 {#问题}

语义应该是和相对位置有关的，而不是绝对位置


### 2. 修改Attention，加入相对位置信息 {#2-dot-修改attention-加入相对位置信息}


#### 回顾Attention {#回顾attention}

\\(\begin{aligned}Q\_{i} &= X\_{i} W\_{Q} \\\\
   K\_{i} &= X\_{i} W\_{K}\\\\
   V\_{i} &= X\_{i} W\_{V}\\\\
   Y\_{i} &= \sum\_{j=1}^{N}sim(Q\_i,K\_{j}) V\_j\\\\
   sim(Q\_{i},K\_j) &= \frac{exp(\frac{Q\_{i}K\_{j}^{T}}{\sqrt{d}})}
{\sum\_{j=1}^N exp(\frac{Q\_iK\_j^{T}}{\sqrt{d}})}\\\\
   \end{aligned}\\)

可以从相似性入手，让位置的相对关系反应到q，k的相似性中来。


#### 希望 {#希望}

相似性计算只依赖向量还有相对距离, 而不依赖于其绝对的位置。

\\(Q\_{i}K\_j^T=g(X\_{i},X\_j,i-j)\\)


## 旋转位置编码 {#旋转位置编码}


### 在二维空间中看motivation {#在二维空间中看motivation}

假设\\(Q\_{i}, K\_j\\) 都是二维的向量，\\(i, j\\) 是它们对应的position，
这里\\(\eta\_{i},\eta\_{j}\\) 是$Q_i, K_j$向量的弧度表示对应的角度.

-   点击只和模长和夹角有关
    -   \\(Q\_iK\_j^T=\\|Q\_i\\|\\|K\_j\\| cos(\eta\_{i}-\eta\_j)\\),
-   如果: 基于位置乘倍数旋转之后做点击
    -   我们把两个向量各自旋转\\(i\theta,j\theta\\) 后再来计算点击
    -   其中\\(\theta\\) 是一个单位角度，
    -   应该就只和\\(Q\_i,Q\_j,i-j\\) 相关了，
-   因为: 模长没有变，只是夹角变了，夹角增加了 \\((i-j)\theta\\).
    -   \\(Q\_iR(i\theta)(K\_jR(j\theta))^T=\\|Q\_i\\|\\|K\_j\\| cos(\eta\_{i}-\eta\_{j}+(i-j)\theta)\\)


### 回顾矩阵的知识 {#回顾矩阵的知识}


#### 关于行向量和矩阵 {#关于行向量和矩阵}

-   定义线性算子 \\(\mathcal{A}\\)
    -   可以作用到行向量  \\(\mathcal{A}(X\_i) = X\_{i} A\\)
    -   也可以作用到矩阵  \\(\mathcal{A}(X) = XA\\)
-   右乘矩阵等于对每个行向量逐个施加行变换

    \\(XA=\begin{pmatrix}
       X\_1\\\\
       X\_2\\\\
       \vdots\\\\
       X\_N
       \end{pmatrix}A=
       \begin{pmatrix}
       X\_1 A\\\\
       X\_2 A\\\\
       \vdots\\\\
       X\_N A
       \end{pmatrix}=
       \begin{pmatrix}
       \mathcal{A}(X\_1) \\\\
       \mathcal{A}(X\_2) \\\\
       \vdots\\\\
       \mathcal{A}(X\_N)
       \end{pmatrix}=\mathcal{A}(X)\\)
-   算子是对矩阵乘法的一种物理理解
    -   旋转矩阵

        \\(R(\theta)=
             \begin{pmatrix}
             cos\theta& sin\theta\\\\
             -sin\theta& cos\theta
             \end{pmatrix}\\)
    -   缩放变换

        \\(R(\lambda\_1,\lambda\_2)=\begin{pmatrix} \lambda\_1 & \\\\
                & \lambda\_2 \\\ \end{pmatrix}\\)


#### 关于旋转矩阵 {#关于旋转矩阵}

-   旋转矩阵

    \\(R(\theta)=
       \begin{pmatrix}
       cos\theta& sin\theta\\\\
       -sin\theta& cos\theta
       \end{pmatrix}\\)

-   物理意义

    \\(X\_iR(i\theta)\\) 对位置在\\(i\\) 的语义向量\\(X\_i\\) 逆时针旋转\\(i\theta\\)
-   性质
    -   \\(R(\theta)^T=R(-\theta)\\)
    -   \\(R(\theta\_1)(\theta\_2)=R(\theta\_1+\theta\_{2})\\)

{{< figure src="images/2024-03-08_11-42-35_screenshot.png" width="200px" >}}


### 二维空间的一个解 {#二维空间的一个解}


#### 基于旋转矩阵的一个解 {#基于旋转矩阵的一个解}

\begin{equation\*}
\begin{split}
Q\_{i}&= X\_{i} W\_{Q} R(i\theta) \\\\
K\_{j}&= X\_j W\_{K} R(j\theta)\\\\
Q\_{i}K\_j^T &=X\_{i}W\_QR(i\theta)R(j\theta)^{T}W\_K^{T}X\_{j}^T\\\\
&=X\_{i}W\_QR(i\theta)R(j\theta)^{T}W\_K^{T}X\_{j}^T\\\\
&=X\_{i}W\_QR(i\theta)R(-j\theta)W\_K^{T}X\_{j}^T\\\\
&=X\_{i}W\_QR((i-j)\theta)W\_K^{T}X\_{j}^T\\\\
& =g(X\_i,X\_j,i-j)\\\\
     \end{split}
     \end{equation\*}


#### 为什么是在投影之后旋转，不在投影之前转？ {#为什么是在投影之后旋转-不在投影之前转}

\begin{equation\*}
\begin{split}
Q\_{i}&= f\_{Q}(X\_{i}, i)  = X\_{i} R(i\theta) W\_{Q} \\\\
K\_{j}&= f\_{K}(X\_{j}, j)  = X\_j R(j\theta) W\_{K} \\\\
Q\_{i}K\_j^T &=X\_{i}R(i\theta)W\_QW\_KR(j\theta)^{T}X\_{j}^T\\\\
&=?\\\\
     \end{split}
     \end{equation\*}


### 推广到高纬的空间 {#推广到高纬的空间}

整个空间分割成\\(d/2\\) 个子空间，在各个子空间上分别按照一个位置相关的角度旋转


#### 定义 \\(R(i\Theta)\\) {#定义-r--i-theta}

-   \\(X\_{i}R(i\Theta)\\)
    表示对\\(X\_{i}\\) 在各个子空间分别做角度为\\(i\theta\_1,i\theta\_2,\ldots,i\theta\_{d/2}\\) 的旋转.
    \\(\Theta=(\theta\_{1},\theta\_2,\ldots,\theta\_{d/2})\\)
    \\(R(i \Theta)=\begin{pmatrix}
       cos\\,i\theta\_{1} & sin\\,i\theta\_1 & 0 & 0 \\\\
       -sin\\,i\theta\_{1} & cos\\,i\theta\_1 & 0 & 0 \\\\
       0 & 0 & cos\\,i\theta\_{2} & sin\\,i\theta\_2 \\\\
       0 & 0 & -sin\\,i\theta\_{2} & cos\\,i\theta\_2 \\\\
       \end{pmatrix}=\begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & R(i\theta\_2)
       \end{pmatrix}\\)


#### 物理意义 {#物理意义}

-   依次在独立的二维子空间上做旋转变换

    利用分块矩阵的乘法，我们观察一下, 把对应行向量\\(X\_i\\) 切分为两部分，用上角标来区分

    \\(X\_i = (X\_{i}^1, X\_{i}^2)\\)

    \\(XR(i\Theta)=(X^1, X^2)\begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & R(i\theta\_2)
       \end{pmatrix}=(X^1R(i\theta\_1), X^2R(i\theta\_2))\\)

    可以看出这个矩阵的变化的作用就是在各个独立的二维子空间上分别做独立的旋转变化，最后把变换后的向量拼接即可
-   性质: \\(R(i\Theta)=\widehat{R}(i\theta\_1)\widehat{R}(i\theta\_2)\ldots\widehat{R}(i\theta\_{d/2})\\)

    定义\\(\widehat{R}(i\theta\_1)=
       \begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & 0 \\\\
       \end{pmatrix}\\)

    \\(R(i\Theta)=\begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & R(i\theta\_2)
       \end{pmatrix}=\begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & 0 \\\\
       \end{pmatrix}\begin{pmatrix}
       0 & 0 \\\\
       0 & R(i\theta\_2)
       \end{pmatrix}=\widehat{R}(i\theta\_1)\widehat{R}(i\theta\_2)\\)
-   在第一个二维空间按照 \\(\theta\_{1}\\) 来旋转，第二个 \\(\theta\_{2}\\) 来旋转


#### ROPE在高维空间 {#rope在高维空间}

\begin{equation\*}
\begin{split}
Q\_{i}& = X\_{i} W\_{Q} R(i\Theta) \\\\
K\_{j}& = X\_j W\_{K} R(j\Theta)\\\\
Q\_{i}K\_j^T &=X\_{i}W\_QR(i\Theta)R(j\Theta)^{T}W\_K^{T}X\_{j}\\\\
&=X\_{i}W\_QR(i\Theta)R(j\Theta)^{T}W\_K^{T}X\_{j}\\\\
&=X\_{i}W\_QR((i-j)\Theta)W\_K^{T}X\_{j}\\\\
&=g(X\_i,X\_j,i-j)\\\\
\end{split}
\end{equation\*}

其中

\begin{equation\*}
\begin{split}
R(i\Theta)R(j\Theta)^{T} &= \widehat{R}(i\theta\_1)\widehat{R}(i\theta\_2)\ldots\widehat{R}(i\theta\_{d/2})\widehat{R}(j\theta\_{d/2})^{T}\ldots \widehat{R}(j\theta\_{2})^{T} \widehat{R}(j\theta\_{1})^{T} \\\\
&= (\widehat{R}(i\theta\_1)\widehat{R}(j\theta\_1)^T)(\widehat{R}(i\theta\_2)\widehat{R}(j\theta\_2)^T)\ldots(\widehat{R}(i\theta\_{d/2}\widehat{R}(j\theta\_{d/2})^T)\\\\
&= \widehat{R}((i-j)\theta\_1)\widehat{R}((i-j)\theta\_2)\ldots \widehat{R}((i-j)\theta\_{d/2})\\\\
&= R((i-j)\Theta)\\\\
\end{split}
\end{equation\*}

其中\\(\theta\_{k}\\) 是超参数，\\(\theta\_{k}=10000^{-2(k-1)/d}, k\in[1,2,\ldots,d/2]\\)


### 总结旋转位置编码 {#总结旋转位置编码}


#### 总结 {#总结}

-   旋转位置编码是针对\\(Q,K\\) 的每个行向量做对应的位置旋转变换

    \\(Q\_{i} = X\_{i} W\_{Q} R(i\Theta)\\)

    \\(K\_{j} = X\_{j} W\_{K} R(j\Theta)\\)

-   位置旋转矩阵定义 \\(R(i\Theta)\\)

    其中 \\(\Theta=(\theta\_{1},\theta\_2,\ldots,\theta\_{d/2})\\), \\(\theta\_{k}=10000^{-2(k-1)/d}, k\in[1,2,\ldots,d/2]\\)
    那么

    \\(R(i\theta)=
       \begin{pmatrix}
       cos i\theta& sin i\theta\\\\
       -sin i\theta& cos i\theta
       \end{pmatrix}\\)

    \\(R(i \Theta)=\begin{pmatrix}
       cos\\,i\theta\_{1} & sin\\,i\theta\_1 & 0 & 0 & 0 & 0 &0\\\\
       -sin\\,i\theta\_{1} & cos\\,i\theta\_1 & 0 & 0 & 0 & 0 &0 \\\\
       0 & 0 & cos\\,i\theta\_{2} & sin\\,i\theta\_2 & 0 & 0 &0 \\\\
       0 & 0 & -sin\\,i\theta\_{2} & cos\\,i\theta\_2& 0 & 0 &0  \\\\
       0 & 0 & 0 & 0 & \ldots &0 & 0 \\\\
       0 & 0 & 0 & 0 &\ldots & cos\\,i\theta\_{d/2} & sin\\,i\theta\_{d/2}  \\\\
       0 & 0 & 0 & 0 &\ldots & -sin\\,i\theta\_{d/2} & cos\\,i\theta\_{d/2}
       \end{pmatrix}\\)

    \\(R(i\Theta)=\begin{pmatrix}
       R(i\theta\_{1}) & 0 &0 & 0\\\\
       0 & R(i\theta\_2) & 0 &0 \\\\
       0 & 0 &\ldots &0  \\\\
       0 & 0 & 0 &R(i\theta\_{d/2})\\\\
       \end{pmatrix}\\)


#### 再看下绝对位置编码 {#再看下绝对位置编码}

\\(\begin{aligned}
   P\_{i,2t} &= sin(i/10000^{2t/d}) &&\\\\
   P\_{i,2t+1} &= cos(i/10000^{2t/d})&&\\\\
   \end{aligned}\\)

换个表述的形式，

\\(P\_{i}=\begin{pmatrix}
   B\_1, B\_2, \ldots, B\_{d/2}\end{pmatrix}\\)，   \\(B\_{k}=\begin{pmatrix}
   sin(i\theta\_k),   cos(i\theta\_k)
   \end{pmatrix}\\)
\\(\theta\_{k}=10000^{-2(k-1)/d}, k\in[1,2,\ldots,d/2]\\)


### 代码实现 {#代码实现}


#### 避开旋转矩阵的相乘 {#避开旋转矩阵的相乘}

我们需要对每个\\(Q\_{i}\\) 乘以不同的旋转矩阵，也就是
\\(QR=\begin{pmatrix}
Q\_1 R(1\Theta)\\\\
Q\_2 R(2\Theta)\\\\
\ldots \\\\
Q\_N R(N\Theta)\\\\
\end{pmatrix}\\)

假设是二维空间，把\\(Q\\) 拆分成两个列向量\\(U,V\\), 记录

\\(cos=\begin{pmatrix}cos1\theta \\\\
cos 2\theta\\\ \ldots,\\\ cos N\theta
\end{pmatrix},
sin=\begin{pmatrix}sin 1\theta \\\\
sin 2\theta\\\ \ldots,\\\ sin N\theta
\end{pmatrix}\\)

那么

\\(\begin{aligned}
QR&=\begin{pmatrix}
u\_1 cos 1\theta-v\_1 sin 1\theta, u\_1 sin 1\theta + v\_1 cos 1\theta\\\\
u\_2 cos 2\theta-v\_2 sin 2\theta, u\_2 sin 2\theta + v\_2 cos 2\theta\\\\
\ldots\\\\
u\_N cos N\theta-v\_N sin N\theta, u\_N sin N \theta + v\_N cos N\theta\\\\
\end{pmatrix}\\\\
&=(U \* cos - V\* sin, U\*sin+V\*cos) \\\\
&= (U,V)cos +(V, -U) sin
\end{aligned}\\)

同样的，在高维空间，我们可以把\\(Q\\) 拆分成\\(d/2\\) 个列向量\\(U\_1,V\_1,U\_2,V\_2,\ldots,U\_{d/2},V\_{d/2}\\)


#### tricks {#tricks}

-   trick2：不需要做严格紧密相连的二维子空间序列，将整个空间分成两部分
    -   第一个部分放的是每个子空间的第一维度，第二部分放置的是每个子空间的第二维度
        ```text
        (x1,y1) 是一个子空间，(x2, y2)是一个子空间，(x3, y3)是一个子空间
        before： [(x1,y1), (x2,y2), (x3,y3)]
        after： [(x1,x2,x3), (y1, y2, y3)]
        ```


#### code {#code}

```python
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
class Rotator:
    """根据hidden_dim，和position_ids 生成对应的旋转位置编码, 和论文中定义略有不同，一个个二维的子空间被
    分割到了前后两部分，分别进行旋转，然后拼接起来
    """

    def __init__(self, dim, position_ids):
        """ position_ids: [seq_len], dim 和单个头的hidden_dim对应 """
        base = 10000
        theta_base = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        thetas = position_ids.outer(theta_base)  # [seq_len, D/2]
        full_thetas = torch.cat((thetas, thetas), dim=-1)  # [seq_len, D]
        self.cos = full_thetas.cos()
        self.sin = full_thetas.sin()

    def rotate(self, x):
        """
        x: [bs, num_attention_heads, seq_len, D]
        q: [bs, num_attention_heads, seq_len, D]
        cos: [seq_len, D]
        [x,y] @ [[cos, sin], [-sin, cos]] = [x*cos+y*sin, ycos-x*sin] =[x,y]*cos+[y, -x]*sin
        """
        return x * self.cos + Rotator.reverse_half(x) * self.sin

    @staticmethod
    def reverse_half(q):
        """ q: [bs, num_attention_heads, seq_len, D] """
        x = q[..., : q.shape[-1] // 2]
        y = q[..., q.shape[-1] // 2:]
        return torch.cat((-y, x), dim=-1)


class SelfAttentionWithRoPE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.H = config["n_head"]
        self.F = config["hidden_dim"]  # F
        self.D = self.F // self.H  # D
        # 一次把qkv 全部映射完成，对应W_Q, W_K, W_V
        self.qkv_proj = nn.Linear(self.F, 3 * self.F)
        # 最后的投影，对应于 $W_O$
        self.out_proj = nn.Linear(self.F, self.F)

    def forward(self, x, position_ids):
        # position_ids: [seq_len]
        B, N, _ = x.size()
        q, k, v = self.qkv_proj(x).split(self.F, dim=-1)
        # matmul 只能在最后两个维度相乘，需要对NxD的矩阵相乘，做1,2维度的交换
        k = k.view(B, N, self.H, self.D).transpose(1, 2)
        q = q.view(B, N, self.H, self.D).transpose(1, 2)
        v = v.view(B, N, self.H, self.D).transpose(1, 2)
        # 旋转位置编码
        rotator = Rotator(self.D, position_ids)
        q = rotator.rotate(q)
        k = rotator.rotate(k)
        # 计算相似性
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        # 多头拼接
        y = y.transpose(1, 2).contiguous().view(B, N, self.F)
        y = self.out_proj(y)
        return y


config = {"n_head": 2, "hidden_dim": 16, "batch_size": 3, "seq_len": 5}
attn = SelfAttentionWithRoPE(config)
x = torch.rand(config["batch_size"], config["seq_len"], config["hidden_dim"])
position_ids = torch.arange(config["seq_len"])
y = attn(x, position_ids)
```


## 参考论文 {#参考论文}

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Su, Jianlin, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2022. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv. <a href="https://arxiv.org/abs/2104.09864">https://arxiv.org/abs/2104.09864</a>.</div>
</div>
