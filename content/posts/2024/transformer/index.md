---
title: "深入理解transformer"
description: "Whole hugo blog in plain text!"
date: 2024-02-01
lastmod: 2024-03-12T20:55:01+08:00
tags: ["transformer"]
draft: false
weight: 1001
toc: true
---

## transformer背景 {#transformer背景}


### 主要内容 {#主要内容}

-   主要内容
    -   从高阶语义向量的组合表达来看attention

        \\(\mathcal{A}(X\_i) =  \dfrac{\sum\_{j=1}^{N} sim(Q\_i,K\_j) V\_j}{\sum\_{j=1}^N sim(Q\_i,K\_j)}\\)
    -   提供transformer 的代码讲解
    -   从kernel的角度来看attention

        计算量\\(O(N^2)\\) 源于softmax的存在
    -   linear attention
-   参考
    -   2017. Attention Is All You Need
        (<a href="#citeproc_bib_item_2">Vaswani et al. 2023</a>)
    -   2020. Fast Autoregressive Transformers with Linear Attention
        (<a href="#citeproc_bib_item_1">Katharopoulos et al. 2020</a>)
    -   [mingpt by karpathy](https://github.com/karpathy/minGPT/tree/master/mingpt)


### 矩阵知识 {#矩阵知识}


#### why {#why}

-   原文直接用矩阵来表达
    \\[\begin{aligned}\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\dfrac{QK^T}{\sqrt{d\_k}})V \\\ \mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}\_1,\ldots,\mathrm{head}\_h)W^{O} \\\\
       \mathrm{head}\_i=\mathrm{Attention}(QW\_i^Q, KW^{K}\_i,VW^V\_i)
       \end{aligned}\\]
-   把矩阵剖解成从行向量来看更容易理解


#### 矩阵和行向量 {#矩阵和行向量}

-   矩阵
    \\(X\in R^{N\times F}\\)
    \\(X=\begin{pmatrix}
         X\_{11}, X\_{12},\ldots, X\_{1F} \\\\
         X\_{21}, X\_{22},\ldots, X\_{2F} \\\\
         \vdots\\\\
         X\_{N1}, X\_{N2},\ldots, X\_{NF}
         \end{pmatrix}\\)
-   行向量
    \\(X\_{i}=\begin{pmatrix} X\_{i1}, X\_{i2},\ldots, X\_{iF}\end{pmatrix}, X\_i \in R^{1\times F}\\)
-   分块矩阵
    \\(X=\begin{pmatrix}
       X\_1\\\\
       X\_2\\\\
       \vdots\\\\
       X\_N
       \end{pmatrix}\\)
-   nn.Embedding 按照行向量来组织数据

<!--listend-->

```python
import torch
import torch.nn as nn
N = 3
F = 8
embed = nn.Embedding(30, F)
idx = torch.tensor([1,2,3])
X = embed(idx)
print(X.shape)
```

<!--list-separator-->

-  例子

    -   \\(N\\) 个token，\\(F\\) 是embedding的维度
    -   每行对应于一个token的embedding 行向量

        \\(tokens=\begin{pmatrix}
           \text{hello} \\\\
           \text{world} \\\\
           \text{pad} \\\\
           \text{pad} \\\\
           \text{pad}
           \end{pmatrix}\\)

        \\(X=\begin{pmatrix}
           [0.59, 0.20, 0.04, 0.96] \\\\
           [0.96, 0.30, 0.16, 0.63] \\\\
           [0.02, 0.19, 0.34, 0.25] \\\\
           [0.02, 0.19, 0.34, 0.25] \\\\
           [0.02, 0.19, 0.34, 0.25]
           \end{pmatrix}\\)


#### 矩阵相乘和算子作用 {#矩阵相乘和算子作用}

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
-   代码对应于 nn.Linear

<!--listend-->

```python
import torch
import torch.nn as nn
F = 6
linear = nn.Linear(in_features=F, out_features=F)
X_i = torch.rand(1, 6)
X = torch.rand(3, 6)
print(linear(X_i).shape)
print(linear(X).shape)
```

-   pytorch/tensorflow中的代码都是按照作用于行向量来组织的


#### transformer中的\\(QK^{T}V\\) {#transformer中的-qk-t-v}

-   \\(S=QK^T\\) 行向量两两计算点积相似性
    \\(\begin{pmatrix}
       Q\_{1}\\\\
       Q\_{2}\\\\
       \vdots\\\\
       Q\_N
       \end{pmatrix}
       \begin{pmatrix}
       K\_{1}^T, K\_2^T,\ldots,K\_N^T\\\\
       \end{pmatrix}=(Q\_{i}K\_j^T)\_{ij}=S\\)
-   \\(SV\\) = 对行向量做加权求和
    \\(\begin{pmatrix}
       S\_{11},S\_{12},\ldots, S\_{1N}\\\\
       S\_{21},S\_{22},\ldots, S\_{2N}\\\\
       \vdots\\\\
       S\_{N1},S\_{N2},\ldots, S\_{NN}\\\\
       \end{pmatrix}
       \begin{pmatrix}
       Q\_{1}\\\\
       Q\_{2}\\\\
       \vdots\\\\
       Q\_N
       \end{pmatrix}=
       \begin{pmatrix}
       S\_{1j}Q\_j\\\\
       S\_{2j}Q\_j\\\\
       \vdots\\\\
       S\_{Nj}Q\_j
       \end{pmatrix}\\)


#### 注 {#注}

-   左乘以一个矩阵相当于对每个列向量来施加变化
-   论文：一般会有行/列向量两种表示方式
-   代码：基本都是行向量来作为数据组织的标准
-   本文:
    -   向量都按照行向量的形式来组织
    -   按照作用于单个行向量的方式来讲解transformer


### encoder-decoder {#encoder-decoder}

-   大部分的s2s 的任务建模为 encoder-decoder的结构
    -   机器翻译，语音识别，文本摘要，问答系统等
-   encoder
    -   把token序列\\((x\_{1}, x\_2,\ldots, x\_N)\\) 转化为语义向量序列 \\((Y\_{1}, Y\_2, \ldots, Y\_N)\\)
    -   一般组织为多层的网络的形式
        -   第一层：基础语义向量序列
            \\((x\_{1}, x\_2,\ldots, x\_N)\rightarrow (X\_{1}, X\_2,\ldots, X\_N)\\)
        -   其它层：高阶语义向量序列
            \\((X\_{1}, X\_2,\ldots, X\_N)\rightarrow (Y\_{1}, Y\_2,\ldots, Y\_N)\\)
-   decoder
    基于\\((Y\_{1}, Y\_2, \ldots, Y\_N)\\) 自回归式的逐个token解码

focus到 encoder部分来理解transformer


### 低阶到高阶语义向量的转换 {#低阶到高阶语义向量的转换}

寻找算子 \\(\mathcal{T}\\) 将低阶的语义向量序列变换为高阶的语义向量序列
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

-   输入: \\(X\\) 低阶语义向量序列，输出: \\(Y\\) 高阶语义向量序列
-   意义
    -   \\(Y\_{i}=f(X\_{1}, X\_2, \ldots, X\_{N})\\)
    -   对低阶语义向量做加工组合处理和抽象，变换为一个高阶的语义向量序列
    -   高阶语义向量考虑了 _上下文_ 的语义向量表达
-   motivation
    -   1957. Firth

        > a word is characterized by the company it keeps.

        例子：

        > The **enigmatic** smile on Mona Lisa's face has intrigued art enthusiasts for centuries, leaving them to speculate about its true meaning.
-   用算子作用来表达 \\(Y=\mathcal{T}(X)\\)
    -   \\(X \in R^{N\times F}\\), \\(Y=\mathcal{T}(X): \quad R^{N\times F}\rightarrow R^{N\times F}\\)
    -   这个算子天然可以复合嵌套，形成多层的网络结构
        \\(Y=\mathcal{T}\_{L}\circ \mathcal{T}\_{L-1}\circ \ldots \circ \mathcal{T}\_{1}(X)\\)


### 核心的问题 {#核心的问题}


#### 问题 {#问题}

如何设计 \\(Y\_{i}=f(X\_{1}, X\_2, \ldots, X\_{N})\\)

-   \\(Y\_{1}, \ldots, Y\_N\\) 能否并行得到
-   \\(Y\_{i}\\) 能否高效的建立起远程的依赖


#### RNN {#rnn}

{{< figure src="images/2024-01-18_14-03-26_screenshot.png" width="600px" >}}

-   递归语义序列 \\(Y\_{0}\rightarrow Y\_1 \rightarrow \ldots \rightarrow Y\_{N}\\)
-   \\(Y\_{i}=tanh(X\_{i}W + Y\_{i-1}U)\\)
-   串行
-   单方向的依赖关系
    \\(Y\_{3}\\) 直接依赖于\\(Y\_{2}, X\_{2}\\), 间接依赖于\\(X\_1\\)


#### CNN {#cnn}

{{< figure src="images/2024-01-18_14-04-23_screenshot.png" width="600px" >}}

-   \\(Y\_{i}=(X\_{i-1},X\_i, X\_{i+1}) W\\) 假设窗口宽度是3
-   并行
-   长距离依赖？
    -   一层卷积只能依赖于当前窗口内，不能对窗口外的形成依赖。


#### transformer思路 {#transformer思路}

设计\\(Y\_{i}=f(X\_{1}, X\_2, \ldots, X\_{N})\\)，使得

-   使得 \\(Y\_{1},\ldots, Y\_N\\) 可以做并行计算
-   同时解决长距离依赖的问题

{{< figure src="images/2024-01-18_14-13-40_screenshot.png" width="400px" >}}

\\(Y=\mathcal{F}\circ \mathcal{A}(X)\\) 做两次矩阵的变换

-   \\(Y=\mathcal{A}(X)\\)    MultiHead Attention
    -   高阶的语义等于对 _全部_ 的低阶语义向量基于 _相似性(Attention)_ 做 _加权平均_
    -   \\(\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(X\_i,X\_j) X\_j}{\sum\_{j=1}^N sim(X\_i,X\_j)} \end{aligned}\\)
    -   attention = 相似性

-   \\(Y'=\mathcal{F}(Y)\\)  Position-wise Feedforward
    -   再施加若干非线性变换


## tranformer网络结构 {#tranformer网络结构}


### 基于KV查询的相似性计算 {#基于kv查询的相似性计算}

\\[\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(X\_i,X\_j) X\_j}{\sum\_{j=1}^N sim(X\_i,X\_j)} \end{aligned}\\]


#### 如何来定义相似性 {#如何来定义相似性}

-   \\(sim(X\_{i}, X\_j)= \mathrm{exp}(\dfrac{X\_i X\_{j}^T}{\sqrt{D}})\\)
-   \\(sim(X\_{i}, X\_j)= X\_i X\_{j}^T\\)


#### 直接计算相似性？ {#直接计算相似性}

-   参数太少
-   投影到别的空间来计算相似度   \\(X\_{i}\rightarrow X\_iW\\)

    \\(\begin{aligned}
       \mathcal{A}(X\_i) &=
       \frac{\sum\_{j=1}^{N} sim(X\_iW\_1,X\_jW\_{2}) X\_jW\_3}{\sum\_{j=1}^N sim(X\_iW\_1,X\_jW\_2)}
       \end{aligned}\\)
-   如果我们记 \\(X\_{i}W\_{1}=Q\_i, X\_iW\_2=K\_i, X\_iW\_3=V\_{i}\\)，

    \\(\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(Q\_i,K\_j) V\_j}{\sum\_{j=1}^N sim(Q\_i,K\_j)} \end{aligned}\\)


#### 基于KV查询理解 {#基于kv查询理解}

-   把\\(X\_i\\) 投影出三个向量 \\(Q\_i,K\_i,V\_i\\)
-   QKV
    -   KV 是大家熟悉的key-value存储 \\(K\_{j}\rightarrow V\_{j}\\)
    -   Q 是查询使用的query向量 \\(Q\_{i}\\)
-   QKV的查询方法
    1.  query查询多个key，获取多个value
    2.  最后把这些value加权平均

        \\(Q\_i\Rightarrow \begin{pmatrix}
           K\_{1}\rightarrow V\_{1}\\\\
           K\_2\rightarrow V\_2\\\\
           \vdots\\\\
           K\_N\rightarrow V\_N
           \end{pmatrix}
           \Rightarrow \begin{pmatrix}
           sim(Q\_i,K\_1)V\_{1} \\\\
           sim(Q\_i,K\_2)V\_{2} \\\\
           \vdots\\\\
           sim(Q\_i,K\_N)V\_N
           \end{pmatrix}\Rightarrow\sum\_{j=1}^N sim(Q\_i,K\_j)V\_j\\)
    3.  \\(\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(Q\_i,K\_j) V\_j}{\sum\_{j=1}^N sim(Q\_i,K\_j)} \end{aligned}\\)
-   参数： 对应于\\(Q,K,V\\) 产生了三个投影矩阵矩阵 \\(W\_{Q}, W\_K,W\_V\\)


### 在一个低维空间做attention {#在一个低维空间做attention}


#### 单个头的attention {#单个头的attention}

-   把\\(X\_{i}\\) 从\\(F\\) 维空间投影到\\(D\\) 维空间

    \\(W\_{Q}\in R^{F\times D}, W\_K\in R^{F\times D}, W\_{V} \in R^{F\times M}\\)

    \\(Q\_i = X\_iW\_{Q}, \quad  K\_i = X\_iW\_{K}, \quad  V\_i = X\_iW\_{V}\\)
-   \\(Q\_i\\) 和所有的\\(K\_j\\) 做基于点积的相似度计算，

    这里简单起见，我们省略掉了scaling \\(\frac{1}{\sqrt{D}}\\)

    \\(Q\_iK^{T}=Q\_i(K^T\_1, \ldots, K^T\_N)=(Q\_iK^T\_1, \ldots, Q\_iK^T\_N)\\)
-   对相似度的分布做softmax

    \\(S=\mathrm{soft}(Q\_iK^T\_1, \ldots, Q\_iK^T\_N)=(s\_{i1},\ldots, s\_{iN})\\)

    \\(s\_{i,j}= \dfrac{exp(Q\_iK\_j^T)}{\sum\_{j=1}^N exp(Q\_iK\_j^T)}\\)
-   加权平均

    \\(\mathcal{A}(X\_i)=\sum\_{j=1}^Ns\_jV\_j=(s\_{i1},\ldots, s\_{iN})
       \begin{pmatrix}
       V\_1\\\\
       V\_2\\\\
       \vdots\\\\
       V\_N\end{pmatrix}\\)

\\(\mathcal{A}(X\_i) = \mathrm{soft}(Q\_iK^{T})V = \mathrm{soft}(X\_iW\_QW\_K^TX^T)XW\_V\\)


#### 矩阵表达 {#矩阵表达}

\\(Y=\mathcal{A}(X)
=\begin{pmatrix}
\mathcal{A}(X\_1)\\\\
\mathcal{A}(X\_2)\\\\
\vdots\\\\
\mathcal{A}(X\_N)
\end{pmatrix}
=\begin{pmatrix}
\mathrm{soft}(Q\_1K^T)V\\\\
\mathrm{soft}(Q\_2K^T)V\\\\
\vdots \\\\
\mathrm{soft}(Q\_NK^T)V
\end{pmatrix}=\mathrm{soft}(QK^T)V\\)

简化符号  \\(sim(Q,K)V\\)


#### 代码实现 {#代码实现}

```python
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class SingleHeadAttention(nn.Module):

  def __init__(self, config):
      super().__init__()
      self.F = config["fea_size"] #F
      self.D = config["subspace_dim"] #D
      self.q_proj = nn.Linear(self.F, self.D)
      self.k_proj = nn.Linear(self.F, self.D)
      self.v_proj = nn.Linear(self.F, self.D)

  def forward(self, x):
      B, N, F = x.size()
      q = self.q_proj(x)
      k = self.k_proj(x)
      v = self.v_proj(x)
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = F.softmax(att, dim=-1)
      y = att @ v
      return y
```


#### 注: {#注}

1.  \\(D\neq F\\) 时，\\(\mathcal{A}(X)\\) 还不可用


### 在多个低维空间做attention {#在多个低维空间做attention}


#### why {#why}

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

-   一词多义
-   把\\(F\\) 维的语义向量投影到 \\(H\\) 个不同的子空间中去计算相似加权组合


#### 做法 {#做法}

-   每个头投做独立的Attention变换 \\(\mathcal{A}^{h}(X)\\)
    -   假设有\\(H\\) 个头，每个头作用的低维空间维度是\\(D\\)
    -   \\(D\times H = F\\)
-   对\\(H\\) 个 \\(D\\) 行向量拼接
    -   \\(W\_O\in R^{F\times F}\\)
    -   \\(\mathcal{A}(X) = \mathrm{concat}(\mathcal{A}^1(X), \mathcal{A}^2(X), \ldots, \mathcal{A}^{H}(X) W\_O\\)
-   或者对前面的符号简化
    -   在第\\(j\\) 个子空间做单头注意力 \\(Y^{j}=sim(Q^{j}, K^{j})V^{j}\\)
    -   合并 \\(Y=(Y^{1},\ldots, Y^H)\\)


#### 代码实现 {#代码实现}

```python
# 参考 https://github.com/karpathy/minGPT/tree/master/mingpt
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class SelfAttention(nn.Module):

  def __init__(self, config):
      super().__init__()
      self.H = config["n_head"]
      self.F = config["fea_size"] #F
      self.D = self.fea_size // self.n_head #D
      # 一次把qkv 全部映射完成，对应W_Q, W_K, W_V
      self.qkv_proj = nn.Linear(self.fea_size, 3 * self.fea_size)
      # 最后的投影，对应于 $W_O$
      self.out_proj = nn.Linear(self.fea_size, self.fea_size)

  def forward(self, x):
      B, N, fea_size = x.size()
      q, k, v = self.qkv_proj(x).split(3, dim=2)
      # matmul 只能在最后两个维度相乘，需要对NxD的矩阵相乘，做1,2维度的交换
      k = k.view(B, N, self.H, self.D).transpose(1, 2)
      q = q.view(B, N, self.H, self.D).transpose(1, 2)
      v = v.view(B, N, self.H, self.D).transpose(1, 2)
      # 一次把多个头的映射全部完成
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = F.softmax(att, dim=-1)
      y = att @ v
      # 多头拼接
      y = y.transpose(1, 2).contiguous().view(B, N, F)
      y = self.out_proj(y)
      return y
```


#### 代码示意 {#代码示意}

{{< figure src="images/2024-01-31_11-11-07_screenshot.png" width="600px" >}}


### 位置无关的全连接 {#位置无关的全连接}

-   两层的全连接
    \\(\mathcal{F}(X\_i)=(g(X\_iW\_1)+b\_1)W\_2+b\_2)\\)


#### 代码 {#代码}

```python
import torch
import torch.nn as nn
class PWiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fea_size = config["fea_size"]
        self.proj_wide = nn.Linear(self.fea_size, 4 * self.fea_size)
        self.proj_narrow = nn.Linear(4 * self.fea_size, self.fea_size)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.proj_narrow(self.act(self.proj_wide(x)))
```


### 归一化 + 残差网络 {#归一化-plus-残差网络}

\\(\mathcal{T}(X)=\mathcal{F}\circ\mathcal{A}(X)\\)


#### Layer Normalization {#layer-normalization}

\\(\mathcal{A}'(X)=\mathcal{N}\circ\mathcal{A}(X)\\)
\\(\dfrac{x-\mu}{\sqrt{\sigma}}\gamma + \beta,\mu=\dfrac{1}{d}\sum\limits\_{i=1}^{d}x\_{i}, \sigma=\sqrt{\dfrac{1}{d}\sum\limits\_{i=1}^{d}(x\_{i}-\mu)^{2}}\\)
可以看成是作用在行向量上的算子


#### 输入矩阵例子 {#输入矩阵例子}

\\(\begin{pmatrix}
  \text{hello} \\\\
  \text{world} \\\\
  \text{pad} \\\\
  \text{pad} \\\\
  \text{pad}
  \end{pmatrix}
  \rightarrow X=
  \begin{pmatrix}
  [0.59, 0.20, 0.04, 0.96] \\\\
  [0.96, 0.30, 0.16, 0.63] \\\\
  [0.02, 0.19, 0.34, 0.25] \\\\
  [0.02, 0.19, 0.34, 0.25] \\\\
  [0.02, 0.19, 0.34, 0.25]
  \end{pmatrix}\\)


#### 行归一化 or 列归一化 {#行归一化-or-列归一化}

-   在NLP的序列建模里面，Layer Normalization
-   在CV/CTR预估里面, Batch Normalization


#### Why {#why}

-   padding的影响
    不同batch中&lt;pad&gt;个数不同，沿着token方向做归一化没有意义
-   每个位置做独立的归一化更有意义


#### 其他的可能选择 {#其他的可能选择}

-   RMSNorm

    \\(\dfrac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum\_{i=1}^{d} x\_i^2}\\)


### 整体的变换 {#整体的变换}

\\(Y=\mathcal{T}(X)\\)

1.  Attention \\(Z=\mathcal{N}\circ(X+\mathcal{A}(X))\\)
2.  位置无关的全连接   \\(Y=\mathcal{N}\circ(X+\mathcal{F}(Z))\\)


#### residual network {#residual-network}

\\(\mathcal{A}'(X)=\mathcal{N}\circ(X+\mathcal{A}(X))\\)
\\(\mathcal{F}'(X)=\mathcal{N}\circ(X+\mathcal{F}(X))\\)


#### 多层 {#多层}

一个 \\(L\\) 层的transformer 模型

\begin{equation\*}
\begin{split}
   \mathcal{T}(X) & = \mathcal{T}\_L \circ \ldots \mathcal{T}\_{2}\circ \mathcal{T}\_{1}(X)
\end{split}
\end{equation\*}


#### 代码 {#代码}

```python
import torch.nn as nn
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.fea_size)
        self.attn = SelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.fea_size)
        self.mlp = PWiseFeedForward(config)

    def forward(self, x):
        x = self.layer_norm_1(x + self.attn(x))
        x = self.layer_norm_2(x + self.mlp(x))
        return x
```


## transformer参数和计算量 {#transformer参数和计算量}


### 关于参数量 {#关于参数量}

-   我们需要一种模型能够方便的去增加模型的复杂度
    -   比如增加深度，增加宽度
    -   增加token的embedding size
    -   增加词典的大小
-   transformer模型可以在此之外非常有效的提升模型的参数量


### 参数的分布 {#参数的分布}


#### 多头注意力 \\(4F^2\\) {#多头注意力-4f-2}

-   每个头有
    -   3个投影矩阵 \\(W\_Q, W\_K, W\_V\\)
    -   1个投影concat结果的矩阵 \\(W\_O\\)
-   参数量: 假设投射到的子空间维度是\\(D\\), $H$个子空间，\\(D\times H = F\\)
    -   \\(F\times D \times 3 \times H = 3F^{2}\\)
    -   \\(F^{2}\\)


#### FFW \\(8F^2\\) {#ffw-8f-2}

-   两个矩阵，先从\\(F\\) 变宽到\\(4F\\)，再收窄回来到\\(F\\)
-   参数量\\(F\times4F + 4F\times F= 8F^{2}\\)


#### word embedding {#word-embedding}

\\(E\\) 是token字典的大小

-   \\(E\times F\\)


#### total {#total}

\\(L(12F^{2})+EF\\)

| model     | 维度 | 层数 | 头数 | 字典大小 | 参数量 |
|-----------|----|----|----|------|-----|
| bertBase  | 768  | 12 | 12 | 30000 | 110M |
| bertLarge | 1024 | 24 | 12 | 30000 | 340M |


### linear transformer {#linear-transformer}


#### 两个算子的计算量 {#两个算子的计算量}

-   \\(\mathcal{A}(X)\\) 计算量 \\(O(N^2)\\)
-   \\(\mathcal{F}(X)\\) 计算量 \\(O(N)\\)


#### softmax 导致了\\(O(N^2)\\) {#softmax-导致了-o--n-2}

核心的计算量在这三个矩阵的相乘上，\\(QK^{T}V\\)

-   有softmax的存在的话
    只能先计算\\(H=QK^{T}\\), 对\\(H\\) 做softmax 变换后，再计算\\(HV\\)
    乘法的计算量是 \\(N^2D+N^2M\\), 整体的复杂度是\\(O(N^{2})\\)
    \\(QK^TV=(QK^T)V=\begin{pmatrix}
       H\_{11},H\_{12},\ldots,H\_{1N} \\\\
       \vdots\\\\
       H\_{N1},H\_{N2},\ldots,H\_{NN} \\\\
       \end{pmatrix}V\\)

-   如果没有softmax的话
    可以先计算后两个矩阵相乘\\(H=K^TV\\), 再计算\\(QH\\)
    计算量可以是\\(O(N)\\), 因为\\(K^TV\\) 可以提前算出来缓存，大致如下面这个表达所示
    \\(Q(K^TV)=\begin{pmatrix}
       Q\_1 \\\\
       Q\_2 \\\\
       \vdots\\\\
       Q\_{N}
       \end{pmatrix}(K^TV)\\)


#### kernel {#kernel}

\\(\mathcal{A}(X\_i)=\dfrac{\sum\_{j=1}^{N} sim(Q\_i,K\_j) V\_j}{\sum\_{j=1}^N sim(Q\_i,K\_j)}\\)

-   kernel: \\(k(x,y)=<\phi(x),\phi(y)>\\)
    \\(k(x,y)=(x\cdot z)^2, \phi(x)=(x\_{1}^{2},x\_{2}^2,\sqrt{2}x\_1x\_{2})\\)
    -   kernel 对应一个feature map
    -   可以用非负的kernel来替换掉
    -   当前的sim函数 \\(sim(x,y)=\mathrm{exp}(xy^{T}/\sqrt{D})\\)


#### linear transformer  \\(O(N)\\) {#linear-transformer-o--n}

-   用kernel来替换掉sim
    \\[\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(Q\_i,K\_j) V\_j}{\sum\_{j=1}^N sim(Q\_i,K\_j)} \\\\
       &=\frac{\sum\_{j=1}^{N} \phi(Q\_i)\phi(K\_j)^T V\_j}{\sum\_{j=1}^N \phi(Q\_i)\phi(K\_j)^T} \\\\
       &=\frac{ \phi(Q\_i) \sum\_{j=1}^{N}\phi(K\_j)^T V\_j}{\phi(Q\_i)\sum\_{j=1}^N \phi(K\_j)^T}
       \end{aligned}
       \\]
    -   \\(\sum\_{j=1}^{N}\phi(K\_j)^T V, \sum\_{j=1}^N \phi(K\_j)^T\\) 可以提前算好
    -   去掉归一化来看 \\[(\phi(Q)\phi(K)^{T})V=\phi(Q)(\phi(K)^{T}V)\\]

        \\[\begin{aligned}     \begin{pmatrix}
             \phi(Q\_1)\sum\_{j=1}^{N} \phi(K\_j)^{T} V\_j \\\\
             \vdots \\\\
             \phi(Q\_N)\sum\_{j=1}^{N} \phi(K\_j)^T V\_j \\\\
             \end{pmatrix}& =\begin{pmatrix}
                \phi(Q\_1)\phi(K)^{T}V\\\\
                 \vdots \\\\
                  \phi(Q\_N)\phi(K)^{T}V \\\\
                  \end{pmatrix} \\\\
                  &=
                  \begin{pmatrix}
                  \phi(Q\_{1})\\\\
                  \vdots\\\\
                  \phi(Q\_N)
                  \end{pmatrix}\phi(K)^TV \\\\
                  &=\phi(Q)\phi(K)^TV
                  \end{aligned}\\]

    -   \\(O(N)\\) 复杂度，Linear Transformer
    -   \\(\phi(x)=\mathrm{elu}(x)+1\\)


### 优缺点 {#优缺点}


#### 优点 {#优点}

-   并行
-   长距离依赖
-   可解释性


#### 缺点 {#缺点}

-   本身对顺序无感，操作是在集合层次上的，需要额外加入位置编码

    下面的cls token得到的语义向量是完全一样的。

    -   &lt;cls&gt; 从 北京 到 上海 的 火车票
    -   &lt;cls&gt; 从 上海 到 北京 的 火车票
-   计算的复杂度是序列长度平方


## 参考论文 {#参考论文}

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Katharopoulos, Angelos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. “Transformers Are RNNs: Fast Autoregressive Transformers with Linear Attention.” <i>Arxiv.Org</i>. https://arxiv.org/abs/2006.16236v3.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. “Attention Is All You Need.” arXiv. <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a>.</div>
</div>