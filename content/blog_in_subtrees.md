+++
author = ["连义江"]
lastmod = 2024-02-01T23:47:19+08:00
draft = false
toc = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [大模型](#大模型)
    - [<span class="org-todo done DONE">DONE</span> transformer入门](#about-transformer):hugo:org:
        - [transformer背景](#transformer背景)

</div>
<!--endtoc-->



## 大模型 {#大模型}

关于大模型的文章都在这里


### <span class="org-todo done DONE">DONE</span> transformer入门 <span class="tag"><span class="hugo">hugo</span><span class="org">org</span></span> {#about-transformer}


#### transformer背景 {#transformer背景}

<!--list-separator-->

-  主要内容

    -   参考
        -   2017. (引用 106576)
            Attention Is All You Need
        -   2020. (引用 975)
            Fast Autoregressive Transformers with Linear Attention
        -   [mingpt by karpathy](https://github.com/karpathy/minGPT/tree/master/mingpt)
    -   主要内容
        -   transformer 的设计推演
        -   transformer 的代码讲解
        -   transformer的参数和运算量
        -   linear attention

<!--list-separator-->

-  矩阵知识

    <!--list-separator-->

    -  why

        -   原文直接从整个矩阵作用出发
            \\[\begin{aligned}\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\dfrac{QK^T}{\sqrt{d\_k}})V \\\ \mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}\_1,\ldots,\mathrm{head}\_h)W^{O} \\\\
               \mathrm{head}\_i=\mathrm{Attention}(QW\_i^Q, KW^{K}\_i,VW^V\_i)
               \end{aligned}\\]
        -   从行向量的角度更容易理解

    <!--list-separator-->

    -  矩阵和行向量

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

        <!--list-separator-->

        -  例子

            -   \\(N\\) 个token，\\(F\\) 是embedding的维度
            -   每行对应于一个token的embedding 行向量

            \\(tokens=\begin{pmatrix}
               hello \\\\
               world \\\\
               pad \\\\
               pad \\\\
               pad
               \end{pmatrix}
               \rightarrow X=\begin{pmatrix}
               [0.59, 0.20, 0.04, 0.96] \\\\
               [0.96, 0.30, 0.16, 0.63] \\\\
               [0.02, 0.19, 0.34, 0.25] \\\\
               [0.02, 0.19, 0.34, 0.25] \\\\
               [0.02, 0.19, 0.34, 0.25]
               \end{pmatrix}\\)

    <!--list-separator-->

    -  矩阵相乘和算子作用

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

        -   算子是对矩阵乘法的一种物理理解
            -   旋转矩阵

                \\(R(\theta)=\begin{pmatrix}
                        cos\theta& sin\theta\\\\
                        -sin\theta& cos\theta
                        \end{pmatrix}\\)
            -   缩放变换

                \\(R(\lambda\_1,\lambda\_2)=\begin{pmatrix} \lambda\_1 & \\\\
                        & \lambda\_2  \end{pmatrix}\\)

    <!--list-separator-->

    -  transformer中的\\(QK^{T}V\\)

        -   \\(S=QK^T\\) = 行向量两两计算点积相似性
            \\(\begin{pmatrix}
               Q\_{1}\\\\
               Q\_{2}\\\\
               \vdots\\\\
               Q\_N
               \end{pmatrix}
               \begin{pmatrix}
               K\_{1}^T, K\_2^T,\ldots,K\_N^T
               \end{pmatrix}=(Q\_{i}K\_j^T)\_{ij}=S\\)

        -   \\(SV\\) = 对行向量做加权求和
            $\begin{pmatrix}
            S<sub>11</sub>,S<sub>12</sub>,\ldots, S<sub>1N</sub><br />
            S<sub>21</sub>,S<sub>22</sub>,\ldots, S<sub>2N</sub><br />
            \vdots<br />
            S<sub>N1</sub>,S<sub>N2</sub>,\ldots, S<sub>NN</sub><br />
            \end{pmatrix}

            \begin{pmatrix}
            Q\_{1}\\\\
            Q\_{2}\\\\
            \vdots\\\\
            Q\_N
            \end{pmatrix}

            =(Q<sub>i</sub>K_j^T)<sub>ij</sub>=S$

    <!--list-separator-->

    -  代码

        -   pytorch/tensorflow中的代码都是按照作用于行向量来组织的
        -   nn.Linear 作用于行向量
        -   nn.Embedding 按照行向量来组织数据

        <!--listend-->

        ```python { linenos=true, linenostart=1 }
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

    -  注

        -   左乘以一个矩阵相当于对每个列向量来施加变化
        -   论文：一般会有行/列向量两种表示方式
        -   代码：基本都是行向量来作为数据组织的标准
        -   本文:
            -   向量都按照行向量的形式来组织
            -   按照作用于单个行向量的方式来讲解transformer

<!--list-separator-->

-  encoder-decoder

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

<!--list-separator-->

-  低阶到高阶语义向量的转换

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
    -   用矩阵变换表达 \\(Y=\mathcal{T}(X)\\)
        -   \\(X \in R^{N\times F}\\), \\(Y=\mathcal{T}(X): \quad R^{N\times F}\rightarrow R^{N\times F}\\)
        -   这个算子天然可以复合嵌套，形成多层的网络结构
            \\(Y=\mathcal{T}\_{L}\circ \mathcal{T}\_{L-1}\circ \ldots \circ \mathcal{T}\_{1}(X)\\)

<!--list-separator-->

-  核心的问题

    <!--list-separator-->

    -  问题

        如何设计 \\(Y\_{i}=f(X\_{1}, X\_2, \ldots, X\_{N})\\)

        -   \\(Y\_{1}, \ldots, Y\_N\\) 能否并行得到
        -   \\(Y\_{i}\\) 能否高效的建立起远程的依赖

    <!--list-separator-->

    -  RNN

        {{< figure src="/images/2024-01-18_14-03-26_screenshot.png" width="600px" >}}

        -   递归语义序列 \\(Y\_{0}\rightarrow Y\_1 \rightarrow \ldots \rightarrow Y\_{N}\\)
        -   \\(Y\_{i}=tanh(X\_{i}W + Y\_{i-1}U)\\)
        -   串行
        -   单方向的依赖关系，间接

    <!--list-separator-->

    -  CNN

        {{< figure src="/images/2024-01-18_14-04-23_screenshot.png" width="600px" >}}

        -   \\(Y\_{i}=(X\_{i-1},X\_i, X\_{i+1}) W\\) 假设窗口宽度是3
        -   并行
        -   长距离依赖？
            -   一层卷积只能依赖于当前窗口内，不能对窗口外的形成依赖。

    <!--list-separator-->

    -  transformer思路

        设计\\(Y\_{i}=f(X\_{1}, X\_2, \ldots, X\_{N})\\)，使得

        -   使得 \\(Y\_{1},\ldots, Y\_N\\) 可以做并行计算
        -   同时解决长距离依赖的问题

        {{< figure src="images/2024-01-18_14-13-40_screenshot.png" width="400px" >}}

        \\(Y=\mathcal{F}\circ \mathcal{A}(X)\\) 做两次矩阵的变换

        -   \\(Y=\mathcal{A}(X)\\)    MultiHead Attention
            -   高阶的语义等于对 _全部_ 的低阶语义向量基于 _相似性(Attention)_ 做 _加权平均_
            -   \\[\begin{aligned}\mathcal{A}(X\_i) &=  \frac{\sum\_{j=1}^{N} sim(X\_i,X\_j) X\_j}{\sum\_{j=1}^N sim(X\_i,X\_j)} \\\\end{aligned}\\]
            -   attention = 相似性

        -   \\(Y'=\mathcal{F}(Y)\\)  Position-wise Feedforward
            -   再施加若干非线性变换


[//]: # "Exported with love from a post written in Org mode"
[//]: # "- https://github.com/kaushalmodi/ox-hugo"