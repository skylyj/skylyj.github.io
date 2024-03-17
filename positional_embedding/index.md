# 旋转位置编码


## 背景 {#背景}


### 主要内容 {#主要内容}

-   旋转位置编码在大模型中应用广泛
    -   google PaLM
    -   meta llama
-   内容
    -   旋转位置编码的来龙去脉
        -   基于实数域上的旋转变换来推演
    -   pytorch 代码实现 tricks
    -   重新看解绝对位置编码

        \\(\begin{aligned}
             P\_{i,2t} &= sin(i/10000^{2t/D}) \\\\
             P\_{i,2t+1} &= cos(i/10000^{2t/D})
             \end{aligned}\\)
-   参考：
    -   2021. Enhanced Transformer with Rotary Position Embedding
        (<a href="#citeproc_bib_item_1">Su et al. 2022</a>)
    -   2017. Attention Is All You Need
        (<a href="#citeproc_bib_item_2">Vaswani et al. 2023</a>)
    -   [hugging face llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)


### 回顾transformer {#回顾transformer}

-   encoder: 寻找算子\\(\mathcal{T}\\) 低阶语义向量序列转化为高阶的语义向量序列

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
-   \\(Y=\mathcal{T}(X)=\mathcal{F}(\mathcal{A}(X))\\)
    -   Attention \\(\mathcal{A}\\)
    -   Feedforward \\(\mathcal{F}\\)
-   Attention

    \\(\begin{aligned}Q\_{i} &= X\_{i} W\_{Q} \\\\
       K\_{i} &= X\_{i} W\_{K}\\\\
       V\_{i} &= X\_{i} W\_{V}\\\\
       Y\_{i} &= \sum\_{j=1}^{N}sim(Q\_i,K\_{j}) V\_j\\\\
       sim(Q\_{i},K\_j) = &= \frac{exp(\frac{Q\_{i}K\_{j}^{T}}{\sqrt{D}})}
       {\sum\_{j=1}^N exp(\frac{Q\_iK\_j^{T}}{\sqrt{D}})}\\\\
       \end{aligned}\\)


### 为什么需要位置编码 {#为什么需要位置编码}

-   因为transformer结构本身是和位置无关的
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


### 直接修改输入 {#直接修改输入}

在\\(X\_i \rightarrow Q\_i, K\_i, V\_i\\) 之前，直接加入位置的embedding
\\(X\_i^{'}=X\_i+P\_i\\)


#### learned embedding {#learned-embedding}

-   优点简单，bert/GPT
-   外推困难，对于超过序列最大长度的位置


#### 自定义绝对位置编码 {#自定义绝对位置编码}

-   二维函数 f(position, dimension)
-   要求
    -   函数随着position,dimension增长应该是有界的
    -   足够的区分度，对每个position,对应的行向量应该是不同的
-   例子

    \\(\begin{aligned}
       P\_{i,2t} &= sin(k/10000^{2t/D}) &&\\\\
       P\_{i,2t+1} &= cos(k/10000^{2t/D})&&\\\\
       \end{aligned}\\)

    \\(t\in[0,1,\ldots,D/2-1], i\ge0\\)
-   问题: 语义应该是和相对位置有关的，而不是绝对位置


### 修改Attention {#修改attention}

-   Attention

    \\(\begin{aligned}Q\_{i} &= X\_{i} W\_{Q} \\\\
       K\_{i} &= X\_{i} W\_{K}\\\\
       V\_{i} &= X\_{i} W\_{V}\\\\
       Y\_{i} &= \sum\_{j=1}^{N}sim(Q\_i,K\_{j}) V\_j\\\\
       sim(Q\_{i},K\_j) &= \frac{exp(\frac{Q\_{i}K\_{j}^{T}}{\sqrt{D}})}
       {\sum\_{j=1}^N exp(\frac{Q\_iK\_j^{T}}{\sqrt{D}})}\\\\
       \end{aligned}\\)
-   想法
    -   可以从相似性入手，i和j之间的语义的相似性应该包含相对的距离信息
    -   希望相似性计算只依赖向量还有相对距离,而不依赖于其绝对的位置。
    -   \\(Q\_{i}K\_j^T=g(X\_{i},X\_j,i-j)\\)


### 回顾矩阵的知识 {#回顾矩阵的知识}


#### 关于行向量和矩阵 {#关于行向量和矩阵}

-   定义线性算子 \\(\mathcal{A}\\)
    -   可以作用到行向量  \\(\mathcal{A}(X\_i) = X\_{i} A\\)
    -   也可以作用到矩阵  \\(\mathcal{A}(X) = XA\\)
-   右乘矩阵等于对每个行向量逐个施加行变换
-   线性算子是对矩阵乘法的一种物理理解
    -   旋转变换

        \\(R(\theta)=
             \begin{pmatrix}
             cos\theta& sin\theta\\\\
             -sin\theta& cos\theta
             \end{pmatrix}\\)
    -   缩放变换

        \\(R(\lambda\_1,\lambda\_2)=\begin{pmatrix} \lambda\_1 & \\\\
                & \lambda\_2 \\\ \end{pmatrix}\\)
-   用对角阵在正交的子空间上施加不同的行变换
    假设有两个方阵A,B，设 \\(X= (X^1, X^2)\\), 那么

    \\((X^1,X^2)\begin{pmatrix}
       A & 0 \\\\
       0 & B
       \end{pmatrix} = (X^1A, X^2 B)\\)

<!--list-separator-->

-  注：

    -   pytorch/tensorflow 中的矩阵相关代码都是按照行向量来组织的
    -   在ROPE 论文是按照列向量来撰写的，表现为是用矩阵左乘以一个列向量
    -   本文中出现的向量全部用行向量来表达，和代码一致


#### 关于旋转矩阵 {#关于旋转矩阵}

-   在二维子空间的旋转矩阵

    \\(R(\theta)=
       \begin{pmatrix}
       cos\theta& sin\theta\\\\
       -sin\theta& cos\theta
       \end{pmatrix}\\)

{{< figure src="images/2024-03-15_23-20-19_screenshot.png" width="200px" >}}

-   物理意义
    -   \\(XR(\theta)\\) 对\\(X\\) 逆时针旋转\\(\theta\\)
    -   证明

        \\(X=\rho(cos\phi, sin\phi)\\)

        \\(\begin{aligned}
             &XR(\theta)\\\\
             =&\rho(cos \phi, sin \phi)
             \begin{pmatrix}
             cos\theta& sin\theta\\\\
             -sin\theta& cos\theta
             \end{pmatrix} \\\\
             =& \rho(
             cos\phi cos\theta - sin\phi sin\theta,
             cos\phi sin\theta + sin\phi cos\theta
             )\\\\
             =& \rho(cos(\phi+\theta), sin(\phi+\theta))
             \end{aligned}\\)
-   性质
    -   \\(R(\theta)^T=R(-\theta)\\)
    -   \\(R(\theta\_1)R(\theta\_2)=R(\theta\_1+\theta\_{2})\\)


#### 在高维空间中旋转 {#在高维空间中旋转}

假设空间是偶数维的，把原始的空间切分成为一个正交的二维子空间，在上面做独立的旋转。

<!--list-separator-->

-  定义

    \\(\Theta=(\theta\_{1},\theta\_2,\ldots,\theta\_{D/2})\\)

    \\(R(\Theta)=\begin{pmatrix}
       cos\\,\theta\_{1} & sin\\,\theta\_1 & 0 & 0 & 0 & 0 &0\\\\
       -sin\\,\theta\_{1} & cos\\,\theta\_1 & 0 & 0 & 0 & 0 &0 \\\\
       0 & 0 & cos\\,\theta\_{2} & sin\\,\theta\_2 & 0 & 0 &0 \\\\
       0 & 0 & -sin\\,\theta\_{2} & cos\\,\theta\_2& 0 & 0 &0  \\\\
       0 & 0 & 0 & 0 & \ldots &0 & 0 \\\\
       0 & 0 & 0 & 0 &\ldots & cos\\,\theta\_{D/2} & sin\\,\theta\_{D/2}  \\\\
       0 & 0 & 0 & 0 &\ldots & -sin\\,\theta\_{D/2} & cos\\,\theta\_{D/2}
       \end{pmatrix}\\)

    \\(R(\Theta)=\begin{pmatrix}
       R(\theta\_{1}) & 0 &0 & 0\\\\
       0 & R(\theta\_2) & 0 &0 \\\\
       0 & 0 &\ldots &0  \\\\
       0 & 0 & 0 &R(\theta\_{D/2})\\\\
       \end{pmatrix}\\)

<!--list-separator-->

-  性质

    -   在独立的二维子空间上做不同角度的旋转

        \\(XR(\Theta)=(X^1, X^2)
           \begin{pmatrix}
           R(\theta\_{1}) & 0 \\\\
           0 & R(\theta\_2)
           \end{pmatrix}=(X^1R(\theta\_1), X^2R(\theta\_2))\\)

    -   \\(R(\Theta)=\widehat{R}(\theta\_1)\widehat{R}(\theta\_2)\ldots\widehat{R}(\theta\_{D/2})\\) 逐个在不同的子空间上做旋转
        定义
        \\(\widehat{R}(\theta)=
          \begin{pmatrix}
          R(\theta) & 0 \\\\
          0 & 1 \\\\
          \end{pmatrix}\\)

        \\(R(\Theta)=\begin{pmatrix}
          R(\theta\_{1}) & 0 \\\\
          0 & R(\theta\_2)
          \end{pmatrix}=\begin{pmatrix}
          R(\theta\_{1}) & 0 \\\\
          0 & 1 \\\\
          \end{pmatrix}\begin{pmatrix}
          1 & 0 \\\\
          0 & R(\theta\_2)
          \end{pmatrix}=\widehat{R}(\theta\_1)\widehat{R}(\theta\_2)\\)


## 旋转位置编码 {#旋转位置编码}


### motivation {#motivation}

{{< figure src="images/2024-03-15_23-23-30_screenshot.png" width="400px" >}}

假设\\(Q\_{i}, K\_j\\) 都是二维的向量，\\(i, j\\) 是它们对应的position，
这里\\(\eta\_{i},\eta\_{j}\\) 是\\(Q\_i, K\_j\\) 弧度表示.

-   点积只和模长和夹角有关
    -   \\(Q\_iK\_j^T=\\|Q\_i\\|\\|K\_j\\| cos(\eta\_{j}-\eta\_i)\\),
    -   如何在这里融入位置的信息？基于位置乘倍数旋转之后做点击
-   做法：
    -   我们把两个向量各自旋转\\(i\theta,j\theta\\) 后再来计算点积
    -   其中\\(\theta\\) 是一个单位角度，
    -   新的向量的内积带上了位置信息，且他们的内积只和\\(Q\_i,Q\_j,i-j\\) 相关
-   因为: 模长没有变，只是夹角变了，夹角增加了 \\((j-i)\theta\\).
    -   \\(Q\_iR(i\theta)(K\_jR(j\theta))^T=\\|Q\_i\\|\\|K\_j\\| cos(\eta\_{j}-\eta\_{i}+(j-i)\theta)\\)


### 二维空间中的一个解 {#二维空间中的一个解}


#### 基于旋转矩阵的一个解 {#基于旋转矩阵的一个解}

\begin{equation\*}
\begin{split}
Q\_{i}&= X\_{i} W\_{Q} R(i\theta) \\\\
K\_{j}&= X\_j W\_{K} R(j\theta)\\\\
Q\_{i}K\_j^T &=X\_{i}W\_QR(i\theta)R(j\theta)^{T}W\_K^{T}X\_{j}^T\\\\
&=X\_{i}W\_QR(i\theta)R(-j\theta)W\_K^{T}X\_{j}^T\\\\
&=X\_{i}W\_QR((i-j)\theta)W\_K^{T}X\_{j}^T\\\\
& =g(X\_i,X\_j,i-j)\\\\
     \end{split}
     \end{equation\*}


#### 为什么是在投影之后旋转，不在投影之前转？ {#为什么是在投影之后旋转-不在投影之前转}

\begin{equation\*}
\begin{split}
Q\_{i}&= X\_{i} R(i\theta) W\_{Q} \\\\
K\_{j}&= X\_j R(j\theta) W\_{K} \\\\
Q\_{i}K\_j^T &=X\_{i}R(i\theta)W\_QW\_KR(j\theta)^{T}X\_{j}^T\\\\
&=?\\\\
     \end{split}
     \end{equation\*}


### 推广到高维空间 {#推广到高维空间}

整个空间分割成\\(D/2\\) 个子空间，在各个子空间上分别按照一个位置相关的角度旋转


#### 定义 \\(R(i\Theta)\\) {#定义-r--i-theta}

-   \\(X\_{i}R(i\Theta)\\)
    表示对\\(X\_{i}\\) 在各个子空间分别做角度为\\(i\theta\_1,i\theta\_2,\ldots,i\theta\_{D/2}\\) 的旋转.
    \\(\Theta=(\theta\_{1},\theta\_2,\ldots,\theta\_{D/2})\\)
    \\(R(i \Theta)=\begin{pmatrix}
       cos\\,i\theta\_{1} & sin\\,i\theta\_1 & 0 & 0 \\\\
       -sin\\,i\theta\_{1} & cos\\,i\theta\_1 & 0 & 0 \\\\
       0 & 0 & cos\\,i\theta\_{2} & sin\\,i\theta\_2 \\\\
       0 & 0 & -sin\\,i\theta\_{2} & cos\\,i\theta\_2 \\\\
       \end{pmatrix}=\begin{pmatrix}
       R(i\theta\_{1}) & 0 \\\\
       0 & R(i\theta\_2)
       \end{pmatrix}\\)


#### ROPE在高维空间 {#rope在高维空间}

\begin{equation\*}
\begin{split}
Q\_{i}& = X\_{i} W\_{Q} R(i\Theta) \\\\
K\_{j}& = X\_j W\_{K} R(j\Theta)\\\\
Q\_{i}K\_j^T &=X\_{i}W\_QR(i\Theta)R(j\Theta)^{T}W\_K^{T}X\_{j}^{T}\\\\
&=X\_{i}W\_QR(i\Theta)R(-j\Theta)W\_K^{T}X\_{j}^{T}\\\\
&=X\_{i}W\_QR((i-j)\Theta)W\_K^{T}X\_{j}^{T}\\\\
&=g(X\_i,X\_j,i-j)\\\\
\end{split}
\end{equation\*}

其中

\begin{equation\*}
\begin{split}
R(i\Theta)R(j\Theta)^{T} &= \widehat{R}(i\theta\_1)\widehat{R}(i\theta\_2)\ldots\widehat{R}(i\theta\_{D/2})\widehat{R}(j\theta\_{D/2})^{T}\ldots \widehat{R}(j\theta\_{2})^{T} \widehat{R}(j\theta\_{1})^{T} \\\\
&= (\widehat{R}(i\theta\_1)\widehat{R}(j\theta\_1)^T)(\widehat{R}(i\theta\_2)\widehat{R}(j\theta\_2)^T)\ldots(\widehat{R}(i\theta\_{D/2}\widehat{R}(j\theta\_{D/2})^T)\\\\
&= \widehat{R}((i-j)\theta\_1)\widehat{R}((i-j)\theta\_2)\ldots \widehat{R}((i-j)\theta\_{D/2})\\\\
&= R((i-j)\Theta)\\\\
\end{split}
\end{equation\*}


### 整体看下 {#整体看下}

-   空间是\\(D\\) 维度，\\(d=D/2\\)
-   有\\(d\\) 个正交的二维子空间 \\(\mathcal{X}\_1, \mathcal{X}\_2, \dots, \mathcal{X}\_{d}\\)
-   每个子空间\\(\mathcal{X}\_{k}\\) 有一个旋转角度基准 \\(\theta\_{k}\\), 一个基准旋转矩阵 \\(R(\theta\_{k})\\)
    -   合并后的基准角度序列和旋转序列是 \\(\Theta, R(\Theta)\\)
    -   每个子空间对应于三角函数中的一个周期 \\(2\pi/\theta\_{k}\\)
-   对于每个位置\\(i\\), 角度序列和旋转序列是 \\(i\Theta, R(i\Theta)\\)

    |                 |                    |                      |                    |              |                      |
    |-----------------|--------------------|----------------------|--------------------|--------------|----------------------|
    | \\(\Theta\\)    | \\(\theta\_1\\)    | \\(\theta\_2\\)      | \\(\theta\_3\\)    | \\(\ldots\\) | \\(\theta\_{d}\\)    |
    | \\(R(\Theta)\\) | \\(R(\theta\_1)\\) | \\(R(\theta\_{2})\\) | \\(R(\theta\_3)\\) | \\(\ldots\\) | \\(R(\theta\_{d})\\) |
    | \\(i\Theta\\)   | \\(i\theta\_1\\)   | \\(i\theta\_{2}\\)   | \\(i\theta\_3\\)   | \\(\ldots\\) | \\(i\theta\_{d}\\)   |


#### 具体化 {#具体化}

-   \\(\theta\_{k}\\) 是超参数

    -   \\(\theta\_{k}=10000^{-2(k-1)/D}, k\in[1,2,\ldots,D/2]\\)，记\\(B=10000^{1/d}\\)
    -   \\(\theta\_{k}=1/B^{k-1}\\) 是一个几何级数
    -   周期随着维度\\(k\\) 逐渐增大

    | \\(\Theta\\) | \\(1\\)    | \\(1/B\\)   | \\(1/B^{2}\\)   | \\(\ldots\\) | \\(1/B^{d-1}\\)   |
    |--------------|------------|-------------|-----------------|--------------|-------------------|
    | \\(T\\)      | \\(2\pi\\) | \\(2B\pi\\) | \\(2B^{2}\pi\\) | \\(\ldots\\) | \\(2B^{d-1}\pi\\) |


#### 随着位置的增大，位置编码是否会重复？ {#随着位置的增大-位置编码是否会重复}

-   如果存在位置$i$和 0 位置的编码撞车了，
-   \\(\Theta\\) 序列在 \\(2\pi\\) 周期整数倍上撞车
    -   \\(\Theta\\) 全部都是 $2&pi;$的整数倍
    -   对dimension的每个\\(k\\), 存在着一个整数 \\(I\_k\\), 使得\\(i\theta\_{k}=2\pi I\_k\\)
    -   不可能，\\(\theta\_k=1/10000^{k/d}\\) 和 \\(2\pi\\) 不会扯上关系


#### RoPE 可能的另外一个优势 {#rope-可能的另外一个优势}

-   在多个block 前向传递的过程中position的信息不会丢失
    -   每个block都会先做QKV的投影，然后QK投影之后会做位置旋转变换


### 再看下绝对位置编码 {#再看下绝对位置编码}

\\(\begin{aligned}
   P\_{i,2t} &= sin(i/10000^{2t/D}) &&\\\\
   P\_{i,2t+1} &= cos(i/10000^{2t/D})&&\\\\
   \end{aligned}\\)

\\(t\in[0,1,\ldots,D/2-1], i\ge0\\)

如果记\\(d=D/2,B=1/10000^{1/d}\\)，
那么\\(\theta\_{k}=1/B^{k-1}, k\in[1,2,\ldots,d]\\)


#### structure {#structure}

-   有\\(d\\) 个正交的二维子空间 \\(\mathcal{X}\_1, \mathcal{X}\_2, \dots, \mathcal{X}\_{d}\\)
-   每个子空间\\(\mathcal{X}\_{k}\\) 有一个基础角度 \\(\theta\_{k}\\)，
    -   两个基底, 记作\\(\text{Tri}(\theta\_k)=(sin(\theta\_k),   cos(\theta\_k))\\)
    -   合并后的基准角度序列和基底序列是 \\(\Theta, \text{Tri} (\Theta)\\)
    -   由 \\(\theta\_{k}\\) 来决定各个子空间的不同
    -   子空间内部由sin,cos 来区分
-   对于每个位置\\(i\\), 基准角度序列和基底序列是 \\(i\Theta, \text{Tri}(i\Theta)\\)

| \\(\Theta\\)             | \\(\theta\_1\\)             | \\(\theta\_{2}\\)             | \\(\theta\_3\\)             | \\(\ldots\\) | \\(\theta\_{d}\\)             |
|--------------------------|-----------------------------|-------------------------------|-----------------------------|--------------|-------------------------------|
| \\(\text{Tri}(\Theta)\\) | \\(\text{Tri}(\theta\_1)\\) | \\(\text{Tri}(\theta\_{2})\\) | \\(\text{Tri}(\theta\_3)\\) | \\(\ldots\\) | \\(\text{Tri}(\theta\_{d})\\) |
| \\(i\Theta\\)            | \\(i\theta\_1\\)            | \\(i\theta\_{2}\\)            | \\(i\theta\_3\\)            | \\(\ldots\\) | \\(i\theta\_{d}\\)            |

<!--list-separator-->

-  具体化

    -   \\(\theta\_{k}=1/B^{k-1}\\) 是一个几何级数序列
    -   周期随着维度\\(k\\) 逐渐增大

    | \\(\Theta\\) | \\(0\\)    | \\(1/B\\)   | \\(1/B^{2}\\)   | \\(\ldots\\) | \\(1/B^{d}\\)   |
    |--------------|------------|-------------|-----------------|--------------|-----------------|
    | \\(T\\)      | \\(2\pi\\) | \\(2B\pi\\) | \\(2B^{2}\pi\\) | \\(\ldots\\) | \\(2B^{d}\pi\\) |

<!--list-separator-->

-  如果我们记录 \\(i=x\\)

    \\(\\{sin(\theta\_k x), cos(\theta\_{k} x)\\}\_{k=1}^{D}\\) 很像对位置函数\\(f(x)\\) 的一个fourier展开


#### 同理，随着位置的增大，位置编码不会重复 {#同理-随着位置的增大-位置编码不会重复}


## 代码实现 {#代码实现}


### trick1: 避开旋转矩阵的相乘 {#trick1-避开旋转矩阵的相乘}


#### why？ {#why}

我们需要对每个\\(Q\_{i}\\) 乘以不同的旋转矩阵，也就是

\\(QR=\begin{pmatrix}
Q\_1 R(1\Theta)\\\\
Q\_2 R(2\Theta)\\\\
\ldots \\\\
Q\_N R(N\Theta)\\\\
\end{pmatrix}\\)

而每个\\(R(i\Theta)\\) 是一个稀疏矩阵，直接matmul代价太大


#### how? {#how}

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
&= (U,V)cos +(-V, U) sin
\end{aligned}\\)

同样的，在高维空间，我们可以把\\(Q\\) 拆分成\\(D/2\\) 个列向量\\(U\_1,V\_1,U\_2,V\_2,\ldots,U\_{D/2},V\_{D/2}\\)


### trick2: 将整个空间分成两部分 {#trick2-将整个空间分成两部分}

不需要做严格紧密相连的二维子空间序列

-   第一个部分放的是每个子空间的第一维度，
-   第二部分放置的是每个子空间的第二维度

<!--listend-->

```text
(x1,y1) 是一个子空间，(x2, y2)是一个子空间，(x3, y3)是一个子空间
before： [(x1,y1), (x2,y2), (x3,y3)]
after： [(x1,x2,x3), (y1, y2, y3)]
```


### code {#code}

```python
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
class Rotator:
    """根据hidden_dim，和position_ids 生成对应的旋转位置编码, 和论文中定义略有不同，一个个二维的子空间被
    分割到了前后两部分，分别进行旋转，然后拼接起来
    """

    def __init__(self, D, position_ids):
        """ position_ids: [seq_len], D 和单个头的hidden_dim对应 """
        base = 10000
        d = D / 2
        B = base ** (1/d)
        theta_base = 1.0 / (B ** (torch.arange(0, d)))    # 几何级数序列
        thetas = position_ids.outer(theta_base)  # [seq_len, D/2]
        full_thetas = torch.cat((thetas, thetas), dim=-1)  # [seq_len, D]
        self.cos = full_thetas.cos()
        self.sin = full_thetas.sin()

    def rotate(self, x):
        """ trick1
        x: [bs, num_attention_heads, seq_len, D]
        q: [bs, num_attention_heads, seq_len, D]
        cos: [seq_len, D]
        [x,y] @ [[cos, sin], [-sin, cos]] = [x*cos-y*sin, ycos+x*sin] =[x,y]*cos+[-y, x]*sin
        """
        return x * self.cos + Rotator.reverse_half(x) * self.sin

    @staticmethod
    def reverse_half(q):
        """ q: [bs, num_attention_heads, seq_len, D] trick2 """
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


### 总结 {#总结}

-   RoPE的motivation：

    -   希望相似性只依赖于向量本身和其相对位置的距离
    -   通过对\\(Q\_i,K\_i\\) 施加 $R(i&Theta;)$变换做到

    \begin{equation\*}
    \begin{split}
    Q\_{i}& = X\_{i} W\_{Q} R(i\Theta) \\\\
    K\_{j}& = X\_j W\_{K} R(j\Theta)\\\\
    Q\_{i}K\_j^T &=X\_{i}W\_QR(i\Theta)R(j\Theta)^{T}W\_K^{T}X\_{j}^{T}\\\\
    &=X\_{i}W\_QR(i\Theta)R(-j\Theta)W\_K^{T}X\_{j}^{T}\\\\
    &=X\_{i}W\_QR((i-j)\Theta)W\_K^{T}X\_{j}^{T}\\\\
    &=g(X\_i,X\_j,i-j)\\\\
    \end{split}
    \end{equation\*}
-   RoPE是什么？

    把原空间切分成为一个个正交的二维子空间，在上面做独立的旋转。
-   RoPE 结构
    -   有\\(d\\) 个正交的二维子空间 \\(\mathcal{X}\_1, \mathcal{X}\_2, \dots, \mathcal{X}\_{d}\\)
    -   每个子空间\\(\mathcal{X}\_{k}\\) 对应一个基础角度和基础矩阵 \\(\theta\_{k}, R(\theta\_{k})\\)
    -   对于每个位置\\(i\\), 对应一个角度序列和矩阵序列 \\(i\Theta, R(i\Theta)\\)

        | \\(\Theta\\)    | \\(\theta\_1\\)    | \\(\theta\_2\\)      | \\(\theta\_3\\)    | \\(\ldots\\) | \\(\theta\_{d}\\)    |
        |-----------------|--------------------|----------------------|--------------------|--------------|----------------------|
        | \\(R(\Theta)\\) | \\(R(\theta\_1)\\) | \\(R(\theta\_{2})\\) | \\(R(\theta\_3)\\) | \\(\ldots\\) | \\(R(\theta\_{d})\\) |
        | \\(i\Theta\\)   | \\(i\theta\_1\\)   | \\(i\theta\_{2}\\)   | \\(i\theta\_3\\)   | \\(\ldots\\) | \\(i\theta\_{d}\\)   |

-   绝对位置编码和RoPE 有相似的结构


## 参考论文 {#参考论文}

<style>.csl-entry{text-indent: -1.5em; margin-left: 1.5em;}</style><div class="csl-bib-body">
  <div class="csl-entry"><a id="citeproc_bib_item_1"></a>Su, Jianlin, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2022. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv. <a href="https://arxiv.org/abs/2104.09864">https://arxiv.org/abs/2104.09864</a>.</div>
  <div class="csl-entry"><a id="citeproc_bib_item_2"></a>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. “Attention Is All You Need.” arXiv. <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a>.</div>
</div>

