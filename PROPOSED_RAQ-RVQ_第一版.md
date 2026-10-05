# III. PROPOSED RAQ-RVQ

如 Fig. [2] 所示，针对上述问题，我们提出了融合残差向量量化的速率自适应量化方法（Rate-Adaptive Quantization with Residual Vector Quantization, RAQ-RVQ）。该方法以固定规模的 SimVQ 源码本为条件，通过基于 Transformer 的速率自适应码本生成模块（Transformer-Based Rate-Adaptive Codebook Generation, RTG）按需生成目标码本，并在每个语义尺度上逐级完成残差量化。在此基础上，本文进一步引入基于残差能量的索引选择机制，并结合无损掩码编码降低传输开销，从而在无需重新训练的情况下增强码率调节能力。

<!-- 此处插入 Fig. 2 -->

**Fig. 2. 所提出的多尺度独立 RAQ-RVQ 及掩码机制。**

## A. RAQ-RVQ

**1) Multi-Scale Semantic Codec：** 所提模型采用多尺度语义编解码结构，在浅层保留局部纹理与空间细节，在深层提取更加紧凑的高层语义信息，并通过不同尺度信息的互补融合提升语义表示与图像重建能力。

**2) Rate-Adaptive Quantization：**传统 VQ 通常采用固定大小的离散码本，模型训练完成后，其码本容量及对应的索引表示速率也基本固定。因此，当通信资源或目标传输码率发生变化时，往往需要重新训练模型，或者额外维护多组不同规模的码本。为避免这种“一种码率对应一套码本”的设计，我们提出了速率自适应量化（Rate-Adaptive Quantization, RAQ），通过固定源码本与动态目标码本生成机制，使同一语义通信模型能够根据目标码本大小 K 自适应构造不同容量的量化空间。

首先，每个尺度构建一个SimVQ固定源码本，作为后续该尺度速率自适应目标码本生成的输入。对于第 $l$ 个语义尺度，源码本的构建可以表示为：
$$
\mathbf{C}_{l,\cdot}^{\mathrm{src}}=\left\{
\mathbf{c}_{l,\cdot}^{\mathrm{src},(1)},\mathbf{c}_{l,\cdot}^{\mathrm{src},(2)},\ldots,\mathbf{c}_{l,\cdot}^{\mathrm{src},(k)},\ldots,\mathbf{c}_{l,\cdot}^{\mathrm{src},({K_{l,{\cdot}}^{\mathrm{src}}})}
\right\},  \tag{10}
$$

其中，${K_{l,{\cdot}}^{\mathrm{src}}}$ 表示为源码本的大小，并且 $\mathbf{C}_{l,\cdot}^{\mathrm{src}} \in \mathbb{R}^{K_{l,\cdot}^{\mathrm{src}} \times D_l}$，$D_l$表示源码本的维度。得益于SimVQ 在训练过程中保持码本参数冻结，并通过优化共享投影来调整码字的几何结构的思想，与直接独立更新每个码字相比，这种源码本参数化方式能够维持稳定的码字序列，能为不同目标规模下的码本生成提供统一条件。RAQ 的核心在于 **Rate-Adaptive Target Codebook Generator（RTG）**，我们提出了一种 Transformer encoder-decoder 的结构来实现 RTG，其核心思想是：将源码本中的全部码字作为源序列，并根据当前目标码本大小${K_{l;d}^{\mathrm{trg}}}$ 构造对应长度的目标序列，通过 Transformer 模块建模源码字之间以及源码本与目标码本之间的全局关系，最终生成所需数量的目标码字。换句话说，并不是从源码本中直接选取或截取 ${K_{l;d}^{\mathrm{trg}}}$ 个码字，而是利用源码本所包含的整体离散语义结构，重新生成一组满足当前目标容量的码字表示。具体而言，对于尺度 $l$ 的第 $d$ 级量化阶段，首先，将源码本编织为 Transformer encoder 的源序列：

$$
\mathbf{S}_{l,\cdot}=\mathrm{PE}\left(P_i\left(\mathbf{C}_{l,\cdot}^{\mathrm{src}}\right)\right)=\left\{
\mathbf{s}_{l,\cdot}^{(1)},
\mathbf{s}_{l,\cdot}^{(2)},
\ldots,
\mathbf{s}_{l,\cdot}^{(k)},
\ldots,
\mathbf{s}_{l,\cdot}^{(K_{l,{\cdot}}^{\mathrm{src}})}
\right\}, \tag{11}
$$

其中，$P_i(\cdot)$ 表示源码字维度的线性映射，来满足 Transformer  所要求的 ${d_{model}}$ 维度，并且$\mathbf{S}_{l,\cdot} \in \mathbb{R}^{K_{l,\cdot}^{\mathrm{src}} \times {d_{model}}}$；$\operatorname{PE}(\cdot)$ 表示位置编码操作。然后，Transformer encoder 通过多头自注意力机制对源序列之间的关系进行建模，使每个源码字的表示不仅包含自身信息，也能够拥有源码本中其他码字的信息。其输出记为:
$$
\mathbf{M}_{l,d}=\mathrm{Enc}_{l,d}\left(\mathbf{S}_{l,\cdot}\right)=\left\{
\mathbf{m}_{l,d}^{(1)},
\mathbf{m}_{l,d}^{(2)},
\ldots,\mathbf{m}_{l,d}^{(k)},\ldots
\mathbf{m}_{l,d}^{\left(K_{l,\cdot}^{\mathrm{src}}\right)}
\right\}, \tag{12}
$$
其中，$\mathbf{M}_{l,d} \in \mathbb{R}^{K_{l,\cdot}^{\mathrm{src}} \times D_l}$ 表示经过 Transformer encoder 编码后的源码本信息，作为后续 Transformer decoder 生成目标码本时所需的语义记忆。与此同时，给定当前量化阶段所要求的目标码本大小${K_{l,d}^{\mathrm{trg}}}$，构造目标 ID 序列 $\mathbf{t}_{l,d}=[1,2,\ldots,{K_{l,d}^{\mathrm{trg}}}]$，需要说明的是，这些目标 ID 本身并不是实际码字，而是用于标识待生成目标码本中的不同位置。同样，将这些目标 ID 序列编织为输入 Transformer decoder 的目标查询序列：
$$
\mathbf{T}_{l,d}=\mathrm{PE}\left(F_{l,d}^{\mathrm{trg}}(\mathbf{t}_{l,d})\right)=\left\{\mathbf{q}_{l,d}^{(1)},\mathbf{q}_{l,d}^{(2)},\ldots,\mathbf{q}_{l,d}^{(k)},\ldots,\mathbf{q}_{l,d}^{(K_{l,d}^{\mathrm{trg}})}\right\}, \tag{13}
$$
其中，$F_{l,d}^{\mathrm{trg}}(\cdot)$为可学习目标查询嵌入层，用于将这些离散的 ID 序列映射为 $D_l$ 维目标查询向量，并且$\mathbf{T}_{l,d} \in \mathbb{R}^{K_{l,d}^{\mathrm{trg}} \times D_l}$。再次需要注意的是，这些查询向量并非最终目标码字，而是目标码字生成的初始表示，也就是 $\mathbf{T}_{l,d}$ 中的每一个 query 对应一个待生成的目标码字。恰是这一巧妙的设计，进而成为了RTG 能实现速率自适应的关键，当需要用不同的速率量化时，RTG 无需调整网络参数，只需要改变 ${K_{l,d}^{\mathrm{trg}}}$ 即可获取新的目标 ID 序列长度。接下来，Transformer decoder 主要完成两类信息交互。首先，通过目标自我注意力机制，让目标查询序列之间的 query 可以相互感知，从而使待生成的各个目标码字能够进行联合协调，减少不同目标码字之间不必要的表示冗余。其次，通过交叉注意力机制，让每个 query 都能够从整个源码本信息中选择并聚合与自身最相关的信息，进而使生成的目标码字继承源码本所包含的语义结构。具体来说，先将 $\mathbf{T}_{l,d}$ 和  $\mathbf{M}_{l,d}$ 分别映射为交叉注意力机制所需的 Q、K 和 V 值：

$$
\mathbf{Q}_{l,d}^{\mathrm{att}}=\bar{\mathbf{T}}_{l,d}\mathbf{W}_{l,d}^{Q},\qquad
\mathbf{K}_{l,d}^{\mathrm{att}}=\mathbf{M}_{l,d}\mathbf{W}_{l,d}^{K},\qquad
\mathbf{V}_{l,d}^{\mathrm{att}}=\mathbf{M}_{l,d}\mathbf{W}_{l,d}^{V},\tag{14}
$$
其中，$\bar{\mathbf{T}}_{l,d}$ 为 $\mathbf{T}_{l,d}$ 经过自我注意力机制后的目标查询序列，而$\mathbf{W}_{l,d}^{Q}$、$\mathbf{W}_{l,d}^{K}$、$\mathbf{W}_{l,d}^{V}$ 均为所对应的可学习映射矩阵。然后，通过缩放点积注意力计算目标查询与源码本表示之间的相关性，并对 $\mathbf{M}_{l,d}$ 中的信息进行加权求和得到 Transformer decoder 输出的目标码字隐信息：
$$
\mathbf{H}_{l,d}
=
\mathrm{Dec}_{l,d}\left(\mathbf{T}_{l,d},\mathbf{M}_{l,d}\right)
=
\mathrm{Softmax}
\left(
\frac{
\mathbf{Q}_{l,d}^{\mathrm{att}}
\left(\mathbf{K}_{l,d}^{\mathrm{att}}\right)^{T}
}{
\sqrt{d_{model}}
}
\right)
\mathbf{V}_{l,d}^{\mathrm{att}},\tag{15}
$$
其中，$\mathbf{H}_{l,d} \in \mathbb{R}^{K_{l,d}^{\mathrm{trg}} \times D_l}$ 且 $d_{model}$ 表示 attention 的特征维度。最后再经过输出映射，生成最终所需要的目标码本：
$$
\mathbf{C}_{l,d}^{\mathrm{trg}}=P_o\left(\mathbf{H}_{l,d}\right)=\left\{
\mathbf{c}_{l,d}^{\mathrm{trg},(1)},\mathbf{c}_{l,d}^{\mathrm{trg},(2)},\ldots,\mathbf{c}_{l,d}^{\mathrm{trg},(k)},\ldots,\mathbf{c}_{l,d}^{\mathrm{trg},({K_{l,d}^{\mathrm{trg}}})}
\right\},\tag{16}
$$
其中，$P_0(\cdot)$表示维度线性映射操作。因此，RTG工作的整个流程简单来说就是能生成一个所需的新离散量化空间，并且该量化空间拥有基础量化空间的全部语义信息。

**3) Independent Residual Vector Quantization：** 从所提的 RAQ 模块描述所知，我们借助了残差向量量化（Residual Vector Quantization, RVQ）这个思想，目的是为了解决当 K 较小时，码本表示能力下降，单级量化容易产生较大的量化误差的问题。具体而言，对于第 $l$ 个语义尺度，设其有 $D$ 个残差量化阶段，并且包含 ${N_l}$ 个待量化语义向量，令$\boldsymbol R_{l,0}=\boldsymbol{Z_{l}}=\left\{\boldsymbol r_{l,{0}}^{(1)},\boldsymbol r_{l,{0}}^{(2)},\ldots,\boldsymbol r_{l,{0}}^{(N_l)}\right\}$ ，其中 $\boldsymbol{Z_{l}}$ 为第 $l$ 个尺度的原始语义特征，而 $\boldsymbol R_{l,0}$ 则表示进入第一级全部语义向量构成的初始残差特征。因此该尺度的第 $d$ 个残差量化阶段 RVQ 使用 RTG 生成目标码本 $\mathbf{C}_{l,d}^{\mathrm{trg}}$ 进行残差量化：
$$
\boldsymbol E_{l,d}^{trg},\boldsymbol I_{l,d}^{trg}=\mathcal{Q}(\boldsymbol R_{l,d-1};{\mathbf{C}_{l,d}^{\mathrm{trg}}}), \qquad d=1,2,\ldots,D \tag{17}
$$
其中，$\boldsymbol E_{l,d}^{trg}=\left\{\boldsymbol e_{l,{d}}^{(1)},\boldsymbol e_{l,{d}}^{(2)},\ldots,\boldsymbol e_{l,{d}}^{(N_l)}\right\}$ 和 $\boldsymbol I_{l,d}^{trg}=\left\{\boldsymbol i_{l,{d}}^{(1)},\boldsymbol i_{l,{d}}^{(2)},\ldots,\boldsymbol i_{l,{d}}^{(N_l)}\right\}$ 分别表示该阶段得到的量化特征和离散索引集合，$\mathcal{Q}(\cdot)$ 表示第 $l$ 个语义尺度第 $d$ 个残差量化阶段的量化操作，并且对该尺度内的各语义向量分别进行最近邻码字匹配。更具体地表示可以建模为:


$$
\begin{aligned}
i_{l,d}^{(n)} &=\underset{1\leq k\leq {K_{l,d}^{\mathrm{trg}}}}{\arg\min}\,
\left\|
\boldsymbol r_{l,{d-1}}^{(n)}-\mathbf{c}_{l,d}^{\mathrm{trg},(k)}
\right\|_2^2,\qquad n=1,2,\ldots,{N_l},\\
\boldsymbol e_{l,d}^{(n)}
&={\mathbf{c}_{l,d}^{\mathrm{trg},(i_{l,d}^{(n)})}}.
\end{aligned}
\tag{18}
$$
完成当前阶段量化后，将当前阶段已表示的量化信息从输入残差中去除，得到下一阶段的量化残差：
$$
\boldsymbol{R}_{l,d}=\boldsymbol{R}_{l,d-1}-\boldsymbol{E}_{l,d}^{trg}.\tag{19}
$$
以此重复上述(17)、(18)、(19)的操作，最终，该尺度的量化特征由各残差量化阶段结果累加得到，而各阶段的离散索引分别保留并按阶段顺序组织：
$$
\boldsymbol{I}_l^{trg}=\left(\boldsymbol{I}_{l,1}^{trg},\boldsymbol{I}_{l,2}^{trg},\ldots,\boldsymbol{I}_{l,d}^{trg},\ldots,\boldsymbol{I}_{l,D}^{trg}\right),\qquad \boldsymbol{E}_l^{trg}=\sum_{d=1}^{D} \boldsymbol{E}_{l,d}^{trg}.\tag{20}
$$
其中，索引部分的括号表示有序的阶段索引序列，而非数值相加，并且比特映射时依次拼接，以保留接收端查找对应码字所需的阶段信息。需要强调说明的是，我们在不同尺度及不同残差量化阶段均配置独立的 RTG 与目标码本。这是因为不同尺度的语义层级、空间分辨率以及不同残差阶段的特征分布存在明显差异，独立的码本生成机制可使各量化分支针对自身特征进行专门建模，减少不同尺度与阶段之间的相互干扰，从而提升目标码本与待量化特征的匹配程度及整体量化性能。

**4) Adaptive Masking  for RAQ-RVQ Inference：** 

我们发现，随着残差量化逐级进行，不同位置的残差信息对最终重建质量的贡献并不相同。对于部分语义位置，前级量化结果已经能够较好地表征原始特征，继续增加后续残差量化阶段所带来的重建增益较为有限，却会引入额外的索引传输开销。基于此，我们在训练好模型的推理阶段上引入了基于残差能量的掩码机制，根据实际比特开销的需求，通过目标激活比例控制后续残差量化阶段参与传输量化索引的数量。具体而言，对于第 $l$ 个语义尺度第 $d$ 个残差量化阶段，同样设其有 $D$ 个残差量化阶段，并包含 $N_l$ 个参与量化的语义向量数，则残差能量可以定义为：
$$
\varepsilon_{l,d-1}^{(n)}=\frac{1}{D_l}\left\|\boldsymbol r_{l,d-1}^{(n)}\right\|_2^2,\qquad n=1,2,\ldots,N_l,\tag{21}
$$
其中，$D_l$ 表示该尺度语义向量的特征维度，$\boldsymbol r_{l,d-1}^{(n)}$ 表示第 $n$ 个位置的残差向量。若残差能量较大，则表明该位置仍存在较多未被前级量化表示的信息，因此具有更高的进一步量化需求。给定目标激活比例 $\rho_{l,d}\in[0,1]$，则对应目标激活位置数定义为：
$$
A_{l,d}
=
\min\left\{
N_l,\,
\max\left(
0,\,
\left\lfloor \rho_{l,d}N_l+0.5 \right\rfloor
\right)
\right\}.\tag{22}
$$
随后，根据残差能量对各位置进行排序，并选取能量最大的 $A_{l,d}$个位置形成激活集合:
$$
\Omega_{l,d} = \operatorname{Top}_{A_{l,d}} \left( \left\{ \varepsilon_{l,d-1}^{(n)} \right\}_{n=1}^{N_l} \right),\tag{23}
$$
所以最终相对应的二值掩码可以定义为：
$$
\mathbf{u}_{l,d}
=\left[
u_{l,d}^{(n)}
\right]_{n=1}^{N_l},
\qquad

u_{l,d}^{(n)}
=\begin{cases}
1, & n\in\Omega_{l,d},\\
0, & n\notin\Omega_{l,d}.
\end{cases}
\tag{24}
$$
其中，当 $u_{l,d}^{(n)}=1$ 时，表示第 $n$ 个位置在该残差量化阶段被激活，其对应的量化结果和离散索引被保留；当 $u_{l,d}^{(n)}=0$ 时，则跳过该位置在当前阶段的残差量化，不参与后面的传输过程。因此，该语义尺度经过掩码选择后的量化表示可统一写为：
$$
\boldsymbol{I}_{l}^{trg}
=
\left(
\boldsymbol{I}_{l,1}^{trg}[\Omega_{l,1}],\boldsymbol{I}_{l,2}^{trg}[\Omega_{l,2}],\ldots,\boldsymbol{I}_{l,d}^{trg}[\Omega_{l,d}],\ldots,
\boldsymbol{I}_{l,D}^{trg}[\Omega_{l,D}]
\right),
\qquad
\boldsymbol{E}_{l}^{trg}
=
\sum_{d=1}^{D}
\mathbf{u}_{l,d}
\odot
\boldsymbol{E}_{l,d}^{trg},
\tag{25}
$$
其中，$\boldsymbol{I}_{l,d}^{trg}$ 与 $\boldsymbol{E}_{l,d}^{trg}$ 的含义与式(17)一致，而 $\boldsymbol{I}_{l,d}^{trg}[\Omega_{l,d}]$ 表示按空间顺序提取激活集合中的索引，$\odot$ 表示按位置选择量化贡献。第一级始终参与表示，即 $\Omega_{l,1}=\{1,\ldots,N_l\}$ 且 $\mathbf{u}_{l,1}=\mathbf{1}$。对于后续阶段，接收端无法仅凭保留的索引确定其空间位置，因此二值掩码也属于有效载荷。另外当采用二值掩码时，式(19)的残差更新去除实际保留的量化信息变为：

$$
\boldsymbol R_{l,d}
=\boldsymbol R_{l,d-1}-\mathbf u_{l,d}\odot\boldsymbol E_{l,d}^{trg}
=\boldsymbol Z_l-\sum_{j=1}^{d}\mathbf u_{l,j}\odot\boldsymbol E_{l,j}^{trg}.\tag{26}
$$
后续阶段据此计算残差能量，并始终在全部 $N_l$ 个位置中选择激活集合，而不局限于前一级已激活的位置。因此，各阶段的掩码不要求相互嵌套，前一级未激活的位置仍可在后级参与残差量化。

**5) insight：** 我们所提方法不仅仅是“ RAQ 与 RVQ 的简单组合”，更是一种从码本容量、残差细化和空间选择三个层面协同分配有限比特的机制。该方法不仅能够根据传输需求动态调整量化能力，还可将额外比特优先用于难以表示的残差信息和关键语义位置，从而减少冗余传输，提高有限码率下的量化效率。为了进一步适应不同语义尺度和残差阶段的特征差异，我们采用独立的码本生成分支，使各量化模块能够针对自身特征分布进行专门建模。因此，我们提出的原生模型能覆盖多种码率配置，并实现更加灵活、细粒度的自适应语义传输。

## B. Training Procedure

与只训练单一固定码本不同，训练多尺度、多残差阶段且容量可变的目标码本时，其难点可以概括为以下几点。首先，最近邻索引选择不可微，如果仅依赖最终图像重建误差，则 RTG 生成的码字难以获得直接、稳定的对齐监督。其次，同一 RTG 需要在不同目标容量 $K$ 下生成几何结构有效的码本；尤其是在小 $K$ 下，表示能力骤降会放大量化误差，使不同容量配置产生显著不同的梯度尺度。再次，RVQ 各阶段拟合的是不同残差分布；若只约束全部阶段的最终和，各阶段可能相互补偿甚至重复表示，难以形成明确的逐级残差分工。最后，SimVQ 源分支与 RAQ-RVQ 分支共享语义编解码器，因而既要保持源码本作为稳定语义锚点的基础量化能力，又要使动态目标码本学会对齐各尺度的特征分布，避免某一分支或某一尺度主导联合优化。

基于上述困难，我们将训练目标分解为三类互补约束：用双分支图像重建损失保证最终语义通信任务；用源码本量化损失维持稳定的基准量化空间；用“当前阶段拟合当前残差”与“阶段累积结果迫近原始特征”的双重约束训练各级 RTG。图像重建梯度通过最终直通估计器传回语义编码器，而冻结的基础嵌入不更新；源码本的可训练投影、各级 RTG、语义编解码器等参数由上述目标联合优化。

#### **1) Loss Function：**

**a) Image Reconstruction Distortion**

量化损失只能保证语义特征与码字在特征空间中对齐，不能直接表征最终图像的任务失真。因此，我们同时约束源分支和目标分支的重建结果，使共享解码器能够处理两类量化特征，并防止动态码本训练以牺牲源分支的基准重建能力为代价。记两条分支的重建图像分别为 $\widehat{\boldsymbol x}^{\mathrm{src}}$ 和 $\widehat{\boldsymbol x}^{\mathrm{trg}}$，则图像重建损失定义为：
$$
\mathcal L_{\mathrm{rec}}
=\frac{\lambda_{\mathrm{mse}}}{3HW}\left[
\left\|\boldsymbol x-\widehat{\boldsymbol x}^{\mathrm{src}}\right\|_2^2
+
\left\|\boldsymbol x-\widehat{\boldsymbol x}^{\mathrm{trg}}\right\|_2^2\right],
\tag{27}
$$

其中，$\lambda_{\mathrm{mse}}$ 表示图像重建损失权重，并且两条分支都采用均方误差来衡量图像失真。

**b) Source Codebook Quantization Loss**

源分支既是可独立重建的基准分支，也为各级 RTG 提供统一的源码本语义结构。因此，其量化空间不应随动态目标码本的采样而剧烈漂移。为此，我们冻结 SimVQ 的基础嵌入，仅通过可训练投影调整有效源码字，并同时约束码字与编码特征。对于 SimVQ 源分支，设计的损失函数由下式给出：
$$
\mathcal{L}_{l}^{\mathrm{src}}
=
\left\|
\boldsymbol{E}_{l,\cdot}^{src}
-
\operatorname{sg}\left(\boldsymbol{Z}_l\right)
\right\|_{2}^{2}
+
\beta^{src}
\left\|
\boldsymbol{Z}_l - \operatorname{sg}\left(\boldsymbol{E}_{l,\cdot}^{src}\right)
\right\|_{2}^{2},\tag{28}
$$
其中，$\boldsymbol{Z}_l$ 已经在式(1)给出，$\boldsymbol{E}_{l,\cdot}^{src}$ 为源分支经过 $\mathbf{C}_{l,\cdot}^{\mathrm{src}}$ 的量化特征，$\operatorname{sg}(\cdot)$ 表示 stop-gradient 操作，$\beta^{src}$ 为承诺系数。第一项将经投影后的源码字拉向分配给它们的编码特征，为可训练码本投影提供对齐梯度；第二项则约束编码器输出不要远离已选码字。两者分别更新码字变换与编码特征，从而在保持基础量化和重建能力的同时，为各 RTG 提供可靠的语义基准。

**c) Residual Quantization Constraint**

对于 RAQ-RVQ 分支，仅约束最后的累积量化结果会使阶段间的贡献难以区分：前级码本可能未充分表示主要成分，后级码本则可能用来补偿前级偏差，甚至出现阶段间重复表示。因此，我们同时监督每级的当前残差拟合与截止该级的累积重构，使每个 RTG 都获得与其职责直接对应的训练信号。设计的损失函数由下式给出：
$$
\mathcal{L}_{l}^{\mathrm{trg}}
=
\frac{1}{D}
\sum_{d=1}^{D}
\left[
\left\|
\boldsymbol{E}_{l,d}^{trg}
-
\operatorname{sg}\left(\boldsymbol{R}_{l,d-1}\right)
\right\|_2^2
+
\beta^{trg}
\left\|
\boldsymbol{Z}_l
-
\operatorname{sg}\left(
\sum_{j=1}^{d}\boldsymbol{E}_{l,j}^{trg}
\right)
\right\|_2^2
\right],
\tag{29}
$$
其中，$D$ 表示残差量化阶段数，其余变量的定义均在之前给出。第一项是阶段码本损失，将第 $d$ 级 RTG 生成的码字拉向前 $d-1$ 级尚未表示的残差，使后续阶段真正用于逐级细化；第二项是累积承诺损失，使编码特征在每个阶段前缀上都尽量可由已生成码字的和来表示。对全部 $D$ 个阶段取平均，可避免随残差深度增加而改变该尺度损失的整体量级。

**d) Overall Training Objective**

综合图像重建与多尺度量化的约束，模型的整体训练目标定义为：
$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{rec}}
+
\sum_{l=1}^{L}
w_l
\left(
\mathcal{L}_{l}^{\mathrm{src}}
+
\mathcal{L}_{l}^{\mathrm{trg}}
\right).
\tag{30}
$$
其中，$w_l$ 表示各尺度的约束权重。训练初期适当提高深层语义尺度的约束权重，随后逐步调整为均衡配置，以兼顾不同尺度的收敛与联合优化。

#### **2) Training Details：**

如果从训练开始就在完整尺寸范围内随机切换 $K$，并同时加入信道错误，模型将同时面对目标码本容量变化、小码本引起的强量化失真以及索引扰动三种变化。在 RTG 和语义编解码器尚未形成稳定对齐时，这些高方差训练信号容易导致码字分配频繁改变、各 RVQ 阶段分工不稳定，以及共享解码器过早拟合噪声。因此，我们采用由易到难的三阶段课程：先在无信道干扰的大码本条件下建立语义特征与动态码字之间的基本对应，再增加小码本样本并平衡多尺度优化，最后在完整尺寸范围上引入信道扰动进行鲁棒性微调。

训练过程如 Algorithm 1 所示。本方案共训练 200 个 epoch。为避免自然语言中的 epoch 编号产生歧义，下文以从 0 开始的 $\tau\in\{0,1,\ldots,199\}$ 表示代码中的 epoch。三个阶段的精确设置如下。

| 训练阶段 | 代码 epoch $\tau$（常规编号） | 阶段尺寸集 | 重点尺寸集 | 尺度损失权重 $(w_1,w_2)$ | 跳连 Dropout | 信道启用概率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 阶段 I：表示建立 | $[0,20)$（第 1–20 个 epoch） | $\mathcal B_1=\{32,64\}$ | 无 | $(0.25,0.50)$ | $0.10$ | $0$ |
| 阶段 II：容量扩展 | $[20,80)$（第 21–80 个 epoch） | $\mathcal B_2=\{8,16,32,64\}$ | $\mathcal F_2=\{8,16\}$ | 线性由 $(0.25,0.50)$ 变为 $(0.25,0.25)$ | 线性由 $0.10$ 降至 $0$ | $0$ |
| 阶段 III：鲁棒微调 | $[80,200)$（第 81–200 个 epoch） | $\mathcal B_3=\{2,4,8,16,32,64\}$ | $\mathcal F_3=\{2,4\}$ | $(0.25,0.25)$ | $0$ | $\tau=80,\ldots,119$ 时由 $0$ 按 $0.025$/epoch 增至 $0.975$；$\tau\geq120$ 时为 $1$ |

三个阶段均使用式(30)的联合目标，SimVQ 源分支与 RAQ-RVQ 分支始终同时训练。SimVQ 源码本和目标查询的基础嵌入保持冻结，其可训练线性投影、各尺度各阶段的 RTG、语义编码器和共享解码器参与联合更新。阶段 I 只提供相对充足的码字数，用于降低初期量化难度；阶段 II 纳入 $K=8,16$ 并对它们重点采样，同时逐步撤去深层尺度的额外损失权重和跳连 Dropout；阶段 III 再将范围扩展到 $K=2,4$，并以它们作为重点尺寸，使模型充分接触量化失真最强的低码率配置。

本方案的尺寸采样为 **per-$K$ mixture**，而不是在阶段尺度集上的简单均匀采样。对两个语义尺度和两个残差阶段共四个位置，每个 $K_{l,d}^{\mathrm{trg}}$ 均独立进行一次混合选择。阶段 I 直接使用

$$
K_{l,d}^{\mathrm{trg}}\sim\operatorname{Unif}(\mathcal B_1).
$$

对阶段 $s\in\{2,3\}$，先对每个 $(l,d)$ 独立采样 $m_{l,d}\sim\operatorname{Bernoulli}(0.5)$，然后按

$$
K_{l,d}^{\mathrm{trg}}\sim
\begin{cases}
\operatorname{Unif}(\mathcal F_s), & m_{l,d}=1,\\
\operatorname{Unif}(\mathcal B_s), & m_{l,d}=0.
\end{cases}
$$

因此，阶段 II 中 $K=8,16$ 的边缘采样概率各为 $3/8$，$K=32,64$ 各为 $1/8$；阶段 III 中 $K=2,4$ 的边缘概率各为 $1/3$，$K=8,16,32,64$ 各为 $1/12$。这种采样在保留较大 $K$ 样本以防止已学表示被遗忘的同时，将更多更新机会分配给刚引入且更难优化的小 $K$。四个位置互相独立，因而训练中自然包含尺度间和阶段间的非对称配置，不要求全部量化分支使用相同的码本大小。

在具体实现中，每个小批量开始时采样一组 $\{K_{l,d}^{\mathrm{trg}}\}$，该配置由当前小批量内的所有图像共享，并在该次前向计算、反向传播和参数更新期间保持不变；下一小批量重新独立采样。本实验的总 batch size 与 micro-batch size 均为 24，因而每个小批量都对应一次优化器更新。验证时不使用上述重点混合分布；每个验证小批量都会为四个位置从完整集合 $\{2,4,8,16,32,64\}$ 中重新独立均匀采样，以监测模型对完整目标范围的平均泛化能力。

信道扰动只在阶段 III 引入。其小批量级启用概率为

$$
p_{\mathrm{ch}}(\tau)=
\begin{cases}
0, & 0\leq\tau<80,\\
(\tau-80)/40, & 80\leq\tau<120,\\
1, & 120\leq\tau<200.
\end{cases}
$$

也就是说，阶段 I 和阶段 II 完全使用无噪索引；进入阶段 III 时，第 81 个 epoch（$\tau=80$）的启用概率仍为 0，从第 82 个 epoch 起变为非零，此后以每个 epoch $0.025$ 的步长提高，并从第 121 个 epoch 开始保持为 1。对于每个小批量，先根据 $p_{\mathrm{ch}}(\tau)$ 作一次伯努利决策；若启用信道，则采样 $\mathrm{SNR}\sim\operatorname{Unif}[0,15]$ dB，在编码率 $1/2$ 和块长 256 bit 下用有限码长近似计算 BER，再对源分支与 RAQ-RVQ 各阶段索引的二进制表示施加独立比特翻转。调制阶数随 SNR 分段采样：低于 4 dB 时从 BPSK/QPSK 中选择，$[4,8)$ dB 时从 BPSK/QPSK/16QAM 中选择，不低于 8 dB 时从 QPSK/16QAM 中选择。延后引入信道的目的是先学会稳定的可变容量量化表示，再让共享解码器适应索引错误，从而减少训练初期将量化误差与信道误差混在一起所带来的优化不确定性。

由于最近邻码字选择会阻断重建损失对语义编码器的梯度传播，RAQ-RVQ 分支在全部 $D$ 级量化结果累加后仅设置一次直通估计器：

$$
\widetilde{\boldsymbol E}_l^{\mathrm{trg}}
=\boldsymbol Z_l+\operatorname{sg}\left(
\boldsymbol E_l^{\mathrm{trg}}-\boldsymbol Z_l
\right).
\tag{31}
$$
而 SimVQ 源分支采用对应的单级直通估计。需要强调的是，各残差量化阶段不重复设置直通路径。当阶段 III 的信道伯努利决策为启用时，根据采样的 SNR、调制方式及编码参数，通过有限码长**[这里引用有限码长文章]**近似计算比特错误率 $p_b$，即在两条分支的索引二进制表示上添加独立比特翻转 $\mathbf{\widetilde b}=\mathbf{b}\oplus \xi, \xi_j\sim\operatorname{Bernoulli}(p_b)$。因此，设 $\boldsymbol E_l^{\mathrm{trg,ch}}$ 为 RAQ-RVQ 分支对受扰索引重新查表并累加得到的特征，则送入语义解码器的训练特征可表示为：
$$
\widetilde{\boldsymbol E}_l^{\mathrm{trg,ch}}
=\widetilde{\boldsymbol E}_l^{\mathrm{trg}}
+\operatorname{sg}\left(
\boldsymbol E_l^{\mathrm{trg,ch}}-\boldsymbol E_l^{\mathrm{trg}}
\right).
\tag{32}
$$
而SimVQ源分支采用同样的处理方式。

---

**Algorithm 1. RAQ-RVQ Training Algorithm**

| 行号 | 过程 |
| ---: | :--- |
| 1 | **Function** Joint_Training() |
| 2 | &emsp;**Input:** 训练集 $\mathscr T$、$L$ 个语义尺度、$D$ 个残差阶段、源码本及各级 RTG、阶段尺度集 $\mathcal B_s$、重点尺度集 $\mathcal F_s$ 及信道课程配置。 |
| 3 | 记全部可训练参数为 $\Theta$。 |
| 4 | &emsp;**for** 每个 epoch $\tau$ **do** |
| 5 | &emsp;&emsp;根据 $\tau$ 确定当前阶段 $s$、$\mathcal B_s$、$\mathcal F_s$、尺度权重 $w_l$、跳跃连接 Dropout 和信道启用概率 $p_{\mathrm{ch}}(\tau)$。 |
| 6 | &emsp;&emsp;**for** 每个小批量 $\boldsymbol x\sim\mathscr T$ **do** |
| 8 | &emsp;&emsp;&emsp;对每个 $(l,d)$ 独立采样 $K_{l,d}^{\mathrm{trg}}$：阶段 I 从 $\mathcal B_1$ 均匀采样；阶段 II/III 各以 $0.5$ 概率从 $\mathcal F_s$ 或 $\mathcal B_s$ 中均匀采样。 |
| 9 | &emsp;&emsp;&emsp;$\mathcal Z\leftarrow\boldsymbol S(\boldsymbol x;\boldsymbol\theta_t)$，按式(1)提取多尺度语义特征。 |
| 10 | &emsp;&emsp;&emsp;**for** $l=1,\ldots,L$ **do** |
| 11 | &emsp;&emsp;&emsp;&emsp;利用源码本量化 $\boldsymbol Z_l$，得到 $\boldsymbol E_{l,\cdot}^{\mathrm{src}}$ 与 $\boldsymbol I_{l,\cdot}^{\mathrm{src}}$。 |
| 12 | &emsp;&emsp;&emsp;&emsp;$\boldsymbol R_{l,0}\leftarrow\operatorname{sg}(\boldsymbol Z_l)$，$\boldsymbol E_l^{\mathrm{trg}}\leftarrow\boldsymbol0$。 |
| 13 | &emsp;&emsp;&emsp;&emsp;**for** $d=1,\ldots,D$ **do** |
| 14 | &emsp;&emsp;&emsp;&emsp;&emsp;根据 $K_{l,d}^{\mathrm{trg}}$，由式(11)—式(16)生成 $\mathbf C_{l,d}^{\mathrm{trg}}$。 |
| 15 | &emsp;&emsp;&emsp;&emsp;&emsp;$(\boldsymbol E_{l,d}^{\mathrm{trg}},\boldsymbol I_{l,d}^{\mathrm{trg}})\leftarrow\mathcal Q(\boldsymbol R_{l,d-1};\mathbf C_{l,d}^{\mathrm{trg}})$。 |
| 16 | &emsp;&emsp;&emsp;&emsp;&emsp;$\boldsymbol E_l^{\mathrm{trg}}\leftarrow\boldsymbol E_l^{\mathrm{trg}}+\boldsymbol E_{l,d}^{\mathrm{trg}}$，保存本级残差及累积特征以计算式(29)。 |
| 17 | &emsp;&emsp;&emsp;&emsp;&emsp;$\boldsymbol R_{l,d}\leftarrow\boldsymbol R_{l,d-1}-\operatorname{sg}(\boldsymbol E_{l,d}^{\mathrm{trg}})$。 |
| 18 | &emsp;&emsp;&emsp;&emsp;**end for** |
| 19 | &emsp;&emsp;&emsp;&emsp;按式(20)保存各阶段索引，并根据式(31)构造目标分支的直通特征；源分支采用单级直通估计。 |
| 20 | &emsp;&emsp;&emsp;**end for** |
| 21 | &emsp;&emsp;&emsp;采样 $\chi_{\mathrm{ch}}\sim\operatorname{Bernoulli}(p_{\mathrm{ch}}(\tau))$。 |
| 22 | &emsp;&emsp;&emsp;**if** $\chi_{\mathrm{ch}}=1$ **then** |
| 23 | &emsp;&emsp;&emsp;&emsp;采样 SNR 和调制方式，计算 $p_b$；对源分支及目标分支索引施加独立比特翻转，查表恢复受扰特征，并对两分支应用式(32)处理。 |
| 24 | &emsp;&emsp;&emsp;**else** 使用两分支的干净直通特征；**end if**。 |
| 25 | &emsp;&emsp;&emsp;由共享语义解码器得到 $\widehat{\boldsymbol x}^{\mathrm{src}}$ 与 $\widehat{\boldsymbol x}^{\mathrm{trg}}$。 |
| 26 | &emsp;&emsp;&emsp;按式(27)—式(30)计算当前小批量的损失 $\mathcal L$。 |
| 27 | &emsp;&emsp;&emsp;执行反向传播，计算当前小批量的参数梯度 $\nabla_{\Theta}\mathcal L$。 |
| 28 | &emsp;&emsp;&emsp;使用当前小批量的梯度更新 $\Theta$。 |
| 29 | &emsp;&emsp;**end for** |
| 30 | &emsp;**end for** |
| 31 | &emsp;**Return:** 训练后的语义编解码器、源码本投影及各级 RTG。 |

---

推理过程如 Algorithm 2 所示。无 Mask 的推理路径对全部语义位置执行 $D$ 级残差量化，并分别正常传输各阶段索引；而有 Mask 的路径保持第一级完整量化，后续各级则根据当前残差选择激活位置，其残差更新包含前级实际保留的量化信息。进一步地，有 Mask 路径的激活位置按残差能量排序选取，每个尺度的后续残差量化阶段采用二元算术编码进行二值掩码，概率计数在每幅图像、每个尺度和每个阶段开始时重新初始化。接收端先恢复掩码，再将保留索引映射回对应空间位置，而未激活位置的该级量化信息置零。需要注意的是，Mask 不参与联合训练，其作用仅为推理阶段的残差索引选择，不改变所提网络模型的任何参数 。

---

**Algorithm 2. RAQ-RVQ Inference Algorithm**

**a) Inference without Mask**

| 行号 | 过程 |
| ---: | :--- |
| 1 | **Function** Inference_Without_Mask() |
| 2 | &emsp;**Input:** 图像 $\boldsymbol x$、共享的训练后模型、目标码本 $\{K_{l,d}^{\mathrm{trg}}\}$。 |
| 3 | &emsp;**Transmitter:** |
| 4 | &emsp;&emsp;$\mathcal Z\leftarrow\boldsymbol S(\boldsymbol x;\boldsymbol\theta_t)$。 |
| 5 | &emsp;&emsp;**for** $l=1,\ldots,L$ **do** |
| 6 | &emsp;&emsp;&emsp;$\boldsymbol R_{l,0}\leftarrow\boldsymbol Z_l$。 |
| 7 | &emsp;&emsp;&emsp;**for** $d=1,\ldots,D$ **do** |
| 8 | &emsp;&emsp;&emsp;&emsp;根据共享源码本和 $K_{l,d}^{\mathrm{trg}}$，由式(11)—式(16)生成 $\mathbf C_{l,d}^{\mathrm{trg}}$。 |
| 9 | &emsp;&emsp;&emsp;&emsp;按式(17)、式(18)量化 $\boldsymbol R_{l,d-1}$，得到 $\boldsymbol E_{l,d}^{\mathrm{trg}}$ 与 $\boldsymbol I_{l,d}^{\mathrm{trg}}$。 |
| 10 | &emsp;&emsp;&emsp;&emsp;$\boldsymbol R_{l,d}\leftarrow\boldsymbol R_{l,d-1}-\boldsymbol E_{l,d}^{\mathrm{trg}}$，保存当前阶段索引。 |
| 11 | &emsp;&emsp;&emsp;**end for** |
| 12 | &emsp;&emsp;**end for** |
| 13 | &emsp;&emsp;按约定的尺度、阶段顺序拼接全部索引，并依据式(3)映射为 $\mathbf b$。 |
| 14 | &emsp;&emsp;$\mathbf s\leftarrow\mathcal M(\mathcal C(\mathbf b))$，发送 $\mathbf s$ 经过式(5)所示无线信道。 |
| 15 | &emsp;**Receiver:** |
| 16 | &emsp;&emsp;接收 $\mathbf y$，按式(6)解调与信道解码，得到 $\widehat{\mathbf b}$。 |
| 17 | &emsp;&emsp;按式(7)和分段顺序恢复各阶段索引 $\widehat{\boldsymbol I}_{l,d}^{\mathrm{trg}}$。 |
| 19 | &emsp;&emsp;根据各阶段恢复索引查找对应码字，得到 $\widehat{\boldsymbol E}_{l,d}^{\mathrm{trg}}$。 |
| 20 | &emsp;&emsp;$\widehat{\boldsymbol E}_l^{\mathrm{trg}}\leftarrow\sum_{d=1}^{D}\widehat{\boldsymbol E}_{l,d}^{\mathrm{trg}}$，恢复各尺度空间向量。 |
| 21 | &emsp;&emsp;按式(9)对多尺度恢复特征进行语义解码，得到 $\widehat{\boldsymbol x}$。 |
| 22 | &emsp;**Return:** $\widehat{\boldsymbol x}$。 |

**b) Inference with Mask**

| 行号 | 过程 |
| ---: | :--- |
| 1 | **Function** Inference_With_Mask() |
| 2 | &emsp;**Input:** 图像 $\boldsymbol x$、共享的训练后模型、目标码本 $\{K_{l,d}^{\mathrm{trg}}\}$、后续阶段激活比例 $\{\rho_{l,d}\}_{d=2}^{D}$。 |
| 3 | &emsp;**Transmitter:** |
| 4 | &emsp;&emsp;$\mathcal Z\leftarrow\boldsymbol S(\boldsymbol x;\boldsymbol\theta_t)$。 |
| 5 | &emsp;&emsp;**for** $l=1,\ldots,L$ **do** |
| 6 | &emsp;&emsp;&emsp;$\boldsymbol R_{l,0}\leftarrow\boldsymbol Z_l$。 |
| 7 | &emsp;&emsp;&emsp;**for** $d=1,\ldots,D$ **do** |
| 8 | &emsp;&emsp;&emsp;&emsp;由当前 $K_{l,d}^{\mathrm{trg}}$ 生成目标码本 $\mathbf C_{l,d}^{\mathrm{trg}}$。 |
| 9 | &emsp;&emsp;&emsp;&emsp;**if** $d=1$ **then** $\Omega_{l,1}\leftarrow\{1,\ldots,N_l\}$，$\mathbf u_{l,1}\leftarrow\mathbf1$。 |
| 10 | &emsp;&emsp;&emsp;&emsp;**else** 根据 $\boldsymbol R_{l,d-1}$，按式(21)—式(24)计算残差能量、激活数量、激活集合和二值掩码；**end if**。 |
| 11 | &emsp;&emsp;&emsp;&emsp;将阶段量化特征初始化为零；若 $\Omega_{l,d}$ 非空，按式(18)匹配其中各位置的码字，并仅保存这些位置的索引。 |
| 12 | &emsp;&emsp;&emsp;&emsp;将选中的码字放回原位置，得到 $\boldsymbol E_{l,d}^{\mathrm{trg}}$；未激活位置置零。 |
| 13 | &emsp;&emsp;&emsp;&emsp;$\boldsymbol R_{l,d}\leftarrow\boldsymbol R_{l,d-1}-\mathbf u_{l,d}\odot\boldsymbol E_{l,d}^{\mathrm{trg}}$。 |
| 14 | &emsp;&emsp;&emsp;&emsp;**if** $d\geq2$ **then** 独立初始化算术编码器，并无损编码 $\mathbf u_{l,d}$；**end if**。 |
| 15 | &emsp;&emsp;&emsp;**end for** |
| 16 | &emsp;&emsp;**end for** |
| 17 | &emsp;&emsp;依次拼接全部尺度的第一级索引、后续阶段的压缩 Mask、后续阶段的激活索引，并完成二进制映射，得到 $\mathbf b$。 |
| 18 | &emsp;&emsp;$\mathbf s\leftarrow\mathcal M(\mathcal C(\mathbf b))$，发送 $\mathbf s$ 经过无线信道。 |
| 19 | &emsp;**Receiver:** |
| 20 | &emsp;&emsp;接收 $\mathbf y$，解调与信道解码得到 $\widehat{\mathbf b}$，按分段顺序恢复第一级索引、压缩 Mask 和后续激活索引。 |
| 21 | &emsp;&emsp;分别进行算术解码，得到各阶段恢复掩码 $\widehat{\mathbf u}_{l,d}$；第一级掩码固定为 $\mathbf1$。 |
| 23 | &emsp;&emsp;**for** $l=1,\ldots,L$ **do** |
| 25 | &emsp;&emsp;&emsp;**for** $d=1,\ldots,D$ **do** |
| 26 | &emsp;&emsp;&emsp;&emsp;依据恢复索引查找码字，按 $\widehat{\mathbf u}_{l,d}$ 指示的位置放回 $\widehat{\boldsymbol E}_{l,d}^{\mathrm{trg}}$，其余位置补零。 |
| 27 | &emsp;&emsp;&emsp;&emsp;$\widehat{\boldsymbol E}_l^{\mathrm{trg}}\leftarrow\widehat{\boldsymbol E}_l^{\mathrm{trg}}+\widehat{\mathbf u}_{l,d}\odot\widehat{\boldsymbol E}_{l,d}^{\mathrm{trg}}$。 |
| 28 | &emsp;&emsp;&emsp;**end for** |
| 29 | &emsp;&emsp;**end for** |
| 30 | &emsp;&emsp;恢复各尺度空间排列，并按式(9)进行语义解码，得到 $\widehat{\boldsymbol x}$。 |
| 31 | &emsp;**Return:** $\widehat{\boldsymbol x}$。 |
