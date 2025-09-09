# ILP-Algorithm-for-Graph-Colouring-Problem

## 论文意见
太好了，这次我把你要“查准页码/章节 + 能直接拿来用的思路与参数”的那一套整理成一份可执行的“读法 + 做法”清单。分两块：

1. 先把你导师邮件里的“LP↔ILP交替迭代并逐步固定变量”的核心，配上**至少 5 种**可直接嵌入的“LP→ILP 转换/变量固定/邻域搜索/舍入”技术，并标出原论文**具体章节或页码**；
2. 再把**POP/POP2**模型在原文里该看哪儿、它们的逻辑、强弱点与证明线索标出来，便于你在毕业设计里把POP2作为“主模型”，上面挂载这些迭代技巧。

---

# 一、按“LP→ILP→变量固定→再解LP”的迭代框架，给你的“可插拔模块 + 查读指引”

> 你可以把下面每一种方法理解为：
> **(A)** 从LP解或当前ILP解中拿到信息 → **(B)** 决定“固定哪些变量/收缩哪些邻域/加哪些临时约束” → **(C)** 解一个更小的LP 或 子MIP；
> 用完后“释放/替换”这些临时约束，进入下一轮。

---

## 0) 主模型与总入口：POP/POP2（作为你的基线ILP）

* **看哪里：**

  * 经典赋值模型（ASS）与弱点（对称性、LP松弛值=2）：**Sec. 2.2**；POP 模型：**Sec. 3.1**（指出LP松弛有全局 1.5 的可行值）；POP2 混合模型：**Sec. 3.2**。
* **你要记住：**

  * **POP**把“颜色顺序”建成偏序，天然削弱颜色置换对称性；但密图时系数矩阵更稠。**POP2**在POP的骨架上，针对“密/稀图”做了混合，实践中在**稀疏图**上非常强。
* **实施建议：**

  * 以**POP2**为主模型；初始上界 H 用“最大团大小/启发式着色”（例如 DSATUR）给出；先跑一轮 LP 松弛，进入下面的迭代模块。POP2 的小技巧（预染色 clique、删去“明显跟邻居同色”的顶点、用更紧的 H）都写在 **Sec. 3.2** 末尾的实现提示里。

---

## 1) **Reduced-Cost（/Dual）安全固定（Safe Fixing）** —— “用LP对偶信息，**证明**某些变量在最优解不可能取1”

* **看哪里：**

  * 教学笔记给出清楚的**安全固定公式**（极小化时：若**LP下界** + **变量的正化降成本** ≥ **当前最好上界** → 可把该变量固为0；对取1同理）：T. Ralphs 讲义，**“Reduced Cost”小节**；以及“Hybrid MIP/CP 教程”里**LP-based domain filtering / reduced-cost fixing**页。([COR@L][1], [John Hooker][2])
  * 综述与系统实现也说明“每个结点都做 reduced cost fixing”：Nemhauser–Savelsbergh 的 MINTO 文档（系统函数部分）；以及“分支定界综述”提到“更好上界→更多 reduced cost 固定”。([ISYE Georgia Tech][3], [arXiv][4])
* **怎么用在着色ILP（如POP2/ASS）：**

  * 对二元变量（如“v是否用色i”），在 LP 最优单纯形解处拿到**reduced cost**：

    * 最小化时：若 `LP值 + rc(x_vi=1) ≥ 当前可行着色色数(UB)` → 固定 `x_vi=0`（安全）。
    * 类似地对 `x_vi=0` 的情况，用“相对弛度/替代成本”判定是否可强制为1（或通过对偶影子价推界）。
* **参数与阈值：**

  * 浮点数要留**容差ε**（如 `1e-6~1e-4`）；每轮固定的**最大变量数**设上线（避免过度收缩）；当 UB 改善时**再触发一次** reduced-cost 扫描。
* **为什么“安全”**：这类固定带有**可证明的正确性**（不剔除全局最优）。综述/讲义明确了推理（用 LP 下界 + 降成本界）。([COR@L][1], [John Hooker][2], [ISYE Georgia Tech][3], [arXiv][4])

---

## 2) **RINS（Relaxation Induced Neighborhood Search）** —— “LP解与可行ILP解**一致的位置全部固定**，只在**分歧集合**里搜索”

* **看哪里：**

  * RINS 原文：**Math. Programming 102:71–90 (2005)**；摘要页清楚描述了“用 LP 松弛 + 当前可行解构造邻域”的核心思想。([春erLink][5])
* **怎么用：**

  * 取一份**可行着色**（来自启发式/上一轮舍入），和最新 LP 解 `x*`；对所有变量若 `x*` 与可行解同值→**固定**；其余变量组成**子MIP**（或在 POP2 上加临时约束）求改进。
* **参数：**

  * **邻域大小**＝“不固定的变量数/允许偏离的顶点数”；可通过**最大允许 Hamming 距离**或**最多解开K个顶点**来控。
  * **时间限**对每个邻域（几十秒\~数分钟）+ **多邻域重启次数**。
* **好处：**

  * 很契合“LP→ILP→固定→再解”的循环；对图着色这类 0/1 模型效果稳定。([春erLink][5])

---

## 3) **RENS（Relaxation Enforced Neighborhood Search）** —— “**以LP最优解为中心**，枚举**所有可行舍入**的子MIP”

* **看哪里：**

  * Berthold 的技术报告与期刊稿把 RENS 定义得很清楚（“构造**包含所有可行舍入**的子问题”）：见 ZIB 技术报告与后续论文摘要/说明。([Opus4][6], [ResearchGate][7], [MPC][8])
* **怎么用：**

  * 从 LP 解 `x*` 出发：

    * **把已整数的变量**全固定；
    * **对其余整数变量**加界（如 `x_j ∈ {⌊x*_j⌋, ⌈x*_j⌉}` → 对 0/1 即 `{0,1}`，也可用“就近取整半径/只允许少数翻转”的形式），形成**子MIP**求解；得到一个很强的可行解或改进上界。
* **参数：**

  * **RENS 半径**（允许偏离 `x*` 的变量个数/权重）；**时间限**；可重复用不同 `x*`（比如多解/多参考的 **MRENS** 扩展）。([优化在线][9], [arXiv][10])

---

## 4) **Feasibility Pump（FP）** —— “LP点↔最近整点的**来回投影**，快速找**可行ILP解**”

* **看哪里：**

  * 原论文：**Sec. 2（0–1 MIP 的 FP 算法与伪码）**、**Fig.1**（算法框图）、**关于循环/扰动**与后处理一节；表格里有**时间 & 质量**对比。
* **怎么用在着色：**

  * 目标是**尽快拿到一个可行着色**（上界）。每轮：

    1. 解 LP 得 `x*`；
    2. **四舍五入**得到 `x^I`（必要时加**少量翻转T**来跳出循环）；
    3. 固定 `x^I` 的整数部分，解带固定的 LP/子MIP 校正；
    4. 用 `x^I` 更新 UB，然后触发 **Reduced-Cost 固定**（模块1）。
* **参数：**

  * 每轮**翻转数T**、**最大轮数**、停机阈值；翻转可按“最分数/冲突边最多”的变量选。

---

## 5) **Local Branching（LB）** —— “围绕当前可行解的**Hamming球**，用一条割控制邻域大小”

* **看哪里：**

  * 论文里给出**局部分支约束**（Hamming 距离不超过 k）与流程；实验里还写了**典型 k 值（例如 18/20）**。看 **约束(7)** 与图示、以及“参数 k 可自适应”的讨论。
* **怎么用在着色：**

  * 取当前着色 `x̄`，加

    $$
    \sum_{j: \bar x_j=1}(1-x_j)+\sum_{j:\bar x_j=0}x_j \le k
    $$

    解带此约束的 POP2（或 ASS）几分钟；若改善，就**反向分支**或**扩大/缩小 k**继续；可和 RINS/RENS 交替。
* **参数：**

  * 初始 `k≈10~30`；若无改进，先**软多样化**（换邻域/换起点），再**强多样化**（加 Tabu 约束 `Δ(x, x̄) ≥ 1` 逃离）。细节与策略在文中**Sec. 3–5**有图示与伪码。

---

## 6) **Pipage / Swap Rounding（依赖型舍入）** —— “带结构约束的**可行性保持**舍入”

> 更偏理论，但可用在“**分块变量**”“**预算/基数**约束”场景，避免把 LP 舍入得太乱。

* **看哪里：**

  * **Pipage**：**Sec. 2 “Pipage rounding: A general description”**（给出通用算法），以及示例章节；证明了“沿某个方向函数凸性保证不降质”。([cs.toronto.edu][11])
  * **Swap rounding**：**Section IV / V**（对 matroid 基多面体/交的算法步骤与**Chernoff 类尾界**）；前文还有**算法框（Randomized Swap Rounding Scheme）**。
* **怎么用在着色：**

  * 在 ASS/POP2 的“每色一组”的结构上，可把每个颜色层的“顶点选取”看作**分区/基数**约束的子结构，对**局部的 LP 分量**用 pipage/swap rounding 产出更加“可修复”的 0/1 方案，再用**冲突修复/RENS/LB**清理边冲突。

---

## 7) **Column Generation / Branch-and-Price（独立集（稳定集）模型）** —— “拿到**很强的LP下界**”

* **看哪里：**

  * **Mehrotra–Trick (1996)**：把着色写成**最少稳定集覆盖**，主问题是**集合覆盖 LP**，子问题是**最大加权稳定集**定价；全文详细讲价格、分支。免费稿件（CMU）：建议通读“模型与定价”部分。
  * 另一份小综述也概括了该思路：**Set covering/packing formulations** 说明“变量对应极大稳定集，需列生成”。([优化在线][12])
* **在你的迭代框架里的用法：**

  * 不必自己实现完整 B\&P；但可**用它的 LP 值作强下界**（或用别人跑好的实例数据的界）→ **增强 Reduced-Cost 固定**的判据（更强的 LB ⇒ 固定更多变量）。

---

## 8) **冲突图/探测（Probing）与预处理** —— “把明显互斥的变量提早剪掉”

* **看哪里：**

  * 冲突图与探测在 MIP 预处理中很常见；MINTO 文档与后续综述都写了**节点预处理、探测、Reduced-Price 固定**等机制；“Presolve/Proof-Logging/PaPILO”也把**dual fixing**当作 reduced-cost 固定的推广。([ISYE Georgia Tech][3], [Opus4][13], [arXiv][14])
* **怎么用：**

  * 在 ASS/POP2 层面：对“同色-相邻边”引起的显式互斥做**等价类收缩**、**强制 0**；与**clique 预染色**（POP2 里有提）一起能显著减少变量。

---

# 二、POP/POP2 —— 原文到底怎么讲、你复现/引用时该抓哪些点？

* **ASS（2.2）弱点与对称性处理：**

  * 先给标准 ASS（式(1)–(4)），然后说**颜色置换对称**导致指数多等价可行解；给出 **Méndez-Díaz & Zabala** 的**对称破坏**（例如 `w_i ≤ w_{i-1}` 与 `w_i ≤ Σ_v x_{vi}`），这两条就是你在代码里“开箱即用”的**对称破坏线性约束**。
* **POP（3.1）思想与 LP 弱点：**

  * 把“顶点-颜色”投影到“偏序层级”变量上，**天然减少对称性**；但论文直说它的 LP 松弛**有 1.5 的通用可行值**（弱下界），你要靠第1节那些“迭代固定/邻域”手段来补。
* **POP2（3.2）混合：**

  * 观察到 POP 在**稠密图**里约束矩阵更稠，因而提出 POP2 混合（把 POP 的“顺序骨架”与 ASS 的“稀疏边约束”结合），并在实现建议里强调：
    **(a)** 先找**大团**并**预染色**（直接减少变量/约束）；
    **(b)** 尽量给一个**小的 H 上界**；
    **(c)** 对“度小于预染色代表邻域”的顶点**延后处理/等价回填**。这些都能直接写进你的预处理。
* **和其它模型的横评：**

  * 文中也对比了代表元模型（representatives，适合**密图**），并给出“ASS/POP/REP 各在哪些密度段占优”的经验——你在论文撰写时可据此解释为何选 POP2 做主干。

---

# 三、把这些方法“串成”你毕业设计里**LP↔ILP 交替迭代**的具体流程（建议稿）

**Step 0：** 预处理

* DSATUR 给初解（UB），最大团作 LB/预染色，收缩显然冲突，设置 H。([cse.unl.edu][15], [优化在线][16])

**Step 1：** 解 POP2 的 **LP 松弛** → 得 `x*`

* 记录对偶/化降成本、分数部分、稀疏结构。

**Step 2（可行解获取）**：

* 用 **Feasibility Pump**（或简化版“最近整点 + 少量翻转T”）得到**任一可行着色** `x^I`，更新 **UB**。

**Step 3（安全固定）**：

* 用 **Reduced-Cost 固定**（基于 `LB=LP值` 与 `UB`）把“必为0/1”的变量**严格固定**；留容差 ε。([COR@L][1], [John Hooker][2])

**Step 4（改进邻域）**：

* 交替使用 **RINS**（固定 `x*` 和 `x^I` 一致的位置）与 **Local Branching**（以 `x^I` 为中心、`k≈10–30` 的 Hamming 球）跑**子MIP**若干分钟；若解出了更好解，就更新 **UB** 并回到 Step 3；若没改进，用 **RENS**（以 `x*` 为参考）再求一次。([春erLink][5], [Opus4][6])

**Step 5（收缩LP再解）**：

* 把所有**已固定**变量加到模型中，**重解 LP** 得新的 `x*`；回到 Step 2。
* 直到变量全定或**没有新的安全固定/改进**为止；输出最好的 `x^I`。

> 上面的每个“子问题”都可设**短时间限**（例如 LB/RINS/RENS 2–5 分钟），保证外层能多轮循环。

---

# 四、你问的“文献页码/章节定位”速查表（按你最常用的段落）

* **Jabrayilov & Mutzel (2017)**《New ILP Models for Vertex Coloring》

  * **Sec. 2.2** 赋值模型 ASS + 对称性削弱的两条线性约束 `w_i≤Σ_v x_vi` 与 `w_i ≤ w_{i-1}`；**LP=2 的弱下界**也在此节说明。
  * **Sec. 3.1** POP 模型 + **LP=1.5** 的可行构造（弱点说明）。
  * **Sec. 3.2** POP2 模型 + 实施建议（预染色 clique、减 H、删某些顶点再回补）。

* **Local Branching**（Fischetti & Lodi）

  * 局部分支约束（Hamming 球）与**k 的取值**、**反向分支 / 多样化策略**、算法流程图见**Sec. 3–5**（文中示例常用 `k=18~20`）。

* **Feasibility Pump**（Fischetti–Glover–Lodi）

  * **Sec. 2**：0–1 MIP 的 FP 伪码与最近整点的度量；**循环与扰动**、**后处理**（在第 3 节中段）以及实例表格对比。

* **RINS**（Danna–Rothberg–Le Pape）

  * 文章摘要与引言已准确概述“**LP vs ILP 一致处固定**→ 构造邻域”的主旨；正文细节在方法部分（付费墙下，但摘要页足以引用方法定义）。([春erLink][5])

* **RENS / MRENS**（Berthold；ZIB）

  * 技术报告/论文摘要明确“**所有可行舍入**的子MIP”构造；**MRENS**（2024）给“多参考解”的泛化与在 SCIP 的实现评测。([Opus4][6], [优化在线][9])

* **Pipage / Swap Rounding**

  * **Pipage**：看 **Sec. 2** 的“general description”；
  * **Swap rounding**：看 **Section IV / V**（算法细节与在 matroid / matroid intersection 的推广，含 Chernoff 类尾界）。([cs.toronto.edu][11])

* **Column Generation（独立集覆盖）**

  * Mehrotra–Trick 的 CMU 公开稿：模型、定价与分支结构（整篇都围绕“列生成 + 分支”）。另有一篇短文概述该思路。([优化在线][12])

* **Reduced-Cost 固定（安全固定）入门**

  * Ralphs 讲义 **Lecture 22** 的“Reduced Cost”小节；Hooker 教程里的“LP-based domain filtering / reduced cost fixing”页。([COR@L][1], [John Hooker][2])

* **图着色基础/启发式（DSATUR）**

  * Brélaz 原文（1979）与后续 DSATUR 改进（Furini 等），可作**初始解与团/色数估计**工具。([cse.unl.edu][15], [优化在线][16])

---

# 五、这篇 POP/POP2 能否作为你“LP↔ILP 迭代”框架的**参考基线**？

**可以，而且很合适**：

* 它给了一个**简单、可直接丢进通用 MIP 求解器**的模型（POP/POP2），避免了你在“框架层”还要写复杂的 B\&P；同时明确了**LP 松弛弱**这个事实——恰好为你引入 **Reduced-Cost 固定、RINS/RENS、Local Branching、FP** 等“外部循环”创造空间。
* 你的毕业设计可以定位为：**“在 POP2 上，系统性组合与对比多种 LP→ILP 转换/固定策略”**，并在 **DIMACS** 基准集或 **COLOR02/03** 子集上做实证（POP2 文中也用了这些基准）。

---

## 快速落地清单（你可以直接开工）

1. **实现 POP2**（按 Sec.3.2）+ **ASS 对称破坏**（Sec.2.2附加约束）。
2. **数据集**：用 **DIMACS coloring** 实例做正确性 & 消融实验（同论文）。
3. **外层循环**按“Step 0–5”跑起来，依次接入：

   * FP（拿初可行/上界）；
   * Reduced-Cost 固定（每次 UB 改善后触发一次全局扫描）；([COR@L][1])
   * RINS / LB（交替、小时间限邻域搜索）；([春erLink][5])
   * RENS（以 LP 解为中心的“可行舍入子MIP”）；([Opus4][6])
4. **可选增强**：用公开实现（ColPack/SciCol）给**启发式上界**；用 Mehrotra–Trick 的**下界**（若有）验证“安全固定”的正确性界。([优化在线][12])

如果你愿意，我也可以把上述流程的**伪代码**和\*\*参数表（k、T、时间限、容差ε）\*\*整理成一页清单，或者给你一份最小可运行的 **Python + HiGHS/Gurobi** 原型骨架，直接在 DIMACS 小图上跑通。

[1]: https://coral.ise.lehigh.edu/~ted/files/ie418/lectures/Lecture22.pdf?utm_source=chatgpt.com "Integer Programming ISE 418 Lecture 22"
[2]: https://johnhooker.tepper.cmu.edu/tutorialToulouse.pdf?utm_source=chatgpt.com "Tutorial: Hybrid mixed integer programming and constraint ..."
[3]: https://www2.isye.gatech.edu/people/faculty/Martin_Savelsbergh/software/orl15.pdf?utm_source=chatgpt.com "MINTO, a Mixed INTeger Optimizer"
[4]: https://arxiv.org/pdf/2111.06257?utm_source=chatgpt.com "Branch and Bound in Mixed Integer Linear Programming ..."
[5]: https://link.springer.com/article/10.1007/s10107-004-0518-7 "Exploring relaxation induced neighborhoods to improve MIP solutions | Mathematical Programming
        "
[6]: https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/426?utm_source=chatgpt.com "RENS - Relaxation Enforced Neighborhood Search"
[7]: https://www.researchgate.net/publication/228715008_RENS-relaxation_enforced_neighborhood_search?utm_source=chatgpt.com "(PDF) RENS-relaxation enforced neighborhood search"
[8]: https://mpc.zib.de/archive/2014/1/Berthold2014_Article_RENS.pdf?utm_source=chatgpt.com "The optimal rounding"
[9]: https://optimization-online.org/wp-content/uploads/2024/08/MRENS_in_SCIP.pdf?utm_source=chatgpt.com "A Multi-Reference Relaxation Enforced Neighborhood ..."
[10]: https://arxiv.org/html/2408.00718v1?utm_source=chatgpt.com "A Multi-Reference Relaxation Enforced Neighborhood ..."
[11]: https://www.cs.toronto.edu/tss/files/papers/Ageev-Sviridenko2004_Article_PipageRoundingANewMethodOfCons.pdf "Pipage Rounding: A New Method of Constructing Algorithms with Proven Performance Guarantee"
[12]: https://optimization-online.org/wp-content/uploads/2005/12/1257.pdf?utm_source=chatgpt.com "Set covering and packing formulations of graph coloring"
[13]: https://opus4.kobv.de/opus4-zib/files/6037/Presolve.pdf?utm_source=chatgpt.com "Presolve Reductions in Mixed Integer Programming"
[14]: https://arxiv.org/pdf/2401.09277?utm_source=chatgpt.com "arXiv:2401.09277v2 [math.OC] 20 Mar 2024"
[15]: https://cse.unl.edu/~choueiry/Documents/brelaz.pdf?utm_source=chatgpt.com "New Method to Color the Vertices of a Graph"
[16]: https://optimization-online.org/wp-content/uploads/2015/10/5159.pdf?utm_source=chatgpt.com "An improved DSATUR-based Branch and Bound for the ..."
