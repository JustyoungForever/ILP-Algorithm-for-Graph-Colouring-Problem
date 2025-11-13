# 术语与对象字典（Legend / Terms & Objects)

> 本文仅覆盖 **MAIN / RUN_ITERATIVE_LP_V2** 所依赖的对象与变量，全部命名与含义与论文伪代码保持一致。  
> 语言约定：`G` 为输入图；“coloring”一律指 `Dict[int,int]` 顶点→颜色映射；颜色编号从 0 开始且尽量保持连续前缀。

## 1. 全局对象与基本符号

| 名称 | 类型 | 精确定义 / 含义 | 典型取值/格式 | 产生/使用位置 |
|---|---|---|---|---|
| `G` | `networkx.Graph` | 输入无向简单图；顶点集 `V`、边集 `E`。 | 任意 | `MAIN` 载入；各模块共享 |
| `coloring` | `Dict[int,int]` | **整数着色**：顶点 `v → color`。 | 颜色域 `0..k-1` | DSATUR / R&R / LocalSearch |
| `best_coloring` | `Dict[int,int]` | 当前最好的**可行**着色；其颜色数产生 `UB`。 | 与 `UB` 同步保持紧凑 | 主过程维护 |
| `LB` | `int` | 下界（来自某个团的基数）。 | `|Q|` | `greedy_max_clique` 产出，用于停机/收紧 `K` |
| `UB` | `int` | 上界：当前**可行解**使用的颜色数。 | `|colors(best_coloring)|` | 初始化来自初解；迭代中更新 |
| `K` | `int` | LP/舍入允许使用的**颜色前缀大小**；颜色域 `0..K-1`。 | 初始 `K=UB`；可被 `ceil(zLP)` 收紧 | `IncrementalLP.lock_prefix_K` |
| `zLP` | `float` | 线性松弛目标值（`min sum y_c`），是颜色数的下界估计。 | `≥ LB` | `IncrementalLP.solve()` |
| `ceil(zLP)` | `int` | `zLP` 向上取整。 | `math.ceil(zLP)` | 决定是否能安全收紧 `K` |
| `x_frac` | `Dict[(int,int), float]` | LP 分数解的顶点-颜色分配 `x[v,c]∈[0,1]`。 | 键 `(v,c)`，值 `float` | 用于 R&R、打包、Fixing |
| `y_frac` | `Dict[int, float]` | LP 分数解的颜色启用 `y[c]∈[0,1]`。 | 键 `c`，值 `float` | 用作全局颜色优先序（可小扰动） |
| `restarts` | `int` | R&R 的多启动次数。 | 16/48/… | 命令行参数 |
| `perturb_y` | `float` | R&R 中对 `y_frac` 的**极小扰动幅度**，仅用来**打破平局**（数值近时决定先后）。 | `1e-6` | R&R 入口参数 |
| `verify_report` | `Dict[str,Any]` | 染色可行性报告：`feasible`, `num_conflicts`, … | 字典 | `verify_coloring()` 输出 |
| `BoundsToken` | `List[(Var, lb, ub)]` | **可逆边界变更令牌**：记录变量边界变更，便于回滚。 | 列表 | `IncrementalLP` 内部 |
| `FixPlan` | `Dict[str,Any]` | 固定计划：`{"y_zero":[c], "x_one":[(v,c)], "x_zero":[(v,c)]}`。 | 字典 | `pick_fixings()` 输出 |
| `logs` | `List[Dict]` | 迭代日志（每轮的 `UB, LB, zLP, K` 等）。 | 列表 | `RUN_ITERATIVE_LP_V2` 汇总 |

## 2. 重要不变量与约定

1. **颜色紧凑性**：凡作为最终/阶段性输出向外暴露的 `coloring`，均应在必要时经过 `compact_colors`，使颜色编号为连续前缀 `0..used-1`。  
2. **前缀约束**：`K` 表示可用颜色集合的**前缀**大小；所有 LP 变量/舍入操作仅在 `0..K-1` 上进行。  
3. **回滚安全**：对 LP 变量边界的修改（如 `y_c=0`、`x[v,c]=1`）必须通过 `BoundsToken` 可逆地应用；失败需完整回滚到上一个稳定点。  
4. **判停充分条件**：`ceil(zLP) ≥ UB` 且 `K ≤ UB` 被视为“理论不可再优”信号；或 `UB == LB` 达到最优。  
5. **命名一致性**：统一使用 `x_frac / y_frac` 指 LP 分数解；统一使用 `verify_report` 表述可行性检查的字典返回值。

## 3. 结果包（主流程返回对象）

`res: Dict[str, Any] = {`
- `UB: int` — 最佳可行解使用的颜色数；
- `LB: int` — 下界；
- `coloring: Dict[int,int]` — 最佳着色（已紧凑）；
- `iters: int` — 实际迭代轮数；
- `log: List[Dict]` — 逐轮日志（含 `round, zLP, UB, K` 等）；
- `stop_reason: str` — 终止原因（`"UB==LB"`, `"no_better_than_UB"`, `"stalled_at_K_eq_ceil_zLP"`, `"time_limit"`, `"max_rounds"` 等）；
- `feasible: bool` — 最终可行性；
- `final_check: Dict[str,Any]` — `verify_report` 全量字典（用于审计）。  
`}`

