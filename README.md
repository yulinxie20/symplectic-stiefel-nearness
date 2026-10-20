# symplectic-stiefel-nearness

最近实辛 Stiefel 矩阵问题的 FP64 求解与比较代码。仓库包含原生批量 GPU 对偶 Newton–CG 求解器，以及采用欧氏诱导度量的黎曼梯度和信赖域基线。

代码采用 [MIT 许可证](LICENSE)。论文实验版本为 [`paper-experiments-20261020`](https://github.com/yulinxie20/symplectic-stiefel-nearness/tree/paper-experiments-20261020)，原始记录和汇总位于 [paper_results/2026-10-20/](paper_results/2026-10-20/)。软件引用见 [CITATION.cff](CITATION.cff)。

## 数学问题与约定

给定 \(T\in\mathbb{R}^{2n\times 2k}\)，其中 \(1\leq k\leq n\)，求解

\[
\min_{X\in\mathbb{R}^{2n\times 2k}}
\lVert X-T\rVert_F^2
\quad\text{满足}\quad
X^\top J_{2n}X=J_{2k},
\qquad
J_{2r}=
\begin{bmatrix}
0&I_r\\
-I_r&0
\end{bmatrix}.
\]

目标统一不含 \(1/2\)，环境梯度为 \(2(X-T)\)。可行集是实辛 Stiefel 流形；\(k=n\) 时为辛群。批次中的各个目标相互独立。

对偶变量 \(\omega\) 为实反对称矩阵，驻点方程采用 \(X-T-J_{2n}X\omega=0\)。对偶实现以 \(I+i\omega\succ0\) 的内部迭代和恢复为基础，实际使用实正定矩阵 \(I+\omega^2\) 的 Cholesky 分解。理论上的最优值达到或强对偶不保证这一内部实现能处理所有边界最优解。

## 方法与文献归属

| 实验名称 | 实现 |
| --- | --- |
| `dual` | 小矩阵上的批量对偶 Newton–CG，随后分块恢复原坐标解 |
| `rgd` | 欧氏黎曼梯度，交替 Barzilai–Borwein 试探步长（ABB）及**单调 Armijo** 回溯 |
| `rtr` | 欧氏黎曼信赖域，使用精确黎曼 Hessian 的作用，以截断共轭梯度（tCG）近似求解切空间子问题 |

两个黎曼基线使用相同的欧氏诱导度量、由目标的复极分解构造的正交辛初值，以及同一 Cayley 回缩。几何投影采用 Gao 等的欧氏度量公式；Cayley 回缩源自 Gao 等，按 Bendokat–Zimmermann 命题 5.2 的 \(2k\) 阶等价式计算，只需求解 \(2k\) 阶线性系统。信赖域框架参考 Absil–Baker–Gallivan。

这里的基线是上述方法的明确实现。`rgd` 的单调回溯不同于 Gao 等原算法的非单调线搜索；`rtr` 使用普通 Cayley 回缩，不假定其具有欧氏二阶回缩性质，也不据此主张原文算法的完整复现或局部收敛阶。基线保留设备标量同步，尚未采用 CUDA Graph 或融合核优化。

原始文献：

- Bin Gao, Nguyen Thanh Son, P.-A. Absil, Tatjana Stykel. *Riemannian Optimization on the Symplectic Stiefel Manifold*. SIAM Journal on Optimization 31(2), 1546–1575, 2021. [DOI: 10.1137/20M1348522](https://doi.org/10.1137/20M1348522).
- Bin Gao, Nguyen Thanh Son, P.-A. Absil, Tatjana Stykel. *Geometry of the Symplectic Stiefel Manifold Endowed with the Euclidean Metric*. Geometric Science of Information, LNCS 12829, 789–796, 2021. [DOI: 10.1007/978-3-030-80209-7_85](https://doi.org/10.1007/978-3-030-80209-7_85).
- Thomas Bendokat, Ralf Zimmermann. *The Real Symplectic Stiefel and Grassmann Manifolds: Metrics, Geodesics and Applications*. arXiv:2108.12447, 2021. [原文](https://arxiv.org/abs/2108.12447).
- P.-A. Absil, C. G. Baker, K. A. Gallivan. *Trust-region Methods on Riemannian Manifolds*. Foundations of Computational Mathematics 7(3), 303–330, 2007. [DOI: 10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9).

## 环境与安装

对偶求解器和论文比较入口需要 NVIDIA CUDA，统一使用 FP64。两个黎曼基线本身也支持 CPU 张量，但论文入口将它们放在同一 GPU 上运行。代码不提供 autograd 反向传播。

要求 Python 3.11 或更新版本、CUDA 版 PyTorch 和 `cupy-cuda12x==13.6.0`；完整包依赖见 [pyproject.toml](pyproject.toml)。先安装与驱动兼容的 CUDA 版 PyTorch，再从仓库根目录安装，例如：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

CuPy 还需要可访问的 NVRTC 和 CUDA 头文件；自定义工具链位置可通过 `CUDA_PATH` 指定，参见 [CuPy 安装说明](https://docs.cupy.dev/en/v13.6.0/install.html)。只安装一个 CuPy 分发包。上述命令未固定所有依赖版本；正式实验的 `config.json` 会记录实际 Python 解释器、Python、NumPy、PyTorch、CuPy、CUDA 和驱动版本、硬件、提交及源码哈希，应随结果一并保留。

## 求解接口

```python
import torch
from symplectic_projection import solve

batch_size, n, k = 2, 128, 4
generator = torch.Generator().manual_seed(20261019)
T = torch.randn(
    batch_size, 2 * n, 2 * k,
    dtype=torch.float64,
    device="cpu",
    generator=generator,
) / (2 * n) ** 0.5

X, omega, iterations = solve(T)
```

对偶接口接收有限值的 CPU `torch.float64` 张量，形状为 `[batch_size, 2*n, 2*k]`，批量大小为正。行顺序对应上述标准辛矩阵。返回值均在 CPU：

| 返回值 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `X` | 与 `T` 相同 | `torch.float64` | 恢复得到的辛约束候选解 |
| `omega` | `[batch_size, 2*k, 2*k]` | `torch.float64` | 原坐标中的实反对称对偶变量 |
| `iterations` | `[batch_size]` | `torch.int64` | 每题实际接受更新的 Newton 迭代数 |

黎曼接口处理单个目标矩阵；可传入 CPU 或 CUDA 张量，输出留在相同设备：

```python
from symplectic_projection.baselines import SolverOptions, solve_riemannian

options = SolverOptions(max_iter=1000, grad_tol=1e-9, feas_tol=1e-8)
result = solve_riemannian(T[0].to("cuda"), options, method="abb")  # 或 "rtr"
print(result.status, result.iterations)
```

尺寸和类型满足接口要求不保证求解成功。对偶求解器可能抛出异常；黎曼接口通过 `status` 保留迭代上限、超时、线搜索失败或数值失败等状态。必须检查状态和输出质量。

## 论文实验复现

从仓库根目录**串行**执行以下命令。`--seeds` 指定不同目标；同一目标上的重复计时不算新的随机样本。

```bash
python benchmarks/run_paper.py --suite scaling --seeds 20261021 20261022 20261023 --repeats 6 --warmup 3 --output results/reproduced/scaling
python benchmarks/run_paper.py --suite critical --seeds 20261021 20261022 20261023 --repeats 3 --warmup 2 --output results/reproduced/critical
python benchmarks/run_paper.py --suite boundary --seeds 20261021 --repeats 1 --warmup 0 --output results/reproduced/boundary
python benchmarks/run_paper.py --suite batch --seeds 20261021 --repeats 6 --warmup 3 --output results/reproduced/batch
python benchmarks/summarize_paper.py --source-root results/reproduced --output-root results/reproduced_summary
```

| 组别 | 算例 | 种子数 × 正式重复数 × 预热数 |
| --- | --- | --- |
| `scaling` | 高斯目标，\((n,k)=(256,8),(1024,8),(4096,8),(256,32),(1024,32),(4096,32),(4096,64)\) | 3 × 6 × 3 |
| `critical` | \(n=2k-1\)，\(k=4,8,16\) | 3 × 3 × 2 |
| `boundary` | 已知最优解的近边界族、零目标及 \(n=k=2\) 的严格对偶间隙构造 | 1 × 1 × 0 |
| `batch` | \(n=1024,k=32\)，批量大小 8、32 | 1 × 6 × 3 |

高斯目标在 CPU 上生成，为 `randn(B, 2*n, 2*k) / sqrt(2*n)`。近边界族采用 \(\mu=0.5,10^{-2},10^{-4},10^{-6}\)。完整构造见 [benchmarks/paper_cases.py](benchmarks/paper_cases.py)。三个方法共享同一目标及其哈希；两个黎曼方法共享初始化规则。脚本轮换正式测量中的方法顺序，并分别保存：

- `config.json`：配置、环境、时间定义、阈值及源码哈希。
- `first_call.json`：独立首次调用，用于触发延迟编译。
- `raw.jsonl`：每次预热和正式调用的时间、迭代数、状态、逐题后验指标及异常。

已有 `raw.jsonl` 的目录会被拒绝覆盖；重新运行时另选输出目录。即使 `--warmup 0`，仍会保留独立首次调用。可用 `--methods dual rgd rtr` 选择方法、`--only` 选择算例或 `--history` 保留迭代轨迹；启用轨迹记录会改变计时负担。

只重建已发布数据的表格和图，不重新运行 GPU：

```bash
python benchmarks/summarize_paper.py --source-root paper_results/2026-10-20 --output-root results/paper_summary_reproduced
```

汇总需要 Matplotlib（本次使用 3.10.6），可按需运行 `python -m pip install matplotlib==3.10.6`。脚本在 CPU 上重建目标核对散列，检查正式及预热记录完整性，再按同一目标的时间中位数汇总。公开数据的运行环境为 Python 3.11.14、PyTorch 2.14.1+cu126、NumPy 2.0.1、CuPy 13.6.0、NVIDIA GeForce RTX 3060 Laptop GPU（6 GiB）、驱动 592.82。CPU 数值线程数为 1；PyTorch CUDA 为 12.6，CuPy 报告的 CUDA 运行时为 12.9。

实验在基线提交 `393d129` 上加入对照与驱动后运行，因此 `config.json` 的 `commit` 保留该基线；实际执行文件按 `source_sha256` 追溯。[源码清单](paper_results/2026-10-20/summary/source_manifest.json)同时记录运行时字节散列和统一 LF 行尾后的散列，以兼容 Git 的行尾规范化。未参与求解的汇总脚本不属于执行源码集合。

当前论文入口将黎曼基线的预算设为每题 1000 次外层迭代、60 秒，最多 40 次回溯、100 次内层 tCG 迭代。实际参数随 `config.json` 保存。对偶方法使用自身的停止规则，统一后验验收在计时结束后进行。

## 时间口径与数值验收

时间为 **CPU 输入到 CPU 输出的端到端同步墙钟时间**：输入已生成，计时包含初始化、数据传输、求解、解恢复和回传。输入生成、首次编译、预热、后验验证及输出释放不计入正式求解时间。

`batch` 中对偶方法原生批量求解，两个黎曼基线则在同一 GPU 上逐题调用。该比较衡量这些实现的端到端批量吞吐，不能将其直接解释为同等批量实现下的算法加速比。记录的峰值内存为 PyTorch 分配器统计，不代表包括 CuPy 在内的全部 GPU 内存。

论文入口在 CPU FP64 中重新计算 \(f(X)=\lVert X-T\rVert_F^2\)，并从输出点重新计算欧氏法向投影乘子，统一检查：

- 辛约束残差：\(\lVert X^\top J_{2n}X-J_{2k}\rVert_F/\sqrt{2k}\leq10^{-8}\)。
- 驻点残差：\(\lVert\operatorname{grad} f(X)\rVert_F/[2\max(\lVert T\rVert_F,1)]\leq10^{-9}\)。

`accepted` 同时要求正常返回及上述一阶、可行性验收。`quality_pass` 还要求全局精度检查通过：有已知最优值时比较相对目标误差；否则，在投影乘子给出的 \(I+i\omega\) 数值正定时，独立复数 Cholesky 计算原对偶目标差，使用 \(10^{-8}\) 阈值。严格间隙算例仅已知对偶值，不能把与该值的差当作原问题最优性误差，其全局精度状态保留为未知。

这些检查是浮点数值诊断，不是区间算术证书。近似可行输出的目标差可能为负；接近对偶边界时，形成 \(I+\omega^2\)、Gram 矩阵及恢复步骤均可能放大舍入误差。当前内部对偶实现不声称支持所有秩亏目标或边界解。达到迭代上限、超时、回溯失败及后验验收失败均应保留；批量调用整体中断也单独记录，不只统计成功样本。

## 已完成的实验

共保留 513 条正式方法调用和 261 条预热记录。常规与临界组包含 30 个不同随机目标，三种方法在每次正式调用中均通过验收；重复计时不算独立随机目标。批量组也全部逐题通过。单问题时间先在每个目标上对重复取中位数，再对三个目标取中位数。

| 半维度 `(n,k)` | Dual（毫秒） | RGD（毫秒） | RTR（毫秒） |
| --- | ---: | ---: | ---: |
| `(4096,64)` | 64.8 | 498.8 | 537.3 |
| `(31,16)`，临界尺寸 | 124.2 | 169.3 | 106.9 |

这两行分别说明对偶方法在本次大型矩形目标上的优势，以及其速度并非在所有尺寸上占优。完整结果、种子间变化和残差见 [summary.csv](paper_results/2026-10-20/summary.csv) 与 [尺度图](paper_results/2026-10-20/figures/scaling.pdf)。

边界组保留了四次对偶异常：已知解族 `mu=1e-4`、`mu=1e-6` 和严格间隙目标达到 Newton 上限，零目标线搜索失败。两个黎曼方法在已知解族上均通过验收，但 `mu=1e-6` 时 RGD 的相对解误差仍约为 `8.05e-5`，即使相对目标误差约为 `6.33e-15`。严格间隙目标中，RGD/RTR 的返回目标值约为 `18.15851388`；已知对偶值为 `18`，原最优值未知，故两条记录的全局精度标为未知而非失败。详见 [boundary.csv](paper_results/2026-10-20/boundary.csv)。

## 检查与仓库组织

最小运行和回归检查：

```bash
python benchmarks/run_benchmark.py --preset smoke
python -m unittest discover -s tests -v
```

`run_benchmark.py` 是对偶方法的独立冒烟与计时入口，默认预热 6 次、正式计时 12 次，输出 JSON 到标准输出；论文比较使用上面的 `run_paper.py`。前者另有 `paper-k32`、`paper-k64`、`paper-n8192` 规模预设，预设名称不代表已有论文结果。

```text
src/symplectic_projection/           对偶求解器及 CUDA 核
src/symplectic_projection/baselines/  欧氏黎曼几何、ABB 与 RTR 基线
benchmarks/run_benchmark.py          对偶方法的最小运行与计时入口
benchmarks/run_paper.py              论文比较、失败记录与批量吞吐入口
benchmarks/paper_cases.py            可复现算例及统一后验检查
benchmarks/summarize_paper.py        CPU 汇总、完整性检查与静态表图
tests/                              接口、几何、求解及算例检查
paper_results/2026-10-20/            论文实验原始记录、汇总、表图及散列清单
CITATION.cff                         软件引用信息
LICENSE                              MIT 许可证
```
