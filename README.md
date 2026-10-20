# symplectic-stiefel-nearness

最近实辛 Stiefel 矩阵问题的批量对偶 Newton–CG 求解器，使用 PyTorch 和 CuPy 启动的 CUDA 核进行 FP64 计算。

本仓库目前是私有研究代码，包含当前求解器和最小实验入口。文献对照方法、论文引用和开源许可将在后续补充；当前不包含完整的论文对比实验，也未提供开源许可证。

## 数学问题

给定 \(T\in\mathbb{R}^{2n\times 2k}\)，其中 \(1\leq k\leq n\)，求解

\[
\min_{X\in\mathbb{R}^{2n\times 2k}}
\frac{1}{2}\lVert X-T\rVert_F^2
\quad\text{满足}\quad
X^\top J_{2n}X=J_{2k},
\qquad
J_{2r}=
\begin{bmatrix}
0&I_r\\
-I_r&0
\end{bmatrix}.
\]

可行集是实辛 Stiefel 流形；当 \(k=n\) 时，问题退化为最近实辛方阵问题。一个批次中的各个目标矩阵独立求解。

当前方法在小矩阵上进行对偶迭代，再分块恢复原坐标中的 \(X\)。对偶变量 \(\omega\) 为实反对称矩阵，当前实现以对偶解位于 \(I+i\omega\succ0\) 的内部为前提。实现使用等价的实正定矩阵 \(K=I+\omega^2\)，并通过其 Cholesky 分解处理线搜索和恢复。

## 环境与安装

当前计算后端仅支持 NVIDIA CUDA，使用 FP64。虽然公开接口接收和返回 CPU 张量，求解过程仍需要 GPU；没有纯 CPU 求解路径，也不提供 autograd 反向传播。

已验证的本地环境：

| 项目 | 版本或配置 |
| --- | --- |
| 系统 | Windows |
| Python | 3.11.14 |
| NumPy | 2.0.1 |
| PyTorch | 2.14.1+cu126 |
| PyTorch CUDA | 12.6 |
| CuPy 分发包 | cupy-cuda12x 13.6.0 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB |
| NVIDIA 驱动 | 592.82 |

本项目的 CuPy 依赖面向 CUDA 12.x；运行时还需要可访问的 NVRTC 和 CUDA 头文件。请先准备兼容的 CUDA 12.x 工具链；自定义安装位置可通过 `CUDA_PATH` 指定，参见 [CuPy 安装说明](https://docs.cupy.dev/en/v13.6.0/install.html)。上表记录的是 PyTorch 的 CUDA 版本，不代表所有本地 CUDA 组件的版本。环境中只安装一个 CuPy 分发包。

在独立的 Python 环境中，先安装与驱动兼容的 CUDA 版 PyTorch，再从仓库根目录安装本项目。例如，使用 CUDA 12.6 版 PyTorch：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

上面的安装示例不固定全部依赖版本。复现实验时应记录实际安装版本，或使用与上表一致的环境。首次使用 CUDA 核时会发生编译或缓存加载，正式性能测量应将这些开销与预热分开。

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

输入 `T` 应为有限值的 CPU `torch.float64` 张量，形状为 `[batch_size, 2*n, 2*k]`，其中批量大小为正，且 `1 <= k <= n`。行顺序与上述标准辛矩阵一致：前 `n` 行和后 `n` 行分别组成两部分。

返回值均在 CPU：

| 返回值 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `X` | 与 `T` 相同 | `torch.float64` | 恢复得到的辛约束候选解 |
| `omega` | `[batch_size, 2*k, 2*k]` | `torch.float64` | 原坐标中的实反对称对偶变量 |
| `iterations` | `[batch_size]` | `torch.int64` | 每题实际接受更新的 Newton 迭代数 |

尺寸和类型满足接口要求并不保证任意目标矩阵都能成功求解。达到迭代上限或线搜索失败时，当前实现抛出异常；调用方应记录失败，不能只统计成功样本。

## 最小实验入口

从仓库根目录运行：

```bash
python benchmarks/run_benchmark.py --preset smoke
python benchmarks/run_benchmark.py --preset paper-k32
python benchmarks/run_benchmark.py --preset paper-k64
python benchmarks/run_benchmark.py --preset paper-n8192
```

`smoke` 用于小规模运行检查；其余预设为后续论文规模实验提供固定配置，名称不表示已经完成论文对照或复现。

| 预设 | B | n | k |
| --- | ---: | ---: | ---: |
| `smoke` | 2 | 128 | 8 |
| `paper-k32` | 64 | 4096 | 32 |
| `paper-k64` | 64 | 4096 | 64 |
| `paper-n8192` | 64 | 8192 | 64 |

输入统一为 CPU 上的 `float64 randn(B, 2*n, 2*k) / sqrt(2*n)`，默认种子为 `20261019`。入口先单独运行一次以完成延迟编译，再预热 6 轮、测量 12 轮。GPU 实验应串行运行。

可使用以下参数指定规模和采样设置：

```bash
python benchmarks/run_benchmark.py --preset smoke --batch-size 2 --n 128 --k 4 --seed 20261019 --warmup 2 --repeats 5
```

其中 `--n` 和 `--k` 是半维度，实际输入形状为 `[batch_size, 2*n, 2*k]`。入口默认向标准输出写入 JSON，便于保留参数、环境、验证指标和原始时间样本。

时间口径为 **CPU 输入到 CPU 输出的端到端墙钟时间**：输入已生成，测量包含分块 Gram 构造及传输、对偶求解、解恢复与回传，并在计时边界同步 GPU。输入生成、首次编译、预热和结果验证不属于正式求解时间。该入口不提供完整的 CUPTI 追踪或 FLOP 审计流程。

## 正确性与数值边界

除目标值 \(\frac12\lVert X-T\rVert_F^2\) 外，应逐题检查：

- 辛约束残差：\(\lVert X^\top J_{2n}X-J_{2k}\rVert_F/\sqrt{2k}\)。
- 驻点残差：\(\lVert X-T-J_{2n}X\omega\rVert_F/\max(\lVert T\rVert_F,1)\)。

当前实验使用的验收阈值分别为 \(10^{-8}\) 和 \(10^{-9}\)。这些是验证标准，不是对所有输入的误差保证。

安装后运行 GPU 回归检查：

```bash
python -m unittest discover -s tests -v
```

检查覆盖已知解、初始已收敛、混合活动掩码、非连续输入和尾尺寸，以及 64 阶库求逆回退与 128 阶专用求逆路径。

接近对偶正定域边界时，形成 \(I+\omega^2\)、使用 Gram 矩阵和恢复原始解都可能放大舍入误差。即使内部迭代达到停止条件，恢复后的残差仍可能超过验收阈值。应保留这类结果并报告失败原因，不能通过放宽阈值或删除困难算例使其变为通过。

当前不声称支持所有秩亏输入或边界解，也不将一次数值收敛视为覆盖任意目标矩阵的全局最优保证。

## 仓库组织与后续实验

```text
src/symplectic_projection/  求解器及其 CUDA 编译、启动与融合核实现
benchmarks/run_benchmark.py 最小验证和端到端计时入口
tests/                     接口与正确性检查
```

求解器运行部分由 9 个核心 Python/CUDA 文件组成；安装时需要保留其中的 CUDA 源文件。

后续对照实验将使用统一的输入生成方式、随机种子、目标值与残差定义、成功判据和时间边界。新增方法时需要明确其支持的矩阵形状、计算精度、硬件和初始化方式，并保留逐题失败记录。文献方法实现、完整比较结果、论文引用及许可信息尚待补充。
