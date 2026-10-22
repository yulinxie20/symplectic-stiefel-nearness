# symplectic-stiefel-nearness

最近实辛 Stiefel 矩阵问题的论文复现代码，包含 Dual、Gao 非单调 RSD 和 Jensen–Zimmermann R–TR2。统一入口生成正式实验记录、两张结果表和比较图。

给定 \(T\in\mathbb{R}^{2n\times2k}\)，求解

\[
\min_X\|X-T\|_F^2,\qquad X^\top J_{2n}X=J_{2k},\qquad
J_{2r}=\begin{bmatrix}0&I_r\\-I_r&0\end{bmatrix}.
\]

全部计算使用 FP64；目标函数不含 \(1/2\)。Dual 使用正定内部的实数对偶 Newton–CG 与原解恢复，Gao 使用欧氏诱导度量，R–TR2 使用原作者的右不变度量与回缩。

## 环境与安装

论文原始实验的软件与硬件如下；重新运行时，由首个工作进程将实际 `sys.executable`、Python、NumPy、PyTorch、CuPy、Matplotlib 版本及 CPU、GPU、CUDA、驱动信息集中记录到 `environment.json`。

| 项目 | 原始实验环境 |
| --- | --- |
| Python | 3.11.14 |
| PyTorch | 2.14.1+cu126，CUDA 12.6 |
| NumPy | 2.0.1 |
| CuPy | 13.6.0（`cupy-cuda12x`），报告 CUDA 运行时 12.9 |
| Matplotlib | 3.10.6 |
| GPU / 驱动 | NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB / 592.82 |
| CPU | Intel Core i7-12700H；数值线程数 1 |

使用 Python 3.11 环境和一张 NVIDIA CUDA GPU。从仓库根目录安装：

```bash
python -m pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

`pyproject.toml` 固定直接依赖版本。CuPy 需要可访问的 NVRTC、CUDA 头文件和动态库，参见 [CuPy 安装说明](https://docs.cupy.dev/en/v13.6.0/install.html)。未设置 `CUDA_PATH` 时，入口先尝试 `PATH` 中的 `nvcc`，再检查当前 Python 环境的 `Library` 目录及环境根目录；找到 CUDA 头文件后设置路径，已有配置保持不变。未找到时仍可由 CuPy 使用系统默认位置，或手动设置 `CUDA_PATH`。代码使用当前单张 GPU；PyTorch 与 CuPy 共用同一默认流。每个工作进程在实验前统一编译整组核函数并保存句柄，随后进行预热。

若需手动配置，Windows 下的 CUDA 组件安装在 Conda 环境的 `Library` 目录时，激活该环境后在 PowerShell 中设置：

```powershell
$env:CUDA_PATH = "$env:CONDA_PREFIX\Library"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
```

## 一次运行复现

```bash
python reproduce.py
```

默认使用全新的 `results/` 目录。可指定其他输出目录及可用逻辑 CPU：

```bash
python reproduce.py --output results_new --cpu 2
```

原机工作进程绑定逻辑 CPU 2，即已核对的性能核；其他机器须根据实际拓扑选择 `--cpu`，相同编号不代表相同核类型。每个工作进程启动时绑定一次单核，检查系统调用成功后导入数值运行时。共 18 个独立工作进程，按固定顺序串行使用同一 GPU。

正式协议固定为：

- 六种尺寸：\(n=2048,4096\) 与 \(k=16,32,48\) 的全部组合；每种尺寸固定生成 64 个不同的高斯目标，元素服从 \(N(0,1/(2n))\)。三个单题方法及两条批量路径共用这些目标。
- 完成三轮，每轮按预定顺序运行全部六种尺寸，每个尺寸、每轮使用新进程；按固定种子重排目标并轮换单题方法顺序。
- 单题比较包含 Dual、Gao/RSD、Jensen–Zimmermann R–TR2。每种尺寸、每种方法共 192 次正式调用，取全部调用时间的算术平均。
- 批量比较使用同一个 Dual 求解器，逐题完成 64 个目标，或一次输入全部 64 个目标。各路径每轮运行一次，共三次，取总时间的算术平均。加速比统一为对应均值之比。
- 每轮单题方法各预热两次；顺序和整批路径各完整预热一次。Gao 与 R–TR2 的外层预算为每题 1000 次迭代、60 秒。

计时为预先生成的 CPU 输入到 CPU 输出的同步墙钟时间，包含每次求解的初值构造、数据传输、Gram 矩阵、迭代、原解恢复，以及原解和 `omega` 的 CPU 回传。计时前同步 CUDA，取得 CPU 输出后再次同步 CUDA，再结束计时并独立验收。进程启动、目标生成、核函数编译、预热、独立 CPU 验收及输出释放均在正式计时之外。

结果在 CPU FP64 中独立复算，输出有限、正常终止且同时满足辛约束残差 \(r_F\le10^{-8}\)、驻点残差 \(r_G\le10^{-9}\)、绝对相对原对偶差 \(\delta\le10^{-8}\) 才记为成功。每次求解后判定一次，汇总直接统计。保留全部正式计时及失败记录，慢测量和失败测量也计入算术平均，预热不计入；只有相应比较的全部正式调用成功时，才报告其成功加速比。记录不完整时明确报错。单次求解异常会记录耗时和错误；CUDA 同步失败或工作进程异常退出时停止运行，保留已经写出的记录。

运行输出包括固定参数及 CPU 编号 `protocol.json`、一次环境记录 `environment.json`，各轮各尺寸的输入种子与重排配置 `config.json`、逐次计时与验收记录 `raw.jsonl`，以及汇总 `summary.json`。表图为 `tables/single.csv`、`tables/single.tex`、`tables/batch.csv`、`tables/batch.tex` 两张表和 `figures/comparison_en.pdf`、`figures/comparison_en.png` 四面板比较图。输出目录已存在即拒绝覆盖，包括空目录。

精简后的代码已在上述环境完成三轮六种尺寸实验，全部数值验收通过，并生成两张表和四面板比较图。汇总绘图使用串行 MKL，避免 Windows 下与 PyTorch 的 OpenMP 运行时冲突。不同硬件和运行环境的耗时、加速比会变化。

## 文件与引用

```text
README.md
reproduce.py
pyproject.toml
LICENSE
THIRD_PARTY_NOTICES.txt
src/
```

开发测试、历史方法、旧结果及 MATLAB 对照资料保存在仓库外备份。仓库中的 `src/` 仅保留正式算法、必要 CUDA 源码和复现辅助代码。

软件引用：YulinXie. *symplectic-stiefel-nearness*, version 0.1.0. [代码仓库](https://github.com/yulinxie20/symplectic-stiefel-nearness)。引用时注明实际使用的提交或版本。

方法与程序来源：

- Gao 等，*Riemannian Optimization on the Symplectic Stiefel Manifold*，SIAM Journal on Optimization 31(2), 1546–1575, 2021，[DOI](https://doi.org/10.1137/20M1348522)；*Geometry of the Symplectic Stiefel Manifold Endowed with the Euclidean Metric*，2021，[DOI](https://doi.org/10.1007/978-3-030-80209-7_85)。[原作者 `spopt` 固定版本](https://github.com/opt-gaobin/spopt/tree/94c808dae9da975345b3e23e0e42bffbefbf63ae)。
- Jensen、Zimmermann，*Riemannian optimization on the symplectic Stiefel manifold using second order information*，[论文](https://arxiv.org/abs/2404.08463v2)及[原作者程序固定版本](https://github.com/JensenRasmus/secondorder_spst/tree/c121d28f2116da6a566b8eff44642a42d8971bd0)。
- [Manopt 7.1 固定版本](https://github.com/NicolasBoumal/manopt/tree/909a2f756e691a6297f737bcc45bbff829674833)，用于 R–TR2 的缓存截断共轭梯度子问题求解。

自写代码的 MIT 许可见 [LICENSE](LICENSE)。Gao、Jensen–Zimmermann 与 Manopt 适配代码保留原作者及 GPL-3.0-or-later 标注，底层 GenRTR 保留 BSD 条款；完整版权声明和许可文本集中于 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。包含这些适配代码的组合分发不能整体标为仅 MIT。
