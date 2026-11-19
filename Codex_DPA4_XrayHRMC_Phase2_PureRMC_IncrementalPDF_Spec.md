# Codex 开发任务书｜Phase 2：Pure RMC、增量 X-ray PDF 与三种全局 scale 策略对比

> **执行性质**：这是一份交给已连接 AutoDL CPU 容器和 GitHub 仓库的 Codex 的**可执行开发与科研验证任务书**。请检查现有代码，实际实现、测试、运行、输出数据与结构，并提交 Git；不能只给思路或伪代码。**绝不伪造计算结果、速度数字、结果图或已运行的步数。**
>
> **版本**：2026-11-19；承接 Phase 1 `Codex_DPA4_XrayPDF_Phase1_Spec.md`。
>
> **本阶段范围**：纯 Reverse Monte Carlo（Pure RMC），**仅原子坐标移动 + PDF 残差 + Hard-sphere**。暂不加入 MLIP、DPA4、能量项、配位数约束、键角约束、PO4 刚体约束、`qdamp/qbroad` 拟合、相干尺寸模型及多数据源联合拟合。可实现相关接口，但不得让它们参与本阶段的结果。

## 0. 已确认的用户要求（不得再次询问）

1. **初始结构**：`PDF/PZS_quench_222.cif`，是 PZS 的 **melt–quench (MQ) 无定形结构**，不是此前用于 Phase 1 验证的 `LNOC_opt.cif`。本地 Windows 原始路径 `E:\1_CODEX\dpa4-xray-HRMC\PDF\PZS_quench_222.cif`；**实际开发和运行在已连接的 AutoDL Linux CPU 容器**，请在现有 Git 工作树或已同步的数据目录查找，不要假定 Linux 下有 `E:` 盘。
2. **实验目标**：用户此前提供的 `PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr`，已交给 Codex。使用 `.gr` 的真实数值及头信息；拟合窗口 **`1.5–10.0 Å`**。`.gr` 头部包含 `composition=Li3PO4Zr2Si2Cl16`、`qmin=0.9 Å⁻¹`、`qmax=16.0056 Å⁻¹`、`rstep=0.01 Å`、`mode=xray`、`rpoly=0.9189 Å`。**Codex 已在当前版本将模拟 Qmax 对齐为 ~16，不要无故退回 Phase 1 的 24。** 如实验文件实际内容变化，以其头部数值为准并在报告中说明。
3. **拟合方法**：三种策略（见第 4 节）：(A) `s=1`、(B) 首帧最优 `s0` 后固定、(C) `bounded_dynamic` 每个有效提案重新解析求最优 `s`，限制 `0.8*s0 <= s <= 1.2*s0`。`s0` 仅从**原始第一帧 MQ 的 Gcalc** 和实验 `Gexp` 在统一 `1.5–10 Å` 窗口计算一次；不能硬编码此前的 0.556。
4. **约束**：只启用 **pair-specific Hard-sphere（最短距离）**，禁止添加能量、固定配位数、预设多面体、额外结构正则项。
5. **预算**：功能测试 10,000 次尝试移动；性能测试 100,000 次；正式测试 **1,000,000 次尝试移动 / 链**；正式测试每种 scale 策略 **3 个随机种子**，共 **9 条链**。计数是 **attempted/generated moves**，不是 accepted moves；三种策略同起点、同 seed 组、同物理参数。
6. **运行/版本管理**：利用 **已连接的 AutoDL CPU 容器**运行；现有 GitHub 同步已完成。基于现有仓库和 Conda 环境扩展；不要创建新仓库、重复初始化 Git、覆盖 Phase 1 或擅自重建环境。完成后 push 到现有 GitHub 远程（以实际认证和权限为准）。用户本地原始文件只读。

## 1. 最终科学目标与问题边界

本阶段要用一次公平的纯 RMC benchmark 回答：

- **算法问题**：能否在**完全保留 Phase 1 `Q`-dependent X-ray `G(r)` 前向模型**的前提下，将每次 MC 原子移动的 PDF 更新由全量原子对重算改为可验证的**增量更新**？与完整重算相比能加速多少，内存代价如何？
- **方法问题**：三种全局 scale 策略在相同采样预算和 Hard-sphere 条件下，哪一种能更好地恢复实验 PDF 的**峰位置、相对强度、中程关联趋势**？每种策略自身的采样开销与接受率有何区别？
- **结构问题**：Pure RMC 尽管可能降低 PDF 误差，是否破坏 P–O、Si–O、Zr–O/Cl 等局域化学结构？请用 **post-analysis 诊断**回答，**不能反过来把这些诊断加入接受准则**。

不要声称仅凭 PDF 拟合更好就代表物理结构更好；Pure RMC 无能量项是本阶段有意设置的科学对照。后续才加入 DPA4 的 Inline/Alternating HRMC。

## 2. 启动前检查、项目保护与样品一致性

请先运行并记录（不必把以下命令逐字照搬，按实际环境即可）：

```bash
pwd
ls -lah
find . -maxdepth 4 -iname 'PZS_quench_222.cif' -o -iname '*slice_0_.gr'
git status --short
git branch --show-current
git remote -v
python -V
python -m pip list | grep -Ei 'numpy|scipy|ase|xraydb|numba|pytest'
lscpu | head -35
free -h
```

- **沿用 Phase 1 的包、模块和 CLI 风格**。检查现有代码里 `xray_pdf`、`partial_rdf`、`transform`、`form_factors` 真实 API，不得根据本任务书的建议路径强行覆盖已有架构。
- 检查 `.cif` 和 `.gr` 确实在 AutoDL 容器内；若 `.cif` 没有同步，先检查已连接的本地工作树 / 同步目录，**不要自动凭文件名新建/伪造 CIF**。必要时清楚报告缺失，仍可完成可测试的模块和示例后等待真实文件。
- ASE 读取 `PZS_quench_222.cif` 后记录：元素、原子数、化学计量、体积、密度、PBC、晶胞角度、最小晶胞垂直高度和最短原子间距离；确认与实验 `Li3PO4Zr2Si2Cl16` 的**元素集合和比例**一致（允许整倍超胞以及合理四舍五入；如存在掺杂、非整数配比则必须报告而不静默忽略）。**不要由文件名“222”推断原子数**。
- 保持 **原子数、元素、晶胞尺寸和形状、密度固定**；仅改变原子位置。输出的 `.cif`、`.extxyz` 或 `.vasp` 必须保存完整晶胞与 PBC，读回后验证元素计数和位置。
- 先将输入结构复制到各 run 的不可变快照，所有链从**同一个未经扰动的 MQ 原始结构**开始；本阶段默认不做 seed-dependent rattle 或能量预优化。将初始结构 hash、实验文件 hash、代码 Git commit、配置和环境版本写入元数据。
- 出于性能目的可在 AutoDL 上以 Python 3.11 + NumPy/SciPy/ASE/xraydb/pytest/Numba（如必要）运行；确保 CPU 模式，不安装或调用 DeepMD、GPU/torch。限制 `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` 等，避免 9 条链并发和 BLAS 过度线程竞争。

## 3. 先实现、验证并基准化：增量 X-ray PDF 引擎（本阶段最高优先级）

### 3.1 不允许改变的物理前向模型

沿用 Phase 1 已验证的：

\[
S_{\alpha\beta}(Q)-1=4\pi\rho_0\int_0^{r_c} r^2[g_{\alpha\beta}(r)-1]\,\mathrm{sinc}(Qr)\,dr,\quad \mathrm{sinc}(x)=\frac{\sin x}{x}.
\]

\[
w_{\alpha\beta}(Q)=\frac{(2-\delta_{\alpha\beta})c_\alpha c_\beta f_\alpha(Q)f_\beta(Q)}{\left[\sum_\gamma c_\gamma f_\gamma(Q)\right]^2},\qquad
S(Q)-1=\sum_{\alpha\leq\beta} w_{\alpha\beta}(Q)[S_{\alpha\beta}(Q)-1].
\]

\[
G(r)=\frac{2}{\pi}\int_{Q_\min}^{Q_\max}Q[S(Q)-1]\sin(Qr)\,dQ.
\]

保留 Phase 1 对 `xraydb.f0(element,Q/(4π))`、`Biso=0.3 Å²`（如果当前代码启用且与 Phase 1 完全一致）、`lorch=False`、归一化、自关联、周期镜像的定义。**增量计算和全量重算必须严格一致；不能为了加速改成固定 X-ray 权重或只用 `g(r)` 的直接加权和。**

实验 `Qmin/Qmax` 取 `.gr` 头部；理论上应在与实验相同的 `r` 网格输出计算值。对实验 `.gr` 数值可解析跳过 `#` 头；目标窗口 1.5–10 Å 对齐 `r` 网格并只做必要的插值，禁止额外平滑、基线平移、随 r 缩放或改变实验文件。

### 3.2 单原子移动怎样更新 partial histogram

设存储的物种对原子距离直方图为 `H[p,j]`（`p` 为元素对，`j` 为距离 bin）。一次尝试移动 `i: R_old -> R_new` 时：

1. 根据固定的 cutoff 和正确 PBC，找出移动原子 `i` 与**所有会贡献到直方图的其他原子 / 周期镜像**的旧、新距离（对移动前后使用相同统计约定）。
2. 将旧距离所在 bin `−1`，新距离所在 bin `+1`，得到稀疏 `ΔH[p,j]`；对同元素有序对直方图，按 Phase 1 原有归一化确定是否须 `×2`。**不能漏掉 i 作为中心或邻居时的成对贡献**。
3. `H_trial = H_old + ΔH`。**拒绝**时旧 `H`、位置、计算 `G`、误差与 scale 均不变；**接受**时一次性提交。避免每次 trial 深复制整套 `H`。
4. `i=j` 的周期镜像关联（若 Phase 1 full 计入）**距离不会因单原子平移改变，因而 `ΔH=0`**；但仍必须在完整初始统计与基准测试中保持一致。
5. 当 `r_cut` 超过半个最短周期垂直距离时，必须沿用 Phase 1 **多周期镜像统计**，不能偷偷替换成最小镜像；一并检查斜晶胞（triclinic）的镜像边界。仅在数学上满足 minimum-image 条件时可走更快路径。
6. 针对相邻 bin 跨越、位移越过周期边界、旧新距离恰好在 cutoff 或 bin 边界、同元素/异元素、接受/拒绝连续交替等，写数值测试。

参照用户此前的 `amsrmc-main.zip` 中 `hrmc/histogram.py`、`hrmc/engine.py`：该代码实现了**中子 PDF 的单原子原子对直方图增量更新**；可以参考其组织方式、测试思路和 Hard-sphere 检查，但本项目需要正确推广到 **Q-dependent X-ray PDF**。不得直接将中子散射长度系数替换成一个常数 X-ray 因子。若复制代码，遵守 MIT 许可并保留 Attribution；优先用明确可验证的新实现。

### 3.3 快速计算 `ΔG`：预计算 X-ray 核（重要）

在以下条件均保持固定时：晶胞、物种数量、`Q`/`r` 网格、`Biso`、散射因子、RDF bin 与 cutoff，整个 PDF 前向计算对原子对直方图是**仿射映射**：

\[
G_k=G_{\mathrm{background},k}+\sum_{p,j}K_{k,p,j}H_{p,j},
\quad
\boxed{\Delta G_k=\sum_{p,j}K_{k,p,j}\Delta H_{p,j}}.
\]

其中 `K` 必须从**Phase 1 原方程**推导和数值预计算，含有 partial RDF 的归一化因子、`Q`-dependent X-ray 权重、Biso、sinc 与积分权重、有限 Q 区间和可能存在的真实窗函数。固定 `−1` 均匀背景项属于 `G_background`，不能遗漏；因为本阶段密度与组成固定，其差值为零。

实现接口建议（名称可据现有仓库调整）：

```python
class IncrementalXrayPDF:
    def initialize(self, atoms, pdf_config): ...  # 一次完整重算并构建状态
    def propose_move(self, atom_index, new_position): ...  # delta histogram, delta G; 不修改当前状态
    def commit(self, proposal): ...   # 仅在接受后更新位置、直方图、G 与缓存
    def reject(self, proposal): ...   # 不修改或回滚到精确旧状态
    def full_recompute(self, atoms): ...  # Phase 1 参考算法，用于独立验证
```

**强制性能设计**：

- 不能每次重新 `full_recompute()`，不能为每个原子移动循环全部 `N²` pair，也不能为每次 trial 重建 `S(Q)`/form factor / 大型积分矩阵。
- 大小约为 `n_r_fit × n_pairs × n_bins_rdf` 的三维核可能占用大量内存（6 元素=21 种 partial，`dr=0.01 Å` 和 `r_cut=40 Å` 时尤其明显）。**先估算并记录内存，再选算法**。首选考虑以下两种等价实现并 microbenchmark：
  - **A. 稀疏 ΔH + 直接索引预计算 `K` 列**，求出 `ΔG`；按元素对及改变的 bin 合并相同索引，降低内存带宽和重复计算；必要时分块与 float64 累积。
  - **B. 稀疏 ΔH → ΔS(Q) → ΔG(r)** 的两阶段核；预先缓存 `sinc(Qr_j)`、`w_p(Q)` 和 `sin(Qr_k)` 或其矩阵；对比其性能与内存。
- 可以使用 `Numba` JIT 编译距离/PBC/直方图增量计算；**禁止未经核验的 float32 误差传播**。CPU 上要同时记录编译预热时间与纯 steady-state 速度。
- 还可考虑 cell list / 邻居表，但**必须保证 cutoff 和 skin**、移动跨 cell、邻居表更新正确，不能用近邻截断漏掉对 `G(r)` 有贡献的远程原子对。
- 特别评估 `r_cut=20/30/40 Å` 的**全量曲线差异、每次移动成本与内存**：PZS 无定形结构在超过半箱尺度时会产生周期复制的远程伪相关；不要盲目把 Phase 1 晶体为测试采用的 `40 Å` 当成无定形的唯一正确配置。**先用原有 forward reference 量化收敛，再冻结正式运行参数**。任何修改 `r_cut` 必须对三种策略一致，并在报告中显式记录。
- `Q` 内部积分网格、`.gr` 输出网格、`Qmax=16.0056` 与 `Biso` 处理要在优化前后**逐点一致**。若需要改变 forward solver 或积分数值方法，只能作为单独实验，不能将不同 solver 混入三种 scale 速度比较。

### 3.4 必须通过的 correctness gate

在一小型随机混合原子体系、周期性晶体、以及真实 `PZS_quench_222.cif` 上，至少做：

1. 初始增量引擎生成的 `G(r)` 与 Phase 1 直接全量结果一致。
2. 对几十/几百个**预先存好的可复现 trial moves**，逐一比较 `G_trial_incremental` 与在相同试探坐标下 `G_trial_full`；检查 `ΔG`、partial histograms、`S(Q)`（如实现）与误差函数，比较最大绝对差 / RMS / 相对误差。
3. ≥10,000 个接受/拒绝混合的 MC 操作后，从当前坐标独立全量重算，与增量状态比对；再长时间跑至少一次 ≥100,000 次抽查，监测 drift。
4. 检查 reject 后快照 byte-level 或严格数值无变更；保证 PBC 坐标、全部 pair bins、scale 缓存不漂移。
5. **不得通过“每隔若干步偷偷重建 G”掩盖系统误差**；允许可选重建作为安全校验/恢复机制，但必须记录触发、重建前误差与耗时。在正常模式下误差应足够小。
6. **验收建议**：`max_abs_delta_G < 1e-8 Å^-2`、`RMSE < 1e-9 Å^-2`（float64、数学上等价的离散算法）；若 Phase 1 的求和顺序或预计算精度使这一数值不现实，先分析来源、给出实测稳定阈值并解释，不能仅放宽阈值使测试变绿。尤其确保误差远小于预期真实 single-atom move 造成的 `ΔG`。
7. 数值误差的测试使用**相同位置、相同 PBC 和同一完整前向定义**；不要仅用与旧版代码画出的曲线“肉眼很像”当成正确性证据。

**只有该 correctness gate 通过，才允许正式执行 RMC。**

## 4. 三种残差计算方法（核心对比）

固定 `r ∈ [1.5, 10.0] Å`、同一实验数据 `y_i=Gexp(r_i)`、同一模拟 `x_i=Gcalc(r_i)`、同一非负权重 `w_i`。默认 **uniform weights `w_i=1`**（若实验有每点可靠不确定度才可增加单独的加权模式；本阶段不臆造 `σ_i`）。

定义：

\[
\mathcal L(x,s)=\sum_{i=1}^{M}w_i(sx_i-y_i)^2,\quad
A=\sum w_ix_i^2,\quad B=\sum w_ix_iy_i,\quad C=\sum w_iy_i^2.
\]

**第一帧 scale 只求一次**：

\[
s_0=\frac{B_0}{A_0},\quad A_0>0.
\]

如果 `s0<=0` 或 `A0` 非法，立即报错并检查输入/归一化，不得无提示强制换个值。记录 `s0` 至共享配置与结果目录；**三种策略使用同一个 `s0`**。

| 模式 / ID | 采样时采用的尺度 | 模式自身目标函数 |
|---|---|---|
| **RMC-1** `no_scale` | `s=1.0` | `L1 = C - 2B + A` |
| **RMC-2** `fixed_initial` | `s=s0`，全程固定 | `L2 = C - 2*s0*B + s0*s0*A` |
| **RMC-3** `bounded_dynamic` | `s_trial=clip(B/A, 0.8*s0, 1.2*s0)`；每个合法 trial 解析更新 | `L3 = C - 2*s_trial*B + s_trial*s_trial*A` |

**重要：** RMC-3 的 scale 边界是**第一帧 `s0` 的 ±20%**，**不能以当前上一步的 scale 为中心滚动更新边界**。对 RMC-3 只有在 move 接受时提交新的 `s_trial`；拒绝时保留旧状态和旧 scale。目标始终是给定结构的解析最优**有界**尺度：即使在 Metropolis 流程中每步重新优化，它仍然是明确的配置势函数；但这**不等同于对未知 scale 的统计边缘化**。

### 4.1 计算效率：增量更新残差中的 `A, B`

对一次增量 `d_i=ΔG_i`，避免每次重算整个完整拟合数组的系数：

\[
A_{trial}=A_{old}+2\sum_iw_ix_id_i+\sum_iw_id_i^2,\quad
B_{trial}=B_{old}+\sum_iw_iy_id_i,\quad C=\mathrm{constant}.
\]

这样三种策略可以共享同一份增量 `ΔG`，只用几个 dot products 取得 trial loss。若出现极小负值（浮点消减）要做稳定性检查，不能悄悄 `abs()`。为避免长链增量浮点误差，按 checkpoint 独立全量校验 `A,B`。

### 4.2 三种策略的公平对比（很重要）

- **同一初始结构、同一 `s0`、同一个 delta-PDF 后端、同一个 Hard-sphere 表、同一个可重复 seed 列表**；每种策略 3 条链。对于相同 seed，随机数提案流尽量一致（推荐预生成 `atom_idx, displacement, uniform_acceptance` 流或使用分离 RNG 流）。由于接受率不同，轨迹本来就会分岔，不必强求位置相同；报告此事实。
- **相同 `d` 分布、温度/容忍参数、采样预算**。如果为提高接受率需要调参，先做独立 pilot，冻结全部设置后比较；调参造成的变体要单独标记而非纳入主比较。
- **重点：三种模式各自最小化的目标函数不同，不能简单对比各自记录的最终 `L_mode` 并判定谁更好！** 所有输出的最终结构和中间快照都必须使用下列**公共诊断指标**交叉评分：
  1. `L_no_scale` 与 `Rw_no_scale`（统一 `s=1`）；
  2. `L_fixed_initial` 与 `Rw_fixed_initial`（统一 `s=s0`）；
  3. `L_bounded_dynamic` 与 `Rw_bounded_dynamic`（所有结构统一相同的 `s0±20%`）；
  4. `L_unbounded_positive_opt` 与 `Rw_shape_only`（仅诊断，最佳正 scale，无界；**不参与采样**）；
  5. 按 `1.5–4.5 Å` 与 `4.5–10 Å` 分段计算 RMS 残差、峰位和主要峰相对强度（同一个全局 s，**不得分段重新优化 scale**）。
- 可定义共同 `R_w = sqrt(L / C)`，`C = sum(w_i y_i²)`，注明定义、归一化及 `w`；同时输出 `R²`（谨防常数基线/选择窗口导致负值），不要把不同尺度下的 `R_w` 混为同一评价基准。
- 绘制初始 `G_MQ`、实验 `G_exp`、三种策略的最终 `s_mode G_calc` 曲线；另画每个模式的**未缩放 Gcalc**、**统一固定 s0 后的曲线**以检查真结构变化。不能只输出采样目标随步数下降的曲线。

### 4.3 Metropolis 接受准则：固定且明确定义

Pure RMC 每个 trial：

1. 提案单原子平移；先 Hard-sphere。违反则直接拒绝，不计算 `ΔG`。
2. 未违反时，计算 `ΔH`、`ΔG`、trial loss（对应选择的 scale 模式）。
3. `ΔL = L_trial - L_old`。
4. 对称提案使用：

\[
P_{acc}=\min\{1, \exp[-\Delta\mathcal L/(2\sigma_{RMC}^{2})]\}.
\]

这里 `L=Σ residual²`（`w_i=1`），`σ_RMC` 是 **PDF G(r) 单位的 RMC 容忍参数**（Å⁻²），不是实际样品温度，也不能直接照搬 Zhong 中子实验使用的 0.030 Å⁻² 作为对本 X-ray 数据“已标定”的事实。先用 10k pilot 调整 `d` 和 `σ_RMC`，报告候选试验与接受率，再**冻结同一组 `d,σ_RMC` 给三种策略、三个种子**；若采用预定义值，也必须明确记录。

- 使用 `log_u < -ΔL/(2σ_RMC²)` 实现，避免 `exp()` 溢出；正改善必接收。
- `L` 若选择 `mean` 而非 `sum`，须**同步**对 `σ_RMC` 做一致的数学定义；主实现固定采用 `sum`，不得意外除以 M 或重复除以实验误差。
- Hard-sphere 本质是对称且与构型无关的可行域约束：越界/违反者拒绝。若将来实现不对称 proposals，应加入 proposal ratio；本阶段不需要。
- 最小化 PDF 不代表模拟热平衡；**不输出虚构的能量、温度或电导率**。

## 5. Hard-sphere 约束与移动机制

- 从原始 MQ 结构自动统计元素对最小非零距离，构造可追溯的硬球下限表（建议参照 Wang/Zhong 的方法，设为 `0.95 × min_distance_MQ(pair)`；**仅为初始推荐，不是唯一物理正确值**）。记录每对 `min_input`, `hard_cutoff`, 使用的距离搜索范围、样本计数；检查离群近接触，避免某一种很偶然的异常近接触使约束失效。
- 6 元素体系有 21 类（如果实际 CIF 元素数不同自动调整）；所有有实际关联的元素对均要有安全阈值；对未出现近邻统计的元素对需采用解释清楚的回退规则，禁止缺失约束导致原子重叠。
- `Hard-sphere` 检查新坐标相对于其他原子及必要周期镜像；**不得禁止原始结构中已经存在的接触而使某物种不可移动**，如输入有违反推荐值则需识别原因、由原始数据合理修正并记录，不能偷偷大幅 rattle/改变结构。
- 只移动**单个原子**，优先各笛卡尔坐标独立 `Uniform(-d,+d)`、`d` 全物种相同（可配）；随机选取原子索引，按 PBC wrap 到主胞。
- 可以在 pilot 中试 `d=0.05,0.10,0.20,0.30 Å` 等（**不是最终已确认参数**），选择所有策略共享的可行值并报告；不要根据三个目标分别调最适合自己的 `d` 后声称公平的效果差异。
- **不锁定 P–O、Si–O、Zr–O、Zr–Cl 键**，不固定 PO4 或 SiO4 多面体，也不调节 CN；但是在输出诊断中必须统计这些性质的变化及可能出现的不合理结构。

## 6. 跑法：分阶段执行与 CPU 性能评估

### Stage 0 — 科学与算法验收

- 确认真实 PZS `.cif`、实验 `.gr`、Q 与数据范围；冻结前向物理参数。
- 完成 Section 3 的完整 vs 增量一致性测试；画至少一张 `G_full`/`G_incremental` 重叠图、差值图。
- 实现 Hard-sphere、三种 scale 的**纯函数**及单元测试（解析式对比 SciPy minimize_scalar；边界生效；旧/新回滚）。

### Stage 1 — 10,000 attempted moves 功能测试

- 三种策略、至少 1 个共同 seed，分别运行 10k 次尝试；建议再对 3 种 seed 做短跑确保健壮性。
- 输出接受率、Hard-sphere 拒绝率、loss、最终结构、scale、增量误差抽检；检查没有 NaN、不合理坐标和破坏的 PBC。
- 只有通过后进入 100k。**不允许跳过正确性 gate 直接长跑。**

### Stage 2 — 100,000 attempted moves 性能 benchmark

- 三种策略均执行至少 100k 次尝试，使用相同 seed 和同一参数；建议 3 seeds，用于建立速度离散度。
- 独立测量：
  - 首次初始化（CIF、histogram、form factor、kernel）时间与峰值 RSS；
  - 每个 proposal 的 `hard_sphere_check` 时间；
  - 不违反硬球的 `delta_hist` 时间；
  - `delta_pdf` 时间（包括必要核矩阵乘）；
  - 三种残差计算 `score` 时间（应很低）；
  - accept/reject 和 checkpoint I/O 的时间；
  - **全局端到端 attempts/s**、effective PDF evals/s、accepts/s、完整 100k 的 wall time；
  - 两个增量后端（如都实现）及 FullRecompute baseline 的性能对照。
- 全量参考 baseline 可只跑**固定的 100–1000 个相同 trial moves**进行 microbenchmark；不必为了性能测试把 100k 次全量重算做到不可接受，但必须说明加速比的分母包含哪些操作。推荐分别报告 `speedup_PDF_only` 和 `speedup_end_to_end`，不能混用。
- 为公平测量：**先预热 JIT**，固定 CPU 线程数，运行相同输入与预生成 moves，区分“合法试探（穿过硬球）”与“总尝试移动”，分别统计中位数/p95、总耗时。**不同策略的接受率会改变整体性能；除真实跑速外，还必须用相同已缓存 `ΔG` 序列独立 benchmark 三种 score 的纯计算开销。**

### Stage 3 — 正式研究对比：1,000,000 attempts × 3 seeds × 3 modes

- 每条链**1,000,000 次尝试**，不将 accepted 替代 attempted；共 9 条独立链。
- 三种模式每个 seed 都从同一原始 MQ 文件开始；三种 seed 之间**仅随机流不同**，不改变原始初始结构。
- CPU 并行可按容器核数与内存预算选择，例如 `3` 条链并行、9 条分批，**不得让每条进程又各自开启 32 个 BLAS 线程**；记录实际并行数。作公平的**单链**速度基准时不要同时运行多链。
- 频率建议：trace **每 1,000 attempts**；结构 snapshot **每 50,000 attempts**；checkpoint **每 50,000 attempts**；全量数值校验 **每 10,000 或 50,000 attempts**（可独立配置；测试耗时应另列）。输出初始与最终、若干中间结构与最佳拟合结构。
- **原子移动最终步和 best-so-far 不一定相同**。每条链必须同时输出 `final_structure` 与 `best_structure`（按该模式自己的损失），并记录对应的 loss、scale 与 MC step。不得只保存“最后一次轨迹快照”冒充 best。
- 断点续跑要保存 positions, histogram/G, A/B/C, scale, RNG states, accepted/rejected counters, parameters, Git hash；重启时可重新计算初始 hist 验证缓存。**避免从 0 重跑或恢复后造成随机数重播/状态不一致**。
- 如因明确资源限制无法完成全部 9 条 1M，请先完成可实际运行的测试与高可信度 benchmark，报告已完成的确切步数和阻碍，提供可续跑命令；**绝不写成正式 9M 步已完成**。

## 7. 最终结构的物理与统计诊断（不反馈到 RMC 目标）

运行前、各策略每条最终和最佳结构，至少输出：

- 全部元素对 minimum distance / Hard-sphere 违反数；原子数、密度、晶胞不变量；原子坐标是否 finite。
- `P–O`、`Si–O`、`Zr–O`、`Zr–Cl`、`Li–O`、`Li–Cl` 的 **partial RDF 对比**、指定 cutoff 下的配位分布（写清 cutoff 是如何选择的；优先原始 MQ 的 RDF 第一极小值，不在各模式间随意改 cutoff）。
- P / Si 四面体保留率、P–O 和 Si–O 键长分布（只是诊断，不约束）。适度统计 Zr 配位异常、出现的新短程邻接及异常拉长的键距。
- 结构最大/平均位移（PBC 最小位移意义）、物种分组 RMSD 和多面体连接的变化；**相对于 initial MQ**，不是相对于每条链的第一张扰动帧。
- 计算与实验 G(r) 的峰位置、第一壳相对峰强以及 4.5–10 Å 中程曲线的残差；使用相同仪器 Q 条件和统一评价尺度。
- 输出可被 ASE 读回的 `.cif`、`.extxyz` 或 `.vasp`（建议 `.extxyz` + `.cif` 或 `POSCAR` 双格式），确保对非正交晶胞保持坐标正确。
- 解释：本阶段**不施加能量项**，因此即使获得更低误差，也可能产生不合理键断裂/配位变化；这恰是 Pure RMC 与未来 DPA4-HRMC 的关键对照。

## 8. 文件布局与要求的可复现产物

请适配现有仓库，而不是强行重命名 Phase 1 模块。建议新文件：

```text
<existing repo root>/
├── PDF/
│   ├── PZS_quench_222.cif                 # 用户输入，保持原样/只读
│   └── PDF_ZXS_1_...slice_0_.gr          # 用户提供，保持原样/只读
├── src/<existing_package>/
│   ├── xray_pdf/...                        # 复用 Phase 1
│   └── rmc/
│       ├── incremental_hist.py             # trial delta, commit/reject
│       ├── xray_kernel.py                  # Q-dependent kernel + ΔG
│       ├── hard_sphere.py                  # 21 pairs cutoffs, PBC
│       ├── objectives.py                   # no_scale/fixed/bounded_dynamic
│       ├── metropolis.py                   # RNG, moves, accept
│       ├── runner.py                       # run, resume, checkpoint
│       ├── diagnostics.py                  # 后验结构分析
│       └── benchmark.py                    # 分阶段性能剖析
├── configs/
│   ├── pzs_phase2_common.yaml
│   ├── pzs_phase2_no_scale.yaml
│   ├── pzs_phase2_fixed_initial.yaml
│   └── pzs_phase2_bounded_dynamic.yaml
├── scripts/
│   ├── run_phase2_pzs.py
│   ├── benchmark_incremental_pdf.py
│   ├── compare_phase2_results.py
│   └── validate_incremental.py
├── tests/
│   ├── test_incremental_hist.py
│   ├── test_xray_kernel_delta.py
│   ├── test_scale_objectives.py
│   ├── test_hard_sphere.py
│   ├── test_metropolis_and_rng.py
│   └── test_checkpoint_restart.py
├── results/PZS_phase2/
│   ├── baseline/
│   ├── correctness/
│   ├── benchmarks/
│   ├── no_scale/seed_<id>/
│   ├── fixed_initial/seed_<id>/
│   ├── bounded_dynamic/seed_<id>/
│   ├── comparison_summary.csv
│   ├── comparison_1p5_4p5A.png
│   ├── comparison_1p5_10A.png
│   ├── loss_vs_attempts.png
│   ├── scale_vs_attempts.png
│   ├── speed_comparison.png
│   ├── structural_diagnostics.csv
│   └── phase2_final_report.md
└── docs/
    ├── phase2_incremental_derivation.md
    ├── phase2_rmc_protocol.md
    └── phase2_references.md
```

### 每条链至少产生

- `run_config_resolved.yaml`、`environment.txt`、`input_manifest.json`（hash+Git commit）。
- `trace.csv`: attempted_step, accepted_total, hs_rejected_total, acceptance_rate_window, `s_current`, `loss_mode`, `loss_no_scale`, `loss_fixed_initial`, `loss_bounded_dynamic`, `Rw_*`, wall_s 等。
- `checkpoints/`：可恢复的二进制检查点与 `resume` 命令。
- `final_structure.cif`、`final_structure.extxyz`、`best_structure.cif`、`best_structure.extxyz`。
- `final_calc_Gr.dat`、`best_calc_Gr.dat`：至少含 `r, Gcalc_unscaled, s_mode*Gcalc, Gexp, residual`；另导出 `s0*Gcalc` 与 `best_positive_scale*Gcalc` 诊断数据。
- `traj.extxyz` 或分散 snapshots；不应无节制每一步存大结构。
- `diagnostics.json`：pair distances、coordination summary 等。
- `timings.json`：CPU/线程、初始化和每模块时间拆分、全局 throughput。

### 汇总报告必须回答

1. 是否实现了 `O(N)`（相对于一个移动原子需检查的其他原子数/镜像数）的直方图 delta？实际速度与 `N`、`r_cut` 的关系如何？如 `delta_G` 额外有 `Q`/`r` 维度成本，需分开明确说明，**不可笼统声称整个算法严格 O(N)**。
2. 增量 vs 全量 PDF 的最大误差、累计漂移、性能加速比，内存占用，初始化预计算开销；三种 scoring 模式单步时间有多大差异？
3. 三种模式 9 链最终/最佳结构的 `R_w` 与 `R²`（各自及公共诊断），种子间均值 ± 标准差；哪种“相对变化趋势”改善最多？
4. `s0` 的数值是多少？`bounded_dynamic` scale 是否长期顶在上下界？scale 的变化是否掩盖了配位或 RDF 振幅变化？
5. Pure RMC 对 P–O、Si–O、Zr–O/Cl 局域配位损伤程度如何？对于未来引入 DPA4 能量约束提供什么依据？
6. 与实验条件是否一致：`Qmin/Qmax`、finite-Q transform、Biso、无 Lorch、r-grid、样品组成、数密度以及 `.gr` 的处理定义；**不得用仅缩放图形代替这些检查**。

## 9. 性能与结果的最低验收清单

- [ ] 已在 AutoDL CPU 的当前 repo 上找到真实 `PZS_quench_222.cif` 与 `.gr`，并验证文件内容、组成与 hash。
- [ ] 三种策略使用同一 MQ 首帧、同一 `s0`、同一 `s0±20%` 边界和共用增量引擎。
- [ ] `pytest` 全通过；100–1000 个固定 trial moves 可逐点重现 full `G(r)`；≥10k 长序列无不可解释漂移。
- [ ] CPU profile 清晰区分 `delta_hist`、`delta_G`、`scale_loss`、hard sphere、I/O、总耗时。
- [ ] 10k 与 100k 测试真实运行；正式 1M × 9 链在允许的计算资源下执行，若未完成明确标识哪些未完成。
- [ ] 每条真实完成的链包含 final/best 结构、曲线、trace、scale、检查点；九条完整时提供统计汇总。
- [ ] 三种 scale 策略由**公共交叉评价指标**比较，不仅比较各自采样 loss。
- [ ] 只施加 Hard-sphere，完全没有调用 DPA4 / 能量或配位约束。
- [ ] 代码和文档提交到**已有 GitHub 远程**，保留 Phase 1，用户实验原始文件/大型结果按 `.gitignore` 与现有隐私约定处理。

## 10. 推荐 Codex 执行顺序与交付方式

1. **Repo audit**：输出当前关键文件结构、现有功能、环境、数据查找结果，形成 `docs/phase2_implementation_plan.md`，随后**直接开始实现**，不要停在计划。
2. **Correctness-first**：以 Phase 1 的全量前向实现作为真值，新增 delta pair histogram、X-ray kernel 和严格的移动/回滚单测。
3. **Objective and MC engine**：Hard-sphere、三个 scale 目标、Metropolis、checkpoint、轨迹/图片/性能日志。
4. **Microbenchmark + 10k smoke**：对照算法，修复数值与性能问题。
5. **100k benchmark**：真实测吞吐、拆分各模块耗时，选稳健最快的增量后端。
6. **1M × 9 正式对比**：在 AutoDL CPU 容器合理并发下实际运行。必须记录任一中断，并保证可续跑。
7. **审计与交付**：生成完整科研对比报告、每条链最终和最佳结构，提交代码；在最终 Codex 回复中给出 `git commit`、GitHub branch、运行命令、实际完成步数、各方法分数和实测速度。

### CLI 形式示例（按现有实现调整）

```bash
# 一次正确性验证（真实 PZS 结构 + 实验 G(r)）
python scripts/validate_incremental.py --config configs/pzs_phase2_common.yaml

# 性能比较
python scripts/benchmark_incremental_pdf.py --config configs/pzs_phase2_common.yaml --attempts 100000

# 单条链示例，要求支持 --seed、--steps、--resume
python scripts/run_phase2_pzs.py --mode bounded_dynamic --seed 101 --steps 1000000 \
  --config configs/pzs_phase2_common.yaml

# 三种策略交叉对比报告
python scripts/compare_phase2_results.py --root results/PZS_phase2
```

## 11. 文献、现有实现与必须理解的边界

下列文献/源码供算法设计和科学验证。**明确区分“文献实际做到什么”与“本项目从中延伸开发什么”**，不要声称文献直接提供了本项目严格等价的 6 元素、`Q`-dependent X-ray delta kernel 实现。

1. **McGreevy, R. L.; Pusztai, L. (1988)**. *Reverse Monte Carlo simulation: A new technique for the determination of disordered structures*. **Molecular Simulation** 1, 359–367. DOI: https://doi.org/10.1080/08927028808080958  — RMC 基本方法：移动原子，比较散射数据与模型结构，Metropolis 接受。
2. **Wang, X.; Zhang, X.; Butenko, D.; Hong, B.; Zhu, J.; Zhong, P. (2026)**. *A practical route to amorphous solid-state electrolyte structures by hybrid reverse Monte Carlo with foundation potentials*. arXiv:2610.19846. https://arxiv.org/abs/2610.19846  — 用户主要参考研究，含 pure RMC、inline/alternating HRMC 及“单原子变动更新原子对直方图”的方法；其论文的结构可观测量是**中子** Faber–Ziman PDF，不能直接照搬其散射权重处理。
3. **AM3GroupHub/amsrmc**（Wang/Zhong 论文配套代码，MIT License）: https://github.com/AM3GroupHub/amsrmc  — 特别阅读 `hrmc/histogram.py`、`hrmc/geometry.py`、`hrmc/constraints.py`、`hrmc/engine.py`、`examples/rmc_no_cn/run.py`。源码中 `Histograms.delta` 对 moved atom 的相关距离差分，适合作为 delta 设计参考；需要复核其最小镜像安全要求，不要对 PZS / 大 `r_cut` 不加分析地套用。
4. **Neder, R. B.; Proffen, T. (2020)**. *Exact and fast calculation of the X-ray pair distribution function*. **Journal of Applied Crystallography** 53, 710–721. DOI: https://doi.org/10.1107/S1600576720004616  — 特别关注 `Q`-dependent form factors 及构造修改后的 partial PDFs 以加速计算的原理。**不能据其标题就宣称本项目的有限截断 RDF 算法对任意周期晶体“严格精确”。**
5. **Sławiński et al. (2024)**. *RMCProfile7: reverse Monte Carlo for multiphase systems*. **Journal of Applied Crystallography** 57, 1251–1262. DOI: https://doi.org/10.1107/S1600576724004175  — 提供 big-box RMC、散射数据形式转换和 X-ray PDF back-transform / 分辨率处理的算法与使用场景；本阶段不拟合多相也不精修额外仪器参数。
6. **fullrmc** 官方文档： https://bachiraoun.github.io/fullrmc/fullrmc.Core.html 及 https://bachiraoun.github.io/fullrmc/fullrmc.Constraints.html  — `ExperimentalConstraint.fit_scale_factor` 与 `adjustScaleFactor` 说明 RMC 中自动求 scale 有既有实践；**fullrmc 常用按特定频率更新的策略，与本任务每个合法 trial 都解析选择有界最优 scale 不完全相同**。
7. **RMCProfile 官方手册**：https://rmcprofile.ornl.gov/ （manual、PDF 数据准备、约束与数据归一化）。支持将实验数据的定义和归一化先处理好，而不是让原子移动补偿所有系统误差。
8. **Phase 1 内部文档**：`Codex_DPA4_XrayPDF_Phase1_Spec.md`、本项目 `docs/` 的公式与 `results/LNOC/validation_report.md`（若已同步）。**Phase 1 的全量计算是 delta 引擎的 reference oracle，而不是要求重复开发一次。**

### 引用使用注意事项

- README、报告里给出 DOI / URL；从其他开源仓库复制代码时标注许可证及文件来源，不能未经授权复制非兼容代码。
- `fullrmc` 的 scale 处理可作实践参照，**不能直接当成每步变尺度 Metropolis 算法自动保证物理可辨识性的证明**。
- Zhong 等研究表明 pure RMC 可显著改善 PDF 拟合却损伤局域化学，因此本阶段要把**PDF 拟合质量与结构合理性分开报告**。

---

## 给 Codex 的一句话任务总述

**在已同步 GitHub 的 AutoDL CPU 项目中，沿用 Phase 1 的真实实验 `Q=0.9–16.0056 Å⁻¹` 的 `Q`-dependent X-ray PDF 前向模型；从 `PDF/PZS_quench_222.cif` 的同一个 MQ 首帧开始，优先实现并严格验证单原子 `Δhist → ΔG` 增量计算，再以相同 Hard-sphere 和采样预算对比 `no_scale` / `fixed_initial` / `bounded_dynamic (±20% s0)` 三种 Pure RMC：10k 功能、100k 性能、1M×3 seeds×3 modes 正式对照；保存 final/best 结构、实验曲线叠加、三种公共残差、scale 演化、运行速度和局域配位诊断；不调用任何势能模型，提交现有 GitHub 仓库。**
