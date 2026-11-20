# Phase 3 — PZS 的 DPA4–X-ray PDF Hybrid Reverse Monte Carlo 完整开发规范

**交付对象：Codex（在用户现有 Git 仓库内开发）**  
**版本：v1.0 · 2026-11-20**  
**目标：完成 Phase 3A（DPA4 验证）→ Phase 3B（Alternating HRMC）→ Phase 3C（Inline HRMC）的可复现实现和短程测试；正式百万步计算不属于首轮默认任务。**

> **用户已确认：** (i) 使用已经针对 PZS（Li–P–Si–Zr–O–Cl）体系微调的 DPA4 `.pt2`；(ii) Phase 3B/3C 的全部**正式对比运行均从原始 MQ `PDF/PZS_quench_222.cif` 开始**；(iii) 第一轮只用 Phase 2.5 `no_scale`（s=1）；(iv) 使用 AutoDL 单张 RTX 5090、已有 Conda 环境 `dpa4_torch211`；(v) 模型目录 `/root/autodl-tmp/CODEX`。
>
> **数学基准：直接继承现有 Codex Phase 2.5 的真实实现，不新创配位惩罚形式，不修改 σ、归一化、cutoff 和权重。** 如本文与当前仓库源码有差异，以核实后的源码和用户确认的 Phase 2.5 审计报告为准，先报告差异，禁止悄悄改公式。

---

## 0. 任务边界、执行纪律与验收原则

### 0.1 三阶段任务

1. **Phase 3A — GPU DPA4 接口、模型验证与短程 FIRE**：可靠读取已微调的 `.pt2`，对 MQ 和现有 RMC 结构计算总能量、力，验证单位/元素排序/PBC/有限差分一致性，跑少量固定晶胞 FIRE，检测对实验 PDF 与局域几何的影响，并测量 RTX 5090 性能。
2. **Phase 3B — Alternating HRMC**：从同一 MQ 出发，循环执行 Phase 2.5 `no_scale` 的 PDF+CN RMC 和 DPA4/FIRE；推荐首选 `500 accepted RMC moves + 5 FIRE steps`，保持算法与 Zhong 文献方案可比较。FIRE 不在 RMC 接受概率中引入能量项。
3. **Phase 3C — Inline HRMC**：从同一 MQ 出发，将 DPA4 **整个 896 原子周期超胞的总能量差**（eV）加入原有 Phase 2.5 Metropolis 指数。测试不同有效温度和移动步长，先评估 GPU 性能再确定可负担的尝试数。

### 0.2 绝对不要做

- 不重新训练/微调模型；不把未经允许的模型、实验数据、SSH 凭证上传至 Git；不写入任何明文 SSH 密码/令牌（包括终端命令、报告、checkpoint 和 Git 历史）。
- 不覆盖或删除 Phase 1/2/2.5 的既有代码、报告、结果；不在工作目录执行危险清理；不自动升级/重装 `dpa4_torch211` 的核心 PyTorch、CUDA、DeePMD-kit。
- 不将 `no_scale` 变成自适应 scale；不改变实验 `.gr`、Q grid、PDF 实空间截断、Biso、Lorch、硬球或 CN 参数。
- 不声称已通过任何未实际执行的测试；不以“最终 Rw 最低”替代物理合理性验证；不把 Alternating 当作严格的单一 Boltzmann 采样算法。
- 不在 MC 热循环使用 CrystalNN/ChemEnv；只做离线几何/配位交叉验证；不重复读取 `.pt2` 初始化模型。
- 若模型在 GPU 上不能可靠执行，**停在 Gate A**；不可用随机/虚构能量或悄悄切换成另一模型来完成报告。

### 0.3 起点与对照

| 对象 | 唯一基准/要求 |
|---|---|
| 体系 | `Li3PO4Zr2Si2Cl16` × 32，896 原子，`Cl512 Li96 O128 P32 Si64 Zr64` |
| 正式初始结构 | 原始 `PDF/PZS_quench_222.cif`（不是 Phase 2.5 最终/最佳结构） |
| 主要比较基准 | Phase 2.5 `no_scale`（s=1）的 3-seed Pure RMC；报告平均最终 Rw ≈ 0.12351，只作核对值，不硬编码 |
| X-ray PDF 数据 | 延用现有真实实验 `.gr`；历史名称 `PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr`，在当前仓库内实际发现 |
| 模型 | `/root/autodl-tmp/CODEX` 中用户已微调的 `.pt2`，需先精确识别文件和 SHA256 |
| GPU / conda | 1 × RTX 5090；`dpa4_torch211` |
| 代码 | 现有 `dpa4-xray-HRMC` 仓库（实际 Linux 路径须在服务器上检查） |
| 开发策略 | 新的 feature 分支、三个独立可回归模块/commit、三份 Gate 报告、最终汇总 |

**现有报告事实提醒：** Phase 2.5 几何审计用的是 `final_structure.extxyz`，**不是**每条链的 `best_pdf_structure` 或 `best_total_structure`。如要分析 best 结构必须另行读取和命名，绝不能混为一谈。

---

## 1. Gate 0：服务器、仓库和数据清单（先执行，无计算）

**执行环境：** 用户已配置的 AutoDL SSH 会话内。允许 Codex 在既有登录会话中运行命令；**不要将 SSH 密码写入本文或脚本**。工作根目录 `/root/autodl-tmp/CODEX` 可能是势函数文件夹、代码仓库上层文件夹或两者之一，必须先检查 `pwd`、`git rev-parse --show-toplevel`、`find ... -name '*.pt2'`，而不是假定根目录就是仓库。

先检查并记录：

- `nvidia-smi`、驱动、GPU compute capability、显存；`which python`、`conda info --envs`、`python -V`；`torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`、`torch.cuda.get_device_capability()`；`deepmd`、`ase`、`numpy`、`scipy`、`pymatgen` 版本；DeePMD-kit 后端。
- 是否存在已可工作的 DPA4 LAMMPS 推理示例/日志，核对其 `.pt2` 路径、模型 SHA256、部署与导出环境版本（如果可获取）。
- 仓库状态、分支、未提交更改、远端和最近提交；现有 Phase 2/2.5 单测、`runner_cn.py`、`coordination.py`、增量 PDF 模块、参数 YAML/JSON 的位置。
- 读取 `PDF/PZS_quench_222.cif`、实验 `.gr`；确认实际相同的 896 原子组成、晶胞和坐标，与已有审计一致。
- 在 `/root/autodl-tmp/CODEX` 下寻找 `.pt2`；若有多个候选，**列出候选（路径、大小、SHA256、训练标识）并停止自动选择**，须以确切身份确认一个模型；不能只按 `glob()[0]` 任意选。
- 所有模型推理和结构计算使用**同一模型文件、同一元素 type mapping、同一 PBC/晶胞**；记录环境/模型哈希，禁止默默更换。

建议分支：`feature/phase3-dpa4-hrmc`；若用户仓库已有工作分支规范，遵从现有规范。先跑全部旧回归测试，保留初始通过情况。若存在未提交的用户修改，先保护并说明，不可强制 checkout 或 reset。

---

## 2. Phase 2.5 数学公式：**完全冻结复用**

本节是 Phase 3B/3C 的唯一 PDF+CN 数学真值，来源为用户提供的 Codex `PHASE2_5_COORDINATION_MATH_AND_GEOMETRY_AUDIT.md`，与 `src/dpa4_xray_hrmc/rmc/coordination.py`、`runner_cn.py` 对照。**不要重新定义为 Zhong 的 0.05 容差和 0.9 权重，也不要引入新 PDF 残差归一化。**

### 2.1 实验 PDF 与误差

冻结 Phase 2.5 的正向 PDF：

- Q-dependent X-ray `f0(Q)`、现有 partial RDF → S(Q) → finite-Q `G(r)` 增量实现；
- Q = 0.9–16.0 Å⁻¹，ΔQ=0.05 Å⁻¹；`Biso=0.3 Å²`；`Lorch=false`；RDF `dr=0.01 Å`、`rcut=12 Å`；
- PDF 拟合 r∈[1.5,10.0] Å，沿用现有采样点/权重（均匀权重）、无额外平滑、无基线更改；
- `no_scale`: **s=1，在整个 RMC、FIRE、HRMC 过程中不重新优化**。

\[
X_{\rm PDF}^2({\bf R}) = \sum_{j=1}^{M}\left[G_{\rm calc}(r_j;{\bf R})-G_{\rm exp}(r_j)\right]^2,
\qquad \sigma_{\rm RMC}=0.030\ \mathrm{\AA^{-2}}.
\]

此处 **sum**（不是 mean）且 **仅除一次** \(2\sigma_{\rm RMC}^2\)。报告的 \(R_w\) 使用 Phase 2/2.5 既有定义：

\[
R_w=\sqrt{\frac{\sum_j[G_{\rm calc}(r_j)-G_{\rm exp}(r_j)]^2}{\sum_j G_{\rm exp}(r_j)^2}}.
\]

### 2.2 固定配位 cutoff 与逐中心 CN 误差

\[r_c(\mathrm{P,O})=1.80,\ r_c(\mathrm{Si,O})=1.85,\ r_c(\mathrm{Si,Cl})=2.30,\ r_c(\mathrm{Zr,O})=2.50,\ r_c(\mathrm{Zr,Cl})=3.00\quad(\mathrm{\AA}).\]

在与 Phase 2.5 完全相同的 PBC 最小镜像、边界比较 `<` 的条件下，令 N_P=32、N_Si=64、N_Zr=64：

\[
C_{\rm P}=\frac1{32}\sum_{i\in\mathrm P}[n_{\mathrm{P-O},i}-4]^2,
\qquad
C_{\rm Si}=\frac1{64}\sum_{i\in\mathrm{Si}}[n_{\mathrm{Si-O},i}+n_{\mathrm{Si-Cl},i}-4]^2.
\]

**重要：** P 和 Si 是**按中心原子数取均值**，不是原始平方和；没有另加 ε 容差。一个四配位中心发生 CN=4→3（或 5）的单中心惩罚变化：

- P: \(\Delta C_P=1/32\)，加权惩罚 \(800/32=25\)；
- Si: \(\Delta C_{Si}=1/64\)，加权惩罚 \(2500/64=39.0625\)。

**不得错误地写成 800 和 2500 的单原子变化值。** 若同一移动改动多中心，逐位点差值相加并归一化。

### 2.3 Zr 配位分布及原有自适应 ε（只初始化一次）

对于 total = Zr–(O+Cl) 和 O = Zr–O 两套分布，\(k=0,\ldots,16\)，设初始 MQ 对应直方图为 \(H^{(0)}_k\)，当前为 \(H_k\)：

\[
p^{(0)}_k=\frac{H_k^{(0)}}{64}, \quad
\epsilon_k=\max\left[\sqrt{\frac{p^{(0)}_k(1-p^{(0)}_k)}{64}},\frac1{64}\right],
\]

\[
C_{\mathrm{Zr,total}}=\sum_{k=0}^{16}\left[\frac{(H^{\mathrm{total}}_k-H^{\mathrm{total},0}_k)/64}{\epsilon^{\mathrm{total}}_k}\right]^2,
\qquad
C_{\mathrm{Zr,O}}=\sum_{k=0}^{16}\left[\frac{(H^{\mathrm O}_k-H^{\mathrm O,0}_k)/64}{\epsilon^{\mathrm O}_k}\right]^2.
\]

- ε 来自**初始 MQ** 的目标分布，初始化时确定，后续保持固定；不是每步重新计算；不是 Zhong 的统一 `0.05`。
- 目标总配位：Zr-total `H5=1, H6=63`；目标 Zr–O：`H0=19, H1=23, H2=18, H3=4`，其余初始为 0。
- 对 Zr-total 各 bins 原有 ε 为 1/64=0.015625；Zr–O 的有占据 bins 约 0.0303–0.0600，空 bins 为 0.015625；以现有 `coordination.py` 实际结果为准并生成逐 bin 回归快照。
- 保留原代码对异常/超 bin 配位的处理和显式校验，避免发生范围外 CN 时惩罚被静默忽略。

### 2.4 Phase 2.5 的完整无量纲目标与接受准则

\[
\boxed{
\mathcal L_{2.5}({\bf R})=\frac{X_{\rm PDF}^2({\bf R})}{2\sigma_{\rm RMC}^2}
+800C_{\rm P}({\bf R})+2500C_{\rm Si}({\bf R})
+1.0C_{\rm Zr,total}({\bf R})+0.2C_{\rm Zr,O}({\bf R})
}
\]

对于通过 Hard-sphere 的合法、对称单原子移动：

\[
\boxed{P_{\rm acc}^{(2.5)}=\min[1,\exp(-\Delta\mathcal L_{2.5})]}.\]

`Hard-sphere` 直接拒绝；移动提案每个直角坐标分量 \(U[-d,d]\)；首轮 \(d=0.10\) Å，**不是三维位移模长上界**（最大长度 \(\sqrt3 d\)）。接受概率建议以 `log(u) < -ΔL` 计算；\(\Delta L\le0\) 直接接受；保持 Phase 2.5 原 RNG/位移行为以便退化回归。

**设计原则：** Phase 3A 不用此概率，Phase 3B 的每个 RMC block 完全用此概率，Phase 3C 仅在此基础上添加一个能量差项。三阶段都**不改变**上述四个 CN 项的公式和初始权重。

---

## 3. Phase 3A：DPA4 GPU 推理、能量与力验证

### 3.1 正确加载已微调 `.pt2`

使用 `dpa4_torch211` 环境。优先检查 DeePMD-kit 官方 `deepmd.calculator.DP(model=...)` 的 ASE 支持，并根据**当前实际安装版本**核对 API；不能假定某旧版接口已支持 DPA4 `.pt2`。DPA4 `.pt2` 属于 AOTInductor 编译部署包，可能与导出 GPU 架构、PyTorch/libtorch 版本和编译配置有关。优先采用已经被用户用于 LAMMPS MD 的对应模型/部署栈；如原部署能用而 ASE 接口不能用，先定位版本/接口差异，报告可靠的适配方案。

**强制检查：**

1. `.pt2` 文件绝对路径、大小、SHA256、编译/导出来源（若可得到）；
2. 实际 PZS 模型的 type_map，逐元素映射 Cl、Li、O、P、Si、Zr；**不凭字母顺序推断**；
3. `PZS_quench_222.cif` 读入顺序、元素计数、原子数组与 DPA4 的映射正确；不改动结构坐标和盒子；
4. PBC 全开，周期最小镜像与模型 neighbor 构建一致，盒子/坐标单位 Å，输入顺序正确；
5. 返回的是**整个结构的势能 E_total（eV）**，力 `eV/Å`，必要时应力另记；检查模型有无 per-atom 平均/shift/normalization；
6. 常量零位移、重复同结构调用结果一致；轻微扰动后能量/力有限、没有 NaN/Inf；GPU 真实被使用；
7. 模型只初始化一次，反复更新 ASE Atoms 坐标，不得每个 trial 重建模型对象。

如果现有 `.pt2` 不兼容 RTX 5090：先说明具体错误、CUDA/torch/ABI、模型导出 GPU 信息；检查是否保留同一**已微调模型**的可重新导出 checkpoint。重新导出/重配环境属于另一个明确记录的修复步骤，**不能覆盖原模型，不得擅自训练另一模型**。

### 3.2 模型可靠性测试（不能只测“可运行”）

选结构：

- MQ：原始 `PZS_quench_222.cif`；
- Phase 2.5 `no_scale` 的三条 `final_structure`，若容器中可访问；
- 可选：同策略的 `best_total` 和 `best_pdf` 结构，按标签明确区分；
- MQ 的数个小扰动结构（受约束地生成，不越过极端短接触）；
- 若已有 PZS DFT 验证集，抽取**模型未用于训练**的代表性构型进行能量差、力误差比较；若没有，标记“缺乏独立 DFT 验证”，不要虚构。

对每个输出 `E_total`、`E_total/N`、`ΔE_total=E-E_MQ`、`ΔE/N`、`Force RMS`、`Max |F|`、受力异常位点、模型评估时间、GPU 最大显存。比较相对能量时必须同元素、同原子数、同一模型、相同晶胞边界定义。

**力—能量一致性**：选少数随机原子/方向，以±δ的中心差分计算 `F_iα ≈ -(E(R+δ e_iα)-E(R−δ e_iα))/(2δ)`；试用 1e−2、1e−3 Å 等步长，考虑混合精度数值噪声，避免因为过小步长产生虚假不一致；报告绝对/相对误差、扰动大小和模型精度。若 DPA4 受力 API 存在编译模型限制，记录原因并阻止 FIRE 正式使用。

进行 `N=896` 单点能量评估 benchmark：GPU 同步、排除首次加载/编译时间和预热时间；分别测定 **energy-only** 与 **energy+forces**（若 API 支持），至少 100 次重复或足够稳健的统计；给出 median/p95、GPU 峰值显存、CPU 开销、warmup 时间、总体 wall-time、单位调用吞吐量。不假定 energy-only 一定比 force 快，实际测量。

### 3.3 FIRE 短程试验

从 MQ **复制**结构，以及可选复制一条 Phase 2.5 `no_scale` 终态开展独立短程 FIRE；永不原地覆盖输入。固定晶胞和 PBC，**不做 NPT/晶胞优化**。使用 ASE FIRE 或已经验证的等价实现；推荐先 5、10、20 步对照，收敛阈值/`maxstep`/时间步要显式配置和记录，**这些是试验参数，不是证明已收敛的固定步数**。

每一个 FIRE block 的前后：

- E_total / E_per_atom、Force RMS/Max、实际执行步数；
- 原有 `no_scale` `G(r)`、PDF Rw 与残差曲线（完全同一正向 PDF）；
- P/Si 四配位、Zr 配位分布；P–O / Si–O / Si–Cl 的分键种平均及分布、四面体 O–P–O 和 (O,Cl)–Si–(O,Cl) 键角 RMS、固定中心 CSM、四面体体积；
- hard-sphere 违规数，极短接触、晶胞/成分不变性；
- 弛豫前后原子 RMS/最大位移。

**CSM 定义固定**：之前 Phase 2.5 报告使用固定中心 Kabsch/尺度优化的 CSM，不等同于 pymatgen 默认五点质心 CSM。直接复用既有几何脚本并记录定义；不要把两者数值混在同一张表中。

**Phase 3A Gate 条件**：模型/单位/type_map/PBC可验证；能量与力 finite 并可复现；有限差分检查在实际数值精度范围内合理；FIRE 可运行且不崩溃；能够量化是否改善几何。若 FIRE 明显产生不合理结构或明显异常力，报告而非强行宣布模型有效，停止进入 3B/3C。

---

## 4. Phase 3B：Alternating HRMC（建议先完成）

### 4.1 算法定义（沿用 2.5 数学，不增加能量进 Pacc）

两种操作交替执行：

```text
R ← original MQ
initialize incremental PDF and CN caches
repeat until budget:
    Phase2.5 no_scale RMC:
        attempt symmetric single-atom displacements d=0.10 Å
        hard-sphere filter
        accept by Pacc = min(1, exp[-ΔL_2.5])
        stop this block after 500 ACCEPTED RMC moves
    DPA4 FIRE relaxation:
        run up to 5 FIRE integration/minimization steps at fixed cell
        validate structure; synchronize/rebuild PDF and CN state
        record energy, PDF/CN, geometry and FIRE block diagnostics
end
```

- `500` 指 **被接受的 RMC 移动**，**不是 500 次尝试**；保留尝试计数，若超出 watchdog 上限仍不到 500 接受，不无限循环，要报告停滞。
- `5` 为 **FIRE 最大优化步数**，不是 5 次随机坐标移动；记录实际步数/收敛情况，限制每步最大位移，避免几何异常。
- RMC 每次试探的 \(P_{\rm acc}\) 继续严格执行第 2 节 Phase 2.5 公式；FIRE 用 DPA4 的**力**决定确定性弛豫，**不在 RMC Metropolis 中重复加 ΔE**。
- 当前 RMC `d=0.1 Å` 作为第一轮基准；后续可测试 `d=0.15/0.30 Å`，但必须独立标注，不混入首轮 baseline。

### 4.2 FIRE 导致缓存失效：强制全状态同步

FIRE 可同时移动多个乃至全部原子；现有 2.5 单原子增量 PDF/CN 缓存不能直接继续。每次成功 FIRE 后：

1. 复制/更新全部原子坐标并应用同一周期边界规范；
2. 从当前坐标**完整重建** partial pair histogram、X-ray PDF 状态和全部 CN per-site、Zr hist、CN penalty；
3. 更新 \(\mathcal L_{2.5}\)、`Rw(no_scale)`、当前能量和 best/trace 状态；
4. 比较独立重算的 PDF/CN，检查全量与增量缓存一致性；
5. 确认后继续下一 RMC block；FIRE 后的结构视为新的当前 RMC 状态，不沿用 FIRE 前的总损失。

任何 FIRE 结束后的 NaN/Inf、硬球违规、晶胞/组成不符：**撤销整个 FIRE block，并重新构建原结构全部缓存**，记录失败原因、FIRE step、能量/结构快照和回滚次数；不要只回滚坐标却留下更新后的 cache。是否重试应限定次数并报告，不得在失败状态上继续。FIRE 造成配位变化但未破坏 hard-sphere 时不必自动强行否决；应完整记录 `ΔCN` 和四配位情况，这正是对 DPA4 表现的检测。若违反人为 CN 约束导致高惩罚，可在下一 RMC block 处理，不能不报告地人为改 CN 定义。

**科学说明**：这种交替执行是 refinement/relaxation 非平衡算法，FIRE 步骤并非 Metropolis 核，因此整套流程不能仅凭 RMC 部分满足 detailed balance 就宣称严格采样了 \(\exp(-\mathcal L_{2.5}-E/k_BT)\)。

### 4.3 输出每个 block 的路径与指标

- RMC block 前、RMC block 后、FIRE 后三组结构摘要（其中 RMC/FIRE 对 PDF 与 E 的独立贡献最重要）。
- 每 block attempts、accepted、hard-sphere rejects、Metropolis rejects、CN-break candidates；FIRE steps、energy evaluations、force evaluations、耗时及收敛/回滚次数。
- `G(r)`、Rw、总目标及分项、E、F RMS、P/Si/Zr 配位、键长/角度/CSM。
- `best_pdf`、`best_total` （**明确这是 Phase 2.5 目标，不含能量**）、`lowest_energy`、`final_after_FIRE`、`final_after_RMC`，有些时点可能不存在，要记清阶段。
- 对比初始 MQ、旧 Phase 2.5 no_scale 的相同最终/最佳定义，避免“FIRE 前后 PDF 变化”被平均曲线掩盖。

### 4.4 验证/运行预算

先 1–2 个 FIRE block 的 smoke test；再 10k attempted RMC moves（seed=101）；通过后 3 seeds × 10k/100k（用户同意资源预算后）。若 100k 时仍仅完成部分 block，报告真实次数。暂不做 1M。每 1–5 FIRE blocks 做完整缓存状态与外部重新计算一致性检查；每次 FIRE 后必做全状态同步。

---

## 5. Phase 3C：Inline HRMC（能量直接进入 Metropolis）

### 5.1 完整接受概率（本阶段**唯一**新增的数学项）

对所有通过 Hard-sphere 的对称单原子提案，保留相同第 2 节的 Phase 2.5 PDF 和 CN 项。定义：

\[
\boxed{
\mathcal L_{3C}({\bf R})=\mathcal L_{2.5}({\bf R})+\frac{E_{\rm DPA4}({\bf R})-E_{\rm ref}}{k_BT_{\rm eff}}
}
\]

其中 \(E_{\rm ref}\) 可以取 MQ 总能量以避免大数字，但对**固定组成**的接受比而言会精确抵消；实现上应直接用**总能量差** \(\Delta E=E_{\rm trial}-E_{\rm current}\)，不要在两个巨大的能量偏置值中作数值不稳的处理。

\[
\boxed{
P^{(3C)}_{\rm acc}=\min\left[1,\exp\left(
-\frac{\Delta X^2_{\rm PDF}}{2\sigma_{\rm RMC}^2}
-800\Delta C_{\rm P}-2500\Delta C_{\rm Si}
-\Delta C_{\rm Zr,total}-0.2\Delta C_{\rm Zr,O}
-\frac{\Delta E_{\rm DPA4}}{k_BT_{\rm eff}}
\right)\right]
}
\]

\(k_B=8.617333262145\times10^{-5}\;\mathrm{eV/K}\)。\(T_{\rm eff}\) 是**相对能量项/实验 PDF 项的权衡参数**，不必等于合成温度或室温。Zhong 等使用的 175 K 仅是文献参考，不能自动认为 PZS/DPA4 的最优值。

**下列错误必须防止：**

- 把 energy per atom 的 ΔE/N 直接代替全体系 ΔE；
- 在 PDF/CN 项之外又将同一 CN 惩罚叠加进 DPA4 energy；
- 对 σ 做两次缩放或用 `mean` 替代既有 `sum`；
- 动态优化 scale（当前必须 `s=1`）；
- 把 FIRE 引入 Inline 主采样而称为同一个可详细平衡的 Metropolis 核；
- 修改 proposal 的对称性却不使用必要的提案概率校正。

### 5.2 单个试探的严格顺序

```text
1. choose atom uniformly; propose each Δx,Δy,Δz ~ Uniform(-d,+d)
2. test pair hard-sphere; invalid → reject, no DPA4 call
3. compute incremental trial PDF + CN deltas, without modifying main state
4. call DPA4 for trial total E (model loaded once; GPU)
5. ΔL = ΔL_2.5 + (E_trial - E_current)/(kB*T_eff)
6. if ΔL <= 0 accept; else compare log(U) < -ΔL
7. accepted → commit coords, PDF, CN, current E and counters
   rejected → exact rollback / original state unchanged
8. periodic full PDF/CN/E re-evaluation and checkpoint
```

仅因为 `ΔL_PDF+CN>0` 不可提前拒绝而省略 energy，除非建立**数学上完全等价且严格验证的 delayed-acceptance 方案**；本阶段不实施此优化。`E_current` 在初始化时求一次后缓存，合法试探每次评估 `E_trial`，仅接受后更新当前能量。若 API 刷新 neighbor cache 时对可变 `Atoms` 有隐性状态，需验证拒绝后恢复再次评价能量完全一致。所有尝试保持 PBC/盒子固定，记录是否保持粒子顺序和 type_map 一致。

每 `N` 次接受或尝试通过独立 DPA4 重新计算当前能量，检查 cache drift；每 `N` 步完整重算 PDF/CN，整数 CN 偏差必须为 0；浮点 PDF 差异按原 Phase 2/2.5 回归精度要求，不用人为改成宽松标准。Checkpoint 必须保存当前 E、所有 PDF/CN 缓存、原子坐标、RNG、参数/T、完整模型 hash 和计数器；resume 与连续运行做小链逐步对照（浮点容忍度按实际硬件测量记录）。

### 5.3 能量/位移的 pilot 标定（保留原 Phase 2.5 CN 权重）

从 MQ 用 100–1000 个合法试探抽样 \(\Delta X^2/(2σ^2)\)、四类加权 \(\Delta C\)、\(\Delta E_{\rm total}\) 的分布、分位数与相关性；记录单原子种类、合法率及拒绝原因。

先用 **可配置的** \(T_{\rm eff}\) 扫描（建议候选如 100/175/300 K，必要时根据实际 \(\Delta E\) 调整）和 `d=0.1 Å`；然后在确定的、已冻结的 T 上测试 `d=0.05/0.1/0.2/0.3 Å` 的少量组合。参数均是**探索性候选，不是已确认最佳值**。

- 如果加入能量项使接受率接近 0 或得不到有效构型变化，应停止扩展，调整配置并解释，而不是简单用提高尝试数解决。
- 不以“接受率越高越好”作判据：比较单位墙钟时间的有效位移、局域几何、能量变化以及 Rw 降低速度。
- 优先测 GPU 单次 E 的 median/p95 和实际 `legal_moves/s`。若每次模型 E 推理成本太高，报告 Inline 在现资源下的可行预算；不擅自把全体系能量替换为局域能量、线性近似或低精度代理。

### 5.4 Inline 对照和物理解释

固定 896 原子总数、同一 MQ 初始坐标、同一 PDF+CN 模型和 scale。Inline 理论上在固定参数、对称 proposal、准确能量差与正确 Hard-sphere 限制下，以与 \(\exp[-\mathcal L_{3C}({\bf R})]\) 成正比的目标分布为 detailed balance 基础；但有限步长、强约束、势函数近似、有限采样会影响实际代表性，不能宣称 10k/100k 就达到平衡。

输出 `ΔL_PDF`, `ΔL_CN_P`, `ΔL_CN_Si`, `ΔL_CN_Zr_total`, `ΔL_CN_Zr_O`, `ΔL_E`, `ΔL_total`, 接受率、总能量、PDF、几何和异常事件。给出实际接受/拒绝试探的数值例子与 log 比较，作为可审计证据。

---

## 6. 软件架构及接口建议（以现有仓库结构为准）

先查实旧代码，尽量新增模块，不复制粘贴一套新的 PDF/CN 逻辑。建议结构：

```text
src/dpa4_xray_hrmc/
  rmc/
    ... existing Phase 1/2/2.5 modules ...
    runner_alternating.py       # 3B orchestration
    runner_inline.py            # 3C orchestration
    energy_terms.py             # cached E_total / delta / log-accept math
    hmc_state.py                # proposed vs committed state, sync/checkpoints
  mlip/
    dpa4_backend.py             # ASE/DeePMD API adapter, one-time init, hashes
    fire_relaxation.py          # fixed-cell FIRE wrapper + diagnostics
  analysis/
    ... reuse PDF/CN/geometry analyzers ...
scripts/
  audit_phase3_gpu.py
  validate_dpa4_pzs.py
  benchmark_dpa4_pzs.py
  run_phase3A_fire.py
  run_phase3B_alternating.py
  run_phase3C_inline.py
  compare_phase3.py
configs/
  pzs_phase3_common.yaml
  pzs_phase3A.yaml
  pzs_phase3B.yaml
  pzs_phase3C.yaml
tests/
  test_phase3_energy.py
  test_phase3A_finite_diff.py
  test_phase3B_fire_sync.py
  test_phase3C_acceptance.py
  test_phase3_checkpoint_resume.py
```

可以按项目既有命名修改，但必须保证：

- `EnergyBackend.get_energy(atoms)` 返回总能量 eV；`get_energy_forces(atoms)` 返回 eV 和 eV/Å（统一错误处理），保持 PBC/cell/type_map；
- 3B 使用现有 Phase 2.5 RMC runner 作为内部 block，不自己重写统计和配位；`FIRE` 完成后统一 `rebuild_all_state`；
- 3C 使用现有 2.5 `propose_delta/commit/reject` 路径加 `energy_delta`，不能在不同时刻混用旧/new 坐标；
- 全阶段 `evaluate_structure` 可独立计算 PDF、CN、E、几何，支持初始/终态和 best 的统一后处理；
- 所有出错日志带结构哈希/seed/step/model SHA256，但不输出私密凭证。

### 6.1 配置示意（字段名依照项目配置系统调整）

```yaml
system:
  structure: PDF/PZS_quench_222.cif
  experimental_gr: PDF/PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr  # 实际目录由程序发现后修正
  expected_atoms: 896
pdf:
  scale_mode: no_scale
  scale: 1.0
  fit_r_min_A: 1.5
  fit_r_max_A: 10.0
  qmin_invA: 0.9
  qmax_invA: 16.0
  qstep_invA: 0.05
  histogram_dr_A: 0.01
  histogram_rcut_A: 12.0
  biso_A2: 0.3
  lorch: false
rmc:
  sigma_RMC: 0.030
  proposal_d_A: 0.10
  seed: 101
  hard_sphere: inherit_phase2_5
coordination:
  implementation: inherit_phase2_5
  pair_cutoffs_A:
    P-O: 1.80
    Si-O: 1.85
    Si-Cl: 2.30
    Zr-O: 2.50
    Zr-Cl: 3.00
  lambda_P: 800.0
  lambda_Si: 2500.0
  lambda_Zr_total: 1.0
  lambda_Zr_O: 0.2
mlip:
  backend: deepmd_ase_dpa4
  model_path: /root/autodl-tmp/CODEX/<USER_CONFIRMED_MODEL>.pt2  # 占位符，部署时选定精确文件
  device: cuda:0
  model_sha256: <AUTO_COMPUTED>
phase3B:
  accepted_moves_per_block: 500
  fire_max_steps: 5
  fire_maxstep_A: <PILOT_CONFIG>
phase3C:
  effective_temperature_K: <PILOT_CONFIG>
  kB_eV_per_K: 8.617333262145e-5
output:
  root: results/PZS_phase3
```

**上述 YAML 仅是规范示例，不是可立即执行的现成配置**。`hard_sphere`、物理类型与具体输入路径应按旧仓库配置/API 调整；`<...>` 是必须实际填入的配置占位符。不要让默认路径指向不存在的文件并假装已成功运行。

---

## 7. 测试体系与强制回归（按 Gate 进行）

### 7.1 Phase 2.5 完全退化测试（所有新代码都应通过）

- `energy_weight=0` 或关闭 DPA4，且 3C 仍保留同一 proposal 与 RNG 流时，完整接受概率回到 \(\mathcal L_{2.5}\)；与旧 2.5 测试中相同 seeds/config 的接受决定、PDF/CN 分项、坐标路径一致或在已解释的硬件浮点容差内一致。
- 3B 的 FIRE **禁用/0 步**时，也应退化为相同的 Phase 2.5 RMC（注意 block stop 与预算定义需对齐，禁止随机流偏移造成伪回归失败）。
- 旧 Phase 1/2/2.5 的测试全套继续通过；PDF 增量/完整重算、CN integer histogram 一致性仍满足旧 Gate。

### 7.2 必须新增的单元/集成测试

| 类型 | 测试 | 通过要求 |
|---|---|---|
| 公式 | \(X^2\) 是否 `sum`、σ 单次归一、P/Si `/32` `/64`、Zr ε 固定 | 数学值与 2.5 审计一致 |
| 公式 | CN 单中心 4→3 | P weighted +25；Si weighted +39.0625 |
| Inline | ΔE=0 | 3C 对固定 trial 与 2.5 同概率 |
| Inline | ΔL_total<0 与 >0、log(U) 边界 | 接受/拒绝解析值一致，无 overflow |
| Inline | 非法 hard-sphere trial | DPA4 调用数=0，状态完全不变 |
| Inline | 拒绝后 rollback | atom coords / PDF / CN / E / RNG 后续状态与约定一致 |
| Inline | 有效 E_total 和 E/N 的换算 | 永不误用 E_per_atom 进入指数 |
| PBC | 跨边界移动 / 重新包装 | DPA4 与 RMC 使用一致晶胞、原子顺序 |
| FIRE | 改动全体坐标后重建缓存 | 全量 PDF/CN 对比一致，integer drift 0 |
| FIRE | FIRE 失败/产生非法结构 | 回滚坐标+缓存+计数与快照完整一致 |
| Checkpoint | 停机→resume vs 连续 | 小链的坐标/目标项/能量/RNG 可复现（记录容差） |
| GPU | 同结构重复 E/F、CUDA 同步性能 | 数据 finite、计时准确、没有逐步重新加载 |
| Geometry | 固定中心 CSM、角 RMS、体积 | 同一 Phase 2.5 后处理定义，独立参考校验 |

**性能测试和正确性测试分开**；不要为了省 GPU 推理次数关闭强制一致性测试，反之也不要把所有 GPU 同步插入热循环并以此报告生产性能。

---

## 8. 运行矩阵与性能上限（先短程，后决策）

| Gate | 初始结构 | 首轮规模 | 作用 | 是否立即执行 |
|---|---|---|---|---|
| 0 | — | 仓库、环境、输入审计 | 保护旧代码/文件 | 是 |
| 3A | MQ，及既有 2.5 结构用于离线单点 | 少量单点、100 次推理 benchmark、5/10/20 FIRE | 模型可用性/力和速度 | 是 |
| 3B smoke | MQ | 1–2 FIRE blocks | 缓存同步/回滚 | 3A 通过后 |
| 3B pilot | MQ | 10k attempts，seed101；后续 3 seeds × 10k | 算法与 Rw/E/几何趋势 | Gate B 通过后 |
| 3B 扩展 | MQ | 3 seeds × 100k attempts | 短程方法比较 | 资源允许、测试合格后 |
| 3C smoke | MQ | 100–1000 合法试探、20–100 接受移动 | PDF/CN/E/Metropolis 校验 | 3A 通过后 |
| 3C pilot | MQ | 参数网格，每组 ≤10k attempts | d/T 标定与性能 | Gate C 通过后 |
| 3C 扩展 | MQ | 3 seeds ×100k attempts（仅可负担时） | 同尺度比较 | benchmark 之后再决定 |
| 3B/3C 正式 | MQ | 1M × 3 seeds 等 | 科学正式对照 | **本阶段不自动运行** |

对于 3C：先用 Phase 3A 的 GPU 单次能量调用时间预测 10k、100k、1M 尝试成本，结合合法率，不得未经用户同意把 1M ×多个 seeds 自动展开。可使用单卡多 CPU 线程进行 PDF 计算，但**不要让多个独立 CUDA 进程争用同一 5090 导致假速度评估**；若要批量/并行推理需另做正确性与显存审计。

### 8.1 位移与能量参数的可比性

第一轮所有方法统一 RMC `d=0.1 Å`，以隔离引入 DPA4 的影响；在独立参数敏感性试验中测试其它 d，不能把 `d=0.1` 结果与 `d=0.3` 结果直接归因于 MLIP。使用相同 seeds（101、202、303）仅表示实验设计配对，并不意味着三方法之后应产生相同坐标轨迹。

---

## 9. 输出结构、指标、统计与解释要求

### 9.1 统一科学评价表

| 指标 | 初始 MQ | Phase 2.5 no_scale | Phase 3B | Phase 3C |
|---|---:|---:|---:|---:|
| Rw(no_scale, s=1) | 实际重算 | ≈ 0.12351（3 seeds 平均，仅校验） | 计算 | 计算 |
| DPA4 E/N、ΔE/N | 计算 | Phase 3A 计算 | 计算 | 计算 |
| P–O 与 Si–(O,Cl) 四配位率 | 100% / 100% | 100% / 100% | 计算 | 计算 |
| P–O 键长均值与分布 | 1.5441 Å（审计参考） | ≈1.5863 Å（审计参考） | 计算 | 计算 |
| Si–O、Si–Cl 键长分布 | 计算 | 计算 | 计算 | 计算 |
| O–P–O angle RMS | 3.672°（审计参考） | 14.060°（审计参考） | 计算 | 计算 |
| P 固定中心 CSM | 0.359（审计参考） | 4.304（审计参考） | 计算 | 计算 |
| Si 固定中心 CSM | 0.601（审计参考） | 4.440（审计参考） | 计算 | 计算 |
| Zr-total / Zr–O histogram | 计算 | 计算 | 计算 | 计算 |
| P/Si–配体 topology / shared ligands | 计算 | 计算 | 计算 | 计算 |
| 试探合法率、Metropolis 接受率 | — | 从旧日志读取 | 计算 | 计算 |
| GPU E calls / F calls、耗时 | — | Phase 3A | 计算 | 计算 |

**注意**：初始 MQ 与 Phase 2.5 的几何指标只是现有审计核对参考，务必在相同脚本/cutoff 下实际重算并报告 seed 标准差；不要将平均角度与角度 RMS 混淆。尤其 P/Si 的几何误差不能只对幸存四面体做不加说明的比较；如果 3B/3C 四配位率下降，统计所有位点的 CN 分布，并把几何分布的选择偏差写明。

建议加：`G(r)` 对比图（实验黑线、MQ、2.5、3B、3C）、差值曲线、`PDF residual vs attempts/walltime`、`ΔE/N vs attempts`、`Angle RMS/CSM vs attempts`、`RMC/FIRE block-resolved`、`acceptance vs species`、`energy-vs-PDF tradeoff`，所有图使用明确的 scale、是否单结构和是否跨 seed 平均的标签。

### 9.2 目录组织（大模型/大 checkpoint 留在 AutoDL 数据盘）

```text
results/PZS_phase3/
  phase3A/
    environment_manifest.json
    model_audit.json
    single_point_energy_force.csv
    force_fd_check.csv
    gpu_benchmark.json
    fire_comparison.csv
    structures/  plots/
    PHASE3A_REPORT.md
  phase3B/
    configs/ seed_101/ seed_202/ seed_303/
    block_trace.csv
    rmc_trace.csv
    energy_geometry_trace.csv
    checkpoints/            # 通常数据盘或 gitignored
    structures/              # initial, block-before, block-after, fire-after, final, best
    plots/
    PHASE3B_REPORT.md
  phase3C/
    configs/ pilots/ seed_101/ seed_202/ seed_303/
    acceptance_components.csv
    energy_geometry_trace.csv
    gpu_timing.json
    checkpoints/ structures/ plots/
    PHASE3C_REPORT.md
  comparison/
    summary_per_seed.csv
    summary_aggregate.csv
    geometry_comparison.csv
    pdf_energy_tradeoff.png
    PHASE3_FINAL_REPORT.md
```

每条 run 的 `manifest` 至少含：Git commit、ISO timestamp、代码/环境版本、完整配置 hash、输入 CIF 和 `.gr` SHA256、`.pt2` SHA256、模型 type_map、GPU/CPU 信息、seed、起始结构 SHA256、输出结构阶段（MQ/final/best）、执行步数与真实 wall-time。`checkpoint` 保留必须足以恢复的状态；对性能大文件/模型可用路径引用 + SHA256，不复制到 Git。

---

## 10. 验收 Gate、失败条件、提交与报告格式

### Gate 0 — 基线

- [ ] Git repo/输入/模型/环境可定位；旧测试通过；模型唯一且记录 SHA256；未破坏已有文件与用户修改。

### Gate A — 模型与 FIRE

- [ ] `.pt2` 在 RTX 5090 可稳定加载/推理；能量/受力单位、type_map、PBC 确认；有限差分及重复性报告齐全。
- [ ] 完成 MQ + 现有 RMC 单点评估，5/10/20 步 FIRE；GPU time per call、显存及初始 geometry 分析齐全。
- [ ] 模型异常时 STOP：不得自动进入 3B/3C。

### Gate B — Alternating

- [ ] 500 **accepted** + 5 FIRE 定义正确；每 FIRE 后完整重建/核对 PDF+CN；失败能安全回滚。
- [ ] 旧 RMC 完全退化测试通过；至少完成 1–2 blocks smoke + 短程 pilot；每 block 记录 E/PDF/CN/geometry。
- [ ] 3B 用同一 MQ 和 s=1 起点，并报告与 Phase 2.5 的可比指标。

### Gate C — Inline

- [ ] 第 5 节数学接受公式经解析、随机数、ΔE=0、CN=0 权重等对照单测；无重复 σ / E/N 错用。
- [ ] energy cache/rollback/checkpoint & resume 有实际测试；GPU energy calls 与 timing 统计真实。
- [ ] 已报告 T_eff × d pilot、接受率、能量/PDF 各项规模与方法的现实算力成本。

### Gate D — 方法对照

- [ ] 初始 MQ、Phase 2.5 no_scale、3B、3C 的 PDF、CN、键长/角度/CSM、DPA4 能量与耗时能够公平对比。
- [ ] 至少明确回答：是否保持 PDF 拟合？能否减少四面体畸变？Zr 结构是否有合理重组？两种 HRMC 谁更高效？有无能量/配位过约束？
- [ ] **阶段评估不要求 3B/3C 已达到统计收敛**，明确标记 pilot 结论、局限和未运行项目。

提交安排：Gate 0/3A、3B、3C 各自独立 Git commit（若分支有既存规范按规范），附 `PHASE3A_REPORT.md`、`PHASE3B_REPORT.md`、`PHASE3C_REPORT.md` 和 `PHASE3_FINAL_REPORT.md`。每份报告都写明：实际代码路径和函数、公式、真实运行命令、运行预算、验证结果、数值与异常、性能、输出文件目录、commit ID、尚存风险。**只要一个 Gate 失败，真实报告失败与修复方案，不得越过 Gate 声称已完成 Phase 3。**

---

## 11. Codex 首次执行指令（可直接复制）

> 请严格依据本文件在现有仓库开发 Phase 3。用户已确认 `.pt2` 是 PZS 微调模型，所有 3B/3C 初始结构固定为原始 `PDF/PZS_quench_222.cif`，第一轮仅 `no_scale`；**Phase 2.5 所有 PDF/CN 数学公式、cutoff、归一化、权重和 σ 原样继承**。先做 Gate 0 和 Phase 3A，检查 GPU、环境、模型 type_map、单点 E/F、有限差分及 FIRE 结果，生成 `PHASE3A_REPORT.md`；3A 成功后再实现和验证 3B/3C。首轮仅运行规定 smoke/pilot，不自动执行百万步，必须保留原 2.5 回归。不得在仓库或日志写 SSH 密码，不得覆盖模型/用户文件。碰到模型不兼容、物理异常或关键输入缺失，停止相应 Gate，并报告复现命令和具体证据。

---

## 12. 文献、原始资料与实现依据

### 本项目内部实现依据

1. 用户提供的 `phase2_final_report.md`（Phase 2 增量 PDF 与 no_scale 拟合定义）。
2. 用户提供的 `REPORT.md`（Phase 2.5 1M ×9 链结果、CN cutoff/权重、P/Si 100% 四配位）。
3. 用户提供的 `PHASE2_5_COORDINATION_MATH_AND_GEOMETRY_AUDIT.md`（实际 `runner_cn.py` 数学公式、Zr ε、几何对照、固定中心 CSM 定义）。
4. 当前代码 `src/dpa4_xray_hrmc/rmc/runner_cn.py`、`coordination.py` 与实际仓库内 PDF/硬球/配置模块（具体行号以本次 checkout 版本核查为准）。

### 文献与官方实现文档

5. X. Wang, X. Zhang, D. Butenko, B. Hong, J. Zhu, P. Zhong, *A practical route to amorphous solid-state electrolyte structures by hybrid reverse Monte Carlo with foundation potentials*, arXiv:2610.19846 (2026), https://arxiv.org/abs/2610.19846 。Methods II.D/II.E：RMC Metropolis、配位约束、Inline（能量项）、Alternating（RMC+短程 FIRE）。其 `d=1.0/0.3/0.15 Å`、`T=175 K` 为参考，不是本项目冻结参数。
6. DeePMD-kit DPA4 文档： https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html （DPA4/SeZM、`.pt2` AOTInductor 编译部署、设备与运行选项）。
7. DeePMD-kit ASE `DP` calculator API： https://docs.deepmodeling.com/projects/deepmd/en/latest/autoapi/deepmd/calculator/index.html ；ASE 使用指南：https://docs.deepmodeling.com/projects/deepmd/en/latest/third-party/ase.html 。
8. DeePMD-kit Backend 文档：https://docs.deepmodeling.com/projects/deepmd/en/latest/backend.html （注意与当前安装版本对应，不能凭最新网页强行安装不兼容依赖）。

**最终要求：** 真实可运行、数据可追溯、旧数学不变、物理诊断完整、GPU 算力成本清楚，先给出可信的 Phase 3 短程对照，再决定大规模正式 HRMC。
