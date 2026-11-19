# Phase 2.5 — PZS 配位数约束 Pure RMC：Codex 完整开发说明

> **执行角色：** 你（Codex）已经连接 AutoDL CPU 容器和现有 GitHub 仓库；请在**现有 Phase 2 代码**上增量开发，不从头重写 X-ray PDF 或 RMC。
>
> **任务核心：** 保留 Phase 2 的 3 种全局 scale 策略，给 Pure RMC 加入基于固定原子对距离截断的、可增量维护的配位数约束（P–O、Si–O/Cl、Zr–O/Cl），执行与既有无 CN 结果配对的受控对照。
>
> **禁止事项：** 不使用 DPA4/MACE/其他势函数或能量项；不加入角度项/刚性多面体项到接受准则；不得在 MC 主循环调用 `pymatgen` 的 `CrystalNN`、`ChemEnv` 或全局 Voronoi；不得覆盖 Phase 2 结果；不得硬编码未核实的配位截断距离或宣称“计算了百万步”而未实际完成。

## 0. 已确认的实验与运行条件

- 初始 MQ：Windows 原路径 `E:\1_CODEX\dpa4-xray-HRMC\PDF\PZS_quench_222.cif`；AutoDL 容器中从当前仓库/配置寻找对应文件。**不要假设 Windows E: 路径可在 Linux 访问**。若容器无此数据，只给出缺失文件诊断，保留项目现有隐私与文件同步策略，不尝试猜造数据。
- 原子数 896：Cl512 Li96 O128 P32 Si64 Zr64，32 × Li3PO4Zr2Si2Cl16。晶胞、元素排列、PBC 不变。
- 实验：先前的 `PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr`；保持 Phase 2 对该 `.gr` 的读取方式。拟合 `1.5 ≤ r ≤ 10 Å`。
- **冻结 PDF 前向模型**：Phase 2 的 Q-dependent X-ray f0(Q)，Q 网格 0.9–16.0 Å⁻¹ / 0.05 Å⁻¹（实验头 16.0056 Å⁻¹，沿用此前已验证的实现），Biso=0.3 Å²，Lorch=False，dr=0.01 Å，rcut=12 Å，浮点精度 float64。不要趁本阶段改变 Q 范围、RDF 核、基线或展宽。
- 移动：单原子，每个笛卡尔分量 Uniform(-0.10,+0.10) Å；继承 Phase 2 的各原子对 Hard-sphere 截断、周期最小镜像与 Metropolis 参数 `sigma_RMC=0.030 Å^-2`；保持移动提案对称。
- 三种 scale 模式和既有代码接口严格保留：
  1. `no_scale`：s=1；
  2. `fixed_initial`：使用同一原始 MQ 的最佳正 scale `s0`，之后固定；
  3. `bounded_dynamic`：每一构型解析优化 `s* = B/A`，裁剪在 `[0.8*s0, 1.2*s0]`。
- Phase 2 报告中的 `s0=0.6357769831752518` 仅作为回归测试参考，**必须重新从当前原始 MQ + 实验数据计算并核对**，不可硬编码。
- 随机种子：101、202、303；每模式 10k 功能测试、100k 测试、1M 正式 × 3 seeds；保持与 Phase 2 一样的采样预算与可能的随机流组织。AutoDL 32-vCPU，可并行但应避免线程超额占用。已有 GitHub 同步继续沿用。

## 1. 为什么做 Phase 2.5

Phase 2 报告 `phase2_final_report.md` 显示：没有配位约束时，固定 scale 方案的最终 P 四配位比例只有 13.5%，`bounded_dynamic` 为 40.6%，而初始 MQ 为 100%；虽然总 X-ray PDF 残差已经降低，却伴随局域化学结构劣化。增量 PDF 则已达到约 1.945 ms/提案、相对完整重算约 120.6 倍，最大增量-完整 G(r) 误差约 2.07×10⁻¹³ Å⁻²。

目标：**测试增加有限、透明的化学配位先验后，三种 scale 方案能否同时提高 PDF 拟合与局域结构可信度，并量化新增计算成本。** 不将“优于 Phase 2 的 PDF 拟合”作为唯一成功标准。

## 2. 核心设计：先做 MQ 配位审计，后冻结配位定义

### 2.1 离线配位审计（必须先完成；不能跳过）

对原始 `PZS_quench_222.cif`，在现有 AutoDL Python/conda 环境中：

1. 基于 PBC 计算并绘制（至少）P–O、P–Cl、Si–O、Si–Cl、Zr–O、Zr–Cl 的 partial RDF，以及邻近第一壳层的距离分布、累计配位数随 r 的曲线；对每种 pair 指出选定第一谷值/截断距离及选择依据。
2. 并行进行**离线** `pymatgen.analysis.local_env.CrystalNN` 局域近邻审计，指定版本/参数，并对 O、Cl 配体分类。可以进一步对代表性 P、Si、Zr 多面体用 ChemEnv/连续对称性度量做离线评估，但不要求在每一步 MC 中使用。CrystalNN 对畸变非晶配位的判断并非“真值”，只作对照。
3. 建立固定 pair-specific cutoffs：`r_c(P,O)`、`r_c(Si,O)`、`r_c(Si,Cl)`、`r_c(Zr,O)`、`r_c(Zr,Cl)`；以 Å 写入审计 JSON 和统一配置文件，禁止单一通用半径、禁止运行过程中根据结构动态修改 cutoff。**不要直接把 Zhong 论文的 Nb–O=2.92 Å 用作本体系的任何 cutoff。**
4. 对各阳离子逐个列出：P–O 配位、Si–O、Si–Cl、Si–(O,Cl) 总配位、Zr–O、Zr–Cl、Zr–(O,Cl) 总配位；输出每种配位数的计数与比例及 CSV/JSON。对 Si 必须确认是否有 64/64 个四配位（或说明例外、原因）。
5. 基础自检：此前 Phase 2 计算初始 P–O 均值 4.000 / 四配位比例 100%；Si–O 均值 0.719；Zr–O 1.109、Zr–Cl 4.875（Zr 总均值约 5.984）。这些数字来自**之前的截断定义**，不一定与新的审计 cutoff 完全相同。若差异明显，列明 cutoff 原因、可复现比较和影响，**不要为了匹配旧数值调 cutoff**。
6. 若 Si–(O,Cl) 的四配位比例明显低于预期，或者某一 pair 第一谷值难以唯一确定：停止正式 1M 计算，生成审计报告、两个以上有依据的 cutoff 候选及敏感性测试。可继续模块开发与小测试，但不得强行把错误的初始结构假设写成真实物理约束。

**重要**：最初 Si–O 平均 CN=0.719，不意味着 Si 是不配位。对于混合氧氯化物，约束必须是 **Si–(O,Cl) 总四配位**，不能错误地约束为 Si–O=4。P 以 P–O=4 为目标，允许作为诊断记录 P–Cl 但不默认强行固定。

### 2.2 目标约束的物理定义

| 约束对象 | 定义 | 目标 | 类型 |
|---|---|---|---|
| P | 每个 P 的 O 邻居数 | `CN_P-O(i)=4` | **强的逐原子软约束** |
| Si | 每个 Si 的 O + Cl 邻居数 | `CN_Si-O(i)+CN_Si-Cl(i)=4`（审计通过后） | **强的逐原子软约束** |
| Zr total | 全部 Zr 中 `CN_Zr-(O,Cl)=n` 的分布 | 原始 MQ 的 `p0_total(n)` | **中等群体分布软约束**，不固定每个 Zr=6 |
| Zr–O | 全部 Zr 的 `CN_Zr-O=n` 的分布 | 原始 MQ 的 `p0_O(n)` | **弱群体分布软约束**，防止 O/Cl 配体组成失控 |

Zr 应允许不同配位数及配体组成，**不得用“全部 Zr 六配位”替代 MQ 分布**。Zr total 与 Zr–O 有信息重叠，权重需要防止重复过度惩罚；应进行只使用 Zr total 的简短消融对照。

## 3. 建议的数学目标及 Metropolis 实现

### 3.1 PDF 损失（完全复用 Phase 2）

以 `r=1.5–10 Å` 上统一点权重 w_i=1 为例：

\[
X_{\rm PDF}^2(\mathbf R,s)=\sum_i w_i [sG_{\rm calc}(r_i;\mathbf R)-G_{\rm exp}(r_i)]^2.
\]

Phase 2 的 3 scale 规则不变；`bounded_dynamic` 的 s 是以**当前构型**计算 `B/A` 后裁剪，而不是只在接受后更新。对所有模式使用同一个原始 MQ `s0` 与相同硬球规则。报告裁剪前的 `s_raw=B/A`、裁剪后的 `s` 和边界饱和率。

### 3.2 配位软约束（无量纲，新增）

定义

\[
C_P=\frac1{N_P}\sum_{i\in P}\left[CN_{P-O}(i)-4\right]^2,
\quad
C_{Si}=\frac1{N_{Si}}\sum_{i\in Si}\left[CN_{Si-O}(i)+CN_{Si-Cl}(i)-4\right]^2.
\]

Zr 分布：

\[
p_t(n)=N_{\rm Zr,totalCN=n}/N_{\rm Zr},\quad
p_O(n)=N_{\rm Zr,OCN=n}/N_{\rm Zr}.
\]

\[
C_{Zr,t}=\sum_{n\in\mathcal N_t}\big[(p_t(n)-p_{t,0}(n))/\epsilon_{t,n}\big]^2,
\quad
C_{Zr,O}=\sum_{n\in\mathcal N_O}\big[(p_O(n)-p_{O,0}(n))/\epsilon_{O,n}\big]^2.
\]

- `n` 的范围必须包含初始及试探过程中出现的所有 CN（不能因 n 超出初始分布就丢失惩罚）。实现可用足够宽的非负整数 bins，或延伸计数并显式惩罚尾部。
- `epsilon` 是人为定义的**容差/尺度**，不是实验测量误差。首轮可参考 Zr 数目 64 导致的离散分布统计尺度，并设置有限下限，避免初始 p0(n)=0 时除零或非常强的非物理惩罚。记录采用的公式与数值。
- 记录分项 `C_P,C_Si,C_Zr,t,C_Zr,O` 及其加权贡献。

以 Phase 2 的 Metropolis 写法为准，统一定义**无量纲负对数目标**：

\[
\boxed{\mathcal L(\mathbf R)=X_{\rm PDF}^2(\mathbf R,s(\mathbf R))/(2\sigma_{\rm RMC}^2)
+\lambda_PC_P+\lambda_{Si}C_{Si}+\lambda_{Zr,t}C_{Zr,t}+\lambda_{Zr,O}C_{Zr,O}.}
\]

\[
P_{\rm acc}=\min\left(1,\exp[-(\mathcal L_{\rm trial}-\mathcal L_{\rm current})]\right)
\]

这里假设 Phase 2 的 PDF 残差 `X²` **还没有除以 sigma²**；Codex 需先检查现有代码，防止重复除以 `sigma_RMC²`。若现有实现使用了不同归一化，应显式写出等效形式并通过 `CN_weight=0` 的完全复现测试。

注意：逐个构型优化 s 的 `bounded_dynamic` 是 profile-loss，不等于对未知 s 做严格贝叶斯边缘化；这里只沿用 Phase 2 的一致定义。

### 3.3 配位权重：通过 pilot 标定，不能凭空宣布固定文献数字

**不能直接照搬 Zhong 论文 Nb–O CN 分布约束的权重 0.9、容差 0.05**。他们的目标函数尺度、配位对象和模型不同。

建议流程：

1. 在完成配位审计后，先用 `CN_weight=0` 做 100–1000 个相同试探，保证与 Phase 2 逐步输出一致。
2. 为 `lambda_P`、`lambda_Si`、`lambda_Zr,t`、`lambda_Zr,O` 提供配置项；先在固定 seed 的 10k trial 上做**少量共同候选权重组合**，例如强/中/弱三个相对等级。记录每类破坏配位事件的 `Delta L_CN` 与典型 `Delta L_PDF`，避免一个项数量级独占接受准则。
3. 优先要求初始 100% 四配位的 P 和（如果审计证实）Si 在正式结果中各保持 **≥95% 四配位**。这是本项目的**工程验收阈值，不是文献经验定律**。针对 P 的 32 个位点，每失去一个四配位位点比例变化 3.125%；Si 的 64 个位点，每失去一个变化 1.5625%。
4. 允许 Zr 各位点 CN 转变；其总/氧配位分布不应无故大幅偏离原始 MQ，报告 `L1`、`L2` 距离及逐项 histogram，而不是强求逐原子同一配位。
5. 用同一套权重跑三种 scale 模式 × 至少 pilot seeds，在**看完 10k 结果后冻结权重、截断和接受参数**。禁止为每种 scale 单独调权重到最有利的结果；若共同权重达不到验收阈值，报告失败和权衡，追加一次文档化的全局 pilot，不在正式 1M 过程中在线调权重。
6. 对于 Zr–O 的附加项，至少做一次 `lambda_Zr,O = 0` 的对照（短 pilot 即可），检查它是否过度保留 MQ 配体比例。

## 4. 增量 CN 引擎（性能重点）

必须复用 Phase 2 单原子移动前/后的局域距离扫描，以及周期最小镜像逻辑。禁止每次完整构建 pymatgen Structure；禁止使用 CrystalNN/ChemEnv 判断热循环里的配位。

### 4.1 数据结构

- `pair_cutoffs`: 五个固定原子对截断，含 `P-O`, `Si-O`, `Si-Cl`, `Zr-O`, `Zr-Cl`。
- `cn_per_cation`: 每个 P、Si、Zr 的整数 CN（Zr 保留 O 和 Cl 分项）。
- `zr_hist_total`、`zr_hist_o`: 全部 Zr 的 CN 分布计数，避免每步遍历 64 个 Zr。
- `cn_loss_components` 和 `current_total_loss`，可在接受后增量更新。
- proposal-local delta（在接受前**不修改主状态**），可单独提交/回滚。

### 4.2 单次移动要求

移动原子 a 前后，对于与 a 有关、满足某一受约束 pair 类型的所有 b，比较 `d_old < r_c(pair)` 与 `d_new < r_c(pair)`：

- 如果不跨越 cutoff：该对 CN 不变。
- 如果从壳内变为壳外：对应阳离子 CN -1。
- 如果从壳外变为壳内：对应阳离子 CN +1。
- 若 a 是 P/Si/Zr，被移动原子自己的 CN 也要同步更新；若 a 是 O/Cl，**可能影响多个** P/Si/Zr 位点，不可只更新一个。
- 每个无序原子对只计一次；同一物种的周期镜像、PBC 边界、恰好位于 cutoff 时的 `<` 判定必须一致且可测试。
- 对 Zr，更新受影响 Zr 位点的前后 CN 及分布 histogram（先撤销旧 bin，再添加新 bin）。
- 与 PDF 增量变换可以共享距离数组；但要保证 CN 的 cutoff 不是错误地被 PDF 的 rcut=12 Å 限制（这五个近邻 cutoff 应明显小于 12 Å）。
- 拒绝时 position、PDF histogram、`G(r)`、CN 数组、Zr hist、score、scale、RNG 状态的处理遵守已有 proposal/accept/reject 设计；不允许产生状态漂移。对称提案时保留 Metropolis 原则。

### 4.3 必须通过的正确性测试

- 初始 CN vs 独立直接 O(N²) 距离/最小镜像完整重算：**每个阳离子的整数 CN 完全相同**，Zr histogram 完全相同。
- 随机 100–1000 个移动尝试；10k 个混合接受与拒绝后，以及各 1M 链运行时每 10k–50k 次：对当前结构完整重算 CN，与增量状态要求**完全一致（整数误差 0）**。
- 专项测试：O/Cl 移动造成多个阳离子 CN 改变；P、Si、Zr 中心移动；一次移动跨越多个 cutoff；cutoff 正好附近；跨 PBC；位于不同 pair 类型；拒绝状态不可变；重启恢复增量 CN 和随机数状态。
- `CN_weight=0` 与 Phase 2 原引擎，在相同配置、固定随机流下应保持相同轨迹/接受判定（若代码改动导致浮点运算顺序变化，需给定清楚数值容差并说明）。
- PDF 增量验证仍必须通过，最大误差维持 Phase 2 同数量级；所有中间和最终结构无 Hard-sphere 违规。

## 5. 采样计划及公平比较

### 5.1 运行矩阵

| 阶段 | 模式 | seeds | 每链尝试数 | 目的 |
|---|---|---|---:|---|
| 单元/功能 | 三种 | 101 | 10,000 | 测试输入/输出、增量 CN、loss、checkpoint |
| 性能/初步结果 | 三种 | 101/202/303 | 100,000 | 对照 Phase 2 速度和 CN 分布，冻结参数 |
| 正式 | 三种 | 101/202/303 | 1,000,000 | 9 条各 1M 的科学对照 |

若权重需要 pilot 探索，额外记录所有调参运行并与冻结参数后的 100k/1M 正式对照严格区分。**三种模式共用**：初始 MQ、PDF/实验/Q/grid、截断半径、硬球约束、sigma、提案步长、CN 权重、重复种子与尝试预算。不要用 Phase 2 的最终 RMC 结构作为新一轮起点；Phase 2.5 从原始 MQ 重启。

对比旧 Phase 2 数据时采用“相同 seed / 相同 mode”配对；接受序列因 CN 改变而不同是正常的。保留 Phase 2 原始结果作无 CN 基线。

### 5.2 结果对照方式

**PDF 公平对照**：每个最终结构都报告 4 种独立评估：`Rw(no_scale)`、`Rw(fixed s0)`、`Rw(bounded)`、`Rw(shape_only unconstrained positive)`；且对 1.5–4.5 Å 与 4.5–10 Å 报告分别的 residual RMS。不可只比较各自按其优化目标得到的 Rw。

**Scale 监控**：报告 `s0`、当前 `s_raw`、`s_clipped`、上下限触发比例、随步数演化。Phase 2 动态模式大量抵达 scale 上限，需要检查加 CN 后是否仍饱和。

**结构与化学检查**：
- 每个 P/Si 的 CN、四配位保留百分比、四配位破坏事件次数与恢复次数；Zr–O、Zr–Cl、Zr-total CN histogram，配位分布变化；P–O、Si–O、Si–Cl、Zr–O、Zr–Cl partial RDF。
- 键长分布：P–O、Si–O、Si–Cl、Zr–O/Cl；几何：O–P–O、(O/Cl)–Si–(O/Cl) 键角分布，以及四面体畸变/CSM；若仅有 CN=4 而键角异常，明确报告。CN 约束不能自动保证四面体形状合理。
- 必须记录未被约束的 Li 配位变化、P–Cl 变化、结构密度/晶胞、成分、最大/RMS 位移、最短接触以及是否有不合理的局域键长。
- 初始、终态、按总目标 `L` 最优结构、按 PDF-only 最优结构**分别保存**，不混为一谈。
- 离线 CrystalNN 对初始/终态（必要时各 seed 的 best）做交叉验证；对不一致位点列出具体原子索引、近邻、距离。使用 ChemEnv/CSM 时记录版本与参数、未能分类的比例。不在 MC 热循环调用 pymatgen。

**性能**：同时计时并报告 `hard-sphere`、距离扫描/`PDF delta`、`CN delta`、`score PDF`、`score CN`、`commit/reject`、`checkpoint`；单任务与并行负载分别 benchmark，报告 attempts/s、accepted/s、接受率、p50/p95、peak RSS。与 Phase 2 无 CN 的相同环境/资源/控制线程数量做公平比较。**增量 CN 的开销应量化，不能凭理论宣称“几乎零成本”。**

### 5.3 结果总结必须回答的科学问题

1. 3 种 scale 策略加入 CN 后，与各自无 CN 的 PDF Rw 改变多少？
2. 3 种策略是否保留 ≥95% P–O 和 Si–(O,Cl) 四配位？如没有，哪些位点失败，损失分项和接受率如何？
3. Zr 的总 CN 分布是否仍允许不同配位数？Zr–O/Cl 配体变化幅度如何？是否把 MQ 的假设过度固定？
4. 四面体 CN 保护后，P/Si 键角与键长是否仍不合理？
5. 动态 scale 是否继续触顶？它是否真正改善形状匹配而不是吸收整体振幅？
6. 新增增量 CN 引擎对单步延迟及整链吞吐量的成本是多少？
7. CN 限制是否让 RMC 接受率过低或导致优化停滞？报告可选参数敏感性而非偷偷调参。

## 6. 项目实现、API 与提交要求

- **先检查仓库**：识别 Phase 2 的 config/runner/增量 PDF 模块。优先建立 `feature/phase2-5-cn-constraints` 分支（如果现有工作流有其他分支标准则遵从），不要改动 Phase 1 原物理模型；适配 Linux AutoDL 路径与已存在 conda 环境，必要时仅增加 `pymatgen` 的离线分析依赖。
- 建议新增独立模块：`src/.../coordination.py`（完整与增量 CN）；`scripts/audit_pzs_coordination.py`（RDF、CrystalNN、cutoff 审计）；`scripts/validate_incremental_cn.py`；`scripts/benchmark_cn_rmc.py`；`configs/pzs_phase2_5_cn.yaml`；`scripts/run_phase2_5_pzs.py`；`scripts/compare_phase2_5_results.py`。具体命名按既有代码架构调整，保持接口兼容。
- `coordination.py` 必须有 `full_recompute(state)`, `propose_delta(state, moved_idx, new_pos, old_new_distance_context)`, `commit(proposal)`, `reject(proposal)`, `compute_cn_loss()` 等可单测行为；接口可按仓库风格调整。
- 记录 cutoff 来源、CN 权重、实验与输入文件 hash、Git commit、环境依赖版本、各随机种子、统计配置；checkpoint 必须保存 CN state、Zr hist、scale 相关量与 RNG state，并执行真正的 resume 验证。
- 建议所有成果放在独立 `results/PZS_phase2_5/`（数据磁盘下的大规模 checkpoints/snapshots 不放 Git）。保留每条链的 `config.json/yaml`、`manifest.json`、`trace.csv`、`timing.json`、`best_total.extxyz/cif`、`best_pdf.extxyz/cif`、`final.extxyz/cif`、`coordination_distribution.csv`、`pdf_comparison.dat/png`，以及汇总 `comparison_summary.csv`、`comparison_statistics.json`、`phase2_5_final_report.md`。
- 输入原始 CIF、实验 `.gr`、大型轨迹继续按 Phase 2 的隐私策略排除 Git；只提交可公开的代码、配置、测试、摘要与必要的合规派生结果。严禁在仓库写入 AutoDL 密钥、GitHub 凭证。
- 使用原始 Phase 2 benchmark 的 CPU 配置作为速度比较基准：`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`；9 链并行时记录实际 CPU 使用情况并解释波动。
- 如果发现**无法确认 Si 四配位**、缺失实验数据、CN 正确性不达标、模拟结果异常或 1M 跑不完：停在真实完成的阶段，提供问题/测试证据/恢复命令；不得编造链数、速度、R_w 或结构结论。

## 7. 开发顺序与阶段验收 Gate

**Gate 0 — 文件与代码基线**：当前仓库可运行，Phase 2 回归通过，PZS 和 `.gr` 可读。

**Gate 1 — MQ 配位审计**：完成 partial RDF、候选 cutoff、P/Si/Zr 配位统计及 CrystalNN 差异报告；确认 Si 总四配位的合理性；冻结 cutoff。

**Gate 2 — 增量 CN 单元测试**：所有配位计数/直方图与完整重算整数严格一致、拒绝完全回滚、checkpoint 恢复；CN 关闭时复现 Phase 2。

**Gate 3 — 权重 pilot**：用少量共同候选值探索 CN 约束强度，避免过强抑制所有 MC 移动；冻结共同参数，在报告中保留所有决策证据。若 Si 初始未四配位则不得强行通过此 Gate。

**Gate 4 — 100k 性能与结构测试**：3 modes × 3 seeds、速度剖析、P/Si/Zr 配位分布、PDF 拟合、scale 轨迹均通过。

**Gate 5 — 1M 正式运行和科学汇报**：9 × 1M attempted moves，所有正确性检查、可复现结果与最终结构、跨模式和跨 Phase 2 的对照表/图、Git 提交。不得以“最低 PDF Rw”取代“最低总目标”的结构报告。

## 8. 文献与官方技术依据（与本项目贡献明确区分）

1. **Xintian Wang, Xinyu Zhang, Denys Butenko, Bolong Hong, Jinlong Zhu, Peichen Zhong**. *A practical route to amorphous solid-state electrolyte structures by hybrid reverse Monte Carlo with foundation potentials*. arXiv:2610.19846 (2026). https://arxiv.org/abs/2610.19846 。**证据**：使用 MQ 初始玻璃的 Nb–O **配位分布**作为 RMC 约束目标；讨论有/无配位约束与物理结构的区别。**本项目推广**：按化学角色区分 P、Si 的逐原子四配位和 Zr 的分布约束；不是原文直接提出的 PZS 参数。
2. **RMCProfile User Manual, v6.7.9**, section 5.6.2 “Coordination number”. https://rmcprofile.ornl.gov/wp-content/uploads/2023/11/rmcprofilemanual.pdf 。**证据**：明确区分 fixed coordination number constraint 与 average coordination constraint；示范四面体网络中的 fixed CN。**注意**：手册的权重语义与本项目 `lambda` 不同，数值不可互抄。
3. **RMCProfile7: reverse Monte Carlo for multiphase systems**, *Journal of Applied Crystallography* **57** (2024), DOI: 10.1107/S1600576724004175. https://doi.org/10.1107/S1600576724004175 。**证据**：现代 RMC 软件保留最小距离、配位、键角等约束，以及 X-ray total scattering PDF 拟合能力。
4. **fullrmc — AtomicCoordinationNumberConstraint** 官方 API 文档. https://bachiraoun.github.io/fullrmc/fullrmc.Constraints.html 。**证据**：允许选择中心原子、配体、配位壳层上下界及配位范围，提供可运行的 CN 约束实现思路；**不要求照搬其代码/权重公式**。
5. **pymatgen `CrystalNN`** 官方 API 文档. https://pymatgen.org/pymatgen.core.html 。**证据**：CrystalNN 采用基于 Voronoi、距离/化学权重的近邻判定，适合做一次性/低频率离线交叉验证；**不适合作为每步增量 CN 的定义**。
6. **Cuillier et al.** *Integrating machine learning interatomic potentials with hybrid reverse Monte Carlo structure refinements in RMCProfile*, *Journal of Applied Crystallography* (2024). https://journals.iucr.org/j/issues/2024/06/00/vb5085/ 。**证据**：RMC 中结构约束和能量项能够控制 PDF 反演的非物理自由度；**本 Phase 2.5 不加入能量项**。
7. **AM3GroupHub/amsrmc**（Wang–Zhong 论文源码）https://github.com/AM3GroupHub/amsrmc 。**可参考**：单原子移动与增量 pair histogram 管理；但其 neutron PDF 权重不能直接代替本项目 Q-dependent X-ray 前向核。

## 9. 给 Codex 的最终执行指令

请在当前 AutoDL CPU 容器、已同步的现有仓库中，按 Gate 0→5 顺序执行。第一要务是 MQ 配位审计和增量 CN 的严格数值验证；第二是冻结同一套化学约束与原有三种 scale 设置、完成 9 条 1M RMC 链；第三是独立评估 PDF 与局域化学结构的平衡及新增耗时。以 Phase 2 的无 CN 结果为基准，禁止新增 DPA4 能量项或改变 X-ray PDF 物理模型。所有结果、异常和未完成事项必须实事求是记录。
