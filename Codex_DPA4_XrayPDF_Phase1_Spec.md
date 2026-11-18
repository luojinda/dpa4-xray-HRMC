# Codex 开发任务书｜DPA4 + X-ray PDF + HRMC 项目：Phase 1

> **执行指令**：这是需要你（Codex）实际完成、运行测试、生成图像并提交 Git 的开发任务，不是只给建议或伪代码。当前阶段**仅实现周期性结构的 X-ray PDF 前向计算**；不得提前实现 DPA4、RMC、HRMC 或增量更新。遵循下列验收标准，遇到缺少文件、权限或实际算法障碍时报告实情，不得伪造计算结果。
>
> **开发机器**：用户本地 Windows，PyCharm + Codex；首阶段仅 CPU。未来迁移 AutoDL CPU Linux。用户学校超算暂时无法访问 GitHub，本阶段**不接入超算**。

## 0. 任务背景与优先目标

项目长期目标：以微调后的 `DPA4_Neo_LoRA.pt2` 为物理能量模型，结合实验 **100 keV 高能 X-ray PDF**，建立适用于 Li–Zr–Si–P–O–Cl 等复杂固态电解质的 **Inline HRMC / Alternating HRMC**。原型参考 Wang 等 2026 年 `amsrmc` 框架，但**本阶段不整合、不复制其 RMC 核心代码**。

本阶段目标：从给定的周期性 CIF（测试对象 LNOC）开始，以 partial RDF `g_{αβ}(r)` 为中间量，经 `Q`-dependent X-ray form factors 得到实验定义下的 reduced total PDF `G(r)`；计算数据并输出两张独立的图片。要求可科学复核、参数可配置、结果可重跑。**不用 DebyeCalculator 的原子对完整 Debye 求和作为本阶段的计算引擎。**

### 本阶段交付（必须完成）

1. 能运行的 CPU Python 代码、依赖文件、README、理论推导文档和测试。
2. 从真实 `LNOC_opt.cif` 计算 `g_{αβ}(r)`、`S(Q)`、`F(Q)`、`G(r)`；明确记录元素与晶胞。
3. `results/LNOC/LNOC_xray_Gr.dat`（至少 `r_A, G_r` 两列）。
4. `results/LNOC/LNOC_PDF_1p5_4p5.png`、`results/LNOC/LNOC_PDF_1p5_10.png`，分别画 **1.5–4.5 Å** 和 **1.5–10 Å**；另存 `.pdf` 或 `.svg` 矢量图可选。
5. `results/LNOC/LNOC_partial_RDF.dat`（各 partial RDF）、`LNOC_SQ.dat`（`Q, S(Q), F(Q)`）、`parameters.json`、结构统计与验证报告。
6. `pytest` 测试通过，并执行一次真实结构计算；说明有效 RDF 截断半径、内存、耗时、数值收敛状态。
7. 本地 Git 有清晰的提交；有权限时创建并推送 GitHub 仓库。

## 1. 路径、仓库与文件操作约束

- **用户现有本地项目根目录**：`E:\1_CODEX\dpa4-xray-HRMC`
- **用户已有 CIF 目录**：`E:\1_CODEX\dpa4-xray-HRMC\PDF`
- **预期结构名**：`E:\1_CODEX\dpa4-xray-HRMC\PDF\LNOC_opt.cif`
- **GitHub 所属账号**：`https://github.com/luojinda`
- **建议新仓库**：`luojinda/dpa4-xray-HRMC`，**默认 private**。若已有同名仓库或用户已有 Git 初始化，先检查、保留原数据，不要无条件覆盖；必要时沿用现有 repo。

**文件安全要求**：
- 在用户指定目录中开发；事先用路径检查确认文件存在。不要擅自移动、重命名、覆盖 `PDF/` 原文件。
- `LNOC` 是习惯命名，**以 CIF 实际元素和占位为准**，不得硬编码成 LiNbOCl₄ 或 LiTaOCl₄。
- 读 CIF 必须确认：所有元素、原子数、晶格矢量、体积、周期性、分数坐标、对称性扩展是否正确。ASE 读取后核对 `get_chemical_formula()`、各元素数量；部分占位、无序模型须明确报告，不能静默忽略。
- 仅提交源码、配置、少量必要测试及可公开的小型结果；不上传 SSH 密钥、GitHub token、`.pt2` 模型权重、大型运行数据。若 CIF 有许可限制，则默认不自动提交 CIF；用 `.gitignore` 与示例路径说明管理。
- 首先检查 `git --version`、`conda --version`、`gh auth status`。只有已授权时才用 `gh repo create` 并 push；无权限则完成本地开发并写出用户需手工执行的准确命令，绝不宣称已创建远程仓库。
- 在当前工作目录建立 Python package，不得引入用户电脑上的硬编码盘符作为库内部路径；通过 CLI 输入路径或相对于项目根目录的路径。

## 2. 初始计算参数（当前阶段只改 Qmax）

按照用户此前 `cal_pdf_center_sampling(3).py` 的基本参数，现只将 `QMAX` 由 `20.0` 更改为 `24.0`。其他用户明确设置维持不变：

```yaml
sample: LNOC
structure: PDF/LNOC_opt.cif
radiation_type: xray
qmin_Ainv: 1.0
qmax_Ainv: 24.0
qstep_Ainv: 0.05
output_rmin_A: 0.0
output_rmax_A: 10.0
output_rstep_A: 0.01
biso_A2: 0.3
lorch_mod: false
compute_device: cpu
plot_ranges_A:
  - [1.5, 4.5]
  - [1.5, 10.0]
```

**特别说明**：
- `100 keV` 是用户自己实验 X-ray 的光子能量，不等于 `Qmax`。本次先采用 `Q=1–24 Å⁻¹` 做测试；**没有证据证明这一 Q 范围、峰宽及窗函数与 JACS 参考文献完全一致**。
- 保留 `biso=0.3 Å²` 这个设定，但**必须明确定义和实现 Biso→Uiso→散射/峰展宽的物理关系**，不能把 `0.3` 误写成高斯宽度 0.3 Å，也不能在不同环节重复施加热位移。`Uiso = Biso / (8π²)`（Å²）。计算报告必须说明具体实现。
- 原脚本没有显式设置 `qdamp`、`qbroad`；新实现不得凭空声称匹配旧 DebyeCalculator 的隐式默认值。暂时把额外仪器分辨率修正设置为**不启用**，并注明这使不同程序结果不一定逐点相同。
- `LORCH_MOD=False`：第一版使用矩形 Q 窗；可留功能开关，但不能更改默认值。
- `r=0` 附近的 `g(r)`/`G(r)` 存在数值奇点或无意义值，应安全处理、记录约定；图从 1.5 Å 开始。
- **输出 rmax=10 Å 并不意味着内部 RDF 积分截断只能到 10 Å**。对周期性晶体内部 RDF 积分距离应单独设定，进行 20/30/40 Å 或更大截断的收敛性检查。初次运算不可因成本直接硬截到 10 Å 后仍标称结果严格可靠。`Qstep=0.05 Å⁻¹` 下，晶体结构的有限实空间窗口与倒易空间分辨率相互耦合，应检验峰形和截断振荡。

## 3. 物理方法：必须实现的前向模型

### 3.1 从 CIF 计算 partial RDF

- 使用 `ase.io.read()` 读取 CIF，并正确处理晶体对称操作产生的全部原子。
- 用 PBC 统计每个不同元素对 `g_{αβ}(r)`，其中 `α≤β`；4 元素最多 10 类，6 元素 21 类，**按真实 CIF 自动生成**。
- **对于晶体的小晶胞，若 RDF 截断 `r_cut` 超过晶胞最短垂直高度的一半，单纯最小镜像法是错误的**：必须枚举足够的周期性镜像，或构建足够大的超胞并采用经验证的 PBC 统计实现。避免重复/漏计镜像原子对，避免计入同一原子的自关联（但允许与其不同周期镜像的非零距离关联）。
- 明确 partial RDF 的有序/无序 pair 定义、归一化（物种密度、壳体积、pair multiplicity）；各 partial 在远距离的合理极限应与所用定义一致。
- 内部 `rdf_dr_A` 与输出 `rstep_A` 解耦；从如 `0.01 Å` 的值开始，记录 bin-centering，做至少一次分箱收敛检查。
- 对于小周期性晶胞及完美晶体，长程 `g(r)-1` 并不必然快速衰减；有限 `r_cut` 计算的 `S(Q)` 可能出现显著窗口伪影。**必须在报告里标注这是基于有限 r 截断的近似，并做随 r_cut 的收敛检查**。不得把结果说成无需验证的精确晶体 PDF。

### 3.2 `Q`-dependent X-ray form factors（主方案，不使用 WKM 固定权重代替）

采用 Faber–Ziman 型 partial 定义并清晰注明约定：

\[
S_{\alpha\beta}(Q)-1
=4\pi\rho_0\int_0^{r_c}r^2[g_{\alpha\beta}(r)-1]
\frac{\sin(Qr)}{Qr}\,dr.
\]

\[
w_{\alpha\beta}(Q)=
\frac{(2-\delta_{\alpha\beta})c_\alpha c_\beta
f_\alpha(Q)f_\beta(Q)}{\big[\sum_\gamma c_\gamma f_\gamma(Q)\big]^2},
\quad \alpha\le\beta.
\]

\[
S(Q)-1=\sum_{\alpha\le\beta}
w_{\alpha\beta}(Q)[S_{\alpha\beta}(Q)-1].
\]

\[
F(Q)=Q[S(Q)-1],\qquad
G(r)=\frac{2}{\pi}\int_{Q_{\min}}^{Q_{\max}}F(Q)\sin(Qr)\,dQ.
\]

> 上式假设使用一致的归一化、单原子平均数密度 `rho0=N/V`、实值的中性原子 X-ray `f0(Q)`，且 partial `g_{αβ}(r)` 定义的远程极限为 1。若改用不同的 Faber–Ziman partial convention，必须同步修正所有系数，不得混用文献公式。数值积分可使用 SciPy / NumPy；需在测试中验证单位和 normalization。

**推荐 X-ray form factor 数据源**：`xraydb.f0(element, q)` 或有明确数据源的等价实现。

**极其重要的单位约定**：

```python
Q_Ainv = np.arange(1.0, 24.0 + 0.025, 0.05)
q_for_xraydb = Q_Ainv / (4.0 * np.pi)  # sin(theta)/lambda, Å^-1
f0 = xraydb.f0("Nb", q_for_xraydb)
```

`xraydb.f0` 的参数是 `sin(theta)/lambda`，**不是**通常定义的 `Q=4π sin(theta)/lambda`。这个 4π 错误会显著改变 X-ray 相对峰强，务必加单元测试。

- 默认采用中性原子、弹性 X-ray 散射因子 `f0`。如果后续考虑 100 keV 的 anomalous scattering `f′, f″`，必须另建明确的可选功能和**复数散射权重的正确处理**；当前不强制启用、不得把 `f0` 误标为完整能量依赖散射振幅。
- 各 `Q` 点检查 `sum_{α≤β} w_{αβ}(Q) ≈ 1`（在纯实 `f0` 约定下）；检查 `f0(Q=0)` 合理接近原子序数；检查没有负权重或 NaN；写出 `form_factors.dat` 和 `pair_weights.dat`。
- 建议另行实现只作**方法对比**的 `WKM` 固定权重分支（可选），`Q_f=0` 或 `Qmax/2`，用于后续定量观察固定权重引入的误差；**不可取代主实现**。

### 3.3 热位移与实验处理

- 明确采用 `Biso=0.3 Å²`。若以单原子各向同性位移处理，使用 `Uiso=Biso/(8π²)`；给出与 partial pair correlation 或 `S(Q)-1` 的 Debye–Waller 衰减对应关系，**避免在归一化权重中相互抵消或双重展宽**。
- 如果实现广义 `B_alpha`，第一阶段所有元素都先取 `0.3`；但不要假设这个数是文献拟合精修所得。
- 有限 `Qmin`/`Qmax`、`Qstep`、无 Lorch 窗、数值积分离散都应该真实反映在输出。
- 不得擅自进行平滑、背景调整、幅度缩放、`r` 轴平移来“匹配文献”；可以**另输出**明确标记为仅供观察的对齐图，但本阶段默认禁用。
- 实验的 `Qdamp/Qbroad` 未提供，本阶段不对其做无来源的猜测。

## 4. 数值算法与性能边界

建议先编写**正确性优先的 CPU 版本**，以验证 partial RDF 和 X-ray 归一化为首要目标。

- **切勿用仅统计原始晶胞内的原子对来代替 PBC 全体相结构。**
- 对于晶体，在内部 `r_cut` 太小时，先报告不可收敛的局限；必要时采用按物种分组的多镜像/超胞统计 + 批处理，降低内存成本。不要盲目生成极大超胞以致内存溢出。
- 可预计算积分矩阵 `A(Q, r') ∝ r'^2 sinc(Qr') Δr'`，将 `g-1` 映射为 partial `S(Q)-1`，再进行 `Q` 加权与反傅里叶变换；保留中间数组便于检查。这个静态计算只需 CPU，**不要求第一阶段实现 MC 增量更新**。
- 如果积分范围的长程截断导致晶体 `S(Q)` 产生显著振荡/伪影，可按科学合理的方式实现**标注清楚的实空间窗函数或正则化对照分支**，但绝不暗中改变用户指定的 `lorch_mod=False`（这是 `Q` 空间 Lorch，不要与 `r` 空间截断窗混淆）。
- 若难以在合理成本下取得可信晶体 PDF，先完成算法、误差报告和可视化，明确指出限制；可选择性使用可信周期性计算库做独立交叉验证（例如 DiffPy `PDFCalculator`，但它的固定 `Q` 散射权重近似不能作为 `Q`-dependent 实现的逐点精确金标准）。

## 5. 建议的项目结构

```text
E:\1_CODEX\dpa4-xray-HRMC\
├── PDF\
│   └── LNOC_opt.cif                  # 已有；只读；不得覆盖
├── src\
│   └── dpa4_xray_hrmc\
│       ├── __init__.py
│       └── xray_pdf\
│           ├── __init__.py
│           ├── structure.py         # ASE/CIF, PBC, 组成校验
│           ├── partial_rdf.py       # N 元素 partial RDF，镜像计数
│           ├── form_factors.py      # f0(Q)，Q/(4π)，散射权重
│           ├── transform.py         # RDF -> partial S(Q) -> S(Q) -> F(Q) -> G(r)
│           ├── thermal.py           # Biso、位移模型
│           ├── plotting.py          # 制图
│           └── config.py            # 参数读入和检查
├── scripts\
│   └── calculate_xray_pdf.py        # CLI 主入口
├── configs\
│   └── lnoc_phase1.yaml
├── tests\
│   ├── test_form_factors.py
│   ├── test_pair_counting_pbc.py
│   ├── test_partial_rdf.py
│   ├── test_transforms.py
│   └── test_end_to_end.py
├── results\
│   └── LNOC\
├── docs\
│   ├── theory_and_conventions.md
│   ├── validation.md
│   └── references.md
├── environment.yml
├── pyproject.toml
├── .gitignore
└── README.md
```

原则：`src` 模块不要耦合 `LNOC`，化学组成与参数一律从 `Atoms` + 配置读取，以便未来复用于 Li–Zr–Si–P–O–Cl。输出文件命名可额外使用 `r_cut`、模型版本等后缀；但至少提供上文列出的标准文件。

## 6. Python 环境和运行接口

- 项目创建独立环境，例如 **`dpa4-xray-hrmc`**（Windows 和以后 Linux AutoDL 都要可重建）。
- 优先：Python 3.11、NumPy、SciPy、ASE、Matplotlib、PyYAML、xraydb、pytest。若版本冲突，Codex 要实测兼容后锁定版本，并更新说明。
- **本阶段不安装 DPA4、DeepMD-kit、MACE、PyTorch、CUDA 或 DebyeCalculator 作为运行依赖**。
- 默认 `conda env create -f environment.yml` 或 `conda env update -f environment.yml --prune`；采用 `pip install -e .` 安装本项目。
- CLI 示例：

```powershell
cd 'E:\1_CODEX\dpa4-xray-HRMC'
conda env create -f environment.yml
conda activate dpa4-xray-hrmc
python -m pip install -e .
python scripts/calculate_xray_pdf.py --config configs/lnoc_phase1.yaml
python -m pytest -q
```

- 若本机暂时没有可用 conda，记录阻碍，可使用临时 `.venv` 完成测试，同时保留正式 `environment.yml`；不要声称已用 conda 成功执行。
- `README` 要提供 Windows PowerShell 和 Linux Bash 的可复现运行命令。未来 AutoDL CPU 仅需 Git clone + conda 安装 + 输入 CIF 即可运行。

## 7. 数值验证：必须有，不能只画漂亮曲线

### 7.1 代码级测试

1. **晶胞与元素**：检查 CIF 读入、对称展开、原子总数、体积、归一化。
2. **周期镜像和对计数**：用简单立方单元素模型验证最近邻壳层的位置、配位数和 shell multiplicities；对缩小/扩大的晶胞和超胞结果做一致性检查。
3. **RDF 归一化**：理想均匀点分布或适当随机配置的平均 `g(r)` 在合理范围趋近 1；小有限系统误差应被分析。
4. **X-ray form factor**：检查 `xraydb.f0` 接口 `Q/(4π)`；做至少两个元素的 `Q=0` 与 `Q>0` 测试，`sum(weights)=1`。
5. **数值变换**：检查 `F(Q)` 的单位、`sin(Qr)/(Qr)` 在极限的稳定实现、`G(r)` 的基线、正反变换数值一致性与无 NaN。
6. **热位移**：`Biso=0` 和 `Biso=0.3 Å²` 的差异应合理，且参数变化不应非法改变结构键距。

### 7.2 收敛与独立验证

- `rdf_dr`：至少两个分箱宽度。
- `r_cut`：至少三档（例如 20、30、40 Å；实际按结构与资源调整），对 **1.5–4.5 Å、1.5–10 Å** 的峰位置、峰高、曲线差异进行报告；要求明确标出无法收敛的情形。
- 比较**固定散射权重 WKM**（若实现）与 `Q`-dependent 主方法的变化，不能因为两者峰强不同就判定某一方一定错误。
- 推荐对一个简单周期性测试结构另用成熟 PDF 工具核对峰位置/PBC（可选，依赖应隔离）；注意传统 real-space DiffPy X-ray 权重不是严格同一物理模型。
- 统计 wall time 和峰值内存，保存日志。
- JACS 实验样品可能包含纳米晶、局域无序及实验展宽；优化后的**理想周期性 CIF 不一定与实验 PDF 在全部峰强、峰宽或中程衰减上重合**。第一阶段判断程序正确性的依据应是公式、PBC、收敛与基准测试，而非“画得像文献”。

### 7.3 输出图像

- `LNOC_PDF_1p5_4p5.png`：`xlim(1.5,4.5)`。
- `LNOC_PDF_1p5_10.png`：`xlim(1.5,10.0)`。
- x 轴标 `r (Å)`，y 轴标 `G(r) (Å⁻²)`（按实际选用的 reduced PDF 定义核查）；高清 ≥300 dpi，清晰显示主峰。
- 图题或角标注明 `X-ray`, `Q=1–24 Å⁻¹`, `Biso=0.3 Å²`, `No Lorch`, `r_cut=... Å`；所有曲线来自真实计算，禁止插值出伪造的精细峰。
- 如暂无文献实验原始 XY 数据，仅绘制理论曲线，不要虚构实验对照曲线、Rwp/R² 或在图上声称与实验“吻合”。

## 8. GitHub 工作流与完成时报告

1. 在已有项目根目录检查 `git status`、当前分支和现有文件，避免破坏用户工作区。
2. 若无 repo：`git init`，创建 `.gitignore`、README、环境及项目骨架，先本地 commit。
3. 已认证时使用 `gh repo create luojinda/dpa4-xray-HRMC --private ...` 创建，默认 `main` 为稳定分支；可以建立 `dev` 分支用于开发。**不要为了建仓库强制用户提供 token 或 SSH 私钥到聊天中**。
4. 仅在本地计算成功、测试通过并经过必要的 git diff 检查后推送；远端无法访问时保留本地提交。
5. 在最终回复中真实汇报：`git log -1`、GitHub 实际状态、环境、执行命令、结构组成、实际数值参数、生成文件绝对路径、图像路径、耗时、测试结果和剩余风险。不要声称完成未执行的步骤。

## 9. 后续阶段（只预留扩展接口，不实现）

- **Phase 2**：快速单原子增量 partial RDF / `ΔG(r)`，目标固定网格下近似 `O(N)` 的单步更新；必须与 Phase 1 完整重算对比。
- **Phase 3**：DPA4 `deepmd.calculator.DP(model="DPA4_Neo_LoRA.pt2")` + FIRE 最小化，测试 ASE/LAMMPS 能量和力一致性。
- **Phase 4**：结合 `amsrmc`（保留 MIT License 和 Xintian Wang 版权声明），实现 Inline/Alternating HRMC，检查 Metropolis 接受准则、恢复状态、断点续算、结构拓扑。
- 不要提前引入这些重量级依赖，Phase 1 的 `xray_pdf` 软件包接口要保持独立可测。

## 10. 参考资料（先读、在 docs 中记录如何使用，不要仅罗列）

**A. 最核心的 X-ray PDF 理论与算法**

1. **Neder, R. B. & Proffen, T. (2020)**, *Exact and fast calculation of the X-ray pair distribution function*, **Journal of Applied Crystallography** 53, 710–721. DOI: https://doi.org/10.1107/S1600576720004616
   - 必读：普通实空间固定 X-ray 散射权重的 Warren–Krutter–Morningstar 近似及限制；使用真实 `Q`-dependent form factor 的理由；有限 Q 与仪器分辨率。
2. **Sławiński, W. A. et al. (2024)**, *RMCProfile7: reverse Monte Carlo for multiphase systems*, **Journal of Applied Crystallography** 57, 1251–1262. DOI: https://doi.org/10.1107/S1600576724004175
   - **特别重要：Section 2.6 与 Figure 1 / Figure 10**。该文已经采用从 partial correlations → 倒易空间 X-ray Faber–Ziman 加权 → `G(r)` 反傅里叶变换；是第一阶段数学路径的直接已发表先例。不要把该思路表述成全新发明。
3. **Ehrhardt, K. M. et al. (2026)**, *pyHRMC: Hybrid Reverse Monte Carlo for Electron Total Scattering*, **Journal of Computational Chemistry** 47, e70306. DOI: https://doi.org/10.1002/jcc.70306
   - partial RDF → `S(Q)` → `G(r)`，软件可以将默认电子散射因子改为 X-ray/中子；参考方法而非直接复制未经验证的实现。

**B. 目标结构的对比文献**

4. **Singh, B. et al. (2024)**, *Critical Role of Framework Flexibility and Disorder in Driving High Ionic Conductivity in LiNbOCl₄*, **Journal of the American Chemical Society** 146, 17158–17169. DOI: https://doi.org/10.1021/jacs.4c03142
   - 用其 X-ray PDF 结果作**定性对照**；先核对正文/Supporting Information 中实验 Q 范围、仪器处理与模型来源，不应假设本任务 Q=1–24 Å⁻¹ 与之完全一致。优化晶体与真实样品结构/展宽可能不同。

**C. 计算与后续 HRMC**

5. **Wang, X. et al. (2026)**, *A practical route to amorphous solid-state electrolyte structures by hybrid reverse Monte Carlo with foundation potentials*. arXiv: https://arxiv.org/abs/2610.19846 ; code: https://github.com/AM3GroupHub/amsrmc
   - 学习 partial histogram 与 RMC 更新结构（留待 Phase 2/4）。原项目 MIT License；若未来复制核心代码，保留 MIT License 和版权声明。
6. **diffpy.srreal** 官方文档：https://www.diffpy.org/diffpy.srreal/
   - 了解周期性 `PDFCalculator` 与 `OPTIMIZED` 缓存；其典型 real-space X-ray 计算与完整 `Q`-dependent 路线可能并不完全一致。
7. **XrayDB** 官方 Python API：https://xraypy.github.io/XrayDB/python.html
   - `xraydb.f0(element, q)` 的 `q=sinθ/λ=Q/(4π)`；用于核对 form factor 的函数定义。

## 11. 给 Codex 的立即执行清单

1. 检查项目根目录和 `PDF/LNOC_opt.cif`，读取结构并打印真实元素、原子数、晶胞；检查现有 Git 状态。
2. 创建 Python 3.11 的本地独立 conda 环境和工程骨架；实现 `xraydb.f0` form factors、完整 periodic partial RDF、`Q`-dependent X-ray PDF 正向计算。
3. 用标准参数 `Q=1–24 Å⁻¹`、`Qstep=0.05`、`r=0–10 Å`、`rstep=0.01`、`Biso=0.3 Å²`、无 Lorch 运行真实 `LNOC_opt.cif`；合理选择内部 RDF `r_cut` 并记录收敛测试。
4. 输出要求的两张图片、`G(r)` 数据、partial RDF、`S(Q)`、参数记录、理论说明和单元测试；实际检查文件存在且非空。
5. 本地完成 commit；若 `gh auth` 可用，在 `luojinda` 下创建 private 仓库并推送；否则报告具体障碍和命令。
6. 最后写出已完成事项和未解决的科学/算法限制，不要开始 DPA4 或 HRMC 开发。

**成功标准**：用户无需修改源代码，只需运行一条命令，即可在本地 CPU 由 `LNOC_opt.cif` 重算出可复现的 X-ray `G(r)` 数据与指定两张图；结果含完整的物理约定、PBC 处理说明及数值正确性验证。
