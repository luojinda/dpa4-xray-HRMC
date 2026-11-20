# Phase 2.5 配位约束数学与 P/Si 四面体几何核查

核查对象：当前 `main` 分支 Phase 2 / Phase 2.5 正式 1M × 9 链结果。结构均取各链的 `final_structure.extxyz`，不是 `best_pdf_structure` 或 `best_total_structure`。初始结构为 `PDF/PZS_quench_222.cif`。

## 1. 实际目标函数与四项配位惩罚

记 P、Si、Zr 数分别为 `N_P=32`、`N_Si=64`、`N_Zr=64`。程序的总目标函数为

\[
L=\frac{X_{PDF}^2}{2\sigma_{RMC}^2}
+800C_P+2500C_{Si}+1.0C_{Zr,total}+0.2C_{Zr,O},
\qquad \sigma_{RMC}=0.030\ \mathrm{\AA^{-2}}.
\]

合法试探位移的接受概率为 `min[1, exp(-(L_trial-L_current))]`；若先被 hard-sphere 检查否决，则接受概率为 0。

### P 与 Si

\[
C_P=\frac{1}{N_P}\sum_{i=1}^{N_P}(n_{P-O,i}-4)^2,
\]

\[
C_{Si}=\frac{1}{N_{Si}}\sum_{i=1}^{N_{Si}}
(n_{Si-O,i}+n_{Si-Cl,i}-4)^2.
\]

因此 P、Si 分别按 P、Si 阳离子数取均值，不按配位类别数归一化。P/Si 没有额外的容差参数 ε。

### Zr-total 与 Zr-O

设 `H_k` 是当前 Zr 配位数等于 k 的 Zr 数目，`H^0_k` 是初始 MQ 结构的目标直方图，k=0,...,16。定义

\[
\Delta p_k=\frac{H_k-H^0_k}{N_{Zr}},
\qquad
p^0_k=\frac{H^0_k}{N_{Zr}},
\]

\[
\epsilon_k=\max\left[
\sqrt{\frac{p^0_k(1-p^0_k)}{N_{Zr}}},\frac{1}{N_{Zr}}
\right].
\]

两个 Zr 项分别是

\[
C_{Zr,total}=\sum_{k=0}^{16}
\left(\frac{(H^{total}_k-H^{total,0}_k)/N_{Zr}}
{\epsilon^{total}_k}\right)^2,
\]

\[
C_{Zr,O}=\sum_{k=0}^{16}
\left(\frac{(H^{O}_k-H^{O,0}_k)/N_{Zr}}
{\epsilon^{O}_k}\right)^2.
\]

这里先按 Zr 数把每个 bin 的计数差转成概率差，随后对 17 个类别直接求和；没有再除以 17。因此 Zr 项不是按“配位类别数取均值”。

代码依据：`src/dpa4_xray_hrmc/rmc/coordination.py` 的 `_distribution_epsilon`（109–121 行）、构造器（144–155 行）和 `_components`（173–188 行）；权重组合在 `CoordinationComponents.weighted`（58–64 行）；总目标与 Metropolis 判据在 `src/dpa4_xray_hrmc/rmc/runner_cn.py` 的 `_total`（28 行）及 `run_cn_chain`（54–63 行）。

## 2. Zr 自适应 ε_k

初始目标直方图为：

- Zr-total：`H5=1, H6=63`，其余为 0；
- Zr-O：`H0=19, H1=23, H2=18, H3=4`，其余为 0。

| k | ε_total,k | ε_O,k |
|---:|---:|---:|
| 0 | 0.015625000 | 0.057110123 |
| 1 | 0.015625000 | 0.059977158 |
| 2 | 0.015625000 | 0.056201151 |
| 3 | 0.015625000 | 0.030257682 |
| 4 | 0.015625000 | 0.015625000 |
| 5 | 0.015625000 | 0.015625000 |
| 6 | 0.015625000 | 0.015625000 |
| 7 | 0.015625000 | 0.015625000 |
| 8 | 0.015625000 | 0.015625000 |
| 9 | 0.015625000 | 0.015625000 |
| 10 | 0.015625000 | 0.015625000 |
| 11 | 0.015625000 | 0.015625000 |
| 12 | 0.015625000 | 0.015625000 |
| 13 | 0.015625000 | 0.015625000 |
| 14 | 0.015625000 | 0.015625000 |
| 15 | 0.015625000 | 0.015625000 |
| 16 | 0.015625000 | 0.015625000 |

“自适应”仅表示 ε 由初始目标直方图自动计算。构造 `IncrementalCoordination` 时计算一次后，整条链固定不变；它不会随当前结构或每个试探结构更新。

## 3. 正式 RMC 中 P/Si 配位变化统计

`runner_cn.py` 只在移动已经接受后更新 break/recovery 计数器。计数器记录的是非四配位中心数目的增加/减少，不记录所有试探移动的 `ΔC_P`、`ΔC_Si`，也不记录因配位变化而被拒绝的试探次数。

| scale | seed | attempts | legal | accepted | P break/recover | Si break/recover |
|---|---:|---:|---:|---:|---:|---:|
| no_scale | 101 | 1,000,000 | 717,928 | 334,811 | 0 / 0 | 0 / 0 |
| no_scale | 202 | 1,000,000 | 719,543 | 334,825 | 0 / 0 | 0 / 0 |
| no_scale | 303 | 1,000,000 | 715,972 | 332,715 | 0 / 0 | 1 / 1 |
| fixed_initial | 101 | 1,000,000 | 768,264 | 340,283 | 0 / 0 | 0 / 0 |
| fixed_initial | 202 | 1,000,000 | 769,206 | 339,066 | 0 / 0 | 0 / 0 |
| fixed_initial | 303 | 1,000,000 | 769,120 | 339,148 | 0 / 0 | 0 / 0 |
| bounded_dynamic | 101 | 1,000,000 | 750,587 | 373,309 | 0 / 0 | 0 / 0 |
| bounded_dynamic | 202 | 1,000,000 | 749,440 | 371,006 | 0 / 0 | 0 / 0 |
| bounded_dynamic | 303 | 1,000,000 | 750,755 | 373,561 | 0 / 0 | 1 / 1 |
| **合计** | — | **9,000,000** | **6,710,815** | **3,138,724** | **0 / 0** | **2 / 2** |

可以严格得到：

- 没有任何被接受的移动改变 P 的“四配位/非四配位”状态；
- Si 共有 2 次被接受的破坏事件和 2 次被接受的恢复事件，分别出现在 `no_scale/seed_303` 和 `bounded_dynamic/seed_303`；
- 所有正式链的最终 `C_P=C_Si=0`；每 1000 步采样的 trace 也全部为 0，说明两次 Si 临时偏离及恢复发生在采样点之间；
- 四个已接受 Si 事件相对于全部 attempts、legal evaluations、accepted moves 的发生频率分别为 `4.444e-7`、`5.961e-7`、`1.274e-6`。这些是事件发生频率，不是条件 Metropolis 接受率；
- 全部合法移动的总体 Metropolis 接受率为 `3,138,724 / 6,710,815 = 46.7711%`，但这也不是“配位改变试探”的条件接受率。

单中心配位由 n 变为 n+δ 时，程序中的精确离散变化是

\[
\Delta C_P=\frac{(n+\delta-4)^2-(n-4)^2}{32},
\qquad
\Delta C_{Si}=\frac{(n+\delta-4)^2-(n-4)^2}{64}.
\]

例如从 4 变到 3 或 5，`ΔC_P=+1/32=0.03125`、加权目标增加 25；`ΔC_Si=+1/64=0.015625`、加权目标增加 39.0625。反向恢复取相反数。若一次位移跨越多个 cutoff，ΔC 可以是这些离散值的组合。

现有正式结果没有保存逐试探 `n→n'`、随机数和 rejected-CN-change 计数，因此无法严格恢复四次 Si 事件各自的精确 ΔC，也无法计算

\[
N_{accepted,CN-change}/N_{proposed,CN-change}.
\]

把该接受率写成 0% 或 100% 都不成立。若后续确实需要这一统计，应在新运行中增加逐试探的轻量计数器，而不是根据 1000 步 trace 推断。

## 4. P/Si 四面体几何比较

### 方法

- 使用与 Phase 2.5 一致的 cutoff：P–O 1.80 Å、Si–O 1.85 Å、Si–Cl 2.30 Å，并采用周期性最小镜像；
- P 四面体：恰有 4 个 O；Si 四面体：O+Cl 恰有 4 个；
- 键角为 6 个 ligand–center–ligand 角；理想角为 109.4712206°；
- 四面体体积 `V=|det(v1-v0,v2-v0,v3-v0)|/6`；
- 本报告的 CSM：将四个配体相对中心的向量与理想四面体做 24 种顶点排列、Kabsch 最优旋转及统一尺度拟合，取 `100 × min(sum squared residual / sum squared observed radius)`；理想四面体为 0；
- 键长、键角、CSM 和体积只对恰好四配位的中心统计。Phase 2 四配位保持率很低的组因此存在明显“幸存四面体选择偏差”，不能代表所有 P/Si 中心。

### 四配位保持率与整体几何

表中 Phase 2/2.5 数值为 3 个 seed 的均值；括号为 seed 间标准差。键角 RMS 是每个四面体相对理想角的 RMS 后再平均。

| stage / scale | center | fourfold fraction | all-bond mean (Å) | angle mean (°) | angle RMS (°) | CSM (%) | volume (Å³) |
|---|---|---:|---:|---:|---:|---:|---:|
| initial MQ | P | 100.00% | 1.5441 | 109.367 | 3.672 | 0.359 | 1.878 |
| initial MQ | Si | 100.00% | 1.9576 | 109.439 | 2.949 | 0.601 | 3.832 |
| Phase 2 / no_scale | P | 94.79% (4.77%) | 1.5856 | 108.704 | 13.742 | 4.144 | 1.859 |
| Phase 2 / no_scale | Si | 31.25% (2.71%) | 1.9079 | 108.839 | 13.153 | 4.587 | 3.234 |
| Phase 2 / fixed_initial | P | 27.08% (7.86%) | 1.6029 | 108.496 | 13.191 | 4.080 | 1.939 |
| Phase 2 / fixed_initial | Si | 5.21% (3.61%) | 1.9973 | 108.355 | 14.660 | 5.203 | 3.682 |
| Phase 2 / bounded_dynamic | P | 62.50% (0.00%) | 1.6035 | 108.450 | 14.178 | 4.716 | 1.903 |
| Phase 2 / bounded_dynamic | Si | 14.06% (2.71%) | 1.9672 | 108.082 | 15.179 | 5.954 | 3.449 |
| Phase 2.5 / no_scale | P | 100.00% | 1.5863 | 108.772 | 14.060 | 4.304 | 1.850 |
| Phase 2.5 / no_scale | Si | 100.00% | 1.9647 | 108.752 | 13.168 | 4.440 | 3.571 |
| Phase 2.5 / fixed_initial | P | 100.00% | 1.5936 | 108.360 | 14.076 | 4.784 | 1.874 |
| Phase 2.5 / fixed_initial | Si | 100.00% | 2.0178 | 108.258 | 14.566 | 5.497 | 3.780 |
| Phase 2.5 / bounded_dynamic | P | 100.00% | 1.5990 | 108.475 | 14.177 | 4.644 | 1.894 |
| Phase 2.5 / bounded_dynamic | Si | 100.00% | 1.9980 | 108.481 | 14.114 | 5.031 | 3.706 |

### 分键种平均键长

| stage / scale | P–O (Å) | Si–O (Å) | Si–Cl (Å) |
|---|---:|---:|---:|
| initial MQ | 1.5441 | 1.6512 | 2.0247 |
| Phase 2 / no_scale | 1.5856 | 1.6003 | 2.0076 |
| Phase 2 / fixed_initial | 1.6029 | 1.6814 | 2.0739 |
| Phase 2 / bounded_dynamic | 1.6035 | 1.6532 | 2.0581 |
| Phase 2.5 / no_scale | 1.5863 | 1.6095 | 2.0425 |
| Phase 2.5 / fixed_initial | 1.5936 | 1.6991 | 2.0876 |
| Phase 2.5 / bounded_dynamic | 1.5990 | 1.6586 | 2.0724 |

### 结论

Phase 2.5 的 P/Si 惩罚成功把所有最终结构的四配位保持率锁定为 100%，而 Phase 2 尤其 `fixed_initial` 和 `bounded_dynamic` 会大量丢失 Si 四配位。另一方面，“配位数保持”不等于“理想四面体保持”：Phase 2.5 的角度 RMS 和 CSM 仍显著高于初始 MQ，说明当前目标只约束邻居数，没有直接约束键角、CSM 或四面体体积。后续若这些几何量在物理上必须保持，需要独立论证和引入几何约束；不能从当前 CN=4 自动推断。

## 5. 数据文件与限制

- `geometry_aggregate.csv`：本报告主几何表的机器可读数据；
- `geometry_by_structure.json`：初始结构和 18 个最终结构的逐结构统计；
- `tetrahedra_by_site.csv`：逐四面体的键长、角度、CSM、体积；
- `bond_lengths_by_species.csv`：逐链、分键种统计；
- `bond_lengths_by_species_aggregate.csv`：分键种的跨 seed 汇总。

### pymatgen 参考脚本复核校正

随后已从 `tests/get_site_info_v3.py` 读取参考实现，并在 AutoDL 的 pymatgen 2026.10.24 环境中完成 16 个逐位点对照：

- 本报告体积与参考脚本的 `ConvexHull(...).volume` 最大绝对差为 `7.11e-15 Å³`，数值一致；
- 本报告 CSM 与 pymatgen 的固定中心 `csm_wocs_csc/csm_wcs_csc` 最大绝对差为 `6.31e-14`，数值一致；
- 参考脚本实际读取的无后缀默认 `csm` 等于 `csm_wcs_ctwcc`，它以中心原子和四个配体的五点质心为中心，因此与本报告“中心原子固定为原点”的 CSM 绝对值不同。

所以本文表格中的 CSM 应准确称为“固定中心 CSM”，不能直接当作 `get_site_info_v3.py` 的默认 CSM。四配位率、键长、键角和体积结论不受影响。完整对照见 `PYMATGEN_CSM_VOLUME_COMPARISON.md` 和 `pymatgen_csm_volume_examples.json`。
