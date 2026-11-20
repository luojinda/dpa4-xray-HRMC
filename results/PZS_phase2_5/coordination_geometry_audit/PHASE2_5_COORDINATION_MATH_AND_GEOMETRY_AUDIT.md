# Phase 2.5 配位约束数学与 P/Si 四面体几何核查

Scope:  `main` 的 Phase 2 / Phase 2.5 正式 1M × 9 链结果；几何统计使用每条链的 `final_structure.extxyz`，初始结构为 `PDF/PZS_quench_222.cif`。

## 1. 四项配位惩罚的实际数学定义

System: `N_P=32, N_Si=64, N_Zr=64`。总目标为：

`L = X_PDF²/[2(0.030 Å⁻²)²] + 800 C_P + 2500 C_Si + 1.0 C_Zr,total + 0.2 C_Zr,O`。

 `P_acc=min[1,exp(-(L_trial-L_current))]`；hard-sphere 失败时为 0。

- `C_P = (1/N_P) Σ_i (n_P-O,i - 4)²`
- `C_Si = (1/N_Si) Σ_i (n_Si-O,i + n_Si-Cl,i - 4)²`

P、Si 按各自阳离子数取均值，没有另设 ε 容差。

Definitions:  `H_k`、`H⁰_k` 分别为当前和初始结构中 Zr 配位数等于 k 的计数，`k=0,...,16`：

- `Δp_k=(H_k-H⁰_k)/N_Zr`
- `p⁰_k=H⁰_k/N_Zr`
- `ε_k=max[sqrt(p⁰_k(1-p⁰_k)/N_Zr), 1/N_Zr]`
- `C_Zr,total = Σ_k [(H_total,k-H⁰_total,k)/(N_Zr ε_total,k)]²`
- `C_Zr,O = Σ_k [(H_O,k-H⁰_O,k)/(N_Zr ε_O,k)]²`

Zr 先按 64 个 Zr 把计数差归一化，再对 17 类直接求和；没有再除以 17，因而不是按类别数取均值。

Code: `src/dpa4_xray_hrmc/rmc/coordination.py` 58–64、109–121、144–155、173–188 行；总目标和 Metropolis 判据见 `src/dpa4_xray_hrmc/rmc/runner_cn.py` 28、54–63 行。

## 2. Zr 的 ε_k

Initial target: Zr-total 为 `H5=1,H6=63`；Zr-O 为 `H0=19,H1=23,H2=18,H3=4`；未列类别均为 0。

| k | ε_total | ε_Zr-O |
|---:|---:|---:|
| 0 | 0.015625000 | 0.057110123 |
| 1 | 0.015625000 | 0.059977158 |
| 2 | 0.015625000 | 0.056201151 |
| 3 | 0.015625000 | 0.030257682 |
| 4–16 | 0.015625000 | 0.015625000 |
Epsilon is computed once from the initial MQ target histogram when IncrementalCoordination is constructed and is fixed for the whole chain; it is not recomputed for trial structures.


## 3. 正式链的 P/Si 配位变化

Counters are updated only after acceptance and record increases/decreases in the number of non-fourfold centers. Rejected CN-changing proposals and per-move delta-C values were not recorded.

| scale | seed | attempts | legal | accepted | P break/recover | Si break/recover |
|---|---:|---:|---:|---:|---:|---:|
| no_scale | 101 | 1,000,000 | 717,928 | 334,811 | 0/0 | 0/0 |
| no_scale | 202 | 1,000,000 | 719,543 | 334,825 | 0/0 | 0/0 |
| no_scale | 303 | 1,000,000 | 715,972 | 332,715 | 0/0 | 1/1 |
| fixed_initial | 101 | 1,000,000 | 768,264 | 340,283 | 0/0 | 0/0 |
| fixed_initial | 202 | 1,000,000 | 769,206 | 339,066 | 0/0 | 0/0 |
| fixed_initial | 303 | 1,000,000 | 769,120 | 339,148 | 0/0 | 0/0 |
| bounded_dynamic | 101 | 1,000,000 | 750,587 | 373,309 | 0/0 | 0/0 |
| bounded_dynamic | 202 | 1,000,000 | 749,440 | 371,006 | 0/0 | 0/0 |
| bounded_dynamic | 303 | 1,000,000 | 750,755 | 373,561 | 0/0 | 1/1 |
| **合计** | — | **9,000,000** | **6,710,815** | **3,138,724** | **0/0** | **2/2** |

P 没有被接受的 break/recovery；Si 有 2 次 break 和 2 次 recovery，位于 no_scale/seed_303 与 bounded_dynamic/seed_303。所有最终结构的 C_P=C_Si=0。每 1000 步 trace 也均为 0，说明 Si 的临时偏离和恢复发生在相邻采样点之间。

The four accepted Si events occur at frequencies 4.444e-7 per attempt, 5.961e-7 per legal evaluation, and 1.274e-6 per accepted move. Overall legal-move Metropolis acceptance is 46.7711 percent. None of these is the conditional CN-changing-proposal acceptance rate.

For one center changing from n to n+delta:

- ΔC_P=[(n+δ-4)²-(n-4)²]/32
- ΔC_Si=[(n+δ-4)²-(n-4)²]/64

Minimum  break（4→3 或 5）分别给出 ΔC_P=+0.03125、加权增量 25；ΔC_Si=+0.015625、加权增量 39.0625；恢复取反。一次移动跨越多个 cutoff 时可为这些离散量的组合。

Existing outputs do not save per-proposal n-to-n-prime values or rejected CN-change counts. Therefore the exact four transient Si delta-C values and the conditional acceptance rate Naccepted_CNchange/Nproposed_CNchange cannot be recovered; reporting 0 percent or 100 percent would be incorrect.

## 4. P/Si 四面体几何

Method: P-O 1.80 A, Si-O 1.85 A, Si-Cl 2.30 A with periodic minimum image. A P tetrahedron has exactly four O; a Si tetrahedron has exactly four O+Cl. CSM uses all 24 vertex permutations, Kabsch rotation, and uniform scale; CSM=100*min(SSE/sum observed squared radii), ideal=0. Volume=abs(det(v1-v0,v2-v0,v3-v0))/6.

Aggregation: Phase 2 and Phase 2.5 values are three-seed means. Geometry is evaluated only for exactly four-coordinate surviving tetrahedra, so low-retention Phase 2 groups have selection bias.

| stage / scale | center | fourfold | bond mean Å | angle mean ° | angle RMS ° | CSM % | volume Å³ |
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

| stage / scale | P–O Å | Si–O Å | Si–Cl Å |
|---|---:|---:|---:|
| initial MQ | 1.5441 | 1.6512 | 2.0247 |
| Phase 2 / no_scale | 1.5856 | 1.6003 | 2.0076 |
| Phase 2 / fixed_initial | 1.6029 | 1.6814 | 2.0739 |
| Phase 2 / bounded_dynamic | 1.6035 | 1.6532 | 2.0581 |
| Phase 2.5 / no_scale | 1.5863 | 1.6095 | 2.0425 |
| Phase 2.5 / fixed_initial | 1.5936 | 1.6991 | 2.0876 |
| Phase 2.5 / bounded_dynamic | 1.5990 | 1.6586 | 2.0724 |

Phase 2.5 把所有最终结构的 P/Si 四配位率保持为 100%，但“CN=4”不等于“理想四面体”。其角度 RMS 和 CSM 仍显著高于初始 MQ，因为当前目标没有约束键角、CSM 或体积。

## 5. 机器可读数据与限制

Machine-readable outputs: geometry_aggregate.csv, geometry_by_structure.json, tetrahedra_by_site.csv, bond_lengths_by_species.csv, and bond_lengths_by_species_aggregate.csv.

Limitation: the Windows local executor reported setup refresh errors, so get_site_info_v3.py could not be read. The CSM definition used here is stated explicitly and is internally reproducible, but numerical equivalence to that external script is not claimed. Fourfold fraction, cutoff bond lengths, bond angles, and geometric volumes are unaffected by this CSM convention.
