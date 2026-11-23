# PZS MQ1 与 PZS MQ3 局部环境审计

## 输入与方法

- MQ1：`PDF/PZS_quench_222.cif`
- MQ3：`inputs/PZS_MQ_3/PZS_equilibrated.cif`
- 两者均为 896 原子，组成均为 Li96 Zr64 Si64 P32 Cl512 O128。
- 配位截断沿用 Phase 2.5：P–O 1.80 Å、Si–O 1.85 Å、Si–Cl 2.30 Å、Zr–O 2.50 Å、Zr–Cl 3.00 Å。
- Partial RDF：`dr=0.02 Å`，未额外高斯平滑。

## 晶胞与密度

| 指标 | PZS MQ1 | PZS MQ3 | 变化 |
|---|---:|---:|---:|
| a (Å) | 29.3800 | 30.3740 | +3.383% |
| b (Å) | 24.7420 | 25.5791 | +3.383% |
| c (Å) | 28.0000 | 28.9473 | +3.383% |
| 体积 (Å³) | 20353.759 | 22490.375 | +10.497% |
| 密度 (g cm⁻³) | 2.40603 | 2.17745 | −9.500% |

MQ3 是相对于 MQ1 的近似各向同性膨胀结构。

## 配位分布

| 指标 | PZS MQ1 | PZS MQ3 |
|---|---:|---:|
| P CN3 / CN4 | 0 / 32 | 1 / 31 |
| P 四配位率 | 100% | 96.875% |
| Si CN4 | 64 | 64 |
| Si 四配位率 | 100% | 100% |
| Zr total CN5 / CN6 / CN7 | 1 / 63 / 0 | 6 / 55 / 3 |
| Zr–O CN0 / CN1 / CN2 / CN3 / CN4 | 19 / 23 / 18 / 4 / 0 | 25 / 20 / 11 / 7 / 1 |
| 平均 Zr total CN | 5.9844 | 5.9531 |
| 平均 Zr–O CN | 1.1094 | 1.0469 |
| 平均 Zr–Cl CN | 4.8750 | 4.9063 |

MQ3 的平均 Zr 总配位数变化很小，但分布明显变宽：CN5 增多并出现 3 个 CN7；Zr–O 从以 CN1–2 为主变成同时包含更多 CN0、CN3 和一个 CN4，说明 Zr 周围 O/Cl 分配更不均匀。

## 四面体几何

| 指标 | PZS MQ1 | PZS MQ3 |
|---|---:|---:|
| P 四面体数 | 32 | 31 |
| P angle RMS (°) | 3.6721 | 4.8664 |
| P CSM (%) | 0.35886 | 0.62881 |
| P 四面体体积 (Å³) | 1.87822 | 1.89469 |
| Si 四面体数 | 64 | 64 |
| Si angle RMS (°) | 2.9490 | 4.0459 |
| Si CSM (%) | 0.60128 | 0.74508 |
| Si 四面体体积 (Å³) | 3.83199 | 3.84804 |

MQ3 的 P angle RMS 增加 1.194°，P CSM 增加约 75.2%；Si angle RMS 增加 1.097°，Si CSM 增加约 23.9%。因此 MQ3 的 P、Si 四面体均比 MQ1 更畸变，其中 P 的变化更明显。

## 第一壳层键长

| 键 | MQ1 mean ± std (Å) | MQ3 mean ± std (Å) | MQ3 min–max (Å) |
|---|---:|---:|---:|
| P–O | 1.54405 ± 0.03292 | 1.55111 ± 0.04584 | 1.46770–1.66325 |
| Si–O | 1.65124 ± 0.02355 | 1.65522 ± 0.02916 | 1.60323–1.70706 |
| Si–Cl | 2.02473 ± 0.02209 | 2.02768 ± 0.04031 | 1.92418–2.20231 |

平均键长只增加约 0.003–0.007 Å，但分布均变宽；Si–Cl 标准差从 0.0221 Å 增至 0.0403 Å，是最明显的键长展宽。

## Partial RDF / CN 结论

- P–O、Si–O 与 Si–Cl 第一峰位置接近 MQ1，但 MQ3 峰更宽、峰高更低，与键长标准差增大一致。
- MQ3 的平均第一壳 Zr–O 配位略低，而 Zr–Cl 略高。
- 约 3 Å 以后 MQ3 的累计配位数普遍低于 MQ1，主要与晶胞体积增加 10.5%、数密度降低有关。
- MQ3 并未破坏主体 P/Si 四面体网络，但引入了一个 P-CN3 位点，并显著增加了 Zr 配位异质性。

## 输出

- 审计 JSON：`results/PZS_MQ_3/mq1_mq3_local_environment/mq1_mq3_local_environment_audit.json`
- RDF/CN 图：`results/PZS_MQ_3/mq1_mq3_local_environment/mq1_mq3_partial_rdf_and_CN.png`
- 五组数值数据：同目录下 `Zr_Cl`、`Zr_O`、`P_O`、`Si_Cl`、`Si_O` CSV
- 可复现脚本：`scripts/audit_pzs_mq1_mq3.py`

## 1.4–10 Å X-ray PDF 对比

采用当前 Phase 2/3 的同一前向模型与 PZS ZXS_1 实验数据：Q=0.9–16.0 Å⁻¹、Q 步长 0.05 Å⁻¹、Biso=0.3 Å²、不使用 Lorch modification。

| 比较方式 | PZS MQ1 | PZS MQ3 |
|---|---:|---:|
| no-scale Rw，w=1 | 0.71603 | 0.51810 |
| 独立最优 scale w | 0.63386 | 0.83484 |
| 独立最优 scale 后 Rw | 0.51836 | 0.48849 |

在统一 no-scale 标尺下，MQ3 的 Rw 比 MQ1 低约 27.6%。分别允许独立最优 scale 后，MQ3 仍低约 5.8%。因此 MQ3 的主要优势来自其较低密度使计算 PDF 的绝对振幅更接近实验，但也存在较小而真实的曲线形状改善。

MQ1 与 MQ3 未缩放计算曲线的 Pearson 相关系数为 0.9524，说明整体峰序列相似；相对 MQ1 的 L2 曲线差异为 35.3%。MQ3 对约 3.4–3.7 Å 强峰的过高估计明显小于 MQ1，但 2.5 Å 主峰和 4 Å 之后的中程振荡仍与实验存在差异。

- PDF 指标：`results/PZS_MQ_3/pdf_mq1_mq3_1p4_10A/metrics.json`
- PDF 数表：`results/PZS_MQ_3/pdf_mq1_mq3_1p4_10A/pdf_curves.csv`
- PDF 对比图：`results/PZS_MQ_3/pdf_mq1_mq3_1p4_10A/pdf_mq1_mq3_1p4_10A.png`
- 可复现脚本：`scripts/compare_pzs_mq1_mq3_pdf.py`
