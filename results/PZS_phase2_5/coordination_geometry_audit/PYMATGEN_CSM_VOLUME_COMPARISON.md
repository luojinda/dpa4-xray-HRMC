# pymatgen CSM / volume 对照测试

## 测试范围

- 环境：AutoDL，Python 环境 `dpa4-xray-hrmc`
- pymatgen：2026.10.24
- scipy：1.17.1
- 参考脚本：`tests/get_site_info_v3.py`
- 测试结构：初始 MQ、Phase 2 `no_scale/seed_101`、Phase 2.5 `no_scale/seed_101`、Phase 2.5 `fixed_initial/seed_303`
- 每个结构选择 2 个 P 四面体和 2 个 Si 四面体，共 16 个例子。

参考脚本的两个核心调用是：

- CSM：`LocalGeometryFinder(...).get_coordination_symmetry_measures()["T:4"]["csm"]`
- 体积：`scipy.spatial.ConvexHull(neigh_coords).volume`

## 结论

### Volume：一致

此前审计使用四点行列式体积：

`abs(det(v1-v0, v2-v0, v3-v0))/6`

16 个例子中，它与 `ConvexHull(...).volume` 的最大绝对差为 `7.11e-15 Å³`。这只是浮点舍入误差，所以 volume 定义和数值一致。

### CSM：默认值不一致，但固定中心版本完全一致

此前审计 CSM 与 pymatgen 的 `csm_wocs_csc` / `csm_wcs_csc` 最大绝对差为 `6.31e-14`，达到机器精度一致。

但是，`get_site_info_v3.py` 实际读取的是无后缀默认 `csm`。在当前 pymatgen 中，该值等于 `csm_wcs_ctwcc`：

- `wcs`：计算中包含中心原子；
- `ctwcc`：以“中心原子 + 4 个配体”这 5 个点的质心为中心。

此前审计采用：

- `wocs_csc` / `wcs_csc`：以中心原子位置为固定中心；加入位于原点的中心点不改变结果。

因此两者是不同的 CSM 定义。16 个例子中，审计固定中心 CSM 减去 pymatgen 默认 CSM 的差值范围为 `0.01045–4.63751`，平均差为 `1.32570`。不能把此前报告的 CSM 数值直接称为 `get_site_info_v3.py` 的默认 CSM；应称为“固定中心 CSM”。

## 部分逐位点结果

| structure | site | audit fixed-center CSM | pymatgen default CSM | volume determinant | volume ConvexHull |
|---|---|---:|---:|---:|---:|
| initial MQ | P 224 | 0.598857 | 0.186137 | 1.874250 | 1.874250 |
| initial MQ | Si 160 | 0.894123 | 0.736396 | 3.604919 | 3.604919 |
| Phase 2 no_scale/101 | P 224 | 1.276156 | 0.620853 | 1.757110 | 1.757110 |
| Phase 2 no_scale/101 | Si 162 | 1.905028 | 1.540307 | 2.188030 | 2.188030 |
| Phase 2.5 no_scale/101 | P 224 | 5.261525 | 3.497676 | 1.768721 | 1.768721 |
| Phase 2.5 no_scale/101 | Si 161 | 5.636861 | 2.374649 | 3.636005 | 3.636005 |
| Phase 2.5 fixed_initial/303 | P 224 | 3.820037 | 2.256772 | 1.778680 | 1.778680 |
| Phase 2.5 fixed_initial/303 | Si 160 | 7.801124 | 3.163614 | 3.615380 | 3.615380 |

完整 16 个例子保存在 `pymatgen_csm_volume_examples.json`；可复现脚本为 `tests/compare_pymatgen_csm_volume.py`。

## 对原 Phase 2.5 报告的影响

- 四配位率、键长、键角和体积结论不变。
- 原报告的 CSM 数值内部一致，也与 pymatgen 的固定中心 CSM 一致，但应修正名称，避免与参考脚本默认 CSM 混淆。
- 无论采用固定中心还是 pymatgen 默认 CSM，Phase 2.5 四面体仍比初始 MQ 更畸变；但两种 CSM 的绝对数值不可混用。

原始 `get_site_info_v3.py` 还依赖项目外的 `mapping_sublattice` 包，因此在 AutoDL 中直接运行整个文件会出现 `ModuleNotFoundError`。本对照测试直接复用了脚本中实际负责 CSM 和 volume 的 pymatgen/scipy 调用，不依赖该外部包。
