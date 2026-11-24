# MH8351 Web Analytics Project: 社区检测与关键节点分析 (ego-Facebook)

在 SNAP ego-Facebook 网络（4039 个用户、88234 条好友关系、10 个 ego 的 193 个人工标注圈子）上检测社区，并找出各社区的关键人物。

- 提交截止：12/6 23:59（一个 zip，由一位组员提交）
- 课堂展示：12/8

## 运行顺序

```
pip install -r requirements.txt
python src/01_data/parse_data.py        # 生成 data/processed/ 下统一格式的数据
python src/01_data/data_summary.py      # 10 个 ego 的汇总表和图 (PPT 数据部分)
python src/02_stats/basic_stats.py      # 基础网络统计
python src/03_centrality/centrality.py  # 中心性
python src/04_community/community.py    # 社区检测 -> community.csv
python src/05_roles/homophily.py        # 特征同质性 (依赖 community.csv)
python src/05_roles/roles.py            # 角色分类 (依赖 community.csv)
python src/06_viz/visualize.py          # 出图 -> figures/
```

## 分工

| 成员 | 模块 | 目录 |
|---|---|---|
| wkb | 问题定义、基础统计、整合提交 | src/02_stats |
| dc | 数据解析、特征同质性、角色分类 | src/01_data, src/05_roles |
| zrf | 中心性与鲁棒性 | src/03_centrality |
| ly | 社区检测与评估 | src/04_community |
| tzx | 可视化与 PPT 统稿 | src/06_viz, slides |

## 数据接口 (data/processed/)

| 文件 | 列 | 产出者 |
|---|---|---|
| edges.csv | source, target（无向、去重、source < target） | dc |
| nodes.csv | node, is_ego, n_circles | dc |
| circles.csv | ego, circle, node（节点属于某 ego 的某圈子） | dc |
| node_label.csv | node, label（取“最小圈子”，无圈子为空；用于 NMI/ARI） | dc |
| features.csv | ego, node, feature_id（只保留值为 1 的特征） | dc |
| featnames.csv | ego, feature_id, name, category（类别如 education;school;id，跨 ego 一致） | dc |
| ego_members.csv | ego, node（每个 ego 的全部好友，含特征全为 0 的节点） | dc |
| ego_summary.csv | 每个 ego 的好友数、边数、圈子数、标签覆盖、特征维度 | dc |
| homophily_by_ego.csv / homophily_by_category.csv | 特征同质性检验结果（按 ego；按特征类别汇总） | dc |
| centrality.csv | node, degree, betweenness, closeness, pagerank, eigenvector | zrf |
| community.csv | node, 每种算法一列的社区编号 | ly |
| roles.csv | node, community, degree, within_module_z, participation_coeff, role7, role | dc |
| roles_summary.csv / roles_egos.csv | 各角色数量；10 个 ego 的角色 | dc |

注意：
- 圈子可以重叠，约 29% 的节点没有圈子标签，评估时要剔除。
- 特征编号只在同一个 ego 内有意义，不同 ego 的特征不能直接合并。
- 随机种子统一用 42。

## dc 部分的需求与自检

- 需求清单：`docs/dc_requirements.md`（编号 D1–D6、H1–H6、R1–R8、P1–P3、G1–G2）
- 自检：`python tests/check_dc.py`（数据、同质性、角色分类，另用独立写法重算）；
  `python tests/check_dc_slides.py <deck 目录>`（PPT 部分，deck 目录由 `slides/build_dc_slides.py` 生成）
- `community.csv` 目前由 `src/04_community/community.py` 的 Louvain 结果提供；ly 若更换算法，重跑 `homophily.py` 和 `roles.py` 即可（`--col` 选列）。

## 协作约定

- `main` 保持可运行。每人在自己的分支（如 `ly/community`）开发，完成后提 Pull Request，由 wkb 合并。
- 每人只改自己目录下的文件；代码写好注释。
- 数据文件很小，直接放在仓库里（data/raw）。

## 数据来源

J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.
https://snap.stanford.edu/data/ego-Facebook.html
