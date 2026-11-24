# dc 负责部分：需求清单

对应分工表里 dc 的四项任务。每一条都有编号，由 `tests/check_dc.py` 逐条自动检查（运行后输出 PASS / FAIL）。

## 一、数据解析与建图（任务 1，11/25 前交付）

| 编号 | 需求 |
|---|---|
| D1 | `edges.csv`：列为 source, target；无向、去重、无自环、source < target；与 facebook_combined 的无向边集完全一致（88,234 条） |
| D2 | `nodes.csv`：列为 node, is_ego, n_circles；共 4,039 个节点，与边表中出现的节点集一致；is_ego 恰好标出 10 个 ego（0、107、348、414、686、698、1684、1912、3437、3980） |
| D3 | `circles.csv`：列为 ego, circle, node；共 193 个圈子、覆盖 2,884 个不同节点；所有节点都在 nodes.csv 中；`n_circles` 与它一致 |
| D4 | `node_label.csv`：每个节点一行（4,039 行），无圈子的节点标签为空（1,155 个）；有圈子的节点取其所属圈子中成员数最少的一个，成员数相同时按圈子编号字典序取第一个 |
| D5 | `features.csv` 与 `featnames.csv`：特征按 ego 分别保存，不跨 ego 合并；每个特征都能在 featnames 中找到其类别（如 education;school;id）；`ego_members.csv` 保留每个 ego 的所有 alter（包含全 0 特征的节点） |
| D6 | 一致性校验：每个 ego 的 .edges 文件里的边都在合并图中；合并图里每个 ego 都与其 .edges 中出现的每个节点相连（ego 在 .edges 里不出现，但默认连接所有 alter） |

## 二、特征同质性（任务 2）

| 编号 | 需求 |
|---|---|
| H1 | 按 ego 分别计算，不把不同 ego 的特征编号混在一起 |
| H2 | 检验两件事：特征相似的人是否更容易在同一社区、是否更容易在同一圈子 |
| H3 | 每个 ego 输出：节点数、节点对数、同组与异组的平均相似度（Jaccard）、差值、置换检验 p 值（500 次，固定随机种子） |
| H4 | 按特征类别（education;school、work;employer、hometown、location 等）汇总：共享该类别特征的节点对与不共享的节点对，同社区或同圈子的概率之比；类别名跨 ego 一致，可汇总 |
| H5 | 圈子分析只使用该 ego 圈子里出现过的节点（有标签才能判断）；社区分析使用该 ego 的全部 alter |
| H6 | 输出 `homophily_by_ego.csv`、`homophily_by_category.csv` 和一张图 |

## 三、角色分类（任务 3，依赖 ly 的 community.csv）

| 编号 | 需求 |
|---|---|
| R1 | 输入为 `edges.csv` 和 `community.csv` 中指定的一列（默认 louvain），其他算法可通过参数切换 |
| R2 | 社区内度 z 分数：z_i = (k_i − 该社区度均值) / 该社区度标准差；标准差为 0 的社区 z 记为 0 |
| R3 | participation coefficient：P_i = 1 − Σ_s (k_is / k_i)²；k_i = 0 时记为 0 |
| R4 | 按 Guimerà–Amaral 标准划分 R1–R7：z ≥ 2.5 为枢纽；非枢纽 P 分界 0.05 / 0.625 / 0.8，枢纽 P 分界 0.3 / 0.75 |
| R5 | 另给出四类中文角色：社区内枢纽（R5）、跨社区枢纽（R6、R7）、跨社区桥梁（非枢纽且 P > 0.625，即 R3、R4）、普通节点（R1、R2） |
| R6 | `roles.csv`：列为 node, community, degree, within_module_z, participation_coeff, role7, role；每个节点一行 |
| R7 | 数值抽查：随机取若干节点，用独立的写法重新算 z 和 P，与输出一致；P 取值在 [0, 1) 内 |
| R8 | 输出 z–P 散点图（带分区线，标出 10 个 ego）和各角色的数量统计；说明 10 个 ego 分别落在哪类角色 |

## 四、PPT「数据」与「角色分类」部分（任务 4）

| 编号 | 需求 |
|---|---|
| P1 | 数据部分：数据来源与规模、10 个 ego 的概况（alter 数、圈子数、标签覆盖、特征维度）、标签覆盖率 29% 缺失的说明，以及特征只在 ego 内有意义的注意事项 |
| P2 | 角色分类部分：方法（z、P 的含义和公式）、结果（各角色数量、散点图）、关键节点的解读 |
| P3 | 所有数字来自上面输出的文件，不手写 |

## 通用要求

| 编号 | 需求 |
|---|---|
| G1 | 所有脚本有注释；随机种子固定为 42；可从头一键重跑 |
| G2 | 不改动其他人负责的文件（community.csv 由 ly 的脚本产生，这里只读取） |
