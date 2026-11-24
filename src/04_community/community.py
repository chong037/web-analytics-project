"""
04_community/community.py  (负责人: ly)

作用: 多种社区检测算法 + 与人工圈子对比评估。
输入: data/processed/edges.csv, data/processed/node_label.csv
输出: data/processed/community.csv  (列: node, louvain, label_propagation, ...  每列是一种算法的社区编号)

TODO(ly):
  - 加入 Infomap (pip install infomap) 和一个重叠社区算法 (如 SLPA)
  - 评估: 模块度、社区数与大小分布; 与 node_label.csv 的 NMI / ARI
    (node_label 为空的节点在计算 NMI/ARI 时要剔除; 标签来自 "最小圈子" 规则)
  - 偏差分析: 哪些圈子被拆开或合并, 哪些节点被分到别的圈子
"""
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms import community as nxc

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SEED = 42


def to_membership(parts) -> dict:
    """社区列表 -> {node: community_id}"""
    return {n: i for i, c in enumerate(parts) for n in c}


def main():
    edges = pd.read_csv(PROC / "edges.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target")

    results = {
        "louvain": nxc.louvain_communities(G, seed=SEED),
        "label_propagation": list(nxc.asyn_lpa_communities(G, seed=SEED)),
    }
    out = pd.DataFrame({"node": list(G.nodes())})
    for name, parts in results.items():
        out[name] = out["node"].map(to_membership(parts))
        print(f"{name}: {len(parts)} 个社区, 模块度 {nxc.modularity(G, parts):.4f}")
    out.to_csv(PROC / "community.csv", index=False)


if __name__ == "__main__":
    main()
