"""
03_centrality/centrality.py  (负责人: zrf)

作用: 计算多种中心性, 比较关键节点; 做鲁棒性实验。
输入: data/processed/edges.csv, data/processed/nodes.csv
输出: data/processed/centrality.csv  (列: node, degree, betweenness, closeness, pagerank, eigenvector)

TODO(zrf):
  - 各指标前 10~20 名, 排名差异, 指标间 Spearman 相关性
  - 检验最中心节点是否就是 10 个 ego (nodes.csv 的 is_ego)
  - 鲁棒性: 随机移除 vs 按中心性移除, 记录最大连通子图占比和平均路径长度变化
"""
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def main():
    edges = pd.read_csv(PROC / "edges.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target")
    df = pd.DataFrame({
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),   # 约 4000 节点, 精确计算可接受
        "closeness": nx.closeness_centrality(G),
        "pagerank": nx.pagerank(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
    })
    df.index.name = "node"
    df.reset_index().to_csv(PROC / "centrality.csv", index=False)
    print(df.sort_values("betweenness", ascending=False).head(10))


if __name__ == "__main__":
    main()
