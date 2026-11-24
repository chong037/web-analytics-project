"""
02_stats/basic_stats.py  (负责人: wkb)

作用: 基础网络统计, 描述整张图的整体特征。
输入: data/processed/edges.csv
输出: figures/degree_distribution.png, data/processed/basic_stats.csv

TODO(wkb):
  - 度分布 (线性坐标 + 双对数坐标), 检查是否近似幂律
  - 平均路径长度、直径 (图连通, 可精确计算; 若太慢可抽样)
  - 聚类系数 (平均、全局传递性)
  - 连通分量
"""
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_graph() -> nx.Graph:
    edges = pd.read_csv(PROC / "edges.csv")
    return nx.from_pandas_edgelist(edges, "source", "target")


def main():
    G = load_graph()
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "n_components": nx.number_connected_components(G),
    }
    # TODO: 平均路径长度、直径、度分布图
    pd.Series(stats).to_csv(PROC / "basic_stats.csv", header=["value"])
    print(stats)


if __name__ == "__main__":
    main()
