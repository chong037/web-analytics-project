"""
06_viz/visualize.py  (负责人: tzx)

作用: 统一出图。所有图保存到 figures/, 风格(配色、字体)集中在本文件顶部。
输入: data/processed/ 下各环节的输出
输出: figures/*.png

TODO(tzx):
  - 全图和局部图力导向布局: 按社区着色, 按中心性调节点大小, 单独标出 10 个 ego
  - 对比图: 中心性排名图、社区算法对比柱状图、鲁棒性曲线、角色分类散点图
  - 先用 edges.csv + nodes.csv 把绘图函数写好, 社区/中心性结果出来后只换数据
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

PALETTE = plt.get_cmap("tab20")   # 统一配色, 全组共用


def plot_graph(G: nx.Graph, color_by: dict | None = None, size_by: dict | None = None,
               highlight: set | None = None, path: Path | None = None, seed: int = 42):
    """力导向布局; color_by/size_by 为 {node: value}, highlight 为需要标出的节点集合(如 ego)。"""
    pos = nx.spring_layout(G, seed=seed)
    nodes = list(G.nodes())
    colors = [PALETTE(color_by[n] % 20) if color_by else "#888" for n in nodes]
    sizes = [20 + 400 * size_by[n] if size_by else 8 for n in nodes]
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=sizes, ax=ax)
    if highlight:
        nx.draw_networkx_nodes(G, pos, nodelist=list(highlight), node_color="none",
                               edgecolors="black", linewidths=1.5, node_size=120, ax=ax)
    ax.axis("off")
    if path:
        FIG.mkdir(exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def main():
    edges = pd.read_csv(PROC / "edges.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target")
    nodes = pd.read_csv(PROC / "nodes.csv")
    egos = set(nodes.loc[nodes["is_ego"] == 1, "node"])
    plot_graph(G, highlight=egos, path=FIG / "graph_overview.png")


if __name__ == "__main__":
    main()
