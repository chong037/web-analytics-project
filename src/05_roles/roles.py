"""
05_roles/roles.py  (负责人: dc)

作用: 角色分类 (Guimera & Amaral, 2005)。用两个指标区分 "社区内枢纽" 和 "跨社区桥梁"。

指标 (k_i 为节点 i 的度, k_is 为 i 连到社区 s 内的边数, s_i 为 i 所在的社区):
    社区内度 z 分数   z_i = (k_{i,s_i} - mean_{j in s_i}(k_{j,s_i})) / std_{j in s_i}(k_{j,s_i})
        衡量节点在自己社区内是不是 "枢纽"; 社区内度标准差为 0 时 z 记为 0。
    participation coefficient   P_i = 1 - sum_s (k_is / k_i)^2
        衡量节点的边分散在多少个社区: 全在本社区 P = 0, 均匀连向很多社区 P 趋近 1; k_i = 0 时记 0。

七种角色 (Guimera-Amaral):
    非枢纽 (z < 2.5):  R1 超边缘 P <= 0.05;  R2 边缘 0.05 < P <= 0.625;  R3 连接者 0.625 < P <= 0.8;  R4 无社区归属 P > 0.8
    枢纽   (z >= 2.5): R5 社区内枢纽 P <= 0.3;  R6 连接枢纽 0.3 < P <= 0.75;  R7 无社区归属枢纽 P > 0.75

本项目的四类中文角色:
    社区内枢纽 = R5
    跨社区枢纽 = R6, R7
    跨社区桥梁 = R3, R4  (不是枢纽, 但边明显分散在多个社区)
    普通节点   = R1, R2

输入: data/processed/edges.csv, data/processed/community.csv (由 ly 的 04_community/community.py 产生; 这里只读取)
输出: data/processed/roles.csv, data/processed/roles_summary.csv, data/processed/roles_egos.csv
      figures/roles_scatter.png, figures/roles_counts.png

运行: python src/05_roles/roles.py [--col louvain]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

Z_HUB = 2.5
ROLE7 = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
ROLE7_NAME = {
    "R1": "R1 ultra-peripheral", "R2": "R2 peripheral", "R3": "R3 connector", "R4": "R4 kinless",
    "R5": "R5 provincial hub", "R6": "R6 connector hub", "R7": "R7 kinless hub",
}
ROLE_CN = {"R1": "普通节点", "R2": "普通节点", "R3": "跨社区桥梁", "R4": "跨社区桥梁",
           "R5": "社区内枢纽", "R6": "跨社区枢纽", "R7": "跨社区枢纽"}


def role7(z: float, p: float) -> str:
    """按 Guimera-Amaral 阈值返回 R1..R7。"""
    if z >= Z_HUB:
        return "R5" if p <= 0.3 else ("R6" if p <= 0.75 else "R7")
    if p <= 0.05:
        return "R1"
    if p <= 0.625:
        return "R2"
    return "R3" if p <= 0.8 else "R4"


def compute_roles(edges: pd.DataFrame, comm: pd.Series) -> pd.DataFrame:
    """edges: source,target (无向); comm: index=node, value=社区编号。返回逐节点的 z、P 和角色。"""
    # 无向边展开成 (节点, 邻居) 的两个方向, 再统计每个节点连到各社区的边数 k_is
    a = pd.concat([edges.rename(columns={"source": "node", "target": "nbr"}),
                   edges.rename(columns={"source": "nbr", "target": "node"})], ignore_index=True)
    a["nbr_comm"] = a["nbr"].map(comm)
    k_is = a.groupby(["node", "nbr_comm"]).size().rename("k_is").reset_index()

    deg = a.groupby("node").size().rename("degree")
    df = pd.DataFrame({"node": comm.index, "community": comm.values}).set_index("node")
    df["degree"] = deg.reindex(df.index).fillna(0).astype(int)

    # 社区内度 k_{i,s_i}
    own = k_is[k_is["nbr_comm"] == k_is["node"].map(comm)].set_index("node")["k_is"]
    df["k_in"] = own.reindex(df.index).fillna(0)
    g = df.groupby("community")["k_in"]
    mean, std = g.transform("mean"), g.transform("std", ddof=0)
    df["within_module_z"] = np.where(std > 0, (df["k_in"] - mean) / std.replace(0, np.nan), 0.0)

    # participation coefficient
    k_is["frac_sq"] = (k_is["k_is"] / k_is["node"].map(deg)) ** 2
    df["participation_coeff"] = (1 - k_is.groupby("node")["frac_sq"].sum()).reindex(df.index).fillna(0.0)
    df.loc[df["degree"] == 0, "participation_coeff"] = 0.0

    df["role7"] = [role7(z, p) for z, p in zip(df["within_module_z"], df["participation_coeff"])]
    df["role"] = df["role7"].map(ROLE_CN)
    return df.reset_index().drop(columns="k_in")


def plot_scatter(df: pd.DataFrame, egos: set, col: str):
    """z-P 平面散点图: 带 Guimera-Amaral 分区线, 标出 10 个 ego。"""
    FIG.mkdir(exist_ok=True)
    colors = {"R1": "#BDBDBD", "R2": "#9ECAE1", "R3": "#F58518", "R4": "#E45756",
              "R5": "#4C78A8", "R6": "#54A24B", "R7": "#B279A2"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r in ROLE7:
        d = df[df["role7"] == r]
        ax.scatter(d["participation_coeff"], d["within_module_z"], s=10, alpha=0.7,
                   color=colors[r], label=f"{ROLE7_NAME[r]} (n={len(d)})", linewidths=0)
    e = df[df["node"].isin(egos)]
    ax.scatter(e["participation_coeff"], e["within_module_z"], s=70, facecolors="none",
               edgecolors="black", linewidths=1.3, label="ego (10 nodes)")
    for _, r in e.iterrows():
        ax.annotate(str(int(r["node"])), (r["participation_coeff"], r["within_module_z"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    # 分区线
    ax.axhline(Z_HUB, color="black", lw=0.8, ls="--")
    xmax = ax.get_ylim()[1]
    for x in (0.05, 0.625, 0.8):
        ax.plot([x, x], [ax.get_ylim()[0], Z_HUB], color="black", lw=0.6, ls=":")
    for x in (0.3, 0.75):
        ax.plot([x, x], [Z_HUB, xmax], color="black", lw=0.6, ls=":")
    ax.set_xlabel("Participation coefficient P")
    ax.set_ylabel("Within-module degree z-score")
    ax.set_title(f"Node roles (Guimerà–Amaral), communities = {col}")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "roles_scatter.png", dpi=200)
    plt.close(fig)


def plot_counts(df: pd.DataFrame):
    order = ["普通节点", "跨社区桥梁", "社区内枢纽", "跨社区枢纽"]
    en = {"普通节点": "Ordinary", "跨社区桥梁": "Cross-community bridge",
          "社区内枢纽": "Provincial hub", "跨社区枢纽": "Cross-community hub"}
    cnt = df["role"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([en[o] for o in order], cnt.values, color=["#BDBDBD", "#F58518", "#4C78A8", "#54A24B"])
    ax.bar_label(bars)
    ax.set_ylabel("Number of nodes")
    ax.set_title("Role counts")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "roles_counts.png", dpi=200)
    plt.close(fig)


def main(col: str):
    edges = pd.read_csv(PROC / "edges.csv")
    comm_df = pd.read_csv(PROC / "community.csv")
    if col not in comm_df.columns:
        raise SystemExit(f"community.csv 里没有列 {col!r}, 可选: {list(comm_df.columns[1:])}")
    comm = comm_df.set_index("node")[col]

    df = compute_roles(edges, comm)
    df.to_csv(PROC / "roles.csv", index=False,
              columns=["node", "community", "degree", "within_module_z", "participation_coeff", "role7", "role"])

    nodes = pd.read_csv(PROC / "nodes.csv")
    egos = set(nodes.loc[nodes["is_ego"] == 1, "node"])
    summary = (df.groupby(["role", "role7"]).size().rename("n_nodes").reset_index())
    summary["share"] = (summary["n_nodes"] / len(df)).round(4)
    summary.to_csv(PROC / "roles_summary.csv", index=False)
    ego_df = df[df["node"].isin(egos)].sort_values("node")
    ego_df.to_csv(PROC / "roles_egos.csv", index=False)

    plot_scatter(df, egos, col)
    plot_counts(df)
    print(f"communities = {col}: {df['community'].nunique()} 个社区")
    print(summary.to_string(index=False))
    print("\n10 个 ego 的角色:")
    print(ego_df[["node", "community", "degree", "within_module_z", "participation_coeff", "role7", "role"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="louvain", help="community.csv 中使用的列")
    main(ap.parse_args().col)
