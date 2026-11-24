"""
05_roles/homophily.py  (负责人: dc)

作用: 特征同质性分析, 检验 "资料特征相似的人, 是否更容易在同一社区 / 同一圈子"。

关键约束:
  * 特征编号只在同一个 ego 内有意义, 所以所有检验都在每个 ego 内部做, 不跨 ego 合并特征列。
  * 特征的"类别"(如 education;school;id、work;employer;id、hometown;id)跨 ego 一致, 所以按类别汇总时可以把各 ego 的节点对计数相加。

两种 "同组" 定义:
  * 同社区: 两个节点在 community.csv 指定列中的社区编号相同 (用该 ego 的全部好友)
  * 同圈子: 两个节点在该 ego 的某个人工圈子里同时出现 (只用该 ego 圈子里出现过的节点, 因为没标签就无法判断)

检验方法 (H3):
  节点对相似度 = Jaccard(两人特征集合), 并集为空时记 0。
  统计量 = 同组节点对的平均相似度 - 异组节点对的平均相似度。
  置换检验: 随机打乱节点身份 500 次 (种子 42), 单侧 p = (1 + 比观测值更大或相等的次数) / (1 + 500)。

按类别汇总 (H4):
  对每个特征类别, 统计 "共享该类别任一特征" 的节点对, 其同组概率 P(同组|共享) 与不共享节点对的 P(同组|不共享) 之比 (lift)。

输入: data/processed/{community,features,featnames,ego_members,circles}.csv
输出: data/processed/homophily_by_ego.csv, homophily_by_category.csv
      figures/homophily_by_category.png

运行: python src/05_roles/homophily.py [--col louvain]
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
SEED = 42
N_PERM = 500


def jaccard_matrix(X: np.ndarray) -> np.ndarray:
    """X: (n, d) 的 0/1 矩阵, 返回 n x n 的 Jaccard 相似度, 并集为空记 0。"""
    X = X.astype(np.float32)
    inter = X @ X.T
    size = X.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J


def perm_test(J: np.ndarray, same: np.ndarray, rng) -> dict:
    """J: 相似度矩阵; same: 同组布尔矩阵 (同一节点顺序)。返回观测差值和置换检验 p 值。"""
    n = J.shape[0]
    iu = np.triu_indices(n, k=1)
    Jv = J[iu]

    def stat(mask):
        if mask.all() or (~mask).all():
            return np.nan
        return Jv[mask].mean() - Jv[~mask].mean()

    mask = same[iu]
    obs = stat(mask)
    out = {"n_pairs": len(Jv), "n_same_pairs": int(mask.sum()),
           "mean_sim_same": Jv[mask].mean() if mask.any() else np.nan,
           "mean_sim_diff": Jv[~mask].mean() if (~mask).any() else np.nan,
           "diff": obs, "p_value": np.nan, "z": np.nan}
    if np.isnan(obs):
        return out
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        p = rng.permutation(n)
        null[b] = stat(same[np.ix_(p, p)][iu])
    out["p_value"] = (1 + np.sum(null >= obs)) / (1 + N_PERM)
    sd = null.std()
    out["z"] = (obs - null.mean()) / sd if sd > 0 else np.nan
    return out


def main(col: str):
    rng = np.random.default_rng(SEED)
    comm = pd.read_csv(PROC / "community.csv").set_index("node")[col]
    feats = pd.read_csv(PROC / "features.csv")
    names = pd.read_csv(PROC / "featnames.csv")
    members = pd.read_csv(PROC / "ego_members.csv")
    circles = pd.read_csv(PROC / "circles.csv")

    ego_rows, cat_rows = [], []
    for ego in sorted(members["ego"].unique()):
        nodes = np.sort(members.loc[members["ego"] == ego, "node"].unique())
        idx = {n: i for i, n in enumerate(nodes)}
        fn = names[names["ego"] == ego].sort_values("feature_id")
        X = np.zeros((len(nodes), len(fn)), dtype=np.uint8)
        f = feats[feats["ego"] == ego]
        X[f["node"].map(idx).to_numpy(), f["feature_id"].to_numpy()] = 1
        categories = fn["category"].to_numpy()

        # ---- 同社区 (全部 alter) ----
        c = comm.reindex(nodes).to_numpy()
        same_comm = c[:, None] == c[None, :]
        # ---- 同圈子 (只用出现在该 ego 圈子里的节点) ----
        cc = circles[circles["ego"] == ego]
        lab_nodes = np.sort(cc["node"].unique())
        lab_nodes = np.array([n for n in lab_nodes if n in idx])
        lab_idx = np.array([idx[n] for n in lab_nodes])
        pos = {n: i for i, n in enumerate(lab_nodes)}
        M = np.zeros((len(lab_nodes), cc["circle"].nunique()), dtype=np.uint8)
        circ_id = {c_: i for i, c_ in enumerate(sorted(cc["circle"].unique()))}
        for node, circle in zip(cc["node"], cc["circle"]):
            if node in pos:
                M[pos[node], circ_id[circle]] = 1
        same_circ = (M @ M.T) > 0

        J_all = jaccard_matrix(X)
        J_lab = J_all[np.ix_(lab_idx, lab_idx)]

        for kind, J, same, n_nodes in (("community", J_all, same_comm, len(nodes)),
                                       ("circle", J_lab, same_circ, len(lab_nodes))):
            res = perm_test(J, same, rng)
            ego_rows.append({"ego": ego, "group_type": kind, "n_nodes": n_nodes, **res})

            # ---- 按特征类别汇总 ----
            Xs = X if kind == "community" else X[lab_idx]
            iu = np.triu_indices(Xs.shape[0], k=1)
            same_v = same[iu]
            for cat in sorted(set(categories)):
                cols = np.where(categories == cat)[0]
                share = (Xs[:, cols].astype(np.float32) @ Xs[:, cols].astype(np.float32).T) > 0
                sh = share[iu]
                cat_rows.append({
                    "ego": ego, "group_type": kind, "category": cat,
                    "pairs_share": int(sh.sum()), "same_given_share": int((same_v & sh).sum()),
                    "pairs_noshare": int((~sh).sum()), "same_given_noshare": int((same_v & ~sh).sum()),
                })

    by_ego = pd.DataFrame(ego_rows)
    by_ego.to_csv(PROC / "homophily_by_ego.csv", index=False)

    cat = pd.DataFrame(cat_rows)
    agg = (cat.groupby(["group_type", "category"])[
        ["pairs_share", "same_given_share", "pairs_noshare", "same_given_noshare"]].sum().reset_index())
    agg["p_same_given_share"] = agg["same_given_share"] / agg["pairs_share"].replace(0, np.nan)
    agg["p_same_given_noshare"] = agg["same_given_noshare"] / agg["pairs_noshare"].replace(0, np.nan)
    agg["lift"] = agg["p_same_given_share"] / agg["p_same_given_noshare"]
    agg = agg.sort_values(["group_type", "lift"], ascending=[True, False])
    agg.to_csv(PROC / "homophily_by_category.csv", index=False)

    print(by_ego.round(4).to_string(index=False))
    plot_categories(agg, col)


def plot_categories(agg: pd.DataFrame, col: str):
    """每个类别的 lift (共享该类别特征的节点对, 同组概率是不共享节点对的多少倍); 只画共享节点对 >= 1000 的类别。"""
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for ax, kind, title in zip(axes, ("community", "circle"),
                               (f"Same community ({col})", "Same circle")):
        d = agg[(agg["group_type"] == kind) & (agg["pairs_share"] >= 1000)].sort_values("lift")
        ax.barh(d["category"], d["lift"], color="#4C78A8")
        ax.axvline(1, color="black", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Lift: P(same group | share) / P(same group | not share)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Which profile attributes predict being in the same group? (pooled over 10 egos)")
    fig.tight_layout()
    fig.savefig(FIG / "homophily_by_category.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="louvain", help="community.csv 中使用的列")
    main(ap.parse_args().col)
