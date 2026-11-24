"""
01_data/data_summary.py  (负责人: dc)

作用: 生成 PPT "数据" 部分所需的汇总表和图。
输入: data/processed/{nodes,circles,ego_members,featnames}.csv, data/raw/facebook.tar.gz
输出: data/processed/ego_summary.csv
      figures/ego_overview.png

ego_summary.csv 列:
    ego, n_alters (ego 的好友数), n_alter_edges (好友之间的边数, 来自 .edges),
    n_circles, n_labelled (出现在该 ego 圈子中的节点数), label_coverage, n_features (特征维度)

运行: python src/01_data/data_summary.py
"""
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"


def count_alter_edges() -> dict:
    """每个 ego 的 .edges 文件行数 (好友之间的边, 不含 ego 自己)。"""
    out = {}
    with tarfile.open(RAW / "facebook.tar.gz", "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".edges"):
                ego = int(Path(m.name).name.split(".")[0])
                lines = [l for l in tar.extractfile(m).read().decode().splitlines() if l.strip()]
                out[ego] = len(lines)
    return out


def main():
    members = pd.read_csv(PROC / "ego_members.csv")
    circles = pd.read_csv(PROC / "circles.csv")
    featnames = pd.read_csv(PROC / "featnames.csv")
    alter_edges = count_alter_edges()

    rows = []
    for ego, g in members.groupby("ego"):
        c = circles[circles["ego"] == ego]
        rows.append({
            "ego": ego,
            "n_alters": g["node"].nunique(),
            "n_alter_edges": alter_edges[ego],
            "n_circles": c["circle"].nunique(),
            "n_labelled": c["node"].nunique(),
            "n_features": int((featnames["ego"] == ego).sum()),
        })
    df = pd.DataFrame(rows)
    df["label_coverage"] = (df["n_labelled"] / df["n_alters"]).round(3)
    df.to_csv(PROC / "ego_summary.csv", index=False)
    print(df.to_string(index=False))

    # 图: 每个 ego 的好友数与有圈子标签的节点数
    FIG.mkdir(exist_ok=True)
    d = df.sort_values("n_alters", ascending=False)
    x = range(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - 0.2 for i in x], d["n_alters"], width=0.4, label="Friends (alters)", color="#4C78A8")
    ax.bar([i + 0.2 for i in x], d["n_labelled"], width=0.4, label="Alters in a labelled circle", color="#F58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["ego"].astype(str))
    ax.set_xlabel("Ego node id")
    ax.set_ylabel("Number of nodes")
    ax.set_title("The 10 ego networks: size and label coverage")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "ego_overview.png", dpi=200)


if __name__ == "__main__":
    main()
