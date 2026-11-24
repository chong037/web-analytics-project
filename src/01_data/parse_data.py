"""
01_data/parse_data.py  (负责人: dc)

作用: 解析 SNAP ego-Facebook 原始数据, 输出后续所有环节共用的统一格式文件。

输入 (data/raw/):
    facebook_combined.txt.gz  合并后的无向边表, 每行 "u v"
    facebook.tar.gz           10 个 ego 的 .edges / .circles / .feat / .egofeat / .featnames

输出 (data/processed/):
    edges.csv         列: source, target                  (无向, 已去重, source < target)
    nodes.csv         列: node, is_ego, n_circles         (n_circles = 该节点所属圈子总数)
    circles.csv       列: ego, circle, node               (一行一个 "节点属于某 ego 的某圈子")
    node_label.csv    列: node, label                     (每个节点取其"最小圈子"作为单一标签, 无圈子则为空)
    features.csv      列: ego, node, feature_id           (稀疏表示: 值为 1 的特征; 特征编号只在同一 ego 内有意义)
    featnames.csv     列: ego, feature_id, name, category (category 如 education;school;id, 跨 ego 一致)
    ego_members.csv   列: ego, node                       (每个 ego 的全部 alter, 含特征全为 0 的节点)

运行: python src/01_data/parse_data.py
"""
import gzip
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_edges() -> pd.DataFrame:
    """读取合并边表, 统一成 source < target 并去重。"""
    with gzip.open(RAW / "facebook_combined.txt.gz", "rt") as f:
        df = pd.read_csv(f, sep=" ", header=None, names=["source", "target"])
    lo = df.min(axis=1)
    hi = df.max(axis=1)
    return pd.DataFrame({"source": lo, "target": hi}).drop_duplicates().reset_index(drop=True)


def parse_ego_files():
    """遍历 tar 包, 返回 (egos, circles, feats, ego_members, featnames)。"""
    egos, circle_rows, feat_rows, member_rows, name_rows = [], [], [], [], []
    with tarfile.open(RAW / "facebook.tar.gz", "r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        egos = sorted({int(n.split(".")[0]) for n in members if n.endswith(".edges")})
        for ego in egos:
            # circles: 每行第一项是圈子名, 其余是节点编号
            for line in tar.extractfile(members[f"{ego}.circles"]).read().decode().splitlines():
                parts = line.split()
                if not parts:
                    continue
                name = parts[0]
                for n in parts[1:]:
                    circle_rows.append((ego, name, int(n)))
            # feat: 每行 "节点编号 0/1 0/1 ...", 只保留值为 1 的特征
            for line in tar.extractfile(members[f"{ego}.feat"]).read().decode().splitlines():
                parts = line.split()
                if not parts:
                    continue
                node = int(parts[0])
                member_rows.append((ego, node))
                for fid, v in enumerate(parts[1:]):
                    if v == "1":
                        feat_rows.append((ego, node, fid))
            # featnames: 每行 "编号 名称", 名称形如 "education;school;id;anonymized feature 53"
            for line in tar.extractfile(members[f"{ego}.featnames"]).read().decode().splitlines():
                if not line.strip():
                    continue
                fid, name = line.split(" ", 1)
                category = name.split("anonymized feature")[0].rstrip("; ")
                name_rows.append((ego, int(fid), name, category))
    circles = pd.DataFrame(circle_rows, columns=["ego", "circle", "node"])
    feats = pd.DataFrame(feat_rows, columns=["ego", "node", "feature_id"])
    members_df = pd.DataFrame(member_rows, columns=["ego", "node"])
    names_df = pd.DataFrame(name_rows, columns=["ego", "feature_id", "name", "category"])
    return egos, circles, feats, members_df, names_df


def smallest_circle_label(circles: pd.DataFrame) -> pd.DataFrame:
    """每个节点取其所属的"最小圈子"(成员数最少)作为单一标签, 用于 NMI/ARI 评估。"""
    circles = circles.copy()
    circles["circle_id"] = circles["ego"].astype(str) + ":" + circles["circle"]
    size = circles.groupby("circle_id")["node"].nunique().rename("size")
    circles = circles.join(size, on="circle_id")
    best = circles.sort_values(["node", "size", "circle_id"]).drop_duplicates("node")
    return best[["node", "circle_id"]].rename(columns={"circle_id": "label"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    edges = load_edges()
    egos, circles, feats, members_df, names_df = parse_ego_files()

    all_nodes = sorted(set(edges["source"]) | set(edges["target"]))
    n_circ = circles.groupby("node")["circle"].count()
    nodes = pd.DataFrame({"node": all_nodes})
    nodes["is_ego"] = nodes["node"].isin(egos).astype(int)
    nodes["n_circles"] = nodes["node"].map(n_circ).fillna(0).astype(int)

    labels = smallest_circle_label(circles)
    label_df = nodes[["node"]].merge(labels, on="node", how="left")

    edges.to_csv(OUT / "edges.csv", index=False)
    nodes.to_csv(OUT / "nodes.csv", index=False)
    circles.to_csv(OUT / "circles.csv", index=False)
    label_df.to_csv(OUT / "node_label.csv", index=False)
    feats.to_csv(OUT / "features.csv", index=False)
    names_df.to_csv(OUT / "featnames.csv", index=False)
    members_df.to_csv(OUT / "ego_members.csv", index=False)

    print(f"edges={len(edges)}  nodes={len(nodes)}  egos={egos}")
    print(f"circles={circles.groupby(['ego','circle']).ngroups}  "
          f"labelled nodes={label_df['label'].notna().sum()}")


if __name__ == "__main__":
    main()
