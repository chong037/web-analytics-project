"""
tests/check_dc.py

逐条检查 docs/dc_requirements.md 里 dc 负责部分的需求 (D1-D6, H1-H6, R1-R8, G1-G2)。
校验用的是"另一套独立写法": 直接从原始文件重新计算, 不复用 src/ 里的解析函数, 以免同一个错误被两边同时带过。

运行 (在仓库根目录): python tests/check_dc.py
全部通过时退出码为 0, 否则为 1。
"""
import gzip
import hashlib
import math
import random
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
EGOS = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

results = []


def check(code: str, desc: str, ok: bool, detail: str = ""):
    results.append((code, desc, bool(ok), detail))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 原始数据 (独立读取)
def read_raw():
    comb = set()
    with gzip.open(RAW / "facebook_combined.txt.gz", "rt") as f:
        for line in f:
            a, b = map(int, line.split())
            comb.add((min(a, b), max(a, b)))
    ego_edges, circles, feats, featdim, featcat = {}, {}, {}, {}, {}
    with tarfile.open(RAW / "facebook.tar.gz", "r:gz") as tar:
        files = {Path(m.name).name: tar.extractfile(m).read().decode()
                 for m in tar.getmembers() if m.isfile()}
    for ego in EGOS:
        ego_edges[ego] = [tuple(map(int, l.split())) for l in files[f"{ego}.edges"].splitlines() if l.strip()]
        circ = {}
        for l in files[f"{ego}.circles"].splitlines():
            p = l.split()
            if p:
                circ[p[0]] = [int(x) for x in p[1:]]
        circles[ego] = circ
        rows = {}
        for l in files[f"{ego}.feat"].splitlines():
            p = l.split()
            if p:
                rows[int(p[0])] = [i for i, v in enumerate(p[1:]) if v == "1"]
        feats[ego] = rows
        names = [l.split(" ", 1) for l in files[f"{ego}.featnames"].splitlines() if l.strip()]
        featdim[ego] = len(names)
        featcat[ego] = {int(i): n.split("anonymized feature")[0].rstrip("; ") for i, n in names}
    return comb, ego_edges, circles, feats, featdim, featcat


def main():
    comb, ego_edges, circles, feats, featdim, featcat = read_raw()
    edges = pd.read_csv(PROC / "edges.csv")
    nodes = pd.read_csv(PROC / "nodes.csv")
    circ_df = pd.read_csv(PROC / "circles.csv")
    label = pd.read_csv(PROC / "node_label.csv")
    feat_df = pd.read_csv(PROC / "features.csv")
    names_df = pd.read_csv(PROC / "featnames.csv")
    members = pd.read_csv(PROC / "ego_members.csv")

    # ------------------------------------------------------------ D1
    es = set(zip(edges["source"], edges["target"]))
    check("D1", "edges.csv 列名为 source,target", list(edges.columns) == ["source", "target"])
    check("D1", "共 88,234 行且无重复", len(edges) == 88234 and len(es) == len(edges), f"{len(edges)} 行")
    check("D1", "无自环且 source < target", bool((edges["source"] < edges["target"]).all()))
    check("D1", "与 facebook_combined 的无向边集完全一致", es == comb)

    # ------------------------------------------------------------ D2
    node_set = {n for e in comb for n in e}
    check("D2", "nodes.csv 列名为 node,is_ego,n_circles", list(nodes.columns) == ["node", "is_ego", "n_circles"])
    check("D2", "共 4,039 个节点且无重复", len(nodes) == 4039 and nodes["node"].is_unique)
    check("D2", "节点集与边表一致", set(nodes["node"]) == node_set)
    check("D2", "is_ego 恰好标出 10 个 ego 且编号正确",
          sorted(nodes.loc[nodes["is_ego"] == 1, "node"]) == EGOS)

    # ------------------------------------------------------------ D3
    raw_circ = {(ego, c, n) for ego in EGOS for c, ns in circles[ego].items() for n in ns}
    got_circ = set(zip(circ_df["ego"], circ_df["circle"], circ_df["node"]))
    check("D3", "circles.csv 列名为 ego,circle,node", list(circ_df.columns) == ["ego", "circle", "node"])
    check("D3", "与原始 .circles 文件逐行一致", raw_circ == got_circ and len(circ_df) == len(got_circ))
    n_circles = len({(e, c) for e, c, _ in raw_circ})
    n_lab = len({n for _, _, n in raw_circ})
    check("D3", "193 个圈子、覆盖 2,884 个不同节点", n_circles == 193 and n_lab == 2884, f"{n_circles} / {n_lab}")
    check("D3", "圈子里的节点都在 nodes.csv 中", {n for _, _, n in raw_circ} <= set(nodes["node"]))
    cnt = defaultdict(int)
    for _, _, n in raw_circ:
        cnt[n] += 1
    check("D3", "nodes.n_circles 与圈子成员关系一致",
          all(int(r.n_circles) == cnt.get(r.node, 0) for r in nodes.itertuples()))

    # ------------------------------------------------------------ D4
    size = defaultdict(set)
    for e, c, n in raw_circ:
        size[(e, c)].add(n)
    best = {}
    for e, c, n in raw_circ:
        cand = (len(size[(e, c)]), f"{e}:{c}")
        if n not in best or cand < best[n]:
            best[n] = cand
    exp = {n: v[1] for n, v in best.items()}
    got = dict(zip(label["node"], label["label"]))
    check("D4", "node_label.csv 每个节点一行(4,039 行)", len(label) == 4039 and label["node"].is_unique
          and set(label["node"]) == node_set)
    check("D4", "无圈子的节点标签为空, 共 1,155 个", int(label["label"].isna().sum()) == 4039 - 2884 == 1155)
    check("D4", "有圈子的节点取成员数最少的圈子, 并列取字典序最小",
          all((got[n] == exp[n]) for n in exp) and all(pd.isna(got[n]) for n in node_set - set(exp)))

    # ------------------------------------------------------------ D5
    raw_feat = {(e, n, f) for e in EGOS for n, fs in feats[e].items() for f in fs}
    got_feat = set(zip(feat_df["ego"], feat_df["node"], feat_df["feature_id"]))
    check("D5", "features.csv 与原始 .feat 中值为 1 的项逐项一致", raw_feat == got_feat and len(feat_df) == len(raw_feat))
    check("D5", "特征编号都在该 ego 的维度内(未跨 ego 混用)",
          all(f < featdim[e] for e, _, f in got_feat))
    nm = {(r.ego, r.feature_id): r.category for r in names_df.itertuples()}
    check("D5", "featnames.csv 维度与原始一致且类别与独立解析一致",
          all(len([k for k in nm if k[0] == e]) == featdim[e] for e in EGOS)
          and all(nm[(e, i)] == c for e in EGOS for i, c in featcat[e].items()))
    check("D5", "每个特征都能找到类别", all((e, f) in nm and nm[(e, f)] for e, _, f in got_feat))
    raw_members = {(e, n) for e in EGOS for n in feats[e]}
    check("D5", "ego_members.csv 保留全部 alter(含特征全为 0 的节点)",
          set(zip(members["ego"], members["node"])) == raw_members)
    zero = sum(1 for e in EGOS for n, fs in feats[e].items() if not fs)
    inm = sum(1 for e in EGOS for n, fs in feats[e].items() if not fs
              and ((members["ego"] == e) & (members["node"] == n)).any())
    check("D5", "特征全为 0 的节点确实在 ego_members 里", inm == zero, f"{zero} 个")

    # ------------------------------------------------------------ D6
    ok1 = all((min(a, b), max(a, b)) in comb for e in EGOS for a, b in ego_edges[e])
    ok2 = all((min(e, n), max(e, n)) in comb for e in EGOS for n in {x for ab in ego_edges[e] for x in ab})
    check("D6", "每个 ego 的 .edges 里的边都在合并图中", ok1)
    check("D6", "合并图里每个 ego 都连着其 .edges 中出现的所有节点", ok2)

    # ------------------------------------------------------------ 重跑前记录 community.csv 的指纹 (G2)
    comm_hash = md5(PROC / "community.csv")

    # ------------------------------------------------------------ H
    he = pd.read_csv(PROC / "homophily_by_ego.csv")
    hc = pd.read_csv(PROC / "homophily_by_category.csv")
    check("H1", "10 个 ego × 2 种分组 = 20 行, 逐 ego 计算",
          len(he) == 20 and set(he["ego"]) == set(EGOS) and set(he["group_type"]) == {"community", "circle"})
    check("H2", "同时检验了同社区与同圈子", {"community", "circle"} <= set(he["group_type"]))
    cols = {"ego", "group_type", "n_nodes", "n_pairs", "n_same_pairs", "mean_sim_same", "mean_sim_diff",
            "diff", "p_value", "z"}
    check("H3", "输出节点数、节点对数、同组/异组平均相似度、差值、p 值", cols <= set(he.columns))
    pv = he["p_value"].dropna()
    check("H3", "p 值在 [1/501, 1] 内(500 次置换)", bool(((pv >= 1 / 501 - 1e-12) & (pv <= 1)).all()))

    # H3 独立复核: 在 ego 414 上用纯 Python 集合重新算 Jaccard 差值
    comm = pd.read_csv(PROC / "community.csv").set_index("node")["louvain"].to_dict()
    ego = 414
    ns = sorted(feats[ego])
    fs = {n: set(feats[ego][n]) for n in ns}

    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    s_same = s_diff = c_same = c_diff = 0
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            v = jac(fs[ns[i]], fs[ns[j]])
            if comm[ns[i]] == comm[ns[j]]:
                s_same += v; c_same += 1
            else:
                s_diff += v; c_diff += 1
    row = he[(he["ego"] == ego) & (he["group_type"] == "community")].iloc[0]
    exp_diff = s_same / c_same - s_diff / c_diff
    check("H3", "独立复核 ego 414 的同社区相似度差值", abs(row["diff"] - exp_diff) < 1e-5,
          f"{row['diff']:.6f} vs {exp_diff:.6f}")
    # 圈子版独立复核 (ego 414)
    node_circ = defaultdict(set)
    for c, nn in circles[ego].items():
        for n in nn:
            node_circ[n].add(c)
    ln = sorted(n for n in node_circ if n in fs)
    s_same = s_diff = c_same = c_diff = 0
    for i in range(len(ln)):
        for j in range(i + 1, len(ln)):
            v = jac(fs[ln[i]], fs[ln[j]])
            if node_circ[ln[i]] & node_circ[ln[j]]:
                s_same += v; c_same += 1
            else:
                s_diff += v; c_diff += 1
    row = he[(he["ego"] == ego) & (he["group_type"] == "circle")].iloc[0]
    exp_diff = s_same / c_same - s_diff / c_diff
    check("H3", "独立复核 ego 414 的同圈子相似度差值", abs(row["diff"] - exp_diff) < 1e-5,
          f"{row['diff']:.6f} vs {exp_diff:.6f}")

    cats = set(names_df["category"])
    check("H4", "类别汇总覆盖全部特征类别", set(hc["category"]) == cats, f"{len(cats)} 类")
    # H4 独立复核: education;school;id, 同圈子, 汇总所有 ego
    ps = ss = pn = sn = 0
    for e in EGOS:
        cn = defaultdict(set)
        for c, nn in circles[e].items():
            for n in nn:
                cn[n].add(c)
        lab = sorted(n for n in cn if n in feats[e])
        sch = {n: {f for f in feats[e][n] if featcat[e][f] == "education;school;id"} for n in lab}
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                same = bool(cn[lab[i]] & cn[lab[j]])
                if sch[lab[i]] & sch[lab[j]]:
                    ps += 1; ss += same
                else:
                    pn += 1; sn += same
    r = hc[(hc["group_type"] == "circle") & (hc["category"] == "education;school;id")].iloc[0]
    check("H4", "独立复核 education;school;id(同圈子)汇总计数",
          (r["pairs_share"], r["same_given_share"], r["pairs_noshare"], r["same_given_noshare"]) == (ps, ss, pn, sn),
          f"{(ps, ss, pn, sn)}")
    es_ = pd.read_csv(PROC / "ego_summary.csv").set_index("ego")
    c_rows = he[he["group_type"] == "circle"].set_index("ego")["n_nodes"]
    m_rows = he[he["group_type"] == "community"].set_index("ego")["n_nodes"]
    check("H5", "圈子分析的节点数 = 该 ego 圈子里出现过且有特征的节点数",
          all(c_rows[e] == len([n for n in {x for v in circles[e].values() for x in v} if n in feats[e]])
              for e in EGOS))
    check("H5", "社区分析使用该 ego 的全部 alter", all(m_rows[e] == es_.loc[e, "n_alters"] for e in EGOS))
    check("H6", "输出两张 csv 和一张图", (FIG / "homophily_by_category.png").stat().st_size > 5000)

    # ------------------------------------------------------------ R
    roles = pd.read_csv(PROC / "roles.csv")
    comm_df = pd.read_csv(PROC / "community.csv")
    check("R1", "可通过参数选择 community.csv 的列(louvain 与 label_propagation 均可运行)",
          _runs_with_other_col())
    check("R6", "roles.csv 列名正确、每个节点一行",
          list(roles.columns) == ["node", "community", "degree", "within_module_z", "participation_coeff",
                                  "role7", "role"] and len(roles) == 4039 and roles["node"].is_unique
          and set(roles["node"]) == node_set)

    # R2/R3/R7 独立复核: 用邻接表字典, 不用 pandas groupby
    adj = defaultdict(set)
    for a, b in comb:
        adj[a].add(b); adj[b].add(a)
    cm = comm_df.set_index("node")["louvain"].to_dict()
    kin = {n: sum(1 for m in adj[n] if cm[m] == cm[n]) for n in node_set}
    by_c = defaultdict(list)
    for n in node_set:
        by_c[cm[n]].append(kin[n])
    stat = {c: (sum(v) / len(v), math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)))
            for c, v in by_c.items()}
    rng = random.Random(42)
    sample = sorted(set(rng.sample(sorted(node_set), 200)) | set(EGOS))
    rr = roles.set_index("node")
    zmax = pmax = 0.0
    for n in sample:
        mu, sd = stat[cm[n]]
        z = (kin[n] - mu) / sd if sd > 0 else 0.0
        cntc = defaultdict(int)
        for m in adj[n]:
            cntc[cm[m]] += 1
        k = len(adj[n])
        p = 1 - sum((v / k) ** 2 for v in cntc.values()) if k else 0.0
        zmax = max(zmax, abs(z - rr.loc[n, "within_module_z"]))
        pmax = max(pmax, abs(p - rr.loc[n, "participation_coeff"]))
    check("R2", "z 分数与独立算法一致(抽样 200 个 + 10 个 ego)", zmax < 1e-9, f"最大误差 {zmax:.2e}")
    check("R2", "标准差为 0 的社区 z 记为 0", all(rr.loc[n, "within_module_z"] == 0.0
          for n in node_set if stat[cm[n]][1] == 0) )
    check("R3", "P 与独立算法一致(抽样 200 个 + 10 个 ego)", pmax < 1e-9, f"最大误差 {pmax:.2e}")
    check("R7", "P 取值在 [0, 1) 内", bool(((roles["participation_coeff"] >= 0) & (roles["participation_coeff"] < 1)).all()))
    check("R3", "度数与邻接表一致", all(rr.loc[n, "degree"] == len(adj[n]) for n in sample))

    # R4 边界值单元测试 (阈值: z=2.5; 非枢纽 P=0.05/0.625/0.8; 枢纽 P=0.3/0.75)
    sys.path.insert(0, str(ROOT / "src" / "05_roles"))
    import roles as rl  # noqa: E402
    cases = [(2.5, 0.0, "R5"), (2.5, 0.3, "R5"), (2.5, 0.3001, "R6"), (2.5, 0.75, "R6"), (2.5, 0.7501, "R7"),
             (2.4999, 0.0, "R1"), (0.0, 0.05, "R1"), (0.0, 0.0501, "R2"), (0.0, 0.625, "R2"),
             (0.0, 0.6251, "R3"), (0.0, 0.8, "R3"), (0.0, 0.8001, "R4")]
    bad = [(z, p, exp, rl.role7(z, p)) for z, p, exp in cases if rl.role7(z, p) != exp]
    check("R4", "Guimerà–Amaral 阈值边界值测试(12 组)", not bad, str(bad))
    check("R4", "roles.csv 的 role7 与逐行重新分类一致",
          all(rl.role7(z, p) == r for z, p, r in zip(roles["within_module_z"], roles["participation_coeff"], roles["role7"])))
    cn = {"R1": "普通节点", "R2": "普通节点", "R3": "跨社区桥梁", "R4": "跨社区桥梁",
          "R5": "社区内枢纽", "R6": "跨社区枢纽", "R7": "跨社区枢纽"}
    check("R5", "四类中文角色与 R1–R7 的对应关系正确", all(cn[a] == b for a, b in zip(roles["role7"], roles["role"])))

    ro = pd.read_csv(PROC / "roles_egos.csv")
    check("R8", "roles_egos.csv 给出 10 个 ego 的角色", sorted(ro["node"]) == EGOS)
    check("R8", "散点图和数量统计图已生成",
          (FIG / "roles_scatter.png").stat().st_size > 5000 and (FIG / "roles_counts.png").stat().st_size > 5000)
    summ = pd.read_csv(PROC / "roles_summary.csv")
    check("R8", "各角色数量之和等于节点总数", int(summ["n_nodes"].sum()) == 4039)

    # ------------------------------------------------------------ G
    srcs = [ROOT / "src" / "01_data" / "parse_data.py", ROOT / "src" / "01_data" / "data_summary.py",
            ROOT / "src" / "05_roles" / "homophily.py", ROOT / "src" / "05_roles" / "roles.py"]
    check("G1", "脚本均有说明文档和注释", all(p.read_text().lstrip().startswith('"""') and
                                          p.read_text().count("#") >= 3 for p in srcs))
    check("G1", "随机种子固定为 42", "SEED = 42" in (srcs[2]).read_text())
    h_before = md5(PROC / "homophily_by_ego.csv"), md5(PROC / "homophily_by_category.csv")
    r_before = md5(PROC / "roles.csv")
    for s in ("src/05_roles/homophily.py", "src/05_roles/roles.py"):
        subprocess.run([sys.executable, "-I", str(ROOT / s)], check=True, capture_output=True)
    check("G1", "重跑结果完全一致(可复现)",
          h_before == (md5(PROC / "homophily_by_ego.csv"), md5(PROC / "homophily_by_category.csv"))
          and r_before == md5(PROC / "roles.csv"))
    check("G2", "没有改动 community.csv(只读取)", comm_hash == md5(PROC / "community.csv"))

    # ------------------------------------------------------------ 汇总
    width = max(len(d) for _, d, _, _ in results)
    for code, desc, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {code:<3} {desc}{'  (' + detail + ')' if detail and not ok else ''}")
    n_fail = sum(1 for r in results if not r[2])
    print(f"\n共 {len(results)} 项检查, 通过 {len(results) - n_fail}, 失败 {n_fail}")
    sys.exit(1 if n_fail else 0)


def _runs_with_other_col() -> bool:
    """用 label_propagation 列跑一遍 roles.py, 看能否成功; 随后用默认列恢复输出。"""
    ok = subprocess.run([sys.executable, "-I", str(ROOT / "src/05_roles/roles.py"), "--col", "label_propagation"],
                        capture_output=True).returncode == 0
    subprocess.run([sys.executable, "-I", str(ROOT / "src/05_roles/roles.py")], check=True, capture_output=True)
    bad = subprocess.run([sys.executable, "-I", str(ROOT / "src/05_roles/roles.py"), "--col", "nope"],
                         capture_output=True).returncode != 0
    return ok and bad


if __name__ == "__main__":
    main()
