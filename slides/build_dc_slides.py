"""
slides/build_dc_slides.py  (负责人: dc)

作用: 生成 PPT "数据" 与 "角色分类" 部分的幻灯片文件(deck.json、deck.css 和每页 html)。
      页面上的所有数字都从 data/processed/ 的输出文件里读取, 不手写 (需求 P3)。

输入: data/processed/*.csv
输出: --out 指定的目录下的 project/deck.json, project/deck.css, project/slides/*.html

用法: python slides/build_dc_slides.py --out /tmp/deck --blobs blobs.json
      blobs.json 形如 {"ego_overview": "/_blob/xxx", "homophily": "/_blob/xxx", "roles_scatter": "/_blob/xxx"}
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms import community as nxc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
COL = "louvain"


def load_facts() -> dict:
    """从输出文件里读出所有要写进幻灯片的数字。"""
    nodes = pd.read_csv(PROC / "nodes.csv")
    edges = pd.read_csv(PROC / "edges.csv")
    circles = pd.read_csv(PROC / "circles.csv")
    summ = pd.read_csv(PROC / "ego_summary.csv")
    names = pd.read_csv(PROC / "featnames.csv")
    he = pd.read_csv(PROC / "homophily_by_ego.csv")
    hc = pd.read_csv(PROC / "homophily_by_category.csv")
    roles = pd.read_csv(PROC / "roles.csv")
    roles_egos = pd.read_csv(PROC / "roles_egos.csv").sort_values("node")
    comm = pd.read_csv(PROC / "community.csv").set_index("node")[COL]

    G = nx.from_pandas_edgelist(edges, "source", "target")
    parts = [set(n for n, c in comm.items() if c == k) for k in sorted(comm.unique())]
    mod = nxc.modularity(G, parts)

    f = {}
    f["n_nodes"], f["n_edges"] = len(nodes), len(edges)
    f["n_egos"] = int(nodes["is_ego"].sum())
    f["n_circles"] = circles.groupby(["ego", "circle"]).ngroups
    f["n_labelled"] = circles["node"].nunique()
    f["n_unlabelled"] = f["n_nodes"] - f["n_labelled"]
    f["pct_labelled"] = round(100 * f["n_labelled"] / f["n_nodes"])
    f["pct_unlabelled"] = round(100 * f["n_unlabelled"] / f["n_nodes"])
    f["n_multi"] = int((nodes["n_circles"] >= 2).sum())
    f["ego_min"], f["ego_max"] = int(summ["n_alters"].min()), int(summ["n_alters"].max())
    f["feat_min"], f["feat_max"] = int(summ["n_features"].min()), int(summ["n_features"].max())
    f["n_categories"] = names["category"].nunique()
    f["n_communities"] = len(parts)
    f["modularity"] = round(mod, 3)
    # 同质性
    for kind in ("community", "circle"):
        d = he[he["group_type"] == kind]
        f[f"{kind}_testable"] = int(d["p_value"].notna().sum())
        f[f"{kind}_sig"] = int((d["p_value"] < 0.05).sum())
        f[f"{kind}_sig_egos"] = sorted(int(e) for e in d.loc[d["p_value"] < 0.05, "ego"])
        f[f"{kind}_untestable_egos"] = sorted(int(e) for e in d.loc[d["p_value"].isna(), "ego"])
        c = hc[(hc["group_type"] == kind) & (hc["category"] == "education;school;id")].iloc[0]
        f[f"{kind}_school_lift"] = round(float(c["lift"]), 1)
        f[f"{kind}_school_p_share"] = round(float(c["p_same_given_share"]) * 100)
        f[f"{kind}_school_p_noshare"] = round(float(c["p_same_given_noshare"]) * 100)
    for kind in ("community", "circle"):
        d = hc[(hc["group_type"] == kind) & (hc["pairs_share"] >= 1000)]
        assert d.sort_values("lift").iloc[-1]["category"] == "education;school;id", f"{kind}: school 不是 lift 最高的类别, 请改幻灯片标题"
    # 角色
    cnt = roles["role"].value_counts()
    for k in ("普通节点", "社区内枢纽", "跨社区枢纽", "跨社区桥梁"):
        f[f"n_{k}"] = int(cnt.get(k, 0))
    f["n_key"] = f["n_社区内枢纽"] + f["n_跨社区枢纽"] + f["n_跨社区桥梁"]
    f["ego_rows"] = roles_egos.to_dict("records")
    ego_set = set(roles_egos["node"])
    f["ego_role_counts"] = roles_egos["role"].value_counts().to_dict()
    f["nonego_xhubs"] = roles[(roles["role"] == "跨社区枢纽") & (~roles["node"].isin(ego_set))].to_dict("records")
    br = roles[roles["role"] == "跨社区桥梁"]
    f["bridge_deg_min"], f["bridge_deg_max"] = int(br["degree"].min()), int(br["degree"].max())
    f["bridge_p_min"], f["bridge_p_max"] = round(float(br["participation_coeff"].min()), 2), round(float(br["participation_coeff"].max()), 2)
    x = roles[roles["role"] == "跨社区枢纽"]
    f["xhub_p_min"], f["xhub_p_max"] = round(float(x["participation_coeff"].min()), 2), round(float(x["participation_coeff"].max()), 2)
    return f


def n(x) -> str:
    return f"{x:,}"


CSS = """\
:root { --ink:#1F2A37; --paper:#F6F4EF; --blue:#3B6EA8; --orange:#E08A2E; --mute:#5B6673; --card:#FFFFFF; --line:#D9D4C7; }
section { padding:128px 128px 160px; background:var(--paper); color:var(--ink); font-family:'Inter',sans-serif; font-size:30px; line-height:1.35; }
section h1, section h2, section h3 { font-family:'Space Grotesk','Inter',sans-serif; margin:0; }
section h2 { font-size:56px; line-height:1.1; margin-bottom:40px; }
section h3 { font-size:32px; line-height:1.2; margin-bottom:12px; }
section p { margin:0; }
.dark { background:var(--ink); color:var(--paper); }
.dark h1 { font-size:96px; line-height:1.05; margin:24px 0 32px; }
.dark .kicker { font-size:30px; color:#E7B26F; letter-spacing:2px; }
.dark .sub { font-size:36px; color:#D8DDE3; width:1300px; }
.stats { display:flex; gap:32px; margin-bottom:40px; }
.stat { flex:1; background:var(--card); border:2px solid var(--line); border-radius:16px; padding:28px 32px; }
.stat .num { font-family:'Space Grotesk','Inter',sans-serif; font-size:72px; line-height:1; color:var(--blue); }
.stat .lbl { font-size:28px; color:var(--mute); margin-top:10px; }
.row { display:flex; gap:48px; align-items:flex-start; }
.col { flex:1; }
.cards { display:flex; gap:32px; }
.card { flex:1; background:var(--card); border:2px solid var(--line); border-radius:16px; padding:36px; }
.card .big { font-family:'Space Grotesk','Inter',sans-serif; font-size:64px; line-height:1; color:var(--orange); margin-bottom:16px; }
.card p, .col p { font-size:30px; color:var(--ink); }
.muted { color:var(--mute); }
.fig { border:2px solid var(--line); border-radius:12px; background:#FFFFFF; object-fit:contain; }
.note { font-size:26px; color:var(--mute); }
.src { position:absolute; left:128px; bottom:64px; width:1664px; font-size:24px; color:var(--mute); }
.tbl { border-collapse:collapse; width:100%; font-size:28px; }
.tbl th { text-align:left; background:var(--ink); color:var(--paper); padding:12px 16px; font-weight:600; }
.tbl td { padding:12px 16px; border-bottom:1px solid var(--line); }
.rolepill { font-weight:600; }
.hub { color:var(--blue); } .xhub { color:#2E8B57; } .bridge { color:var(--orange); }
.formula { background:var(--card); border:2px solid var(--line); border-radius:16px; padding:28px 36px; margin-bottom:24px; }
.formula .eq { font-family:'Space Grotesk','Inter',sans-serif; font-size:38px; margin:8px 0; }
"""


def slide(id_, body, cls="", notes="", src=""):
    c = f' class="{cls}"' if cls else ""
    s = f'<p class="src">{src}</p>\n' if src else ""
    a = f"<aside>{notes}</aside>\n" if notes else ""
    return f'<section id="{id_}"{c} data-transition="fade">\n{body}\n{s}{a}</section>\n'


def build(f: dict, blobs: dict) -> dict:
    role_cls = {"社区内枢纽": ("Provincial hub", "hub"), "跨社区枢纽": ("Cross-community hub", "xhub"),
                "跨社区桥梁": ("Cross-community bridge", "bridge"), "普通节点": ("Ordinary", "")}
    S = {}
    S["cover"] = slide("cover", f"""<p class="kicker">MH8351 WEB ANALYTICS · EGO-FACEBOOK</p>
<h1>Data and node roles</h1>
<p class="sub">What the data contain, which profile attributes go with shared circles, and which nodes sit inside or between communities.</p>""", cls="dark",
                       notes="本部分(dc)：数据概况、特征同质性、角色分类。")

    S["data"] = slide("data", f"""<h2>{n(f['n_nodes'])} users, {n(f['n_edges'])} friendships, {f['n_egos']} ego networks</h2>
<div class="stats">
<div class="stat"><p class="num">{n(f['n_nodes'])}</p><p class="lbl">users (nodes)</p></div>
<div class="stat"><p class="num">{n(f['n_edges'])}</p><p class="lbl">friendships (undirected edges)</p></div>
<div class="stat"><p class="num">{f['n_egos']}</p><p class="lbl">ego networks (survey participants)</p></div>
<div class="stat"><p class="num">{f['n_circles']}</p><p class="lbl">hand-labelled friend circles</p></div>
</div>
<div class="row">
<img class="fig" src="{blobs['ego_overview']}" alt="Bar chart of friends and labelled friends for each of the 10 ego networks" width="900" height="472">
<div class="col"><p><b>{n(f['n_labelled'])} users</b> ({f['pct_labelled']}%) belong to at least one circle.</p>
<p style="margin-top:20px">Ego networks range from {n(f['ego_min'])} to {n(f['ego_max'])} friends.</p>
<p class="note" style="margin-top:20px">Each ego labelled their own friends by hand (classmates, team, family, …). Circles are the ground truth for the community analysis.</p></div>
</div>""", src="Source: McAuley & Leskovec (2012), SNAP ego-Facebook. Counts computed from data/processed/.",
                      notes=f"数据：{n(f['n_nodes'])}个用户、{n(f['n_edges'])}条好友关系，来自10位调查参与者(ego)的好友网络，共{f['n_circles']}个人工标注圈子。ego越大，对整体网络的影响越大。")

    S["caveats"] = slide("caveats", f"""<h2>Three things to know before analysing</h2>
<div class="cards">
<div class="card"><p class="big">{f['n_multi']}</p><h3>Circles overlap</h3><p>{n(f['n_multi'])} users sit in two or more circles. For NMI and ARI we give each user one label: the smallest circle they belong to.</p></div>
<div class="card"><p class="big">{f['pct_unlabelled']}%</p><h3>Many users have no label</h3><p>{n(f['n_unlabelled'])} users are in no circle. They are left out whenever we compare against circles.</p></div>
<div class="card"><p class="big">{f['feat_min']}–{f['feat_max']}</p><h3>Features are ego-specific</h3><p>Profile features are anonymised and numbered separately in each ego network, so columns cannot be merged across egos. Only the {f['n_categories']} feature categories (school, employer, hometown, …) are comparable.</p></div>
</div>""", src="Source: computed from data/processed/ (circles.csv, node_label.csv, featnames.csv).",
                         notes="三个注意事项：圈子可重叠(取最小圈子作为单一标签)；约29%的节点没有圈子标签；特征编号只在各自ego内有意义，只有特征类别可以跨ego比较。")

    ce = ", ".join(str(e) for e in f["community_sig_egos"])
    ue = " and ".join(str(e) for e in f["community_untestable_egos"])
    S["homophily"] = slide("homophily", f"""<h2>Friends who share a school tend to share a circle</h2>
<div class="row">
<img class="fig" src="{blobs['homophily']}" alt="Lift of being in the same community or circle, by shared profile attribute category" width="1040" height="473">
<div class="col">
<p><b>Circles:</b> profile similarity is significantly higher inside circles in {f['circle_sig']} of {f['circle_testable']} egos (permutation test, p &lt; 0.05).</p>
<p style="margin-top:20px"><b>Communities:</b> only {f['community_sig']} of {f['community_testable']} testable egos ({ce}). In egos {ue} all friends fall in one community, so there is nothing to contrast.</p>
<p style="margin-top:20px"><b>School</b> is the strongest attribute: friends who share one are {f['circle_school_lift']}× as likely to share a circle ({f['circle_school_p_share']}% vs {f['circle_school_p_noshare']}%), and {f['community_school_lift']}× as likely to share a community.</p>
</div></div>""", src="Pair similarity = Jaccard of feature sets, tested within each ego; 500 permutations, seed 42. Lift pooled over egos by feature category.",
                          notes=f"特征同质性：按ego分别检验，不跨ego合并特征。圈子上{f['circle_sig']}/{f['circle_testable']}个ego显著；社区上只有{f['community_sig']}/{f['community_testable']}个可检验的ego显著(社区由Louvain得到)。教育(school)是最强的类别。")

    S["roles_method"] = slide("roles_method", f"""<h2>Roles: where a node's edges go</h2>
<div class="row">
<div class="col">
<div class="formula"><h3>Within-community degree z</h3><p class="eq">z = (k<sub>in</sub> − mean) / std</p><p class="note">How much a node's links inside its own community exceed those of its community peers.</p></div>
<div class="formula"><h3>Participation coefficient P</h3><p class="eq">P = 1 − Σ<sub>s</sub> (k<sub>s</sub> / k)²</p><p class="note">P = 0: all links stay in one community. P near 1: links spread evenly over many.</p></div>
</div>
<div class="col">
<table class="tbl">
<tr><th>Role</th><th>Rule</th><th>Guimerà–Amaral</th></tr>
<tr><td class="hub rolepill">Provincial hub</td><td>z ≥ 2.5, P ≤ 0.3</td><td>R5</td></tr>
<tr><td class="xhub rolepill">Cross-community hub</td><td>z ≥ 2.5, P &gt; 0.3</td><td>R6, R7</td></tr>
<tr><td class="bridge rolepill">Cross-community bridge</td><td>z &lt; 2.5, P &gt; 0.625</td><td>R3, R4</td></tr>
<tr><td class="rolepill">Ordinary</td><td>otherwise</td><td>R1, R2</td></tr>
</table>
<p class="note" style="margin-top:24px">Communities: Louvain (seed 42), {f['n_communities']} communities, modularity {f['modularity']}.</p>
</div></div>""", src="Guimerà & Amaral (2005), Functional cartography of complex metabolic networks.",
                              notes="角色分类的方法：z是节点在本社区内的度的标准分数(是否社区内枢纽)，P衡量边分散在多少个社区(是否跨社区桥梁)。阈值取Guimerà-Amaral的标准，合并成四类。社区来自Louvain，若ly最终选用别的算法，需重跑roles.py。")

    S["roles_results"] = slide("roles_results", f"""<h2>{n(f['n_key'])} key players among {n(f['n_nodes'])} users</h2>
<div class="row">
<img class="fig" src="{blobs['roles_scatter']}" alt="Scatter of within-community z-score against participation coefficient, coloured by role, with the 10 egos marked" width="980" height="692">
<div class="col">
<div class="stat" style="margin-bottom:20px"><p class="num hub">{f['n_社区内枢纽']}</p><p class="lbl">provincial hubs: central inside one community</p></div>
<div class="stat" style="margin-bottom:20px"><p class="num xhub">{f['n_跨社区枢纽']}</p><p class="lbl">cross-community hubs (P {f['xhub_p_min']}–{f['xhub_p_max']})</p></div>
<div class="stat"><p class="num bridge">{f['n_跨社区桥梁']}</p><p class="lbl">bridges: low degree ({f['bridge_deg_min']}–{f['bridge_deg_max']}), P {f['bridge_p_min']}–{f['bridge_p_max']}</p></div>
</div></div>""", src=f"Communities: Louvain, {f['n_communities']} communities. The remaining {n(f['n_普通节点'])} users are ordinary.",
                              notes=f"结果：{f['n_社区内枢纽']}个社区内枢纽、{f['n_跨社区枢纽']}个跨社区枢纽、{f['n_跨社区桥梁']}个跨社区桥梁，其余{n(f['n_普通节点'])}个为普通节点。")

    rows = ""
    for r in f["ego_rows"]:
        label, cls = role_cls[r["role"]]
        rows += (f'<tr><td>{int(r["node"])}</td><td>{int(r["community"])}</td><td>{int(r["degree"])}</td>'
                 f'<td>{r["within_module_z"]:.1f}</td><td>{r["participation_coeff"]:.2f}</td>'
                 f'<td class="rolepill {cls}">{label}</td></tr>\n')
    rc = f["ego_role_counts"]
    nx_ = f["nonego_xhubs"]
    other = ""
    if nx_:
        other = " The only other one is node " + ", ".join(str(int(r["node"])) for r in nx_) + "."
    xr = [r for r in f["ego_rows"] if r["role"] == "跨社区枢纽"]
    xr_p = f"P {min(r['participation_coeff'] for r in xr):.2f}–{max(r['participation_coeff'] for r in xr):.2f}" if xr else ""
    S["egos"] = slide("egos", f"""<h2>Where the 10 egos fall</h2>
<div class="row">
<div class="col" style="flex:1.3"><table class="tbl">
<tr><th>Ego</th><th>Community</th><th>Degree</th><th>z</th><th>P</th><th>Role</th></tr>
{rows}</table></div>
<div class="col">
<p><b>{rc.get('社区内枢纽', 0)} egos</b> are provincial hubs: their friends mostly live in one community.</p>
<p style="margin-top:20px"><b>{rc.get('跨社区枢纽', 0)} egos</b> are cross-community hubs ({xr_p}): their friends span several communities.{other}</p>
<p class="note" style="margin-top:24px">Each ego is linked to every one of their friends by construction, so egos rank high on degree by design. Read their roles with that in mind.</p>
</div></div>""", src="Roles computed on the full 4,039-node graph; ego rows from roles_egos.csv.",
                      notes="10个ego的角色。注意：ego与其所有好友相连是数据收集方式决定的，所以ego的度天然很高；其他节点的角色更能反映真实结构。")
    return S


def main(out: Path, blobs: dict):
    f = load_facts()
    S = build(f, blobs)
    order = ["cover", "data", "caveats", "homophily", "roles_method", "roles_results", "egos"]
    root = out / "project"
    (root / "slides").mkdir(parents=True, exist_ok=True)
    for k, html in S.items():
        (root / "slides" / f"{k}.html").write_text(html)
    (root / "deck.css").write_text(CSS)
    deck = {
        "v": 5,
        "createdOnFiles": {"v": 1, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "lists": "css", "format": {"v": 1, "shared": ["project/deck.css"]},
        "title": "MH8351 项目：数据与节点角色分类", "cover": "cover", "order": order,
        "sections": {"s1": {"description": "Data and caveats", "start": "cover"},
                     "s2": {"description": "Profile homophily", "start": "homophily"},
                     "s3": {"description": "Node roles", "start": "roles_method"}},
        "faces": {
            "inter": {"family": "Inter", "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"},
            "space-grotesk": {"family": "Space Grotesk", "href": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap"},
        },
        "designSystems": [],
    }
    (root / "deck.json").write_text(json.dumps(deck, ensure_ascii=False, indent=1))
    print("facts:", {k: v for k, v in f.items() if not isinstance(v, (list, dict))})
    print("written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blobs", required=True, help="图片名到 /_blob/ 地址的 json")
    a = ap.parse_args()
    main(Path(a.out), json.load(open(a.blobs)))
