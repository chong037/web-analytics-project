"""
tests/check_dc_slides.py

检查 docs/dc_requirements.md 里的 PPT 需求 P1-P3。
用法: python tests/check_dc_slides.py <deck 目录, 即包含 project/ 的那一层>
数字用另一套写法从 data/processed/ 重新算一遍, 与幻灯片文本逐个比对。
"""
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
deck = Path(sys.argv[1]) / "project"

text = {p.stem: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p.read_text()))) for p in (deck / "slides").glob("*.html")}
raw = {p.stem: p.read_text() for p in (deck / "slides").glob("*.html")}
allt = " ".join(text.values())
res = []


def check(code, desc, ok):
    res.append((code, desc, bool(ok)))


# ---- 独立重算
circ = pd.read_csv(PROC / "circles.csv")
nodes = pd.read_csv(PROC / "nodes.csv")
roles = pd.read_csv(PROC / "roles.csv")
he = pd.read_csv(PROC / "homophily_by_ego.csv")
summ = pd.read_csv(PROC / "ego_summary.csv")
n_circ = len(set(zip(circ["ego"], circ["circle"])))
n_lab = circ["node"].nunique()
rc = Counter(roles["role"])

# ---- P1 数据部分
d = text["data"] + " " + text["caveats"]
check("P1", "数据来源与规模(节点、边、ego 数、圈子数)",
      all(s in d for s in (f"{len(nodes):,}", "88,234", "10 ego", str(n_circ))) and "McAuley" in raw["data"])
check("P1", "10 个 ego 的概况(图 + 规模范围)",
      "_blob/" in raw["data"] and f"{summ['n_alters'].min()} to {summ['n_alters'].max():,}" in d)
check("P1", "标签覆盖率与 29% 缺失的说明",
      f"{round(100 * n_lab / len(nodes))}%" in d and f"{len(nodes) - n_lab:,}" in d)
check("P1", "特征只在 ego 内有意义的注意事项", "ego-specific" in d and "cannot be merged across egos" in d)
check("P1", "圈子重叠与取最小圈子的说明", "smallest circle" in d and f"{int((nodes['n_circles'] >= 2).sum()):,}" in d)

# ---- P2 角色分类部分
r = text["roles_method"] + " " + text["roles_results"] + " " + text["egos"]
check("P2", "方法: z 与 P 的公式和含义",
      all(s in text["roles_method"] for s in ("z =", "P = 1", "Participation coefficient", "Within-community degree")))
check("P2", "方法: 四类角色的规则与 R1-R7 对应",
      all(s in text["roles_method"] for s in ("z ≥ 2.5", "P ≤ 0.3", "P > 0.625", "R5", "R6, R7", "R3, R4", "R1, R2")))
check("P2", "结果: 各角色数量与散点图",
      all(str(rc[k]) in text["roles_results"] for k in ("社区内枢纽", "跨社区枢纽", "跨社区桥梁"))
      and f"{rc['普通节点']:,}" in text["roles_results"] and "_blob/" in raw["roles_results"])
egos = pd.read_csv(PROC / "roles_egos.csv")
check("P2", "关键节点解读: 10 个 ego 的角色表与说明",
      all(f"<td>{e}</td>" in raw["egos"] for e in egos["node"]) and "by construction" in text["egos"])
nonego = roles[(roles["role"] == "跨社区枢纽") & (~roles["node"].isin(egos["node"]))]
check("P2", "解读中提到非 ego 的跨社区枢纽", all(str(int(x)) in text["egos"] for x in nonego["node"]))

# ---- P3 数字来自文件
sig_c = int((he[he.group_type == "circle"]["p_value"] < 0.05).sum())
sig_m = int((he[he.group_type == "community"]["p_value"] < 0.05).sum())
test_m = int(he[he.group_type == "community"]["p_value"].notna().sum())
check("P3", "同质性检验的显著个数与数据一致",
      f"in {sig_c} of 10 egos" in text["homophily"] and f"only {sig_m} of {test_m} testable" in text["homophily"])
hc = pd.read_csv(PROC / "homophily_by_category.csv")
sc = hc[(hc.group_type == "circle") & (hc.category == "education;school;id")].iloc[0]
check("P3", "school 的 lift 与概率与数据一致",
      f"{sc['lift']:.1f}×" in text["homophily"] and f"{round(100 * sc['p_same_given_share'])}%" in text["homophily"]
      and f"{round(100 * sc['p_same_given_noshare'])}%" in text["homophily"])
rows_ok = True
for _, e in egos.iterrows():
    m = re.search(rf"<tr><td>{int(e['node'])}</td><td>{int(e['community'])}</td><td>{int(e['degree'])}</td>"
                  rf"<td>{e['within_module_z']:.1f}</td><td>{e['participation_coeff']:.2f}</td>", raw["egos"])
    rows_ok &= bool(m)
check("P3", "ego 角色表每一行的社区、度、z、P 与 roles_egos.csv 一致", rows_ok)
check("P3", "总数 62 = 三类关键角色之和", f"{rc['社区内枢纽'] + rc['跨社区枢纽'] + rc['跨社区桥梁']} key players" in text["roles_results"])
check("P3", "页面里没有手写的、与文件不符的 4,039 以外的节点总数",
      all(int(x.replace(",", "")) in {len(nodes), 88234, n_lab, len(nodes) - n_lab, rc["普通节点"], 1045, 59, 760,
                                        int(summ['n_alters'].max()), int(summ['n_alters'].min())}
          or int(x.replace(",", "")) < 1000 or int(x.replace(",", "")) >= 1900 and int(x.replace(",", "")) <= 2100
          for x in re.findall(r"\b\d{1,3}(?:,\d{3})+\b", allt)))
check("G1", "每页都有演讲备注(中文)", all("<aside>" in h for k, h in raw.items()))
deckjson = (deck / "deck.json").read_text()
check("G1", "deck.json 的 order 覆盖全部 7 页且文件齐全",
      all(f'"{k}"' in deckjson for k in raw) and len(raw) == 7)

w = 0
for code, desc, ok in res:
    print(f"[{'PASS' if ok else 'FAIL'}] {code} {desc}")
    w += not ok
print(f"\n共 {len(res)} 项, 失败 {w}")
sys.exit(1 if w else 0)
