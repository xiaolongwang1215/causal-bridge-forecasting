# -*- coding: utf-8 -*-
"""
Plot DPAG (direct causes only) for Leaf Temperature from PCMCI+ significant edges
===============================================================================

✅ 目标
- 只绘制：所有“源变量 -> 叶温”的直接因果边（仅 Parents(LT) -> LT）
- 红色：促进 (val > 0)
- 蓝色：抑制 (val < 0)
- 边标签：汇总该 (source -> LT) 的所有显著滞后，例如 "1,3,6"
- 输入支持两种格式（可混合合并）：
  1) edges 格式：source,target,lag,pvalue,val
  2) bootstrap 频率格式：source,target,lag,freq,mean_val

✅ 输出
- 默认输出：dpag_direct_to_leaf_temperature.png
- 输出目录：优先输出到“找到的第一个输入文件”所在目录

依赖：
pip install pandas numpy matplotlib networkx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
import matplotlib


# =============================
# Config (可按需改)
# =============================
CSV_ENCODING = "utf-8-sig"

DEFAULT_INPUT_FILES = (
    "baseline_pcmciplus_edges.csv",
    "baseline_pcmciplus_edges_growth.csv",
    "baseline_bootstrap_freq.csv",
    "baseline_bootstrap_freq_growth.csv",
)

# >>> 修改：输出名字（只画直接父边）
DEFAULT_OUT_PNG = "dpag_direct_to_leaf_temperature.png"

KEEP_LAG_GE_1 = True
INCLUDE_SELF_LOOPS = False
BOOT_FREQ_MIN = 0.0

# --------- 叶温目标变量配置 ----------
LEAF_TEMP_CANDIDATES = (
    "compartment/leaf_temperature",
    "leaf_temperature",
    "LT",
    "compartment/leaf_temperature.microclimate",
)

# 视觉风格
FIGSIZE = (12, 7.5)
DPI = 220

NODE_SIZE = 1400
NODE_EDGE_COLOR = "white"
NODE_LINEWIDTH = 1.2

EDGE_ALPHA = 0.92
EDGE_WIDTH_MIN = 1.5
EDGE_WIDTH_SCALE = 10.0

CMAP_NAME = "RdBu_r"
BASE_RAD = 0.12

EDGE_LABEL_FONTSIZE = 9
EDGE_LABEL_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65)
NODE_LABEL_FONTSIZE = 11

NAME_MAP: Dict[str, str] = {
    # Bamboo set
    "II": "II", "AT": "AT", "AH": "AH", "CO2C": "CO2C", "SM": "SM", "ST": "ST", "SF": "SF",
    # greenhouse common
    "compartment/air_temperature": "AT",
    "compartment/humidity_deficit": "VPD",
    "compartment/relative_humidity": "RH",
    "compartment/co2_concentration": "CO2C",
    "substrate_relative_permittivity_mean": "SM",
    "substrate_temperature_mean": "ST",
    "compartment/leaf_temperature": "LT",
    "compartment/leaf_temperature.microclimate": "LT",
    "leaf_temperature": "LT",
    "LT": "LT",
    "compartment/mass.plant": "Mass",
    "delta_mass_24h": "dMass24h",
    "delta_mass_7d": "dMass7d",
}


def short_name(x: str) -> str:
    return NAME_MAP.get(x, x.split("/")[-1])


def _get_cmap(name: str):
    try:
        return matplotlib.colormaps.get_cmap(name)
    except Exception:
        return cm.get_cmap(name)


@dataclass
class EdgeAgg:
    rep_val: float
    rep_sig: float
    rep_lag: int
    lags: List[int]


# -----------------------------
# 读取/规范化不同输入格式
# -----------------------------
def _standardize_edges_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    cols = set(df.columns)

    # PCMCI+ edges: source,target,lag,pvalue,val
    if {"source", "target", "lag", "pvalue", "val"} <= cols:
        out = df[["source", "target", "lag", "pvalue", "val"]].copy()
        out["lag"] = pd.to_numeric(out["lag"], errors="coerce")
        out["pvalue"] = pd.to_numeric(out["pvalue"], errors="coerce")
        out["val"] = pd.to_numeric(out["val"], errors="coerce")
        out = out.dropna(subset=["source", "target", "lag", "pvalue", "val"]).copy()

        eps = 1e-300
        out["rep_sig"] = -np.log10(np.clip(out["pvalue"].values.astype(float), eps, 1.0))
        out["rep_val"] = out["val"].astype(float)
        out["sig_type"] = "pvalue"
        out["source_file"] = source_name
        return out[["source", "target", "lag", "rep_val", "rep_sig", "sig_type", "source_file"]]

    # bootstrap freq: source,target,lag,freq,mean_val
    if {"source", "target", "lag", "freq", "mean_val"} <= cols:
        out = df[["source", "target", "lag", "freq", "mean_val"]].copy()
        out["lag"] = pd.to_numeric(out["lag"], errors="coerce")
        out["freq"] = pd.to_numeric(out["freq"], errors="coerce")
        out["mean_val"] = pd.to_numeric(out["mean_val"], errors="coerce")
        out = out.dropna(subset=["source", "target", "lag", "freq", "mean_val"]).copy()

        out = out[out["freq"] >= float(BOOT_FREQ_MIN)].copy()
        out["rep_sig"] = out["freq"].astype(float)
        out["rep_val"] = out["mean_val"].astype(float)
        out["sig_type"] = "freq"
        out["source_file"] = source_name
        return out[["source", "target", "lag", "rep_val", "rep_sig", "sig_type", "source_file"]]

    raise ValueError(
        f"Unrecognized CSV schema in {source_name}. "
        f"Need either (source,target,lag,pvalue,val) or (source,target,lag,freq,mean_val). "
        f"Got columns: {list(df.columns)}"
    )


def load_and_merge_inputs(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding=CSV_ENCODING)
        frames.append(_standardize_edges_df(df, p.name))

    merged = pd.concat(frames, ignore_index=True)

    merged["source"] = merged["source"].astype(str)
    merged["target"] = merged["target"].astype(str)
    merged["lag"] = pd.to_numeric(merged["lag"], errors="coerce")
    merged = merged.dropna(subset=["source", "target", "lag", "rep_val", "rep_sig"]).copy()

    if KEEP_LAG_GE_1:
        merged = merged[merged["lag"] >= 1].copy()
    if not INCLUDE_SELF_LOOPS:
        merged = merged[merged["source"] != merged["target"]].copy()

    return merged


def aggregate_edges(df: pd.DataFrame) -> Dict[Tuple[str, str], EdgeAgg]:
    """同一 (source,target) 多滞后合并，并选择代表性滞后来定颜色/粗细。"""
    if df.empty:
        return {}

    grouped = df.groupby(["source", "target"])
    lag_map = {k: sorted({int(v) for v in g["lag"].tolist()}) for k, g in grouped}

    df2 = df.copy()
    df2["abs_val"] = df2["rep_val"].abs()
    df2 = df2.sort_values(["source", "target", "rep_sig", "abs_val"], ascending=[True, True, False, False])

    agg: Dict[Tuple[str, str], EdgeAgg] = {}
    for _, r in df2.iterrows():
        key = (r["source"], r["target"])
        if key in agg:
            continue
        agg[key] = EdgeAgg(
            rep_val=float(r["rep_val"]),
            rep_sig=float(r["rep_sig"]),
            rep_lag=int(r["lag"]),
            lags=lag_map.get(key, [int(r["lag"])]),
        )
    return agg


def build_graph(edges: Dict[Tuple[str, str], EdgeAgg]) -> nx.DiGraph:
    G = nx.DiGraph()
    for (u, v), e in edges.items():
        G.add_node(u)
        G.add_node(v)
        G.add_edge(u, v, rep_val=e.rep_val, rep_sig=e.rep_sig, rep_lag=e.rep_lag, lags=e.lags)
    return G


def format_lags(lags: List[int]) -> str:
    return ",".join(str(x) for x in sorted(lags))


def discover_inputs(base_dir: Path) -> List[Path]:
    candidate_dirs = [
        base_dir,
        base_dir / "step2_results",
        base_dir / "step2_baseline_pcmciplus" / "step2_results",
        base_dir.parent / "step2_baseline_pcmciplus" / "step2_results",
    ]

    found: List[Path] = []
    for d in candidate_dirs:
        for name in DEFAULT_INPUT_FILES:
            p = d / name
            if p.exists():
                found.append(p)

    uniq, seen = [], set()
    for p in found:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


def resolve_leaf_temp_node(nodes: List[str]) -> str:
    node_set = set(nodes)
    for cand in LEAF_TEMP_CANDIDATES:
        if cand in node_set:
            return cand

    # 兜底：短名匹配
    for n in nodes:
        if short_name(n).lower() in {"lt", "leaf_temperature", "leaf temp", "leaf_temp"}:
            return n

    raise ValueError(
        "Cannot find leaf temperature target in nodes.\n"
        f"Tried candidates: {list(LEAF_TEMP_CANDIDATES)}\n"
        "Please add your real leaf temperature column name into LEAF_TEMP_CANDIDATES."
    )


def direct_parent_subgraph(G_full: nx.DiGraph, target: str) -> nx.DiGraph:
    """只保留 Parents(target) -> target 的边。"""
    parents: Set[str] = set(G_full.predecessors(target))
    keep_nodes = parents | {target}
    H = G_full.subgraph(sorted(keep_nodes)).copy()

    # 删除所有不是指向 target 的边（确保只有 parent->target）
    for u, v in list(H.edges()):
        if v != target:
            H.remove_edge(u, v)

    # 清理可能孤立节点（理论上 parents 都连着 target，不会孤立；以防万一）
    isolates = [n for n in H.nodes() if (n != target and H.degree(n) == 0)]
    H.remove_nodes_from(isolates)

    return H


def layout_star(target: str, parents: List[str]) -> Dict[str, np.ndarray]:
    """
    布局：target 在右侧，parents 环绕（更像“源 -> 叶温”）。
    """
    pos: Dict[str, np.ndarray] = {}
    pos[target] = np.array([1.25, 0.0])
    if not parents:
        return pos

    # parents 做圆环分布
    tmpG = nx.DiGraph()
    tmpG.add_nodes_from(parents)
    ppos = nx.circular_layout(tmpG) if len(parents) > 1 else {parents[0]: np.array([0.0, 0.0])}

    # 把parents整体往左挪，避免与target重叠
    for n, xy in ppos.items():
        pos[n] = np.array([xy[0] - 0.25, xy[1]])
    return pos


def main() -> int:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent

    input_paths = discover_inputs(base_dir)
    if not input_paths:
        for name in DEFAULT_INPUT_FILES:
            p = Path(name)
            if p.exists():
                input_paths.append(p)

    if not input_paths:
        raise FileNotFoundError(
            "No input CSV found.\n"
            f"Tried: {list(DEFAULT_INPUT_FILES)}\n"
            "Put CSVs next to this script or in step2_results / step2_baseline_pcmciplus/step2_results."
        )

    out_dir = input_paths[0].parent
    out_png = out_dir / DEFAULT_OUT_PNG

    print("====================================")
    print("Plot | DPAG (direct causes -> Leaf Temperature only)")
    print(f"[SCRIPT] {script_path}")
    print("[INPUTS]")
    for p in input_paths:
        print(f"  - {p}")
    print(f"[OUT  ] {out_png}")
    print("====================================")

    df = load_and_merge_inputs(input_paths)
    if df.empty:
        raise RuntimeError("No edges to plot (after filtering). Check your CSV contents/filters.")

    edges = aggregate_edges(df)
    if not edges:
        raise RuntimeError("No aggregated edges found.")

    G_full = build_graph(edges)
    leaf_node = resolve_leaf_temp_node(list(G_full.nodes()))
    print(f"[TARGET] Leaf Temperature node = {leaf_node} ({short_name(leaf_node)})")

    # >>> 核心：只画 parents -> leaf_node
    G = direct_parent_subgraph(G_full, leaf_node)
    if G.number_of_edges() == 0:
        raise RuntimeError(
            f"Leaf temperature '{leaf_node}' has no significant direct parent edges in the inputs."
        )

    parents = sorted([n for n in G.nodes() if n != leaf_node])
    print(f"[PARENTS] {len(parents)} parents: {', '.join(short_name(x) for x in parents)}")
    print(f"[SUBG ] Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

    # 节点着色：parents 用其 -> leaf 的 rep_val；leaf 用 0（中性）
    node_scores = {leaf_node: 0.0}
    for p in parents:
        node_scores[p] = float(G.edges[p, leaf_node]["rep_val"])

    # scales
    edge_vals = np.array([abs(G.edges[u, v]["rep_val"]) for (u, v) in G.edges()], dtype=float)
    max_abs_edge = float(np.max(edge_vals)) if len(edge_vals) else 1.0
    max_abs_edge = max(max_abs_edge, 1e-12)

    node_vals = np.array([node_scores[n] for n in G.nodes()], dtype=float)
    max_abs_node = float(np.max(np.abs(node_vals))) if len(node_vals) else 1.0
    max_abs_node = max(max_abs_node, 1e-12)

    cmap = _get_cmap(CMAP_NAME)
    norm_edge = TwoSlopeNorm(vmin=-max_abs_edge, vcenter=0.0, vmax=max_abs_edge)
    norm_node = TwoSlopeNorm(vmin=-max_abs_node, vcenter=0.0, vmax=max_abs_node)

    pos = layout_star(leaf_node, parents)

    nodes = list(G.nodes())
    node_colors = [cmap(norm_node(node_scores.get(n, 0.0))) for n in nodes]

    edgelist = list(G.edges())
    edge_colors = [cmap(norm_edge(G.edges[u, v]["rep_val"])) for (u, v) in edgelist]
    edge_widths = [
        EDGE_WIDTH_MIN + EDGE_WIDTH_SCALE * (abs(G.edges[u, v]["rep_val"]) / max_abs_edge)
        for (u, v) in edgelist
    ]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=NODE_SIZE,
        linewidths=NODE_LINEWIDTH,
        edgecolors=NODE_EDGE_COLOR,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edgelist,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle=f"arc3,rad={BASE_RAD}",
        alpha=EDGE_ALPHA,
    )

    labels = {n: short_name(n) for n in nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=NODE_LABEL_FONTSIZE, font_color="black", ax=ax)

    edge_labels = {(u, v): format_lags(G.edges[u, v]["lags"]) for (u, v) in edgelist}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=EDGE_LABEL_FONTSIZE,
        label_pos=0.52,
        rotate=False,
        bbox=EDGE_LABEL_BBOX,
        ax=ax,
    )

    # 颜色条：保持和附件1一样（auto-MCI / MCI）
    sm_node = cm.ScalarMappable(norm=norm_node, cmap=cmap)
    sm_node.set_array([])
    sm_edge = cm.ScalarMappable(norm=norm_edge, cmap=cmap)
    sm_edge.set_array([])

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax1 = inset_axes(ax, width="38%", height="3.5%", loc="lower left",
                      bbox_to_anchor=(0.08, -0.10, 0.84, 1), bbox_transform=ax.transAxes, borderpad=0)
    cax2 = inset_axes(ax, width="38%", height="3.5%", loc="lower right",
                      bbox_to_anchor=(0.08, -0.10, 0.84, 1), bbox_transform=ax.transAxes, borderpad=0)

    cb1 = fig.colorbar(sm_node, cax=cax1, orientation="horizontal")
    cb1.set_label("auto-MCI", fontsize=10)

    cb2 = fig.colorbar(sm_edge, cax=cax2, orientation="horizontal")
    cb2.set_label("MCI", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_png}")
    print("====================================")
    print("[DONE] Direct-to-LeafTemperature DPAG generated successfully.")
    print("====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
