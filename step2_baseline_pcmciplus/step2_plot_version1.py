"""
Plot DPAG
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
import matplotlib


# 配置
CSV_ENCODING = "utf-8-sig"

DEFAULT_EDGE_CSV = "baseline_pcmciplus_edges.csv"
DEFAULT_OUT_PNG = "dpag_pcmciplus.png"

MERGE_GROWTH_CSV = False
DEFAULT_GROWTH_EDGE_CSV = "baseline_pcmciplus_edges_growth.csv"

KEEP_LAG_GE_1 = True

INCLUDE_SELF_LOOPS = False

FIGSIZE = (12, 8)
DPI = 220

NODE_SIZE = 1350
NODE_EDGE_COLOR = "white"
NODE_LINEWIDTH = 1.2

EDGE_ALPHA = 0.92
EDGE_WIDTH_MIN = 1.3
EDGE_WIDTH_SCALE = 10.0

CMAP_NAME = "RdBu_r"

BASE_RAD = 0.18

EDGE_LABEL_FONTSIZE = 9
EDGE_LABEL_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.65)

NODE_LABEL_FONTSIZE = 11


NAME_MAP: Dict[str, str] = {
    "II": "II",
    "AT": "AT",
    "AH": "AH",
    "CO2C": "CO2C",
    "SM": "SM",
    "ST": "ST",
    "SF": "SF",
    "compartment/air_temperature": "AT",
    "compartment/humidity_deficit": "VPD",
    "compartment/relative_humidity": "RH",
    "compartment/co2_concentration": "CO2C",
    "substrate_relative_permittivity_mean": "SM",
    "substrate_temperature_mean": "ST",
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
    best_val: float
    best_p: float
    best_lag: int
    lags: List[int]


def load_edges(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Edges file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    required = {"source", "target", "lag", "pvalue", "val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(list(missing))}")

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["lag"] = pd.to_numeric(df["lag"], errors="coerce")
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["source", "target", "lag", "pvalue", "val"]).copy()

    if KEEP_LAG_GE_1:
        df = df[df["lag"] >= 1].copy()

    if not INCLUDE_SELF_LOOPS:
        df = df[df["source"] != df["target"]].copy()

    return df


def aggregate_edges(df: pd.DataFrame) -> Dict[Tuple[str, str], EdgeAgg]:

    agg: Dict[Tuple[str, str], EdgeAgg] = {}
    if df.empty:
        return agg

    df_sorted = df.sort_values(["source", "target", "pvalue", "lag"], ascending=[True, True, True, True])

    grouped = df.groupby(["source", "target"])
    lag_map = {k: sorted({int(v) for v in g["lag"].tolist()}) for k, g in grouped}

    for _, r in df_sorted.iterrows():
        key = (r["source"], r["target"])
        if key in agg:
            continue
        lags = lag_map.get(key, [int(r["lag"])])
        agg[key] = EdgeAgg(
            best_val=float(r["val"]),
            best_p=float(r["pvalue"]),
            best_lag=int(r["lag"]),
            lags=lags,
        )
    return agg


def compute_node_scores(edges: Dict[Tuple[str, str], EdgeAgg], nodes: List[str]) -> Dict[str, float]:
    score = {n: 0.0 for n in nodes}
    for (u, v), e in edges.items():
        score[u] += float(e.best_val)
    return score


def build_graph(edges: Dict[Tuple[str, str], EdgeAgg]) -> nx.DiGraph:
    G = nx.DiGraph()
    for (u, v), e in edges.items():
        G.add_node(u)
        G.add_node(v)
        G.add_edge(u, v, best_val=e.best_val, best_p=e.best_p, best_lag=e.best_lag, lags=e.lags)
    return G


def format_lags(lags: List[int]) -> str:
    return ",".join(str(x) for x in sorted(lags))


def main() -> int:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent

    candidate_dirs = [
        base_dir,
        base_dir / "step2_results",
        base_dir / "step2_baseline_pcmciplus" / "step2_results",
        base_dir.parent / "step2_baseline_pcmciplus" / "step2_results",
    ]

    edge_csv_path = None
    growth_csv_path = None
    for d in candidate_dirs:
        p = d / DEFAULT_EDGE_CSV
        if p.exists():
            edge_csv_path = p
            break
    if edge_csv_path is None:
        edge_csv_path = Path(DEFAULT_EDGE_CSV)

    if MERGE_GROWTH_CSV:
        for d in candidate_dirs:
            p = d / DEFAULT_GROWTH_EDGE_CSV
            if p.exists():
                growth_csv_path = p
                break

    out_dir = edge_csv_path.parent if edge_csv_path.exists() else base_dir
    out_png = out_dir / DEFAULT_OUT_PNG

    print("Plot | DPAG from PCMCI+ significant edges")
    print(f"[SCRIPT] {script_path}")
    print(f"[READ ] {edge_csv_path}")
    if MERGE_GROWTH_CSV:
        print(f"[READ ] {growth_csv_path}")
    print(f"[OUT  ] {out_png}")

    df = load_edges(edge_csv_path)

    if MERGE_GROWTH_CSV and growth_csv_path and growth_csv_path.exists():
        df_g = load_edges(growth_csv_path)
        df = pd.concat([df, df_g], ignore_index=True).drop_duplicates()

    if df.empty:
        raise RuntimeError("No edges to plot (after filtering). Check your CSV and filters.")

    edges = aggregate_edges(df)
    G = build_graph(edges)

    nodes = list(G.nodes())
    node_scores = compute_node_scores(edges, nodes)

    # scales
    edge_vals = np.array([abs(G.edges[u, v]["best_val"]) for (u, v) in G.edges()], dtype=float)
    max_abs_edge = float(np.max(edge_vals)) if len(edge_vals) else 1.0
    max_abs_edge = max(max_abs_edge, 1e-12)

    node_vals = np.array([node_scores[n] for n in nodes], dtype=float)
    max_abs_node = float(np.max(np.abs(node_vals))) if len(node_vals) else 1.0
    max_abs_node = max(max_abs_node, 1e-12)

    cmap = _get_cmap(CMAP_NAME)
    norm_edge = TwoSlopeNorm(vmin=-max_abs_edge, vcenter=0.0, vmax=max_abs_edge)
    norm_node = TwoSlopeNorm(vmin=-max_abs_node, vcenter=0.0, vmax=max_abs_node)

    pos = nx.circular_layout(nodes)

    node_colors = [cmap(norm_node(node_scores[n])) for n in nodes]

    edgelist = list(G.edges())
    edge_colors = [cmap(norm_edge(G.edges[u, v]["best_val"])) for (u, v) in edgelist]
    edge_widths = [EDGE_WIDTH_MIN + EDGE_WIDTH_SCALE * (abs(G.edges[u, v]["best_val"]) / max_abs_edge) for (u, v) in edgelist]

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
        G, pos, edge_labels=edge_labels,
        font_size=EDGE_LABEL_FONTSIZE,
        label_pos=0.52,
        rotate=False,
        bbox=EDGE_LABEL_BBOX,
        ax=ax,
    )

    sm_node = cm.ScalarMappable(norm=norm_node, cmap=cmap)
    sm_node.set_array([])
    sm_edge = cm.ScalarMappable(norm=norm_edge, cmap=cmap)
    sm_edge.set_array([])

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax1 = inset_axes(ax, width="38%", height="3.5%", loc="lower left",
                      bbox_to_anchor=(0.08, -0.09, 0.84, 1), bbox_transform=ax.transAxes, borderpad=0)
    cax2 = inset_axes(ax, width="38%", height="3.5%", loc="lower right",
                      bbox_to_anchor=(0.08, -0.09, 0.84, 1), bbox_transform=ax.transAxes, borderpad=0)

    cb1 = fig.colorbar(sm_node, cax=cax1, orientation="horizontal")
    cb1.set_label("auto-MCI (node score)", fontsize=10)
    cb2 = fig.colorbar(sm_edge, cax=cax2, orientation="horizontal")
    cb2.set_label("MCI (edge val)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_png}")
    print("[DONE] DPAG generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
