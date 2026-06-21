# -*- coding: utf-8 -*-
"""
滞后窗口敏感性分析
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# 主要变量
LEAF_T = "compartment/leaf_temperature"
AIR_T = "compartment/air_temperature"
PAR = "compartment/par"

L_SIDE = "compartment/window_position_lee_side"
W_SIDE = "compartment/window_position_wind_side"
HEAT = "energy/energy_use.heating"
CO2_D = "energy/co2_dosage"
LIGHT = "energy/electricity_use.lighting"
WATER = "compartment/water_supply/water_flow_duration"
PIPE_T = "compartment/heating_lower_circuit/pipe_temperature"
SCREEN_B = "compartment/screen_blackout/screen_position"

SHORT_NAME = {
    LEAF_T: "Leaf_T",
    AIR_T: "Air_T",
    PAR: "PAR",
    L_SIDE: "L_Side",
    W_SIDE: "W_Side",
    HEAT: "Heat",
    CO2_D: "CO2_D",
    LIGHT: "Light",
    WATER: "Water",
    PIPE_T: "Pipe_T",
    SCREEN_B: "Screen_B",
}


CORE_DIRECT_EDGES: List[Tuple[str, str, str]] = [
    ("Air_T_to_Leaf_T", AIR_T, LEAF_T),
    ("PAR_to_Leaf_T", PAR, LEAF_T),
    ("Leaf_T_to_Leaf_T", LEAF_T, LEAF_T),
]

INDIRECT_PATHS: List[Tuple[str, str, str, str]] = [
    ("L_Side_to_Air_T_to_Leaf_T", L_SIDE, AIR_T, LEAF_T),
    ("W_Side_to_Air_T_to_Leaf_T", W_SIDE, AIR_T, LEAF_T),
    ("Heat_to_Air_T_to_Leaf_T", HEAT, AIR_T, LEAF_T),
    ("CO2_D_to_Air_T_to_Leaf_T", CO2_D, AIR_T, LEAF_T),
    ("Light_to_Air_T_to_Leaf_T", LIGHT, AIR_T, LEAF_T),
    ("Water_to_Air_T_to_Leaf_T", WATER, AIR_T, LEAF_T),
    ("Pipe_T_to_Air_T_to_Leaf_T", PIPE_T, AIR_T, LEAF_T),
    ("CO2_D_to_PAR_to_Leaf_T", CO2_D, PAR, LEAF_T),
    ("Light_to_PAR_to_Leaf_T", LIGHT, PAR, LEAF_T),
    ("Screen_B_to_PAR_to_Leaf_T", SCREEN_B, PAR, LEAF_T),
]

CSV_ENCODING = "utf-8-sig"
DT_MINUTES = 10


# 工具
def yes_no(flag: bool) -> str:
    return "Yes" if bool(flag) else "No"


def fmt_float(x, ndigits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{ndigits}f}"


def resolve_result_dir(in_dir: Path) -> Path:

    in_dir = in_dir.resolve()

    if (in_dir / "lag_window_sensitivity.csv").exists():
        return in_dir

    candidate = in_dir / "step3_review_experiments"
    if (candidate / "lag_window_sensitivity.csv").exists():
        return candidate.resolve()

    raise FileNotFoundError(
        "Cannot find lag_window_sensitivity.csv. Please check --in_dir.\n"
        f"Tried:\n"
        f"  1) {in_dir / 'lag_window_sensitivity.csv'}\n"
        f"  2) {candidate / 'lag_window_sensitivity.csv'}"
    )


def find_required_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def infer_tau_values(summary: pd.DataFrame, edges_dir: Path) -> List[int]:
    tau_values = set()

    if "tau_max" in summary.columns:
        tau_values.update(int(x) for x in summary["tau_max"].dropna().tolist())

    for fp in edges_dir.glob("edges_fixed_tau_*.csv"):
        stem = fp.stem.replace("edges_fixed_tau_", "")
        try:
            tau_values.add(int(stem))
        except ValueError:
            pass

    return sorted(tau_values)


def read_edges_for_tau(edges_dir: Path, tau: int) -> pd.DataFrame:
    fp = edges_dir / f"edges_fixed_tau_{tau}.csv"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_csv(fp, encoding=CSV_ENCODING)


def edge_rows(edges: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    needed = {"source", "target", "lag"}
    if not needed.issubset(edges.columns):
        raise ValueError(f"Edge file is missing required columns: {needed - set(edges.columns)}")
    return edges[(edges["source"] == source) & (edges["target"] == target)].copy()


def edge_exists(edges: pd.DataFrame, source: str, target: str) -> bool:
    return not edge_rows(edges, source, target).empty


def edge_details(edges: pd.DataFrame, source: str, target: str) -> str:

    rows = edge_rows(edges, source, target)
    if rows.empty:
        return ""

    pieces = []
    rows = rows.sort_values("lag")
    for _, r in rows.iterrows():
        lag = int(r["lag"])
        if "val" in rows.columns and pd.notna(r["val"]):
            pieces.append(f"lag={lag}, val={float(r['val']):.3f}")
        else:
            pieces.append(f"lag={lag}")
    return "; ".join(pieces)


def path_exists(edges: pd.DataFrame, source: str, mediator: str, target: str) -> bool:

    return edge_exists(edges, source, mediator) and edge_exists(edges, mediator, target)


def path_details(edges: pd.DataFrame, source: str, mediator: str, target: str) -> str:
    if not path_exists(edges, source, mediator, target):
        return ""
    first = edge_details(edges, source, mediator)
    second = edge_details(edges, mediator, target)
    return f"{SHORT_NAME.get(source, source)}→{SHORT_NAME.get(mediator, mediator)} [{first}] ; " \
           f"{SHORT_NAME.get(mediator, mediator)}→{SHORT_NAME.get(target, target)} [{second}]"


def count_leaf_parents_from_edges(edges: pd.DataFrame) -> int:
    if edges.empty or "target" not in edges.columns:
        return 0
    return int((edges["target"] == LEAF_T).sum())


def get_summary_row(summary: pd.DataFrame, tau: int) -> Dict:
    if summary.empty or "tau_max" not in summary.columns:
        return {}

    hit = summary[summary["tau_max"].astype(int) == int(tau)]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def build_direct_indirect_summary(in_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_csv = find_required_file(in_dir / "lag_window_sensitivity.csv", "Lag-window summary CSV")
    edges_dir = find_required_file(in_dir / "lag_window_sensitivity_edges", "Lag-window edge directory")

    summary = pd.read_csv(summary_csv, encoding=CSV_ENCODING)
    tau_values = infer_tau_values(summary, edges_dir)

    if not tau_values:
        raise ValueError("No tau values found in lag_window_sensitivity.csv or lag_window_sensitivity_edges/.")

    rows = []
    path_detail_rows = []
    core_edge_detail_rows = []

    for tau in tau_values:
        edges = read_edges_for_tau(edges_dir, tau)
        base = get_summary_row(summary, tau)

        total_edges = int(base.get("n_total_edges", len(edges)))
        leaf_parents = int(base.get("n_leaf_parents", count_leaf_parents_from_edges(edges)))

        row = {
            "fixed_tau_max": int(tau),
            "time_window_min": int(tau * DT_MINUTES),
            "time_window": f"{int(tau * DT_MINUTES)} min",
            "total_edges": total_edges,
            "leaf_t_parents": leaf_parents,
        }

        retained_direct = []
        for short_name, source, target in CORE_DIRECT_EDGES:
            exists = edge_exists(edges, source, target)
            row[short_name] = yes_no(exists)
            row[f"{short_name}_details"] = edge_details(edges, source, target)
            if exists:
                retained_direct.append(short_name.replace("_to_", "→"))

            matched = edge_rows(edges, source, target)
            if not matched.empty:
                matched = matched.copy()
                matched["fixed_tau_max"] = int(tau)
                matched["time_window_min"] = int(tau * DT_MINUTES)
                matched["core_edge"] = short_name
                core_edge_detail_rows.append(matched)

        row["core_direct_leaf_t_parents_retained"] = ", ".join(retained_direct) if retained_direct else "None"

        air_t_paths = []
        par_paths = []
        retained_indirect = []

        for path_name, source, mediator, target in INDIRECT_PATHS:
            exists = path_exists(edges, source, mediator, target)
            row[path_name] = yes_no(exists)
            row[f"{path_name}_details"] = path_details(edges, source, mediator, target)

            if exists:
                path_readable = f"{SHORT_NAME.get(source, source)}→{SHORT_NAME.get(mediator, mediator)}→{SHORT_NAME.get(target, target)}"
                retained_indirect.append(path_readable)
                if mediator == AIR_T:
                    air_t_paths.append(path_readable)
                elif mediator == PAR:
                    par_paths.append(path_readable)

            path_detail_rows.append({
                "fixed_tau_max": int(tau),
                "time_window_min": int(tau * DT_MINUTES),
                "path_name": path_name,
                "source": source,
                "mediator": mediator,
                "target": target,
                "path_readable": f"{SHORT_NAME.get(source, source)}→{SHORT_NAME.get(mediator, mediator)}→{SHORT_NAME.get(target, target)}",
                "path_exists": yes_no(exists),
                "first_edge_details": edge_details(edges, source, mediator),
                "second_edge_details": edge_details(edges, mediator, target),
                "full_path_details": path_details(edges, source, mediator, target),
            })

        row["n_air_t_mediated_paths"] = len(air_t_paths)
        row["n_par_mediated_paths"] = len(par_paths)
        row["n_core_indirect_paths"] = len(retained_indirect)
        row["air_t_mediated_paths_retained"] = "; ".join(air_t_paths) if air_t_paths else "None"
        row["par_mediated_paths_retained"] = "; ".join(par_paths) if par_paths else "None"
        row["core_indirect_paths_retained"] = "; ".join(retained_indirect) if retained_indirect else "None"

        row["MAE"] = base.get("forecast_mae", base.get("MAE", None))
        row["RMSE"] = base.get("forecast_rmse", base.get("RMSE", None))
        row["R2"] = base.get("forecast_r2", base.get("R2", None))

        # Optional columns for supplementary material.
        row["mean_abs_edge_strength"] = base.get("mean_abs_edge_strength", None)
        row["runtime_seconds"] = base.get("runtime_seconds", None)

        rows.append(row)

    out = pd.DataFrame(rows)
    path_details_df = pd.DataFrame(path_detail_rows)
    core_edge_details_df = pd.concat(core_edge_detail_rows, ignore_index=True) if core_edge_detail_rows else pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_dir / "lag_window_sensitivity_leaf_direct_indirect_summary.csv",
               index=False, encoding=CSV_ENCODING)

    path_details_df.to_csv(out_dir / "lag_window_sensitivity_leaf_direct_indirect_path_details.csv",
                           index=False, encoding=CSV_ENCODING)

    core_edge_details_df.to_csv(out_dir / "lag_window_sensitivity_leaf_core_edge_details.csv",
                                index=False, encoding=CSV_ENCODING)

    word = out.copy()
    rename = {
        "fixed_tau_max": "Fixed τmax",
        "time_window": "Time window",
        "total_edges": "Total edges",
        "leaf_t_parents": "Leaf_T parents",
        "Air_T_to_Leaf_T": "Air_T→Leaf_T",
        "PAR_to_Leaf_T": "PAR→Leaf_T",
        "Leaf_T_to_Leaf_T": "Leaf_T→Leaf_T",
        "n_air_t_mediated_paths": "Air_T-mediated paths",
        "n_par_mediated_paths": "PAR-mediated paths",
        "n_core_indirect_paths": "Core indirect paths",
        "mean_abs_edge_strength": "Mean edge strength",
        "runtime_seconds": "Runtime (s)",
    }
    word = word.rename(columns=rename)

    for col in ["MAE", "RMSE", "R2", "Mean edge strength"]:
        if col in word.columns:
            word[col] = word[col].map(lambda x: fmt_float(x, 4))
    if "Runtime (s)" in word.columns:
        word["Runtime (s)"] = word["Runtime (s)"].map(lambda x: fmt_float(x, 2))

    manuscript_cols = [
        "Fixed τmax",
        "Time window",
        "Total edges",
        "Leaf_T parents",
        "Air_T→Leaf_T",
        "PAR→Leaf_T",
        "Leaf_T→Leaf_T",
        "Air_T-mediated paths",
        "PAR-mediated paths",
        "MAE",
        "RMSE",
        "R2",
    ]
    word_table = word[[c for c in manuscript_cols if c in word.columns]].copy()
    word_table.to_csv(out_dir / "lag_window_sensitivity_leaf_direct_indirect_summary_for_word.csv",
                      index=False, encoding=CSV_ENCODING)

    return out, word_table, path_details_df

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Post-process lag-window sensitivity to check direct and indirect Leaf_T causal pathways."
    )
    p.add_argument(
        "--in_dir",
        type=str,
        default=str(script_dir),
        help=(
            "Directory containing lag_window_sensitivity.csv and "
            "lag_window_sensitivity_edges/. If this script is placed inside "
            "step3_autolag_pcmciplus, it will automatically use "
            "step3_autolag_pcmciplus/step3_review_experiments/."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Default: resolved result directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = resolve_result_dir(Path(args.in_dir))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else in_dir

    summary, word_table, path_details_df = build_direct_indirect_summary(in_dir=in_dir, out_dir=out_dir)

    print("Done.")
    print(f"[IN ] {in_dir}")
    print(f"[OUT] {out_dir}")
    print("\nRecommended manuscript table:")
    print(word_table.to_string(index=False))
    print("\nPath-detail rows:", len(path_details_df))


if __name__ == "__main__":
    main()