# -*- coding: utf-8 -*-
"""
相关性结论
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm


# 配置
INPUT_FILENAME = "data_trigger_mainmodel_z.csv"
TIME_COL = "time"
CSV_ENCODING = "utf-8-sig"

FIG_DPI_MAIN = 220
FIG_DPI_V2 = 240

OUTCOME_LEAF = "compartment/leaf_temperature"
OUTCOME_MASS = "compartment/mass.plant"
OUTCOME_DELTA = "delta_mass_24h"
OUTCOMES_ALL = [OUTCOME_LEAF, OUTCOME_MASS, OUTCOME_DELTA]

OUTCOMES_V2 = [OUTCOME_LEAF, OUTCOME_MASS]

PREDICTOR_COLS = [
    "compartment/air_temperature",
    "compartment/humidity_deficit",
    "compartment/co2_concentration",
    "compartment/par",
    "substrate_relative_permittivity_mean",
    "substrate_bulk_ec_mean",
    "substrate_temperature_mean",

    "energy/energy_use.heating",
    "energy/electricity_use.lighting",
    "energy/co2_dosage",
    "compartment/screen_energy/screen_position",
    "compartment/screen_blackout/screen_position",
    "compartment/water_supply/water_flow_duration",
    "compartment/window_position_lee_side",
    "compartment/window_position_wind_side",
    "compartment/heating_lower_circuit/pipe_temperature",
]

ARC_RULES = {
    OUTCOME_LEAF: {"abs_thresh": 0.25, "topk": 6},
    OUTCOME_MASS: {"abs_thresh": 0.30, "topk": 6},
}

SHORT_NAME: Dict[str, str] = {
    "compartment/leaf_temperature": "Leaf_T",
    "compartment/air_temperature": "T_in",
    "compartment/humidity_deficit": "VPD_in",
    "compartment/co2_concentration": "CO2_in",
    "compartment/par": "PAR_in",
    "substrate_relative_permittivity_mean": "RZ_moist",
    "substrate_bulk_ec_mean": "RZ_EC",
    "substrate_temperature_mean": "RZ_T",
    "energy/energy_use.heating": "Heat",
    "energy/electricity_use.lighting": "Light",
    "energy/co2_dosage": "CO2_dose",
    "compartment/screen_energy/screen_position": "Screen_E",
    "compartment/screen_blackout/screen_position": "Screen_B",
    "compartment/water_supply/water_flow_duration": "Irrig_dur",
    "compartment/window_position_lee_side": "Win_lee",
    "compartment/window_position_wind_side": "Win_wind",
    "compartment/heating_lower_circuit/pipe_temperature": "Pipe_T",
    "compartment/mass.plant": "Mass",
    "delta_mass_24h": "ΔM_24h",
}

HEATMAP_VMIN, HEATMAP_VMAX = -1.0, 1.0
LINE_MIN_W, LINE_MAX_W = 0.6, 4.0
LINE_ALPHA = 0.75
POS_COLOR = "#e76f51"
NEG_COLOR = "#277da1"
NEU_COLOR = "#cfcfcf"

DIVERGE_CMAP = plt.get_cmap("RdBu_r")
DIVERGE_NORM = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _short(col: str) -> str:
    return SHORT_NAME.get(col, col.split("/")[-1].replace(".", "_"))


def _pearson_r(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return np.nan
    return float(np.corrcoef(x[mask].to_numpy(), y[mask].to_numpy())[0, 1])


def _line_width_from_r(r: float) -> float:
    a = abs(r)
    a = max(0.0, min(1.0, a))
    return LINE_MIN_W + (LINE_MAX_W - LINE_MIN_W) * a


def _pick_links(corr: pd.Series, abs_thresh: float, topk: int) -> List[Tuple[str, float]]:
    items = [(k, float(v)) for k, v in corr.items() if np.isfinite(v) and abs(v) >= abs_thresh]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    if len(items) == 0:
        fallback = [(k, float(v)) for k, v in corr.items() if np.isfinite(v)]
        fallback.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return fallback[:topk]
    return items[:topk]


def _make_unique(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            out.append(lab)
        else:
            seen[lab] += 1
            out.append(f"{lab}_{seen[lab]}")
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]  # ROOT
    in_path = root / "step0_clean_build" / "step0_results" / INPUT_FILENAME
    out_dir = root / "step1_correlation" / "step1_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_corr_main_csv = out_dir / "corr_mainmodel.csv"
    out_heatmap_main = out_dir / "corr_heatmap_mainmodel.png"

    out_csv_v2 = out_dir / "corr_predictors_outcomes_v2.csv"
    out_fig_v2 = out_dir / "corr_key_conclusions_v2.png"

    _log("Step1-6(v2) | Corr conclusions (Physiology + Growth outcomes)")
    _log(f"[IN   ] {in_path}")
    _log(f"[OUT  ] {out_corr_main_csv}")
    _log(f"[OUT  ] {out_heatmap_main}")
    _log(f"[OUT  ] {out_csv_v2}")
    _log(f"[OUT  ] {out_fig_v2}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8", low_memory=False)

    cols_all = [c for c in df.columns if c != TIME_COL]
    df_num_all = df[cols_all].apply(pd.to_numeric, errors="coerce")
    corr_all = df_num_all.corr(method="pearson", min_periods=30)
    corr_all.to_csv(out_corr_main_csv, encoding=CSV_ENCODING)

    labels_all = [_short(c) for c in corr_all.columns.tolist()]
    labels_all = _make_unique(labels_all)

    figA = plt.figure(figsize=(12, 10))
    axA = plt.gca()

    imA = axA.imshow(
        corr_all.values.astype(float),
        aspect="auto",
        cmap=DIVERGE_CMAP,
        norm=DIVERGE_NORM,
    )
    axA.set_title("Correlation Heatmap (Main-model variables)", fontsize=12)

    axA.set_xticks(np.arange(len(labels_all)))
    axA.set_yticks(np.arange(len(labels_all)))
    axA.set_xticklabels(labels_all, fontsize=9, rotation=90)
    axA.set_yticklabels(labels_all, fontsize=9)

    cbarA = plt.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
    cbarA.set_label("Pearson r", fontsize=10)
    cbarA.ax.tick_params(labelsize=9)

    plt.tight_layout()
    figA.savefig(out_heatmap_main, dpi=FIG_DPI_MAIN, bbox_inches="tight")
    plt.close(figA)
    _log("[DONE] Saved corr_mainmodel.csv + corr_heatmap_mainmodel.png")

    # predictors: only those present, and not in outcomes (all outcomes)
    predictors = [c for c in PREDICTOR_COLS if c in df.columns and c not in OUTCOMES_ALL]
    missing_pred = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing_pred:
        _log(f"[WARN] Missing predictors (skipped): {missing_pred}")

    for o in OUTCOMES_V2:
        if o not in df.columns:
            raise KeyError(f"Missing outcome column required by v2 figure: {o}")

    df_num = df.copy()
    for c in predictors + OUTCOMES_V2:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    corr_mat = pd.DataFrame(index=predictors, columns=OUTCOMES_V2, dtype=float)
    for p in predictors:
        for o in OUTCOMES_V2:
            corr_mat.loc[p, o] = _pearson_r(df_num[p], df_num[o])

    corr_mat.to_csv(out_csv_v2, encoding=CSV_ENCODING)
    _log("[DONE] Saved corr_predictors_outcomes_v2.csv")

    links_by_outcome: Dict[str, List[Tuple[str, float]]] = {}
    for o in OUTCOMES_V2:
        rule = ARC_RULES[o]
        links_by_outcome[o] = _pick_links(corr_mat[o], abs_thresh=rule["abs_thresh"], topk=rule["topk"])

    fig = plt.figure(figsize=(15.5, 8))

    ax1 = fig.add_axes([0.05, 0.10, 0.42, 0.82])
    heat = corr_mat.values.astype(float)

    im = ax1.imshow(
        heat,
        aspect="auto",
        cmap=DIVERGE_CMAP,
        norm=DIVERGE_NORM,
    )

    ax1.set_title("Predictors × Outcomes (Pearson r)", fontsize=13)
    ax1.set_xticks(np.arange(len(OUTCOMES_V2)))
    ax1.set_xticklabels([_short(o) for o in OUTCOMES_V2], fontsize=11, fontweight="bold")
    ax1.set_yticks(np.arange(len(predictors)))
    ax1.set_yticklabels([_short(p) for p in predictors], fontsize=10)

    for i in range(len(predictors)):
        for j in range(len(OUTCOMES_V2)):
            r = heat[i, j]
            if not np.isfinite(r):
                continue
            ax1.text(j, i, f"{r:+.2f}", ha="center", va="center", fontsize=10, fontweight="bold")

    cax = fig.add_axes([0.48, 0.20, 0.012, 0.60])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Pearson r", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax2 = fig.add_axes([0.52, 0.10, 0.45, 0.82])
    ax2.set_title("Outcome-oriented association overview (pairwise Pearson r)", fontsize=13)

    y_positions = np.linspace(0.90, 0.10, len(predictors))
    pred_pos = {p: (0.06, y_positions[i]) for i, p in enumerate(predictors)}

    for p in predictors:
        x, y = pred_pos[p]
        ax2.plot([x], [y], marker="o", markersize=4, color="#222222")
        ax2.text(x + 0.02, y, _short(p), fontsize=10, va="center", ha="left")

    out_pos = {
        OUTCOME_LEAF: (0.90, 0.72),
        OUTCOME_MASS: (0.90, 0.40),
    }
    for o, (x, y) in out_pos.items():
        ax2.text(x, y, f"★ {_short(o)}", fontsize=14, fontweight="bold", va="center", ha="left")

    def draw_arc(p: str, target: str, r: float):
        x0, y0 = pred_pos[p]
        x1, y1 = out_pos[target]
        color = POS_COLOR if r > 0 else NEG_COLOR if r < 0 else NEU_COLOR
        lw = _line_width_from_r(r)

        rad_map = {OUTCOME_LEAF: 0.18, OUTCOME_MASS: -0.10}
        rad = rad_map.get(target, 0.0)

        con = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=lw,
            color=color,
            alpha=LINE_ALPHA,
        )
        ax2.add_patch(con)

    for o in OUTCOMES_V2:
        for p, r in links_by_outcome[o]:
            draw_arc(p, o, r)

    note_lines = []
    for o in OUTCOMES_V2:
        rule = ARC_RULES[o]
        note_lines.append(f"{_short(o)}: show |r| ≥ {rule['abs_thresh']:.2f}, fallback top-{rule['topk']} if none")
    note_lines.append("Color: red(+), blue(-); width ∝ |r|")

    ax2.text(0.06, 0.02, "\n".join(note_lines), fontsize=9, ha="left", va="bottom", color="#333333")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    fig.savefig(out_fig_v2, dpi=FIG_DPI_V2, bbox_inches="tight")
    plt.close(fig)

    _log("[DONE] Saved key correlation conclusions figure (v2, without ΔM_24h).")
    for o in OUTCOMES_V2:
        _log(f"[INFO] Links shown for {_short(o)}: {len(links_by_outcome[o])}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n[ERROR]", repr(e), flush=True)
        raise
