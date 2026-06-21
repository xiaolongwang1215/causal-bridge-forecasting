"""
相关性分析
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# 配置
INPUT_FILENAME = "data_trigger_mainmodel_z.csv"
FIG_DPI = 220
CSV_ENCODING = "utf-8-sig"

SHORT_NAME: Dict[str, str] = {
    "compartment/mass.plant": "Mass",
    "delta_mass_24h": "ΔM_24h",

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
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _initialism(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[\/\._\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return str(name)

    tokens = s.split(" ")
    out = []
    for t in tokens:
        if not t:
            continue
        if t.isdigit():
            out.append(t)
            continue
        if re.match(r"^\d+[a-zA-Z]+$", t):
            out.append(t.upper())
            continue
        out.append(t[0].upper())
    return "".join(out) if out else str(name)


def _display_label(col: str) -> str:
    if col in SHORT_NAME:
        return SHORT_NAME[col]
    return _initialism(col)


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
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]  # ROOT
    in_path = root / "step0_clean_build" / "step0_results" / INPUT_FILENAME
    out_dir = root / "step1_correlation" / "step1_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_corr_csv = out_dir / "corr_mainmodel.csv"
    out_fig = out_dir / "corr_heatmap_mainmodel.png"

    _log("Step1-1 | Corr heatmap (main-model)")
    _log(f"[ROOT ] {root}")
    _log(f"[IN   ] {in_path}")
    _log(f"[OUT  ] {out_corr_csv}")
    _log(f"[OUT  ] {out_fig}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8", low_memory=False)

    cols = [c for c in df.columns if c != "time"]
    df_num = df[cols].apply(pd.to_numeric, errors="coerce")

    corr = df_num.corr(method="pearson", min_periods=30)
    corr.to_csv(out_corr_csv, encoding=CSV_ENCODING)

    labels = [_display_label(c) for c in corr.columns.tolist()]
    labels = _make_unique(labels)

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    cmap = plt.get_cmap("RdBu_r")  # blue(-) -> white(0) -> red(+)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(corr.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(labels, fontsize=9, rotation=90)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title("Correlation Heatmap (Main-model variables)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_fig, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    _log("[DONE] Saved correlation matrix + heatmap (diverging RdBu colormap).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n[ERROR]", repr(e), flush=True)
        raise
