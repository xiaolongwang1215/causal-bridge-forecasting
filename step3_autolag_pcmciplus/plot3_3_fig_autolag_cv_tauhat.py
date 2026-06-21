"""
绘制AutoLag CV curves
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


TARGETS: Tuple[str, ...] = (
    "compartment/leaf_temperature",
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
)

SHORT_LABELS: Dict[str, str] = {
    "compartment/leaf_temperature": "Leaf_T",
    "compartment/air_temperature": "Air_T",
    "compartment/humidity_deficit": "VPD",
    "compartment/co2_concentration": "CO2_C",
    "compartment/par": "PAR",
    "substrate_relative_permittivity_mean": "S_Perm",
    "substrate_bulk_ec_mean": "S_EC",
    "substrate_temperature_mean": "S_Temp",
    "energy/energy_use.heating": "Heat",
    "energy/electricity_use.lighting": "Light",
    "energy/co2_dosage": "CO2_D",
    "compartment/screen_energy/screen_position": "Screen_E",
    "compartment/screen_blackout/screen_position": "Screen_B",
    "compartment/water_supply/water_flow_duration": "Water",
    "compartment/window_position_lee_side": "L_Side",
    "compartment/window_position_wind_side": "W_Side",
    "compartment/heating_lower_circuit/pipe_temperature": "Pipe_T",
    "compartment/mass.plant": "Mass",
    "delta_mass_24h": "dMass24h",
}

FAST_TARGETS = {
    "compartment/leaf_temperature",
    "compartment/air_temperature",
    "compartment/humidity_deficit",
    "compartment/co2_concentration",
    "compartment/par",
    "energy/co2_dosage",
    "compartment/screen_energy/screen_position",
    "compartment/screen_blackout/screen_position",
    "compartment/window_position_lee_side",
    "compartment/window_position_wind_side",
    "compartment/heating_lower_circuit/pipe_temperature",
}

MEDIUM_TARGETS = {
    "energy/energy_use.heating",
    "energy/electricity_use.lighting",
    "compartment/water_supply/water_flow_duration",
    "substrate_temperature_mean",
}

SLOW_TARGETS = {
    "substrate_relative_permittivity_mean",
    "substrate_bulk_ec_mean",
}


def short_label(target: str) -> str:
    return SHORT_LABELS.get(target, target.replace("/", "_"))


def safe_suffix(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def ensure_required_columns(df: pd.DataFrame, required: Sequence[str], fp: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {fp}: {missing}")


def load_tauhat(tauhat_csv: Optional[Path]) -> Dict[str, int]:
    if tauhat_csv is None or not tauhat_csv.exists():
        return {}

    df = pd.read_csv(tauhat_csv)
    ensure_required_columns(df, ["target", "tau_hat"], tauhat_csv)

    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        if pd.isna(row["target"]) or pd.isna(row["tau_hat"]):
            continue
        out[str(row["target"])] = int(row["tau_hat"])
    return out


def pick_tau_hat_1se(curve: pd.DataFrame, metric_mean: str, metric_std: str) -> int:
    df = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_mean, metric_std]).copy()
    if len(df) == 0:
        return int(curve["tau_max"].max())

    best_idx = df[metric_mean].idxmin()
    best_mean = float(df.loc[best_idx, metric_mean])
    best_std = float(df.loc[best_idx, metric_std])
    threshold = best_mean + best_std

    ok = df[df[metric_mean] <= threshold]
    if len(ok) == 0:
        return int(df.loc[best_idx, "tau_max"])

    return int(ok["tau_max"].min())


def aggregate_new_curve(df_target: pd.DataFrame) -> pd.DataFrame:

    df = df_target.copy()

    df["tau_max"] = pd.to_numeric(df["tau_max"], errors="coerce")
    df = df.dropna(subset=["tau_max"]).copy()
    df["tau_max"] = df["tau_max"].astype(int)

    if "stage" in df.columns:
        stage_order = {"coarse": 0, "fine": 1}
        df["_stage_order"] = df["stage"].map(stage_order).fillna(0).astype(int)
        df = df.sort_values(["tau_max", "_stage_order"]).drop_duplicates(subset=["tau_max"], keep="last")
        df = df.drop(columns=["_stage_order"])
    else:
        df = df.drop_duplicates(subset=["tau_max"], keep="last")

    return df.sort_values("tau_max").reset_index(drop=True)


def convert_lag_to_x(taus: Iterable[int], lag_minutes: int, x_unit: str) -> np.ndarray:
    taus = np.asarray(list(taus), dtype=float)
    if x_unit == "step":
        return taus
    if x_unit == "hour":
        return taus * float(lag_minutes) / 60.0
    raise ValueError("x_unit must be 'step' or 'hour'.")


def x_label(x_unit: str) -> str:
    if x_unit == "step":
        return r"Max lag $\tau_{\max}$ (lag steps)"
    if x_unit == "hour":
        return r"Max lag $\tau_{\max}$ (hours)"
    raise ValueError("x_unit must be 'step' or 'hour'.")


def nice_xmax_from_data(max_x: float) -> float:

    if max_x <= 12:
        return 12.0
    if max_x <= 36:
        return 36.0
    if max_x <= 72:
        return 72.0
    if max_x <= 144:
        return 144.0
    return float(np.ceil(max_x / 12.0) * 12.0)


def group_xmax_for_target(target: str, x_unit: str, lag_minutes: int) -> float:

    if target in SLOW_TARGETS:
        xmax_lag = 72
    elif target in MEDIUM_TARGETS:
        xmax_lag = 36
    else:
        xmax_lag = 12

    if x_unit == "step":
        return float(xmax_lag)
    return float(xmax_lag * lag_minutes / 60.0)


def choose_xmax(
    target: str,
    curve: pd.DataFrame,
    tau_hat: Optional[int],
    x_unit: str,
    lag_minutes: int,
    x_mode: str,
    uniform_xmax: Optional[float],
) -> Optional[float]:

    taus = curve["tau_max"].astype(int).tolist()
    if tau_hat is not None:
        taus.append(int(tau_hat))

    x_vals = convert_lag_to_x(taus, lag_minutes=lag_minutes, x_unit=x_unit)
    actual_max_x = float(np.nanmax(x_vals)) if len(x_vals) else 12.0

    if x_mode == "adaptive":
        return nice_xmax_from_data(actual_max_x)

    if x_mode == "group":
        return group_xmax_for_target(target, x_unit=x_unit, lag_minutes=lag_minutes)

    if x_mode == "fixed":
        if uniform_xmax is not None and uniform_xmax > 0:
            return float(uniform_xmax)
        return nice_xmax_from_data(actual_max_x)

    raise ValueError("x_mode must be 'adaptive', 'group', or 'fixed'.")

def plot_single_curve(
    curve: pd.DataFrame,
    target: str,
    tau_hat: Optional[int],
    out_dir: Path,
    metric: str,
    lag_minutes: int,
    x_unit: str,
    x_mode: str,
    uniform_xmax: Optional[float],
    show_tau_hat: bool,
    dpi: int,
    figure_width: float,
    figure_height: float,
) -> Tuple[Path, float]:
    metric_mean = f"mean_{metric}"
    metric_std = f"std_{metric}"

    ensure_required_columns(curve, ["tau_max", metric_mean, metric_std], Path("<curve>"))

    curve = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau_max", metric_mean, metric_std]).copy()
    if curve.empty:
        raise ValueError(f"No valid curve data for target: {target}")

    taus = curve["tau_max"].astype(int).to_numpy()
    x = convert_lag_to_x(taus, lag_minutes=lag_minutes, x_unit=x_unit)
    y = curve[metric_mean].astype(float).to_numpy()
    s = curve[metric_std].astype(float).to_numpy()

    if tau_hat is None:
        tau_hat = pick_tau_hat_1se(curve, metric_mean, metric_std)

    x_hat = float(convert_lag_to_x([tau_hat], lag_minutes=lag_minutes, x_unit=x_unit)[0])
    hit = curve[curve["tau_max"].astype(int) == int(tau_hat)]
    y_hat = float(hit[metric_mean].values[0]) if len(hit) else float(np.nan)

    x_max = choose_xmax(
        target=target,
        curve=curve,
        tau_hat=tau_hat,
        x_unit=x_unit,
        lag_minutes=lag_minutes,
        x_mode=x_mode,
        uniform_xmax=uniform_xmax,
    )

    plt.figure(figsize=(figure_width, figure_height))
    plt.plot(x, y, marker="o", linewidth=1.8)
    plt.fill_between(x, y - s, y + s, alpha=0.18)

    if show_tau_hat and np.isfinite(y_hat):
        plt.axvline(x_hat, linestyle="--", linewidth=1.4, alpha=0.85)
        plt.scatter([x_hat], [y_hat], s=75, marker="D", zorder=6)
        plt.text(
            x_hat,
            y_hat,
            rf"  $\hat{{\tau}}$={int(tau_hat)}",
            fontsize=15,
            va="center",
        )

    if x_max is not None and x_max > 0:
        plt.xlim(0, x_max)

    plt.xlabel(x_label(x_unit), fontsize=20)
    plt.ylabel(f"CV {metric.upper()}", fontsize=20)
    plt.title(short_label(target), fontsize=20)
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ymin = float(np.nanmin(y - s))
    ymax = float(np.nanmax(y + s))
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.01
    ax.set_ylim(ymin - pad, ymax + pad)

    plt.tick_params(axis="both", labelsize=20)
    plt.tight_layout()

    filename = f"{safe_suffix(short_label(target))}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return out_path, float(x_max if x_max is not None else np.nan)


def plot_all_targets(
    cv_csv: Path,
    tauhat_csv: Optional[Path],
    out_dir: Path,
    metric: str,
    lag_minutes: int,
    x_unit: str,
    x_mode: str,
    uniform_xmax: Optional[float],
    show_tau_hat: bool,
    dpi: int,
    figure_width: float,
    figure_height: float,
) -> pd.DataFrame:
    if not cv_csv.exists():
        raise FileNotFoundError(f"Input CV file not found: {cv_csv}")

    raw = pd.read_csv(cv_csv)
    ensure_required_columns(raw, ["target", "tau_max", f"mean_{metric}", f"std_{metric}"], cv_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    tauhat = load_tauhat(tauhat_csv)

    rows: List[Dict[str, object]] = []

    available_targets = [t for t in TARGETS if t in set(raw["target"].astype(str))]
    extras = sorted(set(raw["target"].astype(str)) - set(available_targets))
    ordered_targets = available_targets + extras

    for target in ordered_targets:
        df_t = raw[raw["target"].astype(str) == target].copy()
        if df_t.empty:
            continue

        curve = aggregate_new_curve(df_t)
        if curve.empty:
            continue

        tau_hat = tauhat.get(target)
        if tau_hat is None:
            tau_hat = pick_tau_hat_1se(curve, f"mean_{metric}", f"std_{metric}")

        out_path, used_xmax = plot_single_curve(
            curve=curve,
            target=target,
            tau_hat=tau_hat,
            out_dir=out_dir,
            metric=metric,
            lag_minutes=lag_minutes,
            x_unit=x_unit,
            x_mode=x_mode,
            uniform_xmax=uniform_xmax,
            show_tau_hat=show_tau_hat,
            dpi=dpi,
            figure_width=figure_width,
            figure_height=figure_height,
        )

        rows.append({
            "target": target,
            "short_label": short_label(target),
            "tau_hat": int(tau_hat),
            "n_points": int(len(curve)),
            "min_tau": int(curve["tau_max"].min()),
            "max_tau": int(curve["tau_max"].max()),
            "x_mode": x_mode,
            "x_unit": x_unit,
            "used_xmax": used_xmax,
            "figure": str(out_path),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "plot_autolag_cv_tauhat_summary.csv", index=False, encoding="utf-8-sig")
    return summary

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    default_in_csv = script_dir / "step3_review_experiments" / "autolag_selection" / "autolag_cv_curves_all_targets.csv"
    default_tauhat_csv = script_dir / "step3_review_experiments" / "autolag_selection" / "tauhat_by_cv_review.csv"
    default_out_dir = script_dir / "step3_review_experiments" / "fig_autolag_cv_tauhat_new_adaptive_x"

    parser = argparse.ArgumentParser(
        description="Plot AutoLag CV curves from new Step3_3 AutoLag-PCMCI+ results with adaptive x-axis."
    )

    parser.add_argument("--in_csv", type=str, default=str(default_in_csv),
                        help="Path to autolag_cv_curves_all_targets.csv.")
    parser.add_argument("--tauhat_csv", type=str, default=str(default_tauhat_csv),
                        help="Path to tauhat_by_cv_review.csv.")
    parser.add_argument("--out_dir", type=str, default=str(default_out_dir),
                        help="Directory for output figures.")

    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse"],
                        help="CV metric to plot.")
    parser.add_argument("--lag_minutes", type=int, default=10,
                        help="Sampling interval in minutes.")
    parser.add_argument("--x_unit", type=str, default="step", choices=["step", "hour"],
                        help="Use lag steps or hours on the x-axis.")

    parser.add_argument("--x_mode", type=str, default="adaptive", choices=["adaptive", "group", "fixed"],
                        help=(
                            "adaptive: x-axis follows actual tau values of each variable; "
                            "group: use 12/36/72 lag caps by variable group; "
                            "fixed: use --uniform_xmax for all variables."
                        ))

    parser.add_argument("--uniform_xmax", type=float, default=12.0,
                        help=(
                            "Only used when --x_mode fixed. "
                            "Set <=0 to behave like adaptive."
                        ))

    parser.add_argument("--show_tau_hat", action="store_true",
                        help="Show vertical dashed line and marker for tau_hat.")

    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--figure_width", type=float, default=6.0)
    parser.add_argument("--figure_height", type=float, default=4.2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_csv = Path(args.in_csv)
    tauhat_csv = Path(args.tauhat_csv) if args.tauhat_csv else None
    out_dir = Path(args.out_dir)

    uniform_xmax = None if args.uniform_xmax is None or args.uniform_xmax <= 0 else float(args.uniform_xmax)

    summary = plot_all_targets(
        cv_csv=in_csv,
        tauhat_csv=tauhat_csv,
        out_dir=out_dir,
        metric=args.metric,
        lag_minutes=args.lag_minutes,
        x_unit=args.x_unit,
        x_mode=args.x_mode,
        uniform_xmax=uniform_xmax,
        show_tau_hat=args.show_tau_hat,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )

    print("Done.")
    print(f"[IN ] {in_csv}")
    print(f"[TAU] {tauhat_csv}")
    print(f"[OUT] {out_dir}")
    print(f"[XMODE] {args.x_mode}")
    print(f"[READ] {len(summary)} target curves")
    if len(summary):
        print(summary[[
            "short_label", "tau_hat", "n_points",
            "min_tau", "max_tau", "used_xmax"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
