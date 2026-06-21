"""
ΔT敏感性分析和稳健性分析
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 600,
})

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_mean_from_mean_std(value) -> float:

    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if not s or s.upper() == "NA":
        return np.nan

    s = re.split(r"\s*(?:±|\+/-)\s*", s)[0]
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else np.nan


def add_metric_mean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    metric_candidates: Dict[str, Iterable[str]] = {
        "roc_auc_mean": ["roc_auc_mean", "roc_auc", "roc_auc_mean_std"],
        "pr_auc_mean": ["pr_auc_mean", "pr_auc", "ap_mean", "ap", "pr_auc_mean_std"],
        "f1_mean": ["f1_mean", "f1", "f1_mean_std"],
        "top10_alert_precision_mean": [
            "top10_alert_precision_mean",
            "top10_alert_precision",
            "top10_precision_mean",
            "top10_precision",
            "top10_alert_precision_mean_std",
        ],
    }

    for target_col, candidates in metric_candidates.items():
        if target_col in out.columns and pd.api.types.is_numeric_dtype(out[target_col]):
            continue

        source_col: Optional[str] = None
        for c in candidates:
            if c in out.columns:
                source_col = c
                break

        if source_col is None:
            raise KeyError(
                f"Cannot find a usable column for {target_col}. "
                f"Existing columns are: {list(out.columns)}"
            )

        out[target_col] = out[source_col].apply(parse_mean_from_mean_std)

    for c in ["positive_rate", "deltaT_threshold", "event_q"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def aggregate_threshold_sensitivity(df_threshold: pd.DataFrame) -> pd.DataFrame:
    df = add_metric_mean_columns(df_threshold)

    if "sensor_scenario" in df.columns:
        df = df[df["sensor_scenario"].astype(str).str.lower().eq("clean")].copy()

    if "event_q" not in df.columns:
        raise KeyError("Missing required column: event_q")

    metric_cols = [
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "top10_alert_precision_mean",
    ]

    agg_dict = {c: "mean" for c in metric_cols}
    if "deltaT_threshold" in df.columns:
        agg_dict["deltaT_threshold"] = "mean"
    if "positive_rate" in df.columns:
        agg_dict["positive_rate"] = "mean"

    out = (
        df.groupby("event_q", as_index=False)
        .agg(agg_dict)
        .sort_values("event_q")
        .reset_index(drop=True)
    )

    if "positive_rate" in out.columns:
        out["event_rate_percent"] = out["positive_rate"] * 100.0
    else:
        out["event_rate_percent"] = np.nan

    return out


def aggregate_sensor_robustness(df_sensor: pd.DataFrame, event_q: float = 0.90) -> pd.DataFrame:

    df = add_metric_mean_columns(df_sensor)

    if "event_q" in df.columns:
        df = df[np.isclose(df["event_q"].astype(float), float(event_q))].copy()

    if "sensor_scenario" not in df.columns:
        raise KeyError("Missing required column: sensor_scenario")

    metric_cols = [
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "top10_alert_precision_mean",
    ]

    agg_dict = {c: "mean" for c in metric_cols}
    if "deltaT_threshold" in df.columns:
        agg_dict["deltaT_threshold"] = "mean"
    if "positive_rate" in df.columns:
        agg_dict["positive_rate"] = "mean"

    out = (
        df.groupby("sensor_scenario", as_index=False)
        .agg(agg_dict)
        .reset_index(drop=True)
    )

    if "positive_rate" in out.columns:
        out["event_rate_percent"] = out["positive_rate"] * 100.0
    else:
        out["event_rate_percent"] = np.nan

    scenario_order = ["clean", "noise_5", "noise_10", "missing_5", "missing_10"]
    scenario_rank = {name: i for i, name in enumerate(scenario_order)}
    out["scenario_rank"] = out["sensor_scenario"].map(lambda x: scenario_rank.get(str(x).lower(), 999))
    out = (
        out.sort_values(["scenario_rank", "sensor_scenario"])
        .drop(columns=["scenario_rank"])
        .reset_index(drop=True)
    )

    display_map = {
        "clean": "No perturbation",
        "noise_5": "5% additive noise",
        "noise_10": "10% additive noise",
        "missing_5": "5% missing + interpolation",
        "missing_10": "10% missing + interpolation",
    }
    out["setting_label"] = out["sensor_scenario"].map(
        lambda x: display_map.get(str(x).lower(), str(x))
    )

    return out


def make_threshold_xtick_labels(df: pd.DataFrame) -> list[str]:

    labels = []
    for _, row in df.iterrows():
        q = float(row["event_q"])
        labels.append(f"{q:.2f}")
    return labels


def make_sensor_xtick_labels(df: pd.DataFrame) -> list[str]:

    short_map = {
        "clean": "No perturb.",
        "noise_5": "5% noise",
        "noise_10": "10% noise",
        "missing_5": "5% missing",
        "missing_10": "10% missing",
    }

    labels = []
    for _, row in df.iterrows():
        scenario = str(row.get("sensor_scenario", "")).lower()
        base = short_map.get(scenario, str(row.get("setting_label", scenario)))
        labels.append(base)
    return labels


def plot_single_metric_figure(
    data: pd.DataFrame,
    x_labels: list[str],
    title: str,
    out_dir: str | Path,
    file_stem: str,
    x_label: str = "",
) -> None:

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    metric_specs = [
        ("roc_auc_mean", "AUROC", "o"),
        ("pr_auc_mean", "AP", "s"),
        ("f1_mean", "F1", "^"),
        ("top10_alert_precision_mean", "Top-10% precision", "D"),
    ]

    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    for col, label, marker in metric_specs:
        ax.plot(
            x,
            data[col].values,
            marker=marker,
            linewidth=2.2,
            markersize=6.5,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=20, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Metric value", fontsize=20, labelpad=10)
    if x_label:
        ax.set_xlabel(x_label, fontsize=20, labelpad=12)
    if title:
        ax.set_title(title, fontsize=14, pad=42)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.tick_params(axis="both", labelsize=20, pad=10)

    # Put legend above the plot area.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        fontsize=20,
        handlelength=2.2,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    for ext in ["png", "pdf", "tiff"]:
        fig.savefig(out_dir / f"{file_stem}.{ext}", dpi=600, bbox_inches="tight")

    plt.close(fig)


def plot_separate_figures(
    threshold_agg: pd.DataFrame,
    sensor_agg: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    threshold_labels = make_threshold_xtick_labels(threshold_agg)
    sensor_labels = make_sensor_xtick_labels(sensor_agg)

    plot_single_metric_figure(
        data=threshold_agg,
        x_labels=threshold_labels,
        title="",
        out_dir=out_dir,
        file_stem="fig7a_deltaT_threshold_sensitivity",
        x_label="ΔT event-threshold quantile",
    )

    plot_single_metric_figure(
        data=sensor_agg,
        x_labels=sensor_labels,
        title="",
        out_dir=out_dir,
        file_stem="fig7b_sensor_robustness",
        x_label="Sensor perturbation scenario",
    )


def find_file_recursively(filename: str, roots: list[Path]) -> Optional[Path]:

    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    candidates = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        direct = root / filename
        if direct.exists():
            candidates.append(direct.resolve())

        try:
            for found in root.rglob(p.name):
                candidates.append(found.resolve())
        except Exception:
            pass

    seen = set()
    unique = []
    for c in candidates:
        if str(c) not in seen:
            unique.append(c)
            seen.add(str(c))

    if not unique:
        return None

    preferred_keywords = [
        "step7_deltaT_proxy_results",
        "step7_deltaT_proxy_results_deltaT_only",
        "step6_deltaT_proxy_results_reviewer_response",
    ]
    for key in preferred_keywords:
        for c in unique:
            if key.lower() in str(c).lower():
                return c

    return unique[0]


def resolve_path(path_or_name: str, results_dir: str | Path, script_dir: Optional[Path] = None) -> Path:
    p = Path(path_or_name)
    if p.is_absolute() or p.exists():
        return p

    roots = [Path(results_dir)]
    if script_dir is not None:
        roots.extend([script_dir, script_dir.parent])

    found = find_file_recursively(path_or_name, roots)
    if found is not None:
        return found

    return Path(results_dir) / p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot separate Fig. 7a and Fig. 7b for ΔT-based proxy early-warning sensitivity and robustness."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=".",
        help="Directory containing Step7 output CSV files.",
    )
    parser.add_argument(
        "--threshold_csv",
        type=str,
        default="table_deltaT_threshold_sensitivity_for_word.csv",
        help="Threshold-sensitivity CSV filename or path.",
    )
    parser.add_argument(
        "--sensor_csv",
        type=str,
        default="table_sensor_robustness_for_word.csv",
        help="Sensor-robustness CSV filename or path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Defaults to results_dir.",
    )
    parser.add_argument(
        "--event_q",
        type=float,
        default=0.90,
        help="Event quantile used for sensor-robustness aggregation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)

    if str(results_dir) == ".":
        results_dir = script_dir

    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    ensure_dir(out_dir)

    threshold_csv = resolve_path(args.threshold_csv, results_dir, script_dir=script_dir)
    sensor_csv = resolve_path(args.sensor_csv, results_dir, script_dir=script_dir)

    if not threshold_csv.exists():
        raise FileNotFoundError(
            "Cannot find threshold-sensitivity CSV. Please pass the Step7 output directory, for example:\n"
            "python plot_step7_fig7_deltaT_sensitivity_robustness_separate.py "
            "--results_dir D:\\...\\step7_deltaT_proxy_results_deltaT_only\n"
            f"Current searched path: {threshold_csv}"
        )

    if not sensor_csv.exists():
        raise FileNotFoundError(
            "Cannot find sensor-robustness CSV. Please pass the Step7 output directory, for example:\n"
            "python plot_step7_fig7_deltaT_sensitivity_robustness_separate.py "
            "--results_dir D:\\...\\step7_deltaT_proxy_results_deltaT_only\n"
            f"Current searched path: {sensor_csv}"
        )

    df_threshold = pd.read_csv(threshold_csv)
    df_sensor = pd.read_csv(sensor_csv)

    threshold_agg = aggregate_threshold_sensitivity(df_threshold)
    sensor_agg = aggregate_sensor_robustness(df_sensor, event_q=args.event_q)

    threshold_agg.to_csv(out_dir / "fig7_threshold_sensitivity_aggregated.csv", index=False)
    sensor_agg.to_csv(out_dir / "fig7_sensor_robustness_aggregated.csv", index=False)

    plot_separate_figures(
        threshold_agg=threshold_agg,
        sensor_agg=sensor_agg,
        out_dir=out_dir,
    )

    print("[OK] Separate Fig. 7 images generated.")
    print(f"  threshold_csv: {threshold_csv}")
    print(f"  sensor_csv:    {sensor_csv}")
    print(f"  out_dir:       {out_dir}")
    print("  files:")
    print(f"    {out_dir / 'fig7a_deltaT_threshold_sensitivity.png'}")
    print(f"    {out_dir / 'fig7a_deltaT_threshold_sensitivity.pdf'}")
    print(f"    {out_dir / 'fig7a_deltaT_threshold_sensitivity.tiff'}")
    print(f"    {out_dir / 'fig7b_sensor_robustness.png'}")
    print(f"    {out_dir / 'fig7b_sensor_robustness.pdf'}")
    print(f"    {out_dir / 'fig7b_sensor_robustness.tiff'}")
    print(f"    {out_dir / 'fig7_threshold_sensitivity_aggregated.csv'}")
    print(f"    {out_dir / 'fig7_sensor_robustness_aggregated.csv'}")


if __name__ == "__main__":
    main()
