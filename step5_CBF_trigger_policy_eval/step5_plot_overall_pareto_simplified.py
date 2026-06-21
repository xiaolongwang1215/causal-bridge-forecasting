"""
绘制因果桥接预测在回放式控制评估下的pareto图
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
})


POLICY_ORDER = ["RULE_BASED", "CBF_BASED"]

POLICY_LABEL = {
    "RULE_BASED": "Rule policy",
    "CBF_BASED": "CBF policy",
}

POLICY_MARKER = {
    "RULE_BASED": "o",
    "CBF_BASED": "^",
}

RESOURCE_ORDER = ["Heat", "Light", "CO2_D"]

RESOURCE_LABEL = {
    "Heat": "Heat",
    "Light": "Light",
    "CO2_D": r"CO$_2$ dosage",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate separated simplified overall Pareto figures for Step6 replay-based control evaluation."
    )

    parser.add_argument(
        "--step6_dir",
        type=str,
        default="step5_replay_policy_eval_from_step5_cbf",
        help="Directory containing Step6 output file policy_summary_grid.csv.",
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        default="",
        help=(
            "Optional direct path to policy_summary_grid.csv. "
            "If empty, the script reads --step6_dir/policy_summary_grid.csv."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="step5_replay_policy_eval_from_step5_cbf",
        help="Output directory for figures and CSV files.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for saved raster figures. Default follows the old Step6 plotting style.",
    )

    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help=(
            "Optional small jitter added to x values for visual separation. "
            "Default 0.0 is recommended for manuscript figures."
        ),
    )

    parser.add_argument(
        "--make_supplement",
        action="store_true",
        help="If set, also generate the full resource-specific supplementary Pareto figure.",
    )

    return parser.parse_args()

def read_summary_csv(args: argparse.Namespace) -> pd.DataFrame:
    if args.summary_csv.strip():
        path = Path(args.summary_csv)
    else:
        path = Path(args.step6_dir) / "policy_summary_grid.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find policy_summary_grid.csv:\n{path}\n"
            f"Please run Step6 first or provide --summary_csv."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[INFO] Loaded: {path.resolve()}")
    print(f"[INFO] Rows: {len(df)}")
    return df


def validate_and_clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "horizon_minutes",
        "policy",
        "resource_short",
        "margin",
        "reduce_ratio",
        "k",
        "saved_pct_mean",
        "obs_violate_frac_mean",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"policy_summary_grid.csv is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    out = out[out["policy"].isin(POLICY_ORDER)].copy()
    out = out[out["resource_short"].isin(RESOURCE_ORDER)].copy()

    numeric_cols = [
        "horizon_minutes",
        "margin",
        "reduce_ratio",
        "k",
        "saved_pct_mean",
        "obs_violate_frac_mean",
    ]

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(
        subset=[
            "horizon_minutes",
            "policy",
            "resource_short",
            "margin",
            "reduce_ratio",
            "k",
            "saved_pct_mean",
            "obs_violate_frac_mean",
        ]
    ).copy()

    out["horizon_minutes"] = out["horizon_minutes"].astype(int)

    if out.empty:
        raise ValueError("No valid rows remain after filtering Rule/CBF policies and resources.")

    return out

def build_overall_pareto_points(df: pd.DataFrame) -> pd.DataFrame:

    group_cols = [
        "horizon_minutes",
        "policy",
        "margin",
        "reduce_ratio",
        "k",
    ]

    optional_cols = []
    for c in ["feature_set", "feature_display_name", "model", "violation_reference"]:
        if c in df.columns:
            optional_cols.append(c)

    group_cols = group_cols + optional_cols

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            overall_saving_score=("saved_pct_mean", "mean"),
            observed_violation=("obs_violate_frac_mean", "mean"),
            n_resources=("resource_short", "nunique"),
        )
    )

    pivot_save = (
        df.pivot_table(
            index=group_cols,
            columns="resource_short",
            values="saved_pct_mean",
            aggfunc="mean",
        )
        .reset_index()
    )

    pivot_save = pivot_save.rename(
        columns={
            "Heat": "heat_saving",
            "Light": "light_saving",
            "CO2_D": "co2_dosage_saving",
        }
    )

    out = agg.merge(pivot_save, on=group_cols, how="left")

    out["policy_label"] = out["policy"].map(POLICY_LABEL)
    out = out.sort_values(
        ["horizon_minutes", "policy", "overall_saving_score", "observed_violation"]
    ).reset_index(drop=True)

    return out

def plot_one_horizon_overall_pareto(
    points: pd.DataFrame,
    horizon: int,
    out_dir: Path,
    dpi: int,
    jitter: float = 0.0,
) -> None:

    sub_h = points[points["horizon_minutes"] == horizon].copy()
    if sub_h.empty:
        print(f"[WARN] No points found for horizon={horizon} min. Skip.")
        return

    plt.figure(figsize=(7.2, 5.4))
    ax = plt.gca()

    for policy in POLICY_ORDER:
        sub_p = sub_h[sub_h["policy"] == policy].copy()
        if sub_p.empty:
            continue

        x = sub_p["overall_saving_score"].astype(float).values
        y = sub_p["observed_violation"].astype(float).values

        if jitter > 0:
            rng = np.random.default_rng(2026 + int(horizon))
            x = x + rng.normal(0.0, jitter, size=len(x))

        ax.scatter(
            x,
            y,
            s=38,
            marker=POLICY_MARKER.get(policy, "o"),
            alpha=0.82,
            label=POLICY_LABEL.get(policy, policy),
        )

    ax.set_xlabel("Overall resource-saving score (%)", fontsize=20)
    ax.set_ylabel("Observed violation fraction (%)", fontsize=20)

    ax.tick_params(axis="both", labelsize=20)

    ax.legend(
        fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=True,
    )

    ax.grid(True, linestyle="--", alpha=0.30)

    plt.tight_layout()

    stem = f"fig6_overall_pareto_{horizon}min"
    plt.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_all_horizon_separate_figures(
    points: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    jitter: float = 0.0,
) -> None:
    horizons = sorted(points["horizon_minutes"].dropna().astype(int).unique().tolist())
    if len(horizons) == 0:
        raise ValueError("No horizons found in overall Pareto points.")

    for horizon in horizons:
        plot_one_horizon_overall_pareto(
            points=points,
            horizon=horizon,
            out_dir=out_dir,
            dpi=dpi,
            jitter=jitter,
        )

def plot_supplement_resource_specific(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:

    horizons = sorted(df["horizon_minutes"].dropna().astype(int).unique().tolist())
    nrows = len(horizons)
    ncols = len(RESOURCE_ORDER)

    if nrows == 0:
        print("[WARN] No horizons found for supplementary figure.")
        return

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.8, 2.35 * nrows),
        sharex=False,
        sharey=False,
    )

    if nrows == 1:
        axes = np.array([axes])
    axes = np.array(axes)

    for row_idx, horizon in enumerate(horizons):
        for col_idx, resource in enumerate(RESOURCE_ORDER):
            ax = axes[row_idx, col_idx]

            for policy in POLICY_ORDER:
                sub = df[
                    (df["horizon_minutes"] == horizon)
                    & (df["resource_short"] == resource)
                    & (df["policy"] == policy)
                ].copy()

                if sub.empty:
                    continue

                ax.scatter(
                    sub["saved_pct_mean"],
                    sub["obs_violate_frac_mean"],
                    s=16,
                    marker=POLICY_MARKER.get(policy, "o"),
                    alpha=0.70,
                    label=POLICY_LABEL.get(policy, policy),
                )

            if row_idx == 0:
                ax.set_title(RESOURCE_LABEL.get(resource, resource), fontweight="bold", fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(f"{horizon} min\nViolation fraction (%)", fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == nrows - 1:
                ax.set_xlabel("Saving ratio (%)", fontsize=10)
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, linestyle="--", alpha=0.30)

            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle(
        "Full resource-specific Pareto trade-offs under replay-based offline control evaluation",
        fontsize=12,
        fontweight="bold",
        y=1.005,
    )

    fig.tight_layout()

    fig.savefig(out_dir / "figS_full_resource_specific_pareto.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figS_full_resource_specific_pareto.pdf", bbox_inches="tight")

    plt.close(fig)

def export_word_table(points: pd.DataFrame, out_dir: Path) -> None:
    table = points.copy()

    keep_cols = [
        "horizon_minutes",
        "policy_label",
        "overall_saving_score",
        "observed_violation",
        "heat_saving",
        "light_saving",
        "co2_dosage_saving",
        "margin",
        "reduce_ratio",
        "k",
    ]

    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    rename = {
        "horizon_minutes": "Horizon",
        "policy_label": "Policy",
        "overall_saving_score": "Overall saving score (%)",
        "observed_violation": "Observed violation (%)",
        "heat_saving": "Heat saving (%)",
        "light_saving": "Light saving (%)",
        "co2_dosage_saving": "CO2 dosage saving (%)",
        "margin": "Safety margin m",
        "reduce_ratio": "Reduction ratio r",
        "k": "Conservativeness coefficient k",
    }

    table = table.rename(columns=rename)
    table["Horizon"] = table["Horizon"].astype(int).astype(str) + " min"

    for col in table.columns:
        if col not in ["Horizon", "Policy"]:
            table[col] = pd.to_numeric(table[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(table[col]):
                table[col] = table[col].round(4)

    table.to_csv(
        out_dir / "fig6_overall_pareto_points_for_word.csv",
        index=False,
        encoding="utf-8-sig",
    )

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_summary_csv(args)
    clean_df = validate_and_clean_summary(raw_df)

    points = build_overall_pareto_points(clean_df)

    clean_df.to_csv(
        out_dir / "figS_resource_specific_pareto_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    points.to_csv(
        out_dir / "fig6_overall_pareto_points.csv",
        index=False,
        encoding="utf-8-sig",
    )

    export_word_table(points, out_dir)

    plot_all_horizon_separate_figures(
        points=points,
        out_dir=out_dir,
        dpi=args.dpi,
        jitter=args.jitter,
    )

    if args.make_supplement:
        plot_supplement_resource_specific(
            df=clean_df,
            out_dir=out_dir,
            dpi=args.dpi,
        )

    print("=" * 80)
    print("Finished plotting separated simplified Fig. 6.")
    print("=" * 80)
    print(f"Output directory: {out_dir.resolve()}")
    print("")
    print("Main-text separated figures:")
    for h in sorted(points["horizon_minutes"].dropna().astype(int).unique().tolist()):
        print(f"  {out_dir / f'fig6_overall_pareto_{h}min.png'}")
        print(f"  {out_dir / f'fig6_overall_pareto_{h}min.pdf'}")
        print(f"  {out_dir / f'fig6_overall_pareto_{h}min.tiff'}")
    print("")
    if args.make_supplement:
        print("Supplementary full Pareto figure:")
        print(f"  {out_dir / 'figS_full_resource_specific_pareto.png'}")
        print(f"  {out_dir / 'figS_full_resource_specific_pareto.pdf'}")
        print("")
    print("Tables:")
    print(f"  {out_dir / 'fig6_overall_pareto_points.csv'}")
    print(f"  {out_dir / 'fig6_overall_pareto_points_for_word.csv'}")
    print(f"  {out_dir / 'figS_resource_specific_pareto_points.csv'}")
    print("")
    print("Preview of overall Pareto points:")
    print(points.head(20).to_string(index=False))


if __name__ == "__main__":
    main()