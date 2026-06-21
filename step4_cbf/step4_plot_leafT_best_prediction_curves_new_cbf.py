# -*- coding: utf-8 -*-
"""
绘制最好预测curves的CBF结果
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


@dataclass
class CFG:
    result_dir: str = os.path.join(SCRIPT_DIR, "step4_cbf_results_leafT_review_forecasting_stats")
    out_dir: str = os.path.join(SCRIPT_DIR, "step4_cbf_results_leafT_review_forecasting_stats")

    horizons_steps: Tuple[int, ...] = (1, 3, 6, 12)
    minutes_per_step: int = 10

    pred_max_points: int = 240

    dpi: int = 300
    fig_format: str = "png"


cfg = CFG()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_find_file(path: str, root_dir: str, filename: str) -> str:
    if os.path.exists(path):
        return path
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Cannot find required file: {filename}\n"
        f"Expected path: {path}\n"
        f"Searched under: {root_dir}"
    )


def load_best_table() -> pd.DataFrame:
    best_path = robust_find_file(
        os.path.join(cfg.result_dir, "cbf_forecasting_best_by_horizon.csv"),
        PROJECT_ROOT,
        "cbf_forecasting_best_by_horizon.csv",
    )
    df_best = pd.read_csv(best_path)

    if "horizon_steps" not in df_best.columns:
        raise ValueError("cbf_forecasting_best_by_horizon.csv is missing column: horizon_steps")
    if "feature_set" not in df_best.columns:
        raise ValueError("cbf_forecasting_best_by_horizon.csv is missing column: feature_set")
    if "model" not in df_best.columns:
        raise ValueError("cbf_forecasting_best_by_horizon.csv is missing column: model")

    df_best["horizon_steps"] = df_best["horizon_steps"].astype(int)
    if "horizon_minutes" not in df_best.columns:
        df_best["horizon_minutes"] = df_best["horizon_steps"] * cfg.minutes_per_step

    return df_best


def load_prediction_table() -> pd.DataFrame:
    pred_path = os.path.join(cfg.result_dir, "cbf_test_predictions_all.csv")
    try:
        pred_path = robust_find_file(pred_path, PROJECT_ROOT, "cbf_test_predictions_all.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Cannot draw best-prediction curves because cbf_test_predictions_all.csv was not found.\n"
            "In the revised CBF script, set `save_predictions: bool = True` in CFG, rerun Step5, "
            "and then rerun this plotting script.\n\n"
            + str(e)
        )

    df_pred = pd.read_csv(pred_path)

    required_cols = ["time", "y_true", "y_pred", "feature_set", "model", "horizon_steps"]
    missing = [c for c in required_cols if c not in df_pred.columns]
    if missing:
        raise ValueError(f"cbf_test_predictions_all.csv is missing required columns: {missing}")

    df_pred["time"] = pd.to_datetime(df_pred["time"])
    df_pred["horizon_steps"] = df_pred["horizon_steps"].astype(int)

    return df_pred


def _thin_series(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].copy()


def save_fig(fig, filename: str) -> None:
    path = os.path.join(cfg.out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved figure:", path)


def read_best_predictions_for_horizon(
    df_best: pd.DataFrame,
    df_pred: pd.DataFrame,
    h: int,
) -> tuple[pd.DataFrame, str, str]:
    best_hit = df_best[df_best["horizon_steps"] == h].copy()
    if best_hit.empty:
        raise ValueError(f"No best-model row found for horizon_steps={h}")

    row = best_hit.iloc[0]
    feature_set = str(row["feature_set"])
    model = str(row["model"])

    pred = df_pred[
        (df_pred["horizon_steps"] == h) &
        (df_pred["feature_set"].astype(str) == feature_set) &
        (df_pred["model"].astype(str) == model)
    ].copy()

    if pred.empty:
        available = (
            df_pred[df_pred["horizon_steps"] == h][["feature_set", "model"]]
            .drop_duplicates()
            .sort_values(["feature_set", "model"])
        )
        raise ValueError(
            f"No prediction rows found for horizon_steps={h}, feature_set={feature_set}, model={model}.\n"
            f"Available feature_set/model pairs at this horizon are:\n{available.to_string(index=False)}"
        )

    pred = pred.sort_values("time").reset_index(drop=True)
    return pred, feature_set, model


def plot_best_prediction_curves(df_best: pd.DataFrame, df_pred: pd.DataFrame) -> None:
    target_h = [h for h in cfg.horizons_steps if h in df_best["horizon_steps"].tolist()]

    for h in target_h:
        pred, feature_set, model = read_best_predictions_for_horizon(df_best, df_pred, h)
        pred = _thin_series(pred, cfg.pred_max_points)

        fig, ax = plt.subplots(figsize=(14, 4.2))

        x = np.arange(len(pred))
        ax.plot(x, pred["y_true"], linewidth=3, label="Observed leaf temperature")
        ax.plot(x, pred["y_pred"], linewidth=3, label=f"Leaf temperature predicted by the {model} model")

        ax.set_ylabel("Leaf temperature (°C)", fontsize=20)
        ax.set_xlabel("Time step (10 min)", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.3)

        ax.legend(
            frameon=True,
            fontsize=20,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.50),
            ncol=2,
        )

        ax.set_ylim(17, 25)

        minutes = h * cfg.minutes_per_step
        save_fig(fig, f"fig_step5_best_prediction_curve_{minutes}min.{cfg.fig_format}")

        print(
            f"[INFO] {minutes} min best curve drawn from feature_set={feature_set}, model={model}, "
            f"n_points_before_thinning={len(df_pred[(df_pred['horizon_steps'] == h) & (df_pred['feature_set'].astype(str) == feature_set) & (df_pred['model'].astype(str) == model)])}"
        )


def main() -> None:
    print("Step5 | Best prediction curve plotting for revised CBF")
    print("[IN ]", cfg.result_dir)
    print("[OUT]", cfg.out_dir)

    ensure_dir(cfg.out_dir)
    df_best = load_best_table()
    df_pred = load_prediction_table()

    df_best = df_best[df_best["horizon_steps"].isin(cfg.horizons_steps)].copy()
    df_pred = df_pred[df_pred["horizon_steps"].isin(cfg.horizons_steps)].copy()

    plot_best_prediction_curves(df_best, df_pred)

    print("[DONE] Best prediction curves were generated successfully.")


if __name__ == "__main__":
    main()
