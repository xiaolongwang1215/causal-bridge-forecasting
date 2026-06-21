"""
因果桥接预测在回放式控制评估下的性能表现
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TIME_COL = "time"
LEAF_T = "compartment/leaf_temperature"

RES_HEAT = "energy/energy_use.heating"
RES_LIGHT = "energy/electricity_use.lighting"
RES_CO2 = "energy/co2_dosage"

DEFAULT_RESOURCES = (RES_HEAT, RES_LIGHT, RES_CO2)

RESOURCE_SHORT = {
    RES_HEAT: "Heat",
    RES_LIGHT: "Light",
    RES_CO2: "CO2_D",
}

FEATURE_DISPLAY_NAME = {
    "AR_only": "AR-only",
    "AR_AirT": "AR-AirT",
    "AR_PAR": "AR-PAR",
    "AR_AirT_PAR": "AR-AirT-PAR",
    "AR-LeafParents": "AR-LeafParents",
    "AR_AirT_PAR_window": "AR-LeafParents-window",
    "AR_AllState": "AR-AllState",
    "TwoStage_CBF_AirT_PAR": "Two-stage CBF",
    "TwoStage_CBF_AirT_PAR_Residual": "Two-stage CBF-residual",
    "TwoStage_CBF_AirT_PAR_ValidationSelected": "Two-stage CBF-validation",
}

@dataclass
class Config:
    raw_csv: str
    step5_dir: str
    predictions_csv: str
    fold_metrics_csv: str
    best_by_horizon_csv: str
    out_dir: str

    horizons_minutes: Tuple[int, ...]
    resources: Tuple[str, ...]

    selection_mode: str
    selected_feature_set: str
    selected_model: str

    auto_Tlow_quantile: float
    margin_grid: Tuple[float, ...]
    reduce_grid: Tuple[float, ...]
    k_grid: Tuple[float, ...]
    default_margin: float
    default_reduce: float
    default_k: float

    violation_reference: str

    rmse_col_for_lower_bound: str

    seed: int


def parse_float_tuple(s: str) -> Tuple[float, ...]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return tuple(vals)


def parse_int_tuple(s: str) -> Tuple[int, ...]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return tuple(vals)


def parse_str_tuple(s: str) -> Tuple[str, ...]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(x)
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay-based control evaluation using NEW Step5 CBF prediction outputs."
    )

    p.add_argument("--raw_csv", type=str, default="data_trigger_mainmodel_raw.csv")
    p.add_argument("--step5_dir", type=str, default="step4_cbf_results_leafT_review_forecasting_stats")
    p.add_argument("--predictions_csv", type=str, default="")
    p.add_argument("--fold_metrics_csv", type=str, default="")
    p.add_argument("--best_by_horizon_csv", type=str, default="")
    p.add_argument("--out_dir", type=str, default="step5_replay_policy_eval_from_step5_cbf")

    p.add_argument("--horizons", type=str, default="10,30,60,120")
    p.add_argument("--resources", type=str, default=",".join(DEFAULT_RESOURCES))

    p.add_argument(
        "--selection_mode",
        type=str,
        default="two_stage_cbf",
        choices=["two_stage_cbf", "best_by_horizon"],
        help="How to select Step5 predictions for replay evaluation.",
    )
    p.add_argument("--selected_feature_set", type=str, default="TwoStage_CBF_AirT_PAR")
    p.add_argument("--selected_model", type=str, default="PatchTST")

    p.add_argument("--auto_Tlow_quantile", type=float, default=0.10)
    p.add_argument("--margin_grid", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--reduce_grid", type=str, default="0.25,0.5,0.75,1.0")
    p.add_argument("--k_grid", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5")
    p.add_argument("--default_margin", type=float, default=0.25)
    p.add_argument("--default_reduce", type=float, default=1.0)
    p.add_argument("--default_k", type=float, default=0.25)

    p.add_argument(
        "--violation_reference",
        type=str,
        default="T_low",
        choices=["T_low", "T_safe"],
        help="Observed violation threshold for safety reporting.",
    )
    p.add_argument(
        "--rmse_col_for_lower_bound",
        type=str,
        default="val_RMSE",
        help="Metric column used in y_pred - k * RMSE. Recommended: val_RMSE.",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def find_file(path_like: str, candidates: Iterable[str], search_roots: Iterable[Path]) -> str:

    path_like = str(path_like or "").strip()
    if path_like:
        p = Path(path_like)
        if p.exists():
            return str(p.resolve())

    direct_candidates: List[Path] = []
    for root in [Path.cwd(), *search_roots]:
        for name in candidates:
            direct_candidates.append(root / name)

    for p in direct_candidates:
        if p.exists():
            return str(p.resolve())

    hits: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for name in candidates:
            hits.extend(root.rglob(name))

    if hits:
        hits = sorted(set(hits), key=lambda x: (len(str(x)), str(x)))
        return str(hits[0].resolve())

    raise FileNotFoundError(
        "Could not find required file. Tried explicit path='{}', candidates={}, roots={}".format(
            path_like, list(candidates), [str(r) for r in search_roots]
        )
    )


def make_config(args: argparse.Namespace) -> Config:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    step5_dir = Path(args.step5_dir)

    search_roots = [
        Path.cwd(),
        script_dir,
        project_root,
        step5_dir,
        project_root / "step0_clean_build" / "step0_results",
    ]

    raw_csv = find_file(
        args.raw_csv,
        candidates=["data_trigger_mainmodel_raw.csv"],
        search_roots=search_roots,
    )

    if args.predictions_csv.strip():
        pred_csv = find_file(args.predictions_csv, [Path(args.predictions_csv).name], search_roots)
    else:
        pred_csv = find_file(
            "",
            candidates=["cbf_test_predictions_all.csv"],
            search_roots=[Path(args.step5_dir), Path.cwd(), script_dir, project_root],
        )

    if args.fold_metrics_csv.strip():
        metrics_csv = find_file(args.fold_metrics_csv, [Path(args.fold_metrics_csv).name], search_roots)
    else:
        metrics_csv = find_file(
            "",
            candidates=["cbf_forecasting_full_results_by_fold.csv"],
            search_roots=[Path(args.step5_dir), Path.cwd(), script_dir, project_root],
        )

    best_csv = ""
    try:
        if args.best_by_horizon_csv.strip():
            best_csv = find_file(args.best_by_horizon_csv, [Path(args.best_by_horizon_csv).name], search_roots)
        else:
            best_csv = find_file(
                "",
                candidates=["cbf_forecasting_best_by_horizon.csv"],
                search_roots=[Path(args.step5_dir), Path.cwd(), script_dir, project_root],
            )
    except FileNotFoundError:
        best_csv = ""

    return Config(
        raw_csv=raw_csv,
        step5_dir=str(Path(args.step5_dir)),
        predictions_csv=pred_csv,
        fold_metrics_csv=metrics_csv,
        best_by_horizon_csv=best_csv,
        out_dir=str(args.out_dir),
        horizons_minutes=parse_int_tuple(args.horizons),
        resources=parse_str_tuple(args.resources),
        selection_mode=args.selection_mode,
        selected_feature_set=args.selected_feature_set,
        selected_model=args.selected_model,
        auto_Tlow_quantile=float(args.auto_Tlow_quantile),
        margin_grid=parse_float_tuple(args.margin_grid),
        reduce_grid=parse_float_tuple(args.reduce_grid),
        k_grid=parse_float_tuple(args.k_grid),
        default_margin=float(args.default_margin),
        default_reduce=float(args.default_reduce),
        default_k=float(args.default_k),
        violation_reference=args.violation_reference,
        rmse_col_for_lower_bound=args.rmse_col_for_lower_bound,
        seed=int(args.seed),
    )

def read_csv_safely(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}\nAvailable columns: {list(df.columns)}")


def load_raw_data(cfg: Config) -> pd.DataFrame:
    df = read_csv_safely(cfg.raw_csv)
    require_columns(df, [TIME_COL, LEAF_T], "raw_csv")
    missing_resources = [r for r in cfg.resources if r not in df.columns]
    if missing_resources:
        raise ValueError(f"raw_csv is missing resource columns: {missing_resources}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    for col in [LEAF_T, *cfg.resources]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    return df


def load_step5_outputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = read_csv_safely(cfg.predictions_csv)
    require_columns(
        pred,
        ["time", "y_true", "y_pred", "fold", "feature_set", "model", "horizon_minutes"],
        "cbf_test_predictions_all.csv",
    )
    pred["time"] = pd.to_datetime(pred["time"])
    pred["horizon_minutes"] = pred["horizon_minutes"].astype(int)
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    pred = pred.dropna(subset=["time", "y_true", "y_pred", "fold", "feature_set", "model", "horizon_minutes"])

    metrics = read_csv_safely(cfg.fold_metrics_csv)
    require_columns(
        metrics,
        ["fold", "feature_set", "model", "horizon_minutes", "test_RMSE"],
        "cbf_forecasting_full_results_by_fold.csv",
    )
    metrics["horizon_minutes"] = metrics["horizon_minutes"].astype(int)

    best = pd.DataFrame()
    if cfg.best_by_horizon_csv:
        try:
            best = read_csv_safely(cfg.best_by_horizon_csv)
            if "horizon_minutes" in best.columns:
                best["horizon_minutes"] = best["horizon_minutes"].astype(int)
        except Exception:
            best = pd.DataFrame()

    return pred, metrics, best

def _metric_mean(metrics: pd.DataFrame, horizon: int, feature_set: str, model: str) -> float:
    sub = metrics[
        (metrics["horizon_minutes"] == int(horizon))
        & (metrics["feature_set"].astype(str) == str(feature_set))
        & (metrics["model"].astype(str) == str(model))
    ]
    if sub.empty:
        return np.inf
    return float(sub["test_RMSE"].mean())


def select_config_for_horizon(
    cfg: Config,
    horizon: int,
    pred: pd.DataFrame,
    metrics: pd.DataFrame,
    best: pd.DataFrame,
) -> Tuple[str, str, str]:
    horizon = int(horizon)
    available = pred[pred["horizon_minutes"] == horizon][["feature_set", "model"]].drop_duplicates()
    if available.empty:
        raise ValueError(f"No Step5 predictions found for horizon={horizon} min.")

    if cfg.selection_mode == "two_stage_cbf":
        exact = available[
            (available["feature_set"].astype(str) == cfg.selected_feature_set)
            & (available["model"].astype(str) == cfg.selected_model)
        ]
        if not exact.empty:
            return cfg.selected_feature_set, cfg.selected_model, "exact selected Two-stage CBF configuration"

        two_stage = available[available["feature_set"].astype(str).str.contains("TwoStage", case=False, na=False)].copy()
        if not two_stage.empty:
            two_stage["mean_test_RMSE"] = two_stage.apply(
                lambda r: _metric_mean(metrics, horizon, str(r["feature_set"]), str(r["model"])), axis=1
            )
            row = two_stage.sort_values(["mean_test_RMSE", "feature_set", "model"]).iloc[0]
            return str(row["feature_set"]), str(row["model"]), "fallback to best available Two-stage CBF configuration"

    if not best.empty and {"horizon_minutes", "feature_set", "model"}.issubset(best.columns):
        hit = best[best["horizon_minutes"] == horizon]
        if not hit.empty:
            row = hit.iloc[0]
            fs, mo = str(row["feature_set"]), str(row["model"])
            exists = available[(available["feature_set"].astype(str) == fs) & (available["model"].astype(str) == mo)]
            if not exists.empty:
                return fs, mo, "best-by-horizon table"

    tmp = available.copy()
    tmp["mean_test_RMSE"] = tmp.apply(
        lambda r: _metric_mean(metrics, horizon, str(r["feature_set"]), str(r["model"])), axis=1
    )
    row = tmp.sort_values(["mean_test_RMSE", "feature_set", "model"]).iloc[0]
    return str(row["feature_set"]), str(row["model"]), "fallback to overall lowest mean test RMSE"


def attach_val_rmse(selected: pd.DataFrame, metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rmse_col = cfg.rmse_col_for_lower_bound
    if rmse_col not in metrics.columns:
        print(f"[WARN] Metric column '{rmse_col}' not found. Falling back to 'test_RMSE'.")
        rmse_col = "test_RMSE"

    mcols = ["fold", "feature_set", "model", "horizon_minutes", rmse_col, "test_RMSE", "test_MAE", "test_R2"]
    mcols = [c for c in mcols if c in metrics.columns]
    met = metrics[mcols].copy()
    met = met.rename(columns={rmse_col: "rmse_for_lower_bound"})

    out = selected.merge(
        met,
        on=["fold", "feature_set", "model", "horizon_minutes"],
        how="left",
    )

    if out["rmse_for_lower_bound"].isna().any():
        grp_rmse = out.groupby(["horizon_minutes", "feature_set", "model"])["test_RMSE"].transform("mean")
        empirical = out.groupby(["horizon_minutes", "feature_set", "model"])["y_true"].transform(lambda x: np.nan)
        out["rmse_for_lower_bound"] = out["rmse_for_lower_bound"].fillna(grp_rmse)

    if out["rmse_for_lower_bound"].isna().any():
        tmp = out.groupby(["horizon_minutes", "feature_set", "model"]).apply(
            lambda g: math.sqrt(np.mean((g["y_true"].values - g["y_pred"].values) ** 2))
        ).reset_index(name="empirical_test_RMSE")
        out = out.merge(tmp, on=["horizon_minutes", "feature_set", "model"], how="left")
        out["rmse_for_lower_bound"] = out["rmse_for_lower_bound"].fillna(out["empirical_test_RMSE"])

    return out


def build_selected_predictions(cfg: Config, pred: pd.DataFrame, metrics: pd.DataFrame, best: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    manifest_rows = []

    for horizon in cfg.horizons_minutes:
        fs, mo, reason = select_config_for_horizon(cfg, horizon, pred, metrics, best)
        sub = pred[
            (pred["horizon_minutes"] == int(horizon))
            & (pred["feature_set"].astype(str) == fs)
            & (pred["model"].astype(str) == mo)
        ].copy()
        if sub.empty:
            raise RuntimeError(f"Selection returned empty predictions for horizon={horizon}, feature_set={fs}, model={mo}")
        selected_parts.append(sub)
        manifest_rows.append({
            "horizon_minutes": int(horizon),
            "selected_feature_set": fs,
            "selected_feature_display_name": FEATURE_DISPLAY_NAME.get(fs, fs),
            "selected_model": mo,
            "n_prediction_rows": int(len(sub)),
            "selection_reason": reason,
        })

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = attach_val_rmse(selected, metrics, cfg)
    manifest = pd.DataFrame(manifest_rows)
    return selected, manifest

def _safe_pct_saved(baseline_sum: float, policy_sum: float) -> float:
    if not np.isfinite(baseline_sum) or abs(baseline_sum) < 1e-12:
        return np.nan
    return float((baseline_sum - policy_sum) / baseline_sum * 100.0)


def _historical_tlow(raw: pd.DataFrame, first_time: pd.Timestamp, quantile: float) -> float:
    hist = raw[raw[TIME_COL] < first_time][LEAF_T].dropna()
    if len(hist) < 100:
        hist = raw[LEAF_T].dropna()
    return float(np.quantile(hist.values, quantile))


def prepare_replay_frame(cfg: Config, raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    raw_small = raw[[TIME_COL, LEAF_T, *cfg.resources]].copy()
    raw_small = raw_small.rename(columns={LEAF_T: "leafT_decision"})

    out = selected.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.merge(raw_small, on="time", how="left")

    out = out.rename(columns={"y_true": "leafT_outcome", "y_pred": "leafT_pred"})
    out["time_outcome_estimated"] = out["time"] + pd.to_timedelta(out["horizon_minutes"].astype(int), unit="m")

    needed = ["leafT_decision", "leafT_outcome", "leafT_pred", "rmse_for_lower_bound", *cfg.resources]
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["time", "fold", "horizon_minutes", "leafT_decision", "leafT_outcome", "leafT_pred", "rmse_for_lower_bound"])
    return out.sort_values(["horizon_minutes", "time", "fold"]).reset_index(drop=True)


def evaluate_one_fold_param(
    g: pd.DataFrame,
    raw: pd.DataFrame,
    cfg: Config,
    margin: float,
    reduce_ratio: float,
    k: float,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    horizon = int(g["horizon_minutes"].iloc[0])
    fold = str(g["fold"].iloc[0])
    feature_set = str(g["feature_set"].iloc[0])
    model = str(g["model"].iloc[0])
    first_time = pd.to_datetime(g["time"].min())

    T_low = _historical_tlow(raw, first_time, cfg.auto_Tlow_quantile)
    T_safe = T_low + float(margin)

    pred_lower = g["leafT_pred"].values - float(k) * g["rmse_for_lower_bound"].values
    rule_allow = g["leafT_decision"].values >= T_safe
    cbf_allow = pred_lower >= T_safe

    safety_ref = T_low if cfg.violation_reference == "T_low" else T_safe

    rows: List[Dict[str, object]] = []
    ts = g[["time", "time_outcome_estimated", "horizon_minutes", "fold", "feature_set", "model", "leafT_decision", "leafT_outcome", "leafT_pred", "rmse_for_lower_bound"]].copy()
    ts["T_low"] = T_low
    ts["T_safe"] = T_safe
    ts["pred_lower"] = pred_lower
    ts["rule_allow_reduce"] = rule_allow.astype(int)
    ts["cbf_allow_reduce"] = cbf_allow.astype(int)
    ts["margin"] = float(margin)
    ts["reduce_ratio"] = float(reduce_ratio)
    ts["k"] = float(k)

    for resource in cfg.resources:
        baseline = g[resource].values.astype(float)
        rule_action = baseline.copy()
        cbf_action = baseline.copy()
        rule_action[rule_allow] = rule_action[rule_allow] * (1.0 - float(reduce_ratio))
        cbf_action[cbf_allow] = cbf_action[cbf_allow] * (1.0 - float(reduce_ratio))

        baseline_sum = float(np.nansum(baseline))
        rule_sum = float(np.nansum(rule_action))
        cbf_sum = float(np.nansum(cbf_action))

        policy_defs = [
            ("LOGGED_BASELINE", np.zeros(len(g), dtype=bool), baseline_sum, baseline, 0.0),
            ("RULE_BASED", rule_allow, rule_sum, rule_action, _safe_pct_saved(baseline_sum, rule_sum)),
            ("CBF_BASED", cbf_allow, cbf_sum, cbf_action, _safe_pct_saved(baseline_sum, cbf_sum)),
        ]

        for policy, mask, action_sum, action_values, saved_pct in policy_defs:
            if policy == "LOGGED_BASELINE":
                reduced_n = 0
                reduced_frac = 0.0
                obs_violate_frac = 0.0
                obs_margin_mean = np.nan
            else:
                reduced_n = int(mask.sum())
                reduced_frac = float(mask.mean()) if len(mask) else np.nan
                if reduced_n > 0:
                    observed_outcome_reduced = g.loc[mask, "leafT_outcome"].values
                    obs_violate_frac = float(np.mean(observed_outcome_reduced < safety_ref) * 100.0)
                    obs_margin_mean = float(np.mean(observed_outcome_reduced - safety_ref))
                else:
                    obs_violate_frac = 0.0
                    obs_margin_mean = np.nan

            rows.append({
                "horizon_minutes": horizon,
                "fold": fold,
                "feature_set": feature_set,
                "feature_display_name": FEATURE_DISPLAY_NAME.get(feature_set, feature_set),
                "model": model,
                "policy": policy,
                "resource": resource,
                "resource_short": RESOURCE_SHORT.get(resource, resource),
                "margin": float(margin),
                "reduce_ratio": float(reduce_ratio),
                "k": float(k),
                "T_low": T_low,
                "T_safe": T_safe,
                "violation_reference": cfg.violation_reference,
                "baseline_sum": baseline_sum,
                "policy_sum": action_sum,
                "saved_pct": saved_pct,
                "reduced_n": reduced_n,
                "reduced_frac": reduced_frac,
                "obs_violate_frac": obs_violate_frac,
                "obs_margin_mean": obs_margin_mean,
                "n_eval": int(len(g)),
                "prediction_MAE_in_fold": float(np.mean(np.abs(g["leafT_outcome"].values - g["leafT_pred"].values))),
                "prediction_RMSE_in_fold": float(math.sqrt(np.mean((g["leafT_outcome"].values - g["leafT_pred"].values) ** 2))),
            })

        ts[f"{RESOURCE_SHORT.get(resource, resource)}_baseline"] = baseline
        ts[f"{RESOURCE_SHORT.get(resource, resource)}_rule"] = rule_action
        ts[f"{RESOURCE_SHORT.get(resource, resource)}_cbf"] = cbf_action

    return rows, ts


def run_replay_evaluation(cfg: Config, raw: pd.DataFrame, replay: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, object]] = []
    default_ts_parts: List[pd.DataFrame] = []

    grouped = replay.groupby(["horizon_minutes", "fold", "feature_set", "model"], sort=True)

    for _, g in grouped:
        g = g.sort_values("time").reset_index(drop=True)
        for margin in cfg.margin_grid:
            for reduce_ratio in cfg.reduce_grid:
                for k in cfg.k_grid:
                    rows, ts = evaluate_one_fold_param(g, raw, cfg, margin, reduce_ratio, k)
                    all_rows.extend(rows)

                    if (
                        abs(float(margin) - cfg.default_margin) < 1e-12
                        and abs(float(reduce_ratio) - cfg.default_reduce) < 1e-12
                        and abs(float(k) - cfg.default_k) < 1e-12
                    ):
                        default_ts_parts.append(ts)

    fold_results = pd.DataFrame(all_rows)
    if default_ts_parts:
        ts_default = pd.concat(default_ts_parts, ignore_index=True)
    else:
        ts_default = pd.DataFrame()

    agg_cols = [
        "horizon_minutes", "feature_set", "feature_display_name", "model", "policy",
        "resource", "resource_short", "margin", "reduce_ratio", "k", "violation_reference",
    ]
    summary = fold_results.groupby(agg_cols, dropna=False).agg(
        saved_pct_mean=("saved_pct", "mean"),
        saved_pct_std=("saved_pct", "std"),
        reduced_frac_mean=("reduced_frac", "mean"),
        reduced_frac_std=("reduced_frac", "std"),
        obs_violate_frac_mean=("obs_violate_frac", "mean"),
        obs_violate_frac_std=("obs_violate_frac", "std"),
        obs_margin_mean=("obs_margin_mean", "mean"),
        prediction_MAE_mean=("prediction_MAE_in_fold", "mean"),
        prediction_RMSE_mean=("prediction_RMSE_in_fold", "mean"),
        n_folds=("fold", "nunique"),
        n_eval_total=("n_eval", "sum"),
    ).reset_index()

    return fold_results, summary, ts_default

def filter_default(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return summary[
        (np.isclose(summary["margin"].astype(float), cfg.default_margin))
        & (np.isclose(summary["reduce_ratio"].astype(float), cfg.default_reduce))
        & (np.isclose(summary["k"].astype(float), cfg.default_k))
    ].copy()


def pick_best_by_policy_resource(summary: pd.DataFrame, policy: str) -> pd.DataFrame:
    sub = summary[summary["policy"] == policy].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(
        ["horizon_minutes", "resource", "saved_pct_mean", "obs_violate_frac_mean"],
        ascending=[True, True, False, True],
    )
    return sub.groupby(["horizon_minutes", "resource"], as_index=False).first()


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_word_table_default(default_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in default_summary.iterrows():
        rows.append({
            "Horizon": f"{int(r['horizon_minutes'])} min",
            "Policy": r["policy"],
            "Resource": r["resource_short"],
            "Saved (%)": fmt_mean_std(r["saved_pct_mean"], r["saved_pct_std"], 2),
            "Reduction fraction": fmt_mean_std(r["reduced_frac_mean"], r["reduced_frac_std"], 3),
            "Observed violation (%)": fmt_mean_std(r["obs_violate_frac_mean"], r["obs_violate_frac_std"], 2),
            "Observed margin": fmt_mean_std(r["obs_margin_mean"], np.nan, 3),
            "Prediction MAE": fmt_mean_std(r["prediction_MAE_mean"], np.nan, 3),
            "Prediction RMSE": fmt_mean_std(r["prediction_RMSE_mean"], np.nan, 3),
        })
    return pd.DataFrame(rows)

def _policy_rank_name(policy: str) -> str:
    if str(policy) == "RULE_BASED":
        return "Rule policy"
    if str(policy) == "CBF_BASED":
        return "CBF policy"
    if str(policy) == "LOGGED_BASELINE":
        return "Logged baseline"
    return str(policy)


def build_overall_policy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    group_cols = [
        "horizon_minutes",
        "feature_set",
        "feature_display_name",
        "model",
        "policy",
        "margin",
        "reduce_ratio",
        "k",
        "violation_reference",
    ]
    group_cols = [c for c in group_cols if c in summary.columns]

    overall = summary.groupby(group_cols, dropna=False).agg(
        overall_saving_score=("saved_pct_mean", "mean"),
        overall_saving_std=("saved_pct_std", "mean"),
        observed_violation_mean=("obs_violate_frac_mean", "mean"),
        observed_violation_std=("obs_violate_frac_std", "mean"),
        mean_reduced_fraction=("reduced_frac_mean", "mean"),
        mean_prediction_MAE=("prediction_MAE_mean", "mean"),
        mean_prediction_RMSE=("prediction_RMSE_mean", "mean"),
        n_resources=("resource_short", "nunique"),
    ).reset_index()

    pivot_save = summary.pivot_table(
        index=group_cols,
        columns="resource_short",
        values="saved_pct_mean",
        aggfunc="mean",
    ).reset_index()

    pivot_save = pivot_save.rename(columns={
        "Heat": "heat_saving_pct",
        "Light": "light_saving_pct",
        "CO2_D": "co2_dosage_saving_pct",
    })

    out = overall.merge(pivot_save, on=group_cols, how="left")
    out["policy_display"] = out["policy"].map(_policy_rank_name)
    return out.sort_values(
        ["horizon_minutes", "policy", "overall_saving_score", "observed_violation_mean"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def _match_param_rows(summary: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=summary.index)

    match_cols = [
        "horizon_minutes",
        "feature_set",
        "model",
        "policy",
        "margin",
        "reduce_ratio",
        "k",
        "violation_reference",
    ]
    for col in match_cols:
        if col in summary.columns and col in row.index:
            val = row[col]
            if pd.isna(val):
                mask &= summary[col].isna()
            elif isinstance(val, (float, np.floating)):
                mask &= np.isclose(pd.to_numeric(summary[col], errors="coerce").astype(float), float(val))
            else:
                mask &= summary[col].astype(str) == str(val)

    return summary[mask].copy()


def select_cbf_under_rule_default_safety(overall: pd.DataFrame, default_summary: pd.DataFrame, tol: float = 1e-12) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if overall.empty:
        return pd.DataFrame(), pd.DataFrame()

    default_overall = build_overall_policy_summary(default_summary)
    selected_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []

    for horizon in sorted(overall["horizon_minutes"].dropna().astype(int).unique()):
        rule_ref = default_overall[
            (default_overall["horizon_minutes"].astype(int) == int(horizon))
            & (default_overall["policy"] == "RULE_BASED")
        ].copy()

        if rule_ref.empty:
            rule_ref = overall[
                (overall["horizon_minutes"].astype(int) == int(horizon))
                & (overall["policy"] == "RULE_BASED")
            ].copy().sort_values(
                ["observed_violation_mean", "overall_saving_score"],
                ascending=[True, False],
            ).head(1)

        if rule_ref.empty:
            continue

        rule_row = rule_ref.iloc[0].copy()
        rule_saving = float(rule_row["overall_saving_score"])
        rule_violation = float(rule_row["observed_violation_mean"])

        rule_dict = rule_row.to_dict()
        rule_dict["selection_type"] = "Rule reference"
        rule_dict["selection_rule"] = "Default Rule policy used as the safety-risk reference"
        rule_dict["reference_rule_overall_saving_score"] = rule_saving
        rule_dict["reference_rule_observed_violation"] = rule_violation
        rule_dict["CBF_dominates_default_rule"] = False
        selected_rows.append(rule_dict)

        cbf_all = overall[
            (overall["horizon_minutes"].astype(int) == int(horizon))
            & (overall["policy"] == "CBF_BASED")
        ].copy()

        if cbf_all.empty:
            continue

        cbf_all["reference_rule_overall_saving_score"] = rule_saving
        cbf_all["reference_rule_observed_violation"] = rule_violation
        cbf_all["saving_minus_rule"] = cbf_all["overall_saving_score"].astype(float) - rule_saving
        cbf_all["violation_minus_rule"] = cbf_all["observed_violation_mean"].astype(float) - rule_violation
        cbf_all["CBF_dominates_default_rule"] = (
            (cbf_all["saving_minus_rule"] > 0)
            & (cbf_all["violation_minus_rule"] <= tol)
        )
        candidate_rows.extend(cbf_all.to_dict("records"))

        feasible = cbf_all[cbf_all["observed_violation_mean"].astype(float) <= rule_violation + tol].copy()

        if not feasible.empty:
            chosen = feasible.sort_values(
                ["overall_saving_score", "observed_violation_mean"],
                ascending=[False, True],
            ).iloc[0].copy()
            chosen["selection_type"] = "CBF selected"
            chosen["selection_rule"] = (
                "Maximum CBF overall saving with observed violation no greater "
                "than the default Rule policy at the same horizon"
            )
        else:
            chosen = cbf_all.sort_values(
                ["observed_violation_mean", "overall_saving_score"],
                ascending=[True, False],
            ).iloc[0].copy()
            chosen["selection_type"] = "CBF fallback"
            chosen["selection_rule"] = (
                "No CBF setting satisfied the Rule-policy safety constraint; "
                "selected the safest available CBF setting"
            )

        chosen["reference_rule_overall_saving_score"] = rule_saving
        chosen["reference_rule_observed_violation"] = rule_violation
        chosen["saving_minus_rule"] = float(chosen["overall_saving_score"]) - rule_saving
        chosen["violation_minus_rule"] = float(chosen["observed_violation_mean"]) - rule_violation
        chosen["CBF_dominates_default_rule"] = (
            (float(chosen["saving_minus_rule"]) > 0)
            and (float(chosen["violation_minus_rule"]) <= tol)
        )
        selected_rows.append(chosen.to_dict())

    selected = pd.DataFrame(selected_rows)
    candidates = pd.DataFrame(candidate_rows)

    if not selected.empty:
        selected = selected.sort_values(["horizon_minutes", "selection_type", "policy"]).reset_index(drop=True)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["horizon_minutes", "CBF_dominates_default_rule", "saving_minus_rule", "violation_minus_rule"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    return selected, candidates


def make_recommended_word_table(selected_overall: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if selected_overall.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for _, row in selected_overall.iterrows():
        sub_res = _match_param_rows(summary, row)
        saving_by_resource = {
            str(r["resource_short"]): float(r["saved_pct_mean"])
            for _, r in sub_res.iterrows()
            if str(r.get("resource_short", "")) in {"Heat", "Light", "CO2_D"}
        }
        saving_std_by_resource = {
            str(r["resource_short"]): r.get("saved_pct_std", np.nan)
            for _, r in sub_res.iterrows()
            if str(r.get("resource_short", "")) in {"Heat", "Light", "CO2_D"}
        }

        rows.append({
            "Horizon": f"{int(row['horizon_minutes'])} min",
            "Policy": _policy_rank_name(str(row["policy"])),
            "Selection type": row.get("selection_type", ""),
            "Heat saving (%)": fmt_mean_std(saving_by_resource.get("Heat", np.nan), saving_std_by_resource.get("Heat", np.nan), 2),
            "Light saving (%)": fmt_mean_std(saving_by_resource.get("Light", np.nan), saving_std_by_resource.get("Light", np.nan), 2),
            "CO2 dosage saving (%)": fmt_mean_std(saving_by_resource.get("CO2_D", np.nan), saving_std_by_resource.get("CO2_D", np.nan), 2),
            "Overall saving score (%)": fmt_mean_std(row.get("overall_saving_score", np.nan), row.get("overall_saving_std", np.nan), 2),
            "Observed violation (%)": fmt_mean_std(row.get("observed_violation_mean", np.nan), row.get("observed_violation_std", np.nan), 2),
            "Safety margin m": row.get("margin", np.nan),
            "Reduction ratio r": row.get("reduce_ratio", np.nan),
            "Conservativeness k": row.get("k", np.nan),
            "CBF dominates default Rule": row.get("CBF_dominates_default_rule", False),
            "Selection rule": row.get("selection_rule", ""),
        })

    return pd.DataFrame(rows)


def write_pareto_selection_notes(out_dir: Path, selected_overall: pd.DataFrame, candidates: pd.DataFrame) -> None:
    with open(out_dir / "pareto_selection_notes.txt", "w", encoding="utf-8") as f:
        f.write("Pareto-based operating-point selection notes\n")
        f.write("Selection rule:\n")
        f.write("- Default Rule policy is used as the safety-risk reference at each horizon.\n")
        f.write("- CBF settings are searched over the full grid of safety margin m, reduction ratio r, and conservativeness k.\n")
        f.write("- The selected CBF point maximizes overall resource-saving score subject to observed violation no greater than the default Rule policy.\n")
        f.write("- If no feasible CBF point exists, the safest CBF point is reported as a fallback.\n\n")

        if selected_overall.empty:
            f.write("No selected operating points were generated.\n")
            return

        cbf_selected = selected_overall[selected_overall["policy"] == "CBF_BASED"].copy()
        if cbf_selected.empty:
            f.write("No CBF selected points were found.\n")
            return

        n_dom = int(cbf_selected.get("CBF_dominates_default_rule", pd.Series(False, index=cbf_selected.index)).sum())
        f.write(f"Number of horizons where selected CBF dominates the default Rule policy: {n_dom}/{len(cbf_selected)}\n\n")

        for _, r in cbf_selected.iterrows():
            f.write(
                f"Horizon {int(r['horizon_minutes'])} min: "
                f"CBF overall saving={float(r['overall_saving_score']):.3f}%, "
                f"Rule reference saving={float(r.get('reference_rule_overall_saving_score', np.nan)):.3f}%, "
                f"CBF violation={float(r['observed_violation_mean']):.3f}%, "
                f"Rule reference violation={float(r.get('reference_rule_observed_violation', np.nan)):.3f}%, "
                f"dominates={bool(r.get('CBF_dominates_default_rule', False))}, "
                f"m={r.get('margin', np.nan)}, r={r.get('reduce_ratio', np.nan)}, k={r.get('k', np.nan)}\n"
            )

        if not candidates.empty:
            dom_candidates = candidates[candidates.get("CBF_dominates_default_rule", False) == True]
            f.write("\n")
            f.write(f"Total CBF candidate settings that dominate default Rule: {len(dom_candidates)}\n")

def write_readme(cfg: Config, out_dir: Path, manifest: pd.DataFrame) -> None:
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("Step6 replay-based control evaluation using NEW Step5 CBF predictions\n")
        f.write("Interpretation:\n")
        f.write("- Results estimate resource-saving potential under historical replay conditions.\n")
        f.write("- Reduced actions were not actually implemented in the greenhouse.\n")
        f.write("- The outputs should not be interpreted as validated closed-loop control savings.\n")
        f.write("- The replay setting cannot fully simulate counterfactual greenhouse dynamics.\n\n")
        f.write("Inputs:\n")
        f.write(f"- raw_csv: {cfg.raw_csv}\n")
        f.write(f"- predictions_csv: {cfg.predictions_csv}\n")
        f.write(f"- fold_metrics_csv: {cfg.fold_metrics_csv}\n")
        f.write(f"- best_by_horizon_csv: {cfg.best_by_horizon_csv}\n\n")
        f.write("Selected prediction configurations:\n")
        f.write(manifest.to_string(index=False))
        f.write("\n\n")
        f.write("Default replay parameters:\n")
        f.write(f"- auto_Tlow_quantile: {cfg.auto_Tlow_quantile}\n")
        f.write(f"- default_margin: {cfg.default_margin}\n")
        f.write(f"- default_reduce: {cfg.default_reduce}\n")
        f.write(f"- default_k: {cfg.default_k}\n")
        f.write(f"- violation_reference: {cfg.violation_reference}\n")

def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step6 | Replay-based control evaluation from NEW Step5 CBF predictions")
    print("=" * 80)
    print(f"[RAW]       {cfg.raw_csv}")
    print(f"[PRED]      {cfg.predictions_csv}")
    print(f"[METRICS]   {cfg.fold_metrics_csv}")
    print(f"[BEST]      {cfg.best_by_horizon_csv if cfg.best_by_horizon_csv else 'not found / not used'}")
    print(f"[OUT]       {out_dir.resolve()}")
    print(f"[SELECTION] mode={cfg.selection_mode}, feature_set={cfg.selected_feature_set}, model={cfg.selected_model}")
    print("=" * 80)

    raw = load_raw_data(cfg)
    pred, metrics, best = load_step5_outputs(cfg)

    selected, manifest = build_selected_predictions(cfg, pred, metrics, best)
    replay = prepare_replay_frame(cfg, raw, selected)

    if replay.empty:
        raise RuntimeError("No replay rows were built. Check time alignment between raw_csv and Step5 predictions.")

    fold_results, summary, ts_default = run_replay_evaluation(cfg, raw, replay)
    default_summary = filter_default(summary, cfg)
    best_cbf = pick_best_by_policy_resource(summary, "CBF_BASED")
    best_rule = pick_best_by_policy_resource(summary, "RULE_BASED")
    word_default = make_word_table_default(default_summary)

    overall_summary = build_overall_policy_summary(summary)
    selected_overall, cbf_candidates_vs_rule = select_cbf_under_rule_default_safety(
        overall=overall_summary,
        default_summary=default_summary,
    )
    recommended_word = make_recommended_word_table(selected_overall, summary)

    manifest.to_csv(out_dir / "selected_prediction_manifest.csv", index=False, encoding="utf-8-sig")
    replay.to_csv(out_dir / "selected_predictions_with_raw_context.csv", index=False, encoding="utf-8-sig")
    fold_results.to_csv(out_dir / "policy_fold_results_grid.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "policy_summary_grid.csv", index=False, encoding="utf-8-sig")
    default_summary.to_csv(out_dir / "policy_summary_default.csv", index=False, encoding="utf-8-sig")
    word_default.to_csv(out_dir / "policy_summary_default_for_word.csv", index=False, encoding="utf-8-sig")
    best_cbf.to_csv(out_dir / "best_cbf_by_resource.csv", index=False, encoding="utf-8-sig")
    best_rule.to_csv(out_dir / "best_rule_by_resource.csv", index=False, encoding="utf-8-sig")

    overall_summary.to_csv(out_dir / "overall_policy_summary_grid.csv", index=False, encoding="utf-8-sig")
    selected_overall.to_csv(out_dir / "pareto_selected_rule_reference_and_cbf.csv", index=False, encoding="utf-8-sig")
    cbf_candidates_vs_rule.to_csv(out_dir / "cbf_candidates_vs_default_rule.csv", index=False, encoding="utf-8-sig")
    recommended_word.to_csv(out_dir / "pareto_selected_policy_summary_for_word.csv", index=False, encoding="utf-8-sig")
    write_pareto_selection_notes(out_dir, selected_overall, cbf_candidates_vs_rule)

    if not ts_default.empty:
        ts_default.to_csv(out_dir / "policy_timeseries_default.csv", index=False, encoding="utf-8-sig")

    write_readme(cfg, out_dir, manifest)

    print("\n[OK] Replay evaluation finished.")
    print(f"[OK] Outputs written to: {out_dir.resolve()}")
    print("\nSelected configurations:")
    print(manifest.to_string(index=False))
    print("\nDefault summary preview:")
    if not word_default.empty:
        print(word_default.head(20).to_string(index=False))
    else:
        print("No default summary rows. Check default_margin/default_reduce/default_k are included in grids.")


if __name__ == "__main__":
    main()
