"""
构建AutoLag机制下的层级约束AutoLag-PCMCI+实验
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tigramite.data_processing import DataFrame as TigDataFrame
    try:
        from tigramite.independence_tests.parcorr import ParCorr  # type: ignore
    except Exception:
        from tigramite.independence_tests import ParCorr  # type: ignore
    from tigramite.pcmci import PCMCI
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires tigramite. Please run it in the same environment used for "
        "step3_1_autolag_module.py and step3_2_autolag_pcmciplus.py."
    ) from exc


# 配置
CSV_ENCODING = "utf-8-sig"
DT_MINUTES = 10

# 主要变量
VARS_MAINMODEL: Tuple[str, ...] = (
    "compartment/mass.plant",
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
    "delta_mass_24h",
)

TARGETS: Tuple[str, ...] = (
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
    "compartment/leaf_temperature",
)

LEAF_T = "compartment/leaf_temperature"
AIR_T = "compartment/air_temperature"

TAU_CAP_HOURS: Dict[str, int] = {
    "compartment/par": 2,
    "energy/co2_dosage": 2,
    "compartment/screen_energy/screen_position": 2,
    "compartment/screen_blackout/screen_position": 2,
    "compartment/window_position_lee_side": 2,
    "compartment/window_position_wind_side": 2,
    "compartment/heating_lower_circuit/pipe_temperature": 2,
    "energy/energy_use.heating": 6,
    "energy/electricity_use.lighting": 6,
    "compartment/water_supply/water_flow_duration": 6,
    "substrate_temperature_mean": 6,
    "substrate_relative_permittivity_mean": 12,
    "substrate_bulk_ec_mean": 12,
}
DEFAULT_TAU_CAP_HOURS = 2

FAST_COARSE = [1, 3, 6, 9, 12]
FAST_FINE_RADIUS = 4
SLOW_COARSE = [12, 24, 36, 48, 72, 96, 120, 144]
SLOW_FINE_RADIUS = 24

# Defaults, overridable by CLI.
PC_ALPHA_DEFAULT = 0.1
ALPHA_LEVEL_DEFAULT = 0.1
TAU_MIN_DEFAULT = 0
FIXED_TAU_MAX_DEFAULT = 12
N_SPLITS_DEFAULT = 5
VAL_DAYS_DEFAULT = 7
MIN_TRAIN_DEFAULT = 2000
RIDGE_ALPHA_DEFAULT = 1.0
BOOTSTRAP_REPS_DEFAULT = 30
BOOTSTRAP_BLOCK_SIZE_DEFAULT = 144  # one day at 10-min resolution
RANDOM_SEED_DEFAULT = 2026


# 工具
def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> float:
    return time.perf_counter()


def seconds_since(t0: float) -> float:
    return float(time.perf_counter() - t0)


def hours_to_lags(hours: int) -> int:
    return int(round(hours * 60 / DT_MINUTES))


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.interpolate(limit_direction="both").ffill().bfill()
    out = out.dropna(axis=0, how="any")
    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def clip_unique_sorted(vals: Iterable[int], lo: int, hi: int) -> List[int]:
    return sorted(set(int(v) for v in vals if lo <= int(v) <= hi))


def fine_grid(center: int, radius: int, lo: int, hi: int) -> List[int]:
    return list(range(max(lo, center - radius), min(hi, center + radius) + 1))


def maybe_get_tigramite_version() -> str:
    try:
        import tigramite  # type: ignore
        return str(getattr(tigramite, "__version__", "unknown"))
    except Exception:
        return "unknown"


# 层级约束
def build_groups(var_names: Sequence[str]) -> Dict[str, Set[str]]:
    action_vars: Set[str] = {
        "energy/co2_dosage",
        "energy/energy_use.heating",
        "energy/electricity_use.lighting",
        "compartment/window_position_lee_side",
        "compartment/window_position_wind_side",
        "compartment/screen_energy/screen_position",
        "compartment/screen_blackout/screen_position",
        "compartment/water_supply/water_flow_duration",
        "compartment/heating_lower_circuit/pipe_temperature",
    }
    state_vars: Set[str] = {
        "compartment/air_temperature",
        "compartment/humidity_deficit",
        "compartment/co2_concentration",
        "compartment/par",
        "substrate_relative_permittivity_mean",
        "substrate_bulk_ec_mean",
        "substrate_temperature_mean",
    }
    physio_vars: Set[str] = {LEAF_T}
    growth_vars: Set[str] = {"delta_mass_24h"}
    vn = set(var_names)
    return {
        "Action": action_vars & vn,
        "State": state_vars & vn,
        "Physiology": physio_vars & vn,
        "Growth": growth_vars & vn,
    }


def group_of(var: str, groups: Dict[str, Set[str]]) -> Optional[str]:
    for g, vals in groups.items():
        if var in vals:
            return g
    return None


def is_allowed_by_hierarchy(src: str, tgt: str, groups: Dict[str, Set[str]]) -> bool:
    if src == tgt:
        return True
    gs = group_of(src, groups)
    gt = group_of(tgt, groups)
    if gs is None or gt is None:
        return False
    return (
        (gs == "Action" and gt == "State")
        or (gs == "State" and gt == "Physiology")
        or (gs == "Physiology" and gt == "Growth")
    )


# 数据加载
def infer_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_in_csv(root: Path) -> Path:
    return root / "step0_clean_build" / "step0_results" / "data_trigger_mainmodel_z.csv"


def load_main_data(in_csv: Path, keep_growth: bool = False) -> pd.DataFrame:
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(in_csv, encoding=CSV_ENCODING)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    needed = [c for c in VARS_MAINMODEL if c in df.columns]
    missing_core = [c for c in TARGETS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing core target columns in input CSV: {missing_core}")

    df = df[needed].copy()
    df = safe_numeric_df(df)

    groups = build_groups(df.columns)
    categorized = [c for c in df.columns if group_of(c, groups) is not None]
    if keep_growth and "delta_mass_24h" in df.columns:
        if "delta_mass_24h" not in categorized:
            categorized.append("delta_mass_24h")
    else:
        categorized = [c for c in categorized if c != "delta_mass_24h"]

    return df[categorized].copy()


def make_pcmci(df: pd.DataFrame, verbosity: int = 0) -> PCMCI:
    tig_df = TigDataFrame(df.values, var_names=list(df.columns))
    parcorr = ParCorr(significance="analytic")
    return PCMCI(dataframe=tig_df, cond_ind_test=parcorr, verbosity=verbosity)


def run_pcmciplus(
    df: pd.DataFrame,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    link_assumptions: Optional[Dict[int, Dict[Tuple[int, int], str]]] = None,
    verbosity: int = 0,
) -> Dict[str, Any]:
    pcmci = make_pcmci(df, verbosity=verbosity)
    kwargs: Dict[str, Any] = dict(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    if link_assumptions is not None:
        kwargs["link_assumptions"] = link_assumptions
    return pcmci.run_pcmciplus(**kwargs)


def build_link_assumptions_for_variant(
    var_names: Sequence[str],
    groups: Dict[str, Set[str]],
    tau_by_target: Dict[str, int],
    use_hierarchy: bool,
    link_type: str = "-?>",
) -> Dict[int, Dict[Tuple[int, int], str]]:

    name_to_idx = {name: idx for idx, name in enumerate(var_names)}
    link_assumptions: Dict[int, Dict[Tuple[int, int], str]] = {j: {} for j in range(len(var_names))}

    for tgt in var_names:
        j = name_to_idx[tgt]
        th = int(tau_by_target.get(tgt, max(tau_by_target.values()) if tau_by_target else 1))
        th = max(1, th)
        for src in var_names:
            if use_hierarchy and not is_allowed_by_hierarchy(src, tgt, groups):
                continue
            i = name_to_idx[src]
            for tau in range(1, th + 1):
                link_assumptions[j][(i, -tau)] = link_type
    return link_assumptions


def collect_edges(
    results: Dict[str, Any],
    var_names: Sequence[str],
    groups: Dict[str, Set[str]],
    alpha_level: float,
    tau_min: int,
    tau_max: int,
    tau_by_target: Optional[Dict[str, int]] = None,
    enforce_hierarchy: bool = False,
    use_fdr: bool = False,
) -> pd.DataFrame:
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n = len(var_names)
    lag_start = max(1, tau_min)

    if use_fdr:
        pvals = []
        for i in range(n):
            for j in range(n):
                for tau in range(lag_start, tau_max + 1):
                    p = p_matrix[i, j, tau]
                    if np.isfinite(p):
                        pvals.append(float(p))
        p_thr = bh_fdr_threshold(np.array(pvals), alpha_level)
    else:
        p_thr = alpha_level

    rows = []
    for i in range(n):
        for j in range(n):
            src, tgt = var_names[i], var_names[j]
            if enforce_hierarchy and not is_allowed_by_hierarchy(src, tgt, groups):
                continue
            target_tau = tau_by_target.get(tgt, tau_max) if tau_by_target else tau_max
            for tau in range(lag_start, min(tau_max, int(target_tau)) + 1):
                pval = p_matrix[i, j, tau]
                if not np.isfinite(pval) or float(pval) > p_thr:
                    continue
                val = val_matrix[i, j, tau]
                rows.append(
                    dict(
                        source=src,
                        target=tgt,
                        lag=int(tau),
                        pval=float(pval),
                        val=float(val) if np.isfinite(val) else np.nan,
                        abs_val=float(abs(val)) if np.isfinite(val) else np.nan,
                        source_group=group_of(src, groups),
                        target_group=group_of(tgt, groups),
                        hierarchy_allowed=bool(is_allowed_by_hierarchy(src, tgt, groups)),
                    )
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source", "target", "lag", "pval", "val", "abs_val",
                "source_group", "target_group", "hierarchy_allowed",
            ]
        )
    return pd.DataFrame(rows).sort_values(["target", "source", "lag"]).reset_index(drop=True)


def bh_fdr_threshold(pvals: np.ndarray, q: float) -> float:
    p = pvals[np.isfinite(pvals)]
    if p.size == 0:
        return 0.0
    p_sorted = np.sort(p)
    m = p_sorted.size
    idx = np.arange(1, m + 1)
    crit = (idx / m) * q
    ok = p_sorted <= crit
    if not np.any(ok):
        return 0.0
    return float(p_sorted[np.max(np.where(ok))])


# AutoLag选择
def build_lagged_features_from_edges(
    df: pd.DataFrame,
    target: str,
    edges: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if edges.empty:
        return np.empty((0, 0)), np.empty((0,)), []
    feats = []
    names = []
    for _, row in edges.iterrows():
        src = str(row["source"])
        lag = int(row["lag"])
        if src not in df.columns or lag <= 0:
            continue
        feats.append(df[src].shift(lag))
        names.append(f"{src}__lag{lag}")
    if not feats:
        return np.empty((0, 0)), np.empty((0,)), []
    xdf = pd.concat(feats, axis=1)
    y = df[target].rename("y")
    all_df = pd.concat([xdf, y], axis=1).dropna(axis=0, how="any")
    if all_df.empty:
        return np.empty((0, 0)), np.empty((0,)), []
    return all_df.drop(columns=["y"]).values.astype(float), all_df["y"].values.astype(float), names


def time_series_splits(n: int, val_size: int, n_splits: int, gap: int, min_train: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n < min_train + gap + val_size + 1:
        return []
    starts = [n - (k + 1) * val_size for k in range(n_splits)][::-1]
    first_val_start = max(min_train + gap, n - n_splits * val_size)
    splits = []
    for start in starts:
        if start < first_val_start:
            continue
        end = min(n, start + val_size)
        train_end = start - gap
        if train_end < min_train:
            continue
        tr = np.arange(0, train_end, dtype=int)
        va = np.arange(start, end, dtype=int)
        if len(va) >= max(50, int(0.1 * val_size)):
            splits.append((tr, va))
    return splits


def cv_score_ridge(
    X: np.ndarray,
    y: np.ndarray,
    tau_max: int,
    n_splits: int,
    val_size: int,
    min_train: int,
    ridge_alpha: float,
) -> Dict[str, float]:
    splits = time_series_splits(len(y), val_size=val_size, n_splits=n_splits, gap=tau_max, min_train=min_train)
    if not splits:
        return dict(mean_mae=np.inf, std_mae=np.inf, mean_rmse=np.inf, std_rmse=np.inf, mean_r2=np.nan, n_folds=0)
    maes, rmses, r2s = [], [], []
    for tr, va in splits:
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        maes.append(mean_absolute_error(y[va], pred))
        rmses.append(rmse(y[va], pred))
        try:
            r2s.append(r2_score(y[va], pred))
        except Exception:
            r2s.append(np.nan)
    return dict(
        mean_mae=float(np.mean(maes)),
        std_mae=float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
        mean_rmse=float(np.mean(rmses)),
        std_rmse=float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
        mean_r2=float(np.nanmean(r2s)) if np.any(np.isfinite(r2s)) else np.nan,
        n_folds=int(len(maes)),
    )


def select_tau_1se(curve: pd.DataFrame) -> int:
    dfc = curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_mae", "std_mae"])
    if dfc.empty:
        return int(curve["tau_max"].max())
    best_idx = dfc["mean_mae"].idxmin()
    best_mean = float(dfc.loc[best_idx, "mean_mae"])
    best_std = float(dfc.loc[best_idx, "std_mae"])
    ok = dfc[dfc["mean_mae"] <= best_mean + best_std].copy()
    if ok.empty:
        return int(dfc.loc[best_idx, "tau_max"])
    ok = ok.sort_values(["n_edges", "tau_max"], ascending=[True, True])
    return int(ok.iloc[0]["tau_max"])


def candidate_tau_grid(target: str) -> List[int]:
    cap_hours = TAU_CAP_HOURS.get(target, DEFAULT_TAU_CAP_HOURS)
    cap = hours_to_lags(cap_hours)
    is_slow = cap_hours >= 12
    if is_slow:
        coarse = clip_unique_sorted(SLOW_COARSE, 1, cap)
        if cap not in coarse:
            coarse.append(cap)
        coarse = sorted(set(coarse))
        center = coarse[0]
        radius = SLOW_FINE_RADIUS
    else:
        coarse = clip_unique_sorted(FAST_COARSE, 1, cap)
        if cap not in coarse:
            coarse.append(cap)
        coarse = sorted(set(coarse))
        center = coarse[0]
        radius = FAST_FINE_RADIUS
    return coarse


def run_autolag_selection(
    df: pd.DataFrame,
    groups: Dict[str, Set[str]],
    targets: Sequence[str],
    pc_alpha: float,
    alpha_level: float,
    tau_min: int,
    n_splits: int,
    val_size: int,
    min_train: int,
    ridge_alpha: float,
    use_hierarchy_for_autolag_search: bool,
    out_dir: Path,
) -> Tuple[Dict[str, int], pd.DataFrame, float]:
    t_start = now()
    var_names = list(df.columns)
    tauhat: Dict[str, int] = {}
    curve_rows: List[Dict[str, Any]] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        if target not in df.columns:
            continue
        cap_hours = TAU_CAP_HOURS.get(target, DEFAULT_TAU_CAP_HOURS)
        tau_cap = hours_to_lags(cap_hours)
        is_slow = cap_hours >= 12
        coarse = clip_unique_sorted(SLOW_COARSE if is_slow else FAST_COARSE, 1, tau_cap)
        if tau_cap not in coarse:
            coarse.append(tau_cap)
        coarse = sorted(set(coarse))
        fine_radius = SLOW_FINE_RADIUS if is_slow else FAST_FINE_RADIUS

        def eval_tau(stage: str, tau: int) -> Dict[str, Any]:
            tau_by_target = {v: int(tau) for v in var_names}
            la = build_link_assumptions_for_variant(
                var_names=var_names,
                groups=groups,
                tau_by_target=tau_by_target,
                use_hierarchy=use_hierarchy_for_autolag_search,
                link_type="-?>",
            )
            t_pcmci = now()
            res = run_pcmciplus(df, tau_min=tau_min, tau_max=tau, pc_alpha=pc_alpha, link_assumptions=la, verbosity=0)
            pcmci_seconds = seconds_since(t_pcmci)
            edges = collect_edges(
                res,
                var_names,
                groups,
                alpha_level=alpha_level,
                tau_min=tau_min,
                tau_max=tau,
                tau_by_target=tau_by_target,
                enforce_hierarchy=use_hierarchy_for_autolag_search,
            )
            target_edges = edges[edges["target"] == target].copy()
            X, y, feat_names = build_lagged_features_from_edges(df, target, target_edges)
            if X.size == 0 or len(y) < min_train + tau + val_size + 1:
                score = dict(mean_mae=np.inf, std_mae=np.inf, mean_rmse=np.inf, std_rmse=np.inf, mean_r2=np.nan, n_folds=0)
            else:
                score = cv_score_ridge(X, y, tau, n_splits, val_size, min_train, ridge_alpha)
            rec = dict(
                target=target,
                stage=stage,
                tau_max=int(tau),
                tau_cap_lags=int(tau_cap),
                n_edges=int(len(target_edges)),
                n_features=int(len(feat_names)),
                pcmci_seconds=float(pcmci_seconds),
                **score,
            )
            return rec

        log(f"[AutoLag] target={target} coarse={coarse} cap={tau_cap}")
        coarse_recs = [eval_tau("coarse", tau) for tau in coarse]
        coarse_df = pd.DataFrame(coarse_recs)
        coarse_selected = select_tau_1se(coarse_df)
        fine = sorted(set(fine_grid(coarse_selected, fine_radius, 1, tau_cap)))
        log(f"[AutoLag] target={target} coarse_selected={coarse_selected}, fine={fine}")
        fine_recs = [eval_tau("fine", tau) for tau in fine]
        full_curve = pd.DataFrame(coarse_recs + fine_recs).drop_duplicates(subset=["target", "tau_max"], keep="last")
        selected = select_tau_1se(full_curve)
        tauhat[target] = int(selected)
        log(f"[AutoLag] target={target} tau_hat={selected}")
        curve_rows.extend(full_curve.to_dict("records"))

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / "autolag_cv_curves_all_targets.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame([{"target": k, "tau_hat": v} for k, v in sorted(tauhat.items())]).to_csv(
        out_dir / "tauhat_by_cv_review.csv", index=False, encoding=CSV_ENCODING
    )
    return tauhat, curve_df, seconds_since(t_start)

# Bootstrap stability
def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= block_size:
        return np.arange(n)
    starts = np.arange(0, n - block_size + 1)
    blocks = []
    while sum(len(b) for b in blocks) < n:
        s = int(rng.choice(starts))
        blocks.append(np.arange(s, s + block_size, dtype=int))
    idx = np.concatenate(blocks)[:n]
    return idx


def run_bootstrap_stability(
    df: pd.DataFrame,
    groups: Dict[str, Set[str]],
    tau_by_target: Dict[str, int],
    use_hierarchy: bool,
    tau_min: int,
    global_tau_max: int,
    pc_alpha: float,
    alpha_level: float,
    reps: int,
    block_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, float]:
    if reps <= 0:
        return pd.DataFrame(columns=["source", "target", "lag", "bootstrap_stability"]), 0.0
    rng = np.random.default_rng(seed)
    var_names = list(df.columns)
    counts: Dict[Tuple[str, str, int], int] = {}
    t0 = now()
    for b in range(reps):
        idx = block_bootstrap_indices(len(df), block_size, rng)
        boot_df = df.iloc[idx].reset_index(drop=True)
        la = build_link_assumptions_for_variant(var_names, groups, tau_by_target, use_hierarchy=use_hierarchy)
        try:
            res = run_pcmciplus(boot_df, tau_min=tau_min, tau_max=global_tau_max, pc_alpha=pc_alpha, link_assumptions=la, verbosity=0)
            edges = collect_edges(
                res,
                var_names,
                groups,
                alpha_level=alpha_level,
                tau_min=tau_min,
                tau_max=global_tau_max,
                tau_by_target=tau_by_target,
                enforce_hierarchy=use_hierarchy,
            )
            for _, r in edges.iterrows():
                key = (str(r["source"]), str(r["target"]), int(r["lag"]))
                counts[key] = counts.get(key, 0) + 1
        except Exception as exc:
            log(f"[WARN] Bootstrap replicate {b+1}/{reps} failed: {exc}")
    rows = [
        dict(source=s, target=t, lag=int(l), bootstrap_stability=float(c / reps), bootstrap_count=int(c), bootstrap_reps=int(reps))
        for (s, t, l), c in counts.items()
    ]
    return pd.DataFrame(rows), seconds_since(t0)

def evaluate_leaf_forecast_from_edges(
    df: pd.DataFrame,
    leaf_edges: pd.DataFrame,
    n_splits: int,
    val_size: int,
    min_train: int,
    ridge_alpha: float,
) -> Dict[str, Any]:
    if LEAF_T not in df.columns:
        return dict(forecast_mae=np.nan, forecast_rmse=np.nan, forecast_r2=np.nan, forecast_n_features=0, forecast_n_folds=0)
    if leaf_edges.empty:
        leaf_edges = pd.DataFrame([dict(source=LEAF_T, target=LEAF_T, lag=1, pval=np.nan, val=np.nan)])
    max_lag = int(leaf_edges["lag"].max()) if not leaf_edges.empty else 1
    X, y, names = build_lagged_features_from_edges(df, LEAF_T, leaf_edges)
    if X.size == 0:
        return dict(forecast_mae=np.nan, forecast_rmse=np.nan, forecast_r2=np.nan, forecast_n_features=0, forecast_n_folds=0)
    score = cv_score_ridge(X, y, max_lag, n_splits, val_size, min_train, ridge_alpha)
    return dict(
        forecast_mae=score["mean_mae"],
        forecast_mae_sd=score["std_mae"],
        forecast_rmse=score["mean_rmse"],
        forecast_rmse_sd=score["std_rmse"],
        forecast_r2=score["mean_r2"],
        forecast_n_features=len(names),
        forecast_n_folds=score["n_folds"],
    )

@dataclass
class Variant:
    method: str
    use_autolag: bool
    use_hierarchy: bool
    use_bootstrap: bool = True
    fixed_tau_max: int = FIXED_TAU_MAX_DEFAULT


ABLATION_VARIANTS: Tuple[Variant, ...] = (
    Variant("standard_pcmciplus", use_autolag=False, use_hierarchy=False, use_bootstrap=False),
    Variant("hierarchical_pcmciplus", use_autolag=False, use_hierarchy=True, use_bootstrap=False),
    Variant("autolag_pcmciplus", use_autolag=True, use_hierarchy=False, use_bootstrap=False),
    Variant("hierarchical_autolag_pcmciplus", use_autolag=True, use_hierarchy=True, use_bootstrap=True),
)


def run_single_variant(
    df: pd.DataFrame,
    groups: Dict[str, Set[str]],
    variant: Variant,
    tauhat: Dict[str, int],
    args: argparse.Namespace,
    out_dir: Path,
    bootstrap_threshold_for_summary: float = 0.6,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    var_names = list(df.columns)
    tau_by_target = {v: int(variant.fixed_tau_max) for v in var_names}
    if variant.use_autolag:
        tau_by_target.update({k: int(v) for k, v in tauhat.items() if k in var_names})
    global_tau_max = max(tau_by_target.values()) if tau_by_target else int(variant.fixed_tau_max)
    la = build_link_assumptions_for_variant(var_names, groups, tau_by_target, use_hierarchy=variant.use_hierarchy)

    method_dir = out_dir / variant.method
    method_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n[Variant] {variant.method} | AutoLag={variant.use_autolag}, Hierarchy={variant.use_hierarchy}, tau_max={global_tau_max}")
    t_final = now()
    res = run_pcmciplus(
        df,
        tau_min=args.tau_min,
        tau_max=global_tau_max,
        pc_alpha=args.pc_alpha,
        link_assumptions=la,
        verbosity=0,
    )
    final_seconds = seconds_since(t_final)

    edges = collect_edges(
        res,
        var_names,
        groups,
        alpha_level=args.alpha_level,
        tau_min=args.tau_min,
        tau_max=global_tau_max,
        tau_by_target=tau_by_target,
        enforce_hierarchy=variant.use_hierarchy,
        use_fdr=args.use_fdr,
    )
    edges["method"] = variant.method
    edges["use_autolag"] = variant.use_autolag
    edges["use_hierarchy"] = variant.use_hierarchy
    edges["tau_setting"] = "target_specific_autolag" if variant.use_autolag else f"fixed_tau_{variant.fixed_tau_max}"

    stability_df = pd.DataFrame(columns=["source", "target", "lag", "bootstrap_stability"])
    bootstrap_seconds = 0.0
    if variant.use_bootstrap and args.bootstrap_reps > 0:
        stability_df, bootstrap_seconds = run_bootstrap_stability(
            df=df,
            groups=groups,
            tau_by_target=tau_by_target,
            use_hierarchy=variant.use_hierarchy,
            tau_min=args.tau_min,
            global_tau_max=global_tau_max,
            pc_alpha=args.pc_alpha,
            alpha_level=args.alpha_level,
            reps=args.bootstrap_reps,
            block_size=args.bootstrap_block_size,
            seed=args.seed,
        )
        if not stability_df.empty:
            edges = edges.merge(stability_df, on=["source", "target", "lag"], how="left")
            edges["bootstrap_stability"] = edges["bootstrap_stability"].fillna(0.0)
        else:
            edges["bootstrap_stability"] = np.nan
    else:
        edges["bootstrap_stability"] = np.nan

    leaf_edges = edges[edges["target"] == LEAF_T].copy()
    forecast = evaluate_leaf_forecast_from_edges(
        df,
        leaf_edges,
        n_splits=args.n_splits,
        val_size=args.val_size,
        min_train=args.min_train,
        ridge_alpha=args.ridge_alpha,
    )

    inadmissible_edges = int((~edges["hierarchy_allowed"].astype(bool)).sum()) if not edges.empty else 0
    air_to_leaf = False
    if not edges.empty:
        air_to_leaf = bool(((edges["source"] == AIR_T) & (edges["target"] == LEAF_T)).any())
    summary = dict(
        method=variant.method,
        use_autolag=variant.use_autolag,
        use_hierarchy=variant.use_hierarchy,
        fixed_tau_max=variant.fixed_tau_max,
        global_tau_max=int(global_tau_max),
        leaf_tau_hat=int(tau_by_target.get(LEAF_T, global_tau_max)),
        n_total_edges=int(len(edges)),
        n_leaf_parents=int(len(leaf_edges)),
        n_inadmissible_edges=inadmissible_edges,
        air_t_to_leaf_t_identified=air_to_leaf,
        mean_abs_edge_strength=float(edges["abs_val"].mean()) if not edges.empty else np.nan,
        max_abs_edge_strength=float(edges["abs_val"].max()) if not edges.empty else np.nan,
        final_pcmciplus_seconds=float(final_seconds),
        bootstrap_seconds=float(bootstrap_seconds),
        total_variant_seconds=float(final_seconds + bootstrap_seconds),
        bootstrap_reps=int(args.bootstrap_reps if variant.use_bootstrap else 0),
        bootstrap_threshold_for_summary=float(bootstrap_threshold_for_summary),
        **forecast,
    )

    edges.to_csv(method_dir / "edge_details.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame([summary]).to_csv(method_dir / "variant_summary.csv", index=False, encoding=CSV_ENCODING)
    return edges, summary, stability_df


# 敏感性分析
def bootstrap_threshold_sensitivity(edges_full: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    if edges_full.empty or "bootstrap_stability" not in edges_full.columns:
        return pd.DataFrame()
    rows = []
    for thr in thresholds:
        stable = edges_full[edges_full["bootstrap_stability"].fillna(0.0) >= float(thr)].copy()
        leaf = stable[stable["target"] == LEAF_T]
        rows.append(
            dict(
                bootstrap_threshold=float(thr),
                n_stable_edges=int(len(stable)),
                n_stable_leaf_parents=int(len(leaf)),
                air_t_to_leaf_t_stable=bool(((stable["source"] == AIR_T) & (stable["target"] == LEAF_T)).any()) if not stable.empty else False,
                mean_abs_edge_strength=float(stable["abs_val"].mean()) if not stable.empty else np.nan,
                max_abs_edge_strength=float(stable["abs_val"].max()) if not stable.empty else np.nan,
            )
        )
    return pd.DataFrame(rows)


def lag_window_sensitivity(
    df: pd.DataFrame,
    groups: Dict[str, Set[str]],
    args: argparse.Namespace,
    tau_values: Sequence[int],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    var_names = list(df.columns)
    out = out_dir / "lag_window_sensitivity_edges"
    out.mkdir(parents=True, exist_ok=True)
    for tau in tau_values:
        tau_by_target = {v: int(tau) for v in var_names}
        la = build_link_assumptions_for_variant(var_names, groups, tau_by_target, use_hierarchy=True)
        t0 = now()
        res = run_pcmciplus(df, args.tau_min, int(tau), args.pc_alpha, la, verbosity=0)
        sec = seconds_since(t0)
        edges = collect_edges(
            res,
            var_names,
            groups,
            alpha_level=args.alpha_level,
            tau_min=args.tau_min,
            tau_max=int(tau),
            tau_by_target=tau_by_target,
            enforce_hierarchy=True,
            use_fdr=args.use_fdr,
        )
        edges["tau_max"] = int(tau)
        edges.to_csv(out / f"edges_fixed_tau_{tau}.csv", index=False, encoding=CSV_ENCODING)
        leaf_edges = edges[edges["target"] == LEAF_T]
        forecast = evaluate_leaf_forecast_from_edges(df, leaf_edges, args.n_splits, args.val_size, args.min_train, args.ridge_alpha)
        rows.append(
            dict(
                tau_max=int(tau),
                n_total_edges=int(len(edges)),
                n_leaf_parents=int(len(leaf_edges)),
                air_t_to_leaf_t_identified=bool(((edges["source"] == AIR_T) & (edges["target"] == LEAF_T)).any()) if not edges.empty else False,
                mean_abs_edge_strength=float(edges["abs_val"].mean()) if not edges.empty else np.nan,
                runtime_seconds=float(sec),
                **forecast,
            )
        )
    return pd.DataFrame(rows)


def write_config_manifest(args: argparse.Namespace, out_dir: Path, df: pd.DataFrame) -> None:
    manifest = {
        "script": Path(__file__).name,
        "purpose": "Reviewer-response AutoLag-PCMCI+ ablation, sensitivity, runtime, and edge-detail experiments",
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tigramite_version": maybe_get_tigramite_version(),
        "input_csv": str(Path(args.in_csv).resolve()),
        "n_samples": int(len(df)),
        "n_variables": int(df.shape[1]),
        "variables": list(df.columns),
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "tau_min": args.tau_min,
        "fixed_tau_max": args.fixed_tau_max,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_block_size": args.bootstrap_block_size,
        "bootstrap_thresholds": args.bootstrap_thresholds,
        "lag_window_values": args.lag_window_values,
        "cv_n_splits": args.n_splits,
        "cv_val_size": args.val_size,
        "cv_val_days": args.val_days,
        "min_train": args.min_train,
        "ridge_alpha": args.ridge_alpha,
        "random_seed": args.seed,
        "notes": [
            "AutoLag variants use target-specific tau_hat only as lag-window limits.",
            "Parent candidates are not pre-restricted to previously significant Step3 parents in ablation variants.",
            "Runtime excludes manual interpretation but includes PCMCI+ and bootstrap computations inside each variant.",
        ],
    }
    with open(out_dir / "experiment_config_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    flat = {k: v for k, v in manifest.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([flat]).to_csv(out_dir / "experiment_config_manifest.csv", index=False, encoding=CSV_ENCODING)

def parse_args() -> argparse.Namespace:
    root = infer_root_from_script()
    p = argparse.ArgumentParser(description="AutoLag-PCMCI+ reviewer-response experiments")
    p.add_argument("--in_csv", type=str, default=str(default_in_csv(root)))
    p.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "step3_review_experiments"))
    p.add_argument("--pc_alpha", type=float, default=PC_ALPHA_DEFAULT)
    p.add_argument("--alpha_level", type=float, default=ALPHA_LEVEL_DEFAULT)
    p.add_argument("--tau_min", type=int, default=TAU_MIN_DEFAULT)
    p.add_argument("--fixed_tau_max", type=int, default=FIXED_TAU_MAX_DEFAULT)
    p.add_argument("--use_fdr", action="store_true")

    p.add_argument("--n_splits", type=int, default=N_SPLITS_DEFAULT)
    p.add_argument("--val_days", type=int, default=VAL_DAYS_DEFAULT)
    p.add_argument("--val_size", type=int, default=-1, help="Validation size in samples. If <0, computed from val_days.")
    p.add_argument("--min_train", type=int, default=MIN_TRAIN_DEFAULT)
    p.add_argument("--ridge_alpha", type=float, default=RIDGE_ALPHA_DEFAULT)

    p.add_argument("--bootstrap_reps", type=int, default=BOOTSTRAP_REPS_DEFAULT)
    p.add_argument("--bootstrap_block_size", type=int, default=BOOTSTRAP_BLOCK_SIZE_DEFAULT)
    p.add_argument("--bootstrap_thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    p.add_argument("--lag_window_values", type=int, nargs="+", default=[3, 6, 9, 12])
    p.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    p.add_argument("--keep_growth", action="store_true", help="Keep delta_mass_24h if present. Default focuses on 17 core variables.")

    p.add_argument("--skip_autolag", action="store_true", help="Use fixed tau for all variants; for debugging only.")
    p.add_argument("--skip_bootstrap", action="store_true", help="Skip bootstrap stability to save time.")
    p.add_argument("--skip_lag_sensitivity", action="store_true", help="Skip fixed lag-window sensitivity analysis.")

    args = p.parse_args()
    if args.val_size < 0:
        args.val_size = int(round(args.val_days * 24 * 60 / DT_MINUTES))
    if args.skip_bootstrap:
        args.bootstrap_reps = 0
    return args


def main() -> int:
    warnings.filterwarnings("ignore")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log("Step3_3 | AutoLag-PCMCI+ reviewer-response experiments")
    log(f"[IN ] {args.in_csv}")
    log(f"[OUT] {out_dir}")
    log("=" * 72)

    total_start = now()
    df = load_main_data(Path(args.in_csv), keep_growth=args.keep_growth)
    groups = build_groups(df.columns)
    log(f"[DATA] T={len(df)}, N={df.shape[1]}")
    for g, vs in groups.items():
        log(f"[GROUP] {g}: {len(vs)} vars")

    write_config_manifest(args, out_dir, df)

    autolag_dir = out_dir / "autolag_selection"
    if args.skip_autolag:
        tauhat = {c: int(args.fixed_tau_max) for c in df.columns}
        autolag_curve = pd.DataFrame()
        autolag_seconds = 0.0
        log("[AutoLag] Skipped. Using fixed tau for all variables.")
    else:
        tauhat, autolag_curve, autolag_seconds = run_autolag_selection(
            df=df,
            groups=groups,
            targets=[t for t in TARGETS if t in df.columns],
            pc_alpha=args.pc_alpha,
            alpha_level=args.alpha_level,
            tau_min=args.tau_min,
            n_splits=args.n_splits,
            val_size=args.val_size,
            min_train=args.min_train,
            ridge_alpha=args.ridge_alpha,
            use_hierarchy_for_autolag_search=False,
            out_dir=autolag_dir,
        )
    log(f"[AutoLag] search_seconds={autolag_seconds:.2f}")

    all_edges = []
    summaries = []
    full_method_edges: Optional[pd.DataFrame] = None

    variants = []
    for v in ABLATION_VARIANTS:
        variants.append(Variant(v.method, v.use_autolag, v.use_hierarchy, v.use_bootstrap, fixed_tau_max=args.fixed_tau_max))

    for variant in variants:
        edges, summary, _ = run_single_variant(
            df=df,
            groups=groups,
            variant=variant,
            tauhat=tauhat,
            args=args,
            out_dir=out_dir / "ablation_runs",
            bootstrap_threshold_for_summary=0.6,
        )
        summary["autolag_search_seconds_shared"] = float(autolag_seconds if variant.use_autolag else 0.0)
        summary["total_with_autolag_seconds"] = float(summary["total_variant_seconds"] + (autolag_seconds if variant.use_autolag else 0.0))
        summaries.append(summary)
        all_edges.append(edges)
        if variant.method == "hierarchical_autolag_pcmciplus":
            full_method_edges = edges.copy()

    ablation_summary = pd.DataFrame(summaries)
    ablation_summary.to_csv(out_dir / "ablation_summary.csv", index=False, encoding=CSV_ENCODING)

    runtime_cols = [
        "method", "use_autolag", "use_hierarchy", "fixed_tau_max", "global_tau_max", "leaf_tau_hat",
        "autolag_search_seconds_shared", "final_pcmciplus_seconds", "bootstrap_seconds",
        "total_variant_seconds", "total_with_autolag_seconds", "bootstrap_reps",
    ]
    ablation_summary[runtime_cols].to_csv(out_dir / "runtime_breakdown.csv", index=False, encoding=CSV_ENCODING)

    edge_details_all = pd.concat(all_edges, axis=0, ignore_index=True) if all_edges else pd.DataFrame()
    edge_details_all.to_csv(out_dir / "edge_details_all_runs.csv", index=False, encoding=CSV_ENCODING)

    if not edge_details_all.empty:
        leaf_parent_summary = edge_details_all[edge_details_all["target"] == LEAF_T].copy()
        leaf_parent_summary.to_csv(out_dir / "leaf_temperature_parent_summary.csv", index=False, encoding=CSV_ENCODING)

    if full_method_edges is not None and not full_method_edges.empty and "bootstrap_stability" in full_method_edges.columns:
        bts = bootstrap_threshold_sensitivity(full_method_edges, args.bootstrap_thresholds)
        bts.to_csv(out_dir / "bootstrap_threshold_sensitivity.csv", index=False, encoding=CSV_ENCODING)
    else:
        pd.DataFrame().to_csv(out_dir / "bootstrap_threshold_sensitivity.csv", index=False, encoding=CSV_ENCODING)

    if not args.skip_lag_sensitivity:
        lws = lag_window_sensitivity(
            df=df,
            groups=groups,
            args=args,
            tau_values=args.lag_window_values,
            out_dir=out_dir,
        )
        lws.to_csv(out_dir / "lag_window_sensitivity.csv", index=False, encoding=CSV_ENCODING)

    total_seconds = seconds_since(total_start)
    with open(out_dir / "step3_3_review_experiment_summary.txt", "w", encoding="utf-8") as f:
        f.write("Step3_3 AutoLag-PCMCI+ review-response experiments\n")
        f.write("=" * 72 + "\n")
        f.write(f"Input: {args.in_csv}\n")
        f.write(f"Output: {out_dir}\n")
        f.write(f"T={len(df)}, N={df.shape[1]}\n")
        f.write(f"pc_alpha={args.pc_alpha}, alpha_level={args.alpha_level}, tau_min={args.tau_min}\n")
        f.write(f"fixed_tau_max={args.fixed_tau_max}\n")
        f.write(f"bootstrap_reps={args.bootstrap_reps}, block_size={args.bootstrap_block_size}\n")
        f.write(f"AutoLag search seconds: {autolag_seconds:.3f}\n")
        f.write(f"Total script seconds: {total_seconds:.3f}\n\n")
        f.write("Ablation summary:\n")
        f.write(ablation_summary.to_string(index=False))
        f.write("\n")

    log("\n[DONE] Review-response outputs saved:")
    for name in [
        "experiment_config_manifest.json",
        "ablation_summary.csv",
        "runtime_breakdown.csv",
        "bootstrap_threshold_sensitivity.csv",
        "lag_window_sensitivity.csv",
        "edge_details_all_runs.csv",
        "leaf_temperature_parent_summary.csv",
        "step3_3_review_experiment_summary.txt",
    ]:
        log(f"  - {out_dir / name}")
    log(f"[TIME] total_seconds={total_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
