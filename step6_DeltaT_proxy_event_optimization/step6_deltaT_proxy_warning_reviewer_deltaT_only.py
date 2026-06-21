"""
ΔT预警系统
ΔT = Leaf_T-Air_T
"""

from __future__ import annotations

import os
import re
import copy
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})


MODEL_DISPLAY_MAP = {
    "DT_PROXY_BASE": "DT-proxy base",
    "CAUSAL_PARENTS": "Causal-parents",
    "CAUSAL_PLUS_ACTION_LAGGED": "Causal+Lagged Controls",
}

DISPLAY_TO_KEY = {v: k for k, v in MODEL_DISPLAY_MAP.items()}


def display_model_name(model_key: str) -> str:
    return MODEL_DISPLAY_MAP.get(model_key, model_key)


def slugify_model_name(model_name: str) -> str:
    s = str(model_name).strip().lower()
    s = s.replace("+", "plus")
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def display_algo_name(algo_key: str) -> str:
    m = {
        "logistic": "Logistic",
        "rf": "RF",
        "xgb": "XGBoost",
    }
    return m.get(algo_key, algo_key)


def slugify_algo_name(algo_name: str) -> str:
    return slugify_model_name(algo_name)


@dataclass
class CommonConfig:
    csv_path: str
    out_dir: str
    time_col: str = "time"
    leaf_col: str = "compartment/leaf_temperature"
    air_col: str = "compartment/air_temperature"

    event_def: str = "deltaT_top"
    event_q: float = 0.90
    event_q_list: Tuple[float, ...] = (0.85, 0.90, 0.95)

    sensor_scenarios: Tuple[str, ...] = ("clean", "noise_5", "noise_10", "missing_5", "missing_10")

    horizons_min: Tuple[int, ...] = (10, 30, 60, 120)
    step_minutes: int = 10
    base_lags: Tuple[int, ...] = (1, 2, 3)

    n_splits: int = 5
    test_size_frac: float = 0.15

    thr_mode: str = "max_f1"
    top_score_quantile: float = 0.90

    random_state: int = 42

    vpd_col: Optional[str] = "compartment/humidity_deficit"
    par_col: Optional[str] = "compartment/par"
    high_env_q: float = 0.75

    algorithms: Tuple[str, ...] = ("logistic", "rf", "xgb")

    logistic_C: float = 1.0
    logistic_max_iter: int = 2000

    rf_n_estimators: int = 300
    rf_min_samples_leaf: int = 5
    rf_max_depth: Optional[int] = None

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_time_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    candidates = ["time", "timestamp", "datetime", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find a valid time column. existing={list(df.columns)[:20]}")


def _assert_raw_input(cfg: CommonConfig, df: pd.DataFrame) -> None:
    base = os.path.basename(str(cfg.csv_path)).lower()
    if "_z.csv" in base or base.endswith("z.csv") or "mainmodel_z" in base:
        raise ValueError(
            "Step7 deployment-oriented ΔT proxy warning must use the RAW dataset, "
            f"but the provided file looks like a z-score file: {cfg.csv_path}\n"
            "Please switch to data_trigger_mainmodel_raw.csv."
        )

    cols_to_check = [cfg.leaf_col, cfg.air_col]
    stats = {}
    for c in cols_to_check:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) > 50:
                stats[c] = (float(s.mean()), float(s.std(ddof=0)))

    suspicious = []
    for c, (m, sd) in stats.items():
        if abs(m) < 0.25 and 0.75 < sd < 1.25:
            suspicious.append(c)

    if len(suspicious) == len(stats) and len(stats) > 0:
        raise ValueError(
            "Step7 detected that the input may be standardized/z-score data "
            f"(columns near mean≈0 and std≈1: {suspicious}).\n"
            "Please use the RAW dataset: data_trigger_mainmodel_raw.csv."
        )


def load_and_prepare_base(cfg: CommonConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    cfg.time_col = infer_time_col(df, cfg.time_col)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)

    if cfg.leaf_col not in df.columns:
        raise KeyError(f"Missing leaf temperature column: {cfg.leaf_col}")
    if cfg.air_col not in df.columns:
        raise KeyError(f"Missing air temperature column: {cfg.air_col}")

    _assert_raw_input(cfg, df)

    df[cfg.leaf_col] = pd.to_numeric(df[cfg.leaf_col], errors="coerce")
    df[cfg.air_col] = pd.to_numeric(df[cfg.air_col], errors="coerce")
    df["deltaT_raw"] = df[cfg.leaf_col] - df[cfg.air_col]
    return df


def _parse_percent_from_scenario(scenario: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", scenario)
    if not m:
        return 0.0
    return float(m.group(1)) / 100.0


def apply_sensor_scenario(df: pd.DataFrame, cfg: CommonConfig, scenario: str) -> pd.DataFrame:
    scenario = scenario.strip().lower()
    out = df.copy()
    rng_seed = cfg.random_state + abs(hash(scenario)) % 100000
    rng = np.random.default_rng(rng_seed)

    sensor_cols = [cfg.leaf_col, cfg.air_col]
    for c in sensor_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if scenario == "clean":
        pass
    elif scenario.startswith("noise"):
        frac = _parse_percent_from_scenario(scenario)
        for c in sensor_cols:
            s = out[c].astype(float)
            sd = float(s.std(ddof=0))
            if np.isfinite(sd) and sd > 0:
                noise = rng.normal(loc=0.0, scale=frac * sd, size=len(out))
                out[c] = s + noise
    elif scenario.startswith("missing"):
        frac = _parse_percent_from_scenario(scenario)
        for c in sensor_cols:
            s = out[c].astype(float).copy()
            mask = rng.random(len(out)) < frac
            s.loc[mask] = np.nan
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
            out[c] = s
    else:
        raise ValueError(
            f"Unknown sensor scenario: {scenario}. Supported examples: "
            "clean, noise_5, noise_10, missing_5, missing_10"
        )

    out["deltaT_raw"] = out[cfg.leaf_col] - out[cfg.air_col]
    out["sensor_scenario"] = scenario
    return out


def save_deltaT_outputs(df: pd.DataFrame, cfg: CommonConfig) -> None:
    out_ts = df[[cfg.time_col, cfg.leaf_col, cfg.air_col, "deltaT_raw"]].copy()
    if "sensor_scenario" in df.columns:
        out_ts.insert(1, "sensor_scenario", df["sensor_scenario"].values)
    out_ts.to_csv(os.path.join(cfg.out_dir, "deltaT_timeseries.csv"), index=False)

    s = out_ts["deltaT_raw"].describe(percentiles=[0.1, 0.5, 0.85, 0.9, 0.95])
    lines = [
        "dataset_mode: raw",
        "task: deployment-oriented ΔT proxy early warning",
        f"csv_path: {cfg.csv_path}",
        f"sensor_scenario: {str(df['sensor_scenario'].iloc[0]) if 'sensor_scenario' in df.columns and len(df) else 'clean'}",
        f"event_def: {cfg.event_def}",
        f"event_q: {cfg.event_q}",
        f"horizons: {cfg.horizons_min}",
        f"thr_mode: {cfg.thr_mode}",
        f"algorithms: {cfg.algorithms}",
        "",
        f"count: {s['count']:.0f}",
        f"mean: {s['mean']:.6f}",
        f"std: {s['std']:.6f}",
        f"min: {s['min']:.6f}",
        f"p10: {s['10%']:.6f}",
        f"p50: {s['50%']:.6f}",
        f"p85: {s['85%']:.6f}",
        f"p90: {s['90%']:.6f}",
        f"p95: {s['95%']:.6f}",
        f"max: {s['max']:.6f}",
    ]
    with open(os.path.join(cfg.out_dir, "deltaT_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_basic_figures(df: pd.DataFrame, cfg: CommonConfig) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(df["deltaT_raw"].dropna().values, bins=50)
    plt.xlabel("ΔT = Leaf_T - Air_T (raw, °C)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "fig_deltaT_distribution.png"), dpi=180)
    plt.close()

    n_tail = min(1500, len(df))
    plt.figure(figsize=(12, 4))
    plt.plot(df[cfg.time_col].tail(n_tail), df["deltaT_raw"].tail(n_tail), linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("ΔT (raw, °C)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "fig_deltaT_timeseries.png"), dpi=180)
    plt.close()

    corr_items = []
    for col in [cfg.leaf_col, cfg.air_col, cfg.vpd_col, cfg.par_col]:
        if col and col in df.columns:
            corr = df[["deltaT_raw", col]].corr().iloc[0, 1]
            corr_items.append((col, corr))
    if corr_items:
        names = [x[0] for x in corr_items]
        vals = [x[1] for x in corr_items]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(names)), vals)
        plt.xticks(range(len(names)), names, rotation=20, ha="right")
        plt.ylabel("Correlation with ΔT (raw)")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "fig_corr_bar.png"), dpi=180)
        plt.close()


def minutes_to_hsteps(minutes: int, step_minutes: int) -> int:
    return int(round(minutes / step_minutes))


def add_lags(df: pd.DataFrame, cols: List[str], lags: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for lag in lags:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def make_future_event_label(df: pd.DataFrame, cfg: CommonConfig, horizon_min: int) -> pd.DataFrame:
    out = df.copy()
    h = minutes_to_hsteps(horizon_min, cfg.step_minutes)

    if cfg.event_def != "deltaT_top":
        raise NotImplementedError("Only deltaT_top is implemented in unified version.")

    thr = out["deltaT_raw"].quantile(cfg.event_q)
    out[f"event_h{horizon_min}"] = (out["deltaT_raw"].shift(-h) >= thr).astype(float)
    out.loc[out["deltaT_raw"].shift(-h).isna(), f"event_h{horizon_min}"] = np.nan
    out[f"event_threshold_h{horizon_min}"] = float(thr)
    return out


def save_event_labels(df: pd.DataFrame, cfg: CommonConfig) -> None:
    cols = [cfg.time_col, "deltaT_raw"]
    if "sensor_scenario" in df.columns:
        cols.append("sensor_scenario")
    for hz in cfg.horizons_min:
        cols.append(f"event_h{hz}")
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(os.path.join(cfg.out_dir, "event_labels.csv"), index=False)


def rolling_origin_splits(n: int, n_splits: int, test_frac: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    test_size = max(50, int(n * test_frac))
    train_min = max(200, n - test_size * n_splits)
    splits = []
    train_end = train_min
    for _ in range(n_splits):
        test_start = train_end
        test_end = min(n, test_start + test_size)
        if test_end - test_start < 20:
            break
        tr = np.arange(0, train_end)
        te = np.arange(test_start, test_end)
        splits.append((tr, te))
        train_end = test_end
    return splits


def make_classifier(cfg: CommonConfig, algorithm: str, random_state: int) -> Pipeline:
    algorithm = algorithm.lower()

    if algorithm == "logistic":
        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=cfg.logistic_C,
                    max_iter=cfg.logistic_max_iter,
                    random_state=random_state,
                    class_weight="balanced",
                    solver="lbfgs",
                )),
            ]
        )

    if algorithm == "rf":
        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=cfg.rf_n_estimators,
                    max_depth=cfg.rf_max_depth,
                    min_samples_leaf=cfg.rf_min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                )),
            ]
        )

    if algorithm == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("XGBoost is not installed. Please run: pip install xgboost") from e

        return Pipeline(
            steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=cfg.xgb_n_estimators,
                    max_depth=cfg.xgb_max_depth,
                    learning_rate=cfg.xgb_learning_rate,
                    subsample=cfg.xgb_subsample,
                    colsample_bytree=cfg.xgb_colsample_bytree,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=4,
                )),
            ]
        )

    raise ValueError(f"Unknown algorithm={algorithm}. Supported: {cfg.algorithms}")


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, mode: str, top_score_quantile: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if mode == "top_quantile":
        return float(np.quantile(y_score, top_score_quantile))

    if mode != "max_f1":
        raise ValueError(f"Unknown thr_mode={mode}")

    p, r, t = precision_recall_curve(y_true, y_score)
    if len(t) == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(t[best_idx])


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    out = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "f1": np.nan,
        "brier": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "top10_alert_precision": np.nan,
    }

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = np.nan

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    out["precision"] = float(tp / max(tp + fp, 1))
    out["recall"] = float(tp / max(tp + fn, 1))

    q = np.quantile(y_score, 0.90)
    mask = y_score >= q
    if mask.sum() > 0:
        out["top10_alert_precision"] = float(y_true[mask].mean())
    return out


def _safe_feature_importance_from_pipeline(clf: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    arr = np.zeros(len(feature_names), dtype=float)
    try:
        est = clf.named_steps.get("clf", None)
    except Exception:
        est = None

    if est is not None:
        if hasattr(est, "coef_"):
            v = np.abs(np.ravel(est.coef_))
            if len(v) == len(feature_names):
                arr = v
        elif hasattr(est, "feature_importances_"):
            v = np.ravel(est.feature_importances_)
            if len(v) == len(feature_names):
                arr = v

    out = pd.DataFrame({"feature": feature_names, "importance": arr})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def fit_predict_single_split(
    X: pd.DataFrame,
    y: pd.Series,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    cfg: CommonConfig,
    algorithm: str,
    random_state: int,
    thr_mode: str,
    top_score_quantile: float,
) -> Dict[str, object]:
    Xtr = X.iloc[tr_idx].copy()
    Xte = X.iloc[te_idx].copy()
    ytr = y.iloc[tr_idx].astype(int).values
    yte = y.iloc[te_idx].astype(int).values

    clf = make_classifier(cfg=cfg, algorithm=algorithm, random_state=random_state)
    clf.fit(Xtr, ytr)

    score_tr = clf.predict_proba(Xtr)[:, 1]
    score_te = clf.predict_proba(Xte)[:, 1]

    thr = choose_threshold(ytr, score_tr, thr_mode, top_score_quantile)
    met = classification_metrics(yte, score_te, thr)
    return {
        "threshold": thr,
        "metrics": met,
        "score_test": score_te,
        "y_test": yte,
        "clf": clf,
    }


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_prefix: str) -> None:
    if len(np.unique(y_true)) <= 1:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, linewidth=2.0)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    plt.savefig(out_prefix + "_roc.png", dpi=180)
    plt.close()

    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, linewidth=2.0)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.tick_params(axis="both", labelsize=20)
    plt.tight_layout()
    plt.savefig(out_prefix + "_pr.png", dpi=180)
    plt.close()


def summarize_high_env(df_eval: pd.DataFrame, cfg: CommonConfig) -> pd.DataFrame:
    out_rows = []
    if cfg.vpd_col and cfg.vpd_col in df_eval.columns:
        vpd_thr = df_eval[cfg.vpd_col].quantile(cfg.high_env_q)
        mask = df_eval[cfg.vpd_col] >= vpd_thr
        if mask.sum() > 0:
            out_rows.append({
                "env_type": "high_vpd",
                "n": int(mask.sum()),
                "positive_rate": float(df_eval.loc[mask, "y_true"].mean())
            })
    if cfg.par_col and cfg.par_col in df_eval.columns:
        par_thr = df_eval[cfg.par_col].quantile(cfg.high_env_q)
        mask = df_eval[cfg.par_col] >= par_thr
        if mask.sum() > 0:
            out_rows.append({
                "env_type": "high_par",
                "n": int(mask.sum()),
                "positive_rate": float(df_eval.loc[mask, "y_true"].mean())
            })
    return pd.DataFrame(out_rows)


def _format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _aggregate_fold_metrics(df_fold: pd.DataFrame) -> Dict[str, float | str]:
    metric_cols = [
        "roc_auc", "pr_auc", "f1", "brier", "precision", "recall", "top10_alert_precision"
    ]
    agg: Dict[str, float | str] = {}
    for col in metric_cols:
        if col not in df_fold.columns:
            continue
        mean = float(df_fold[col].mean())
        std = float(df_fold[col].std(ddof=1)) if len(df_fold) > 1 else 0.0
        # Backward-compatible mean alias.
        agg[col] = mean
        agg[f"{col}_mean"] = mean
        agg[f"{col}_std"] = std
        agg[f"{col}_mean_std"] = _format_mean_std(mean, std)
    return agg


def run_models_unified_single(
    df: pd.DataFrame,
    cfg: CommonConfig,
    model_feature_sets: Dict[str, List[str]],
    sensor_scenario: str,
    event_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(cfg.out_dir)
    save_deltaT_outputs(df, cfg)
    save_basic_figures(df, cfg)

    metrics_rows = []
    thresholds_rows = []
    fi_rows = []

    deltaT_threshold = float(df["deltaT_raw"].quantile(cfg.event_q))

    for horizon in cfg.horizons_min:
        df_h = make_future_event_label(df, cfg, horizon)

        for model_key, feature_cols in model_feature_sets.items():
            used_cols = [c for c in feature_cols if c in df_h.columns]
            used_cols = sorted(set(used_cols))
            if len(used_cols) == 0:
                continue

            df_feat = add_lags(df_h, used_cols, cfg.base_lags)
            lag_cols = [
                f"{c}_lag{lag}"
                for c in used_cols
                for lag in cfg.base_lags
                if f"{c}_lag{lag}" in df_feat.columns
            ]
            y_col = f"event_h{horizon}"

            keep_cols = [cfg.time_col, y_col] + lag_cols
            aux_cols = [x for x in [cfg.vpd_col, cfg.par_col, "deltaT_raw"] if x and x in df_feat.columns]
            keep_cols += aux_cols
            df_model = df_feat[keep_cols].dropna().reset_index(drop=True)
            if len(df_model) < 300:
                continue

            X = df_model[lag_cols].copy()
            y = df_model[y_col].astype(float).copy()
            splits = rolling_origin_splits(len(df_model), cfg.n_splits, cfg.test_size_frac)
            if len(splits) == 0:
                continue

            for algorithm in cfg.algorithms:
                fold_metrics = []
                fold_thresholds = []
                all_eval_rows = []
                last_clf = None

                try:
                    for fold_id, (tr, te) in enumerate(splits, start=1):
                        out = fit_predict_single_split(
                            X=X,
                            y=y,
                            tr_idx=tr,
                            te_idx=te,
                            cfg=cfg,
                            algorithm=algorithm,
                            random_state=cfg.random_state + fold_id,
                            thr_mode=cfg.thr_mode,
                            top_score_quantile=cfg.top_score_quantile,
                        )
                        last_clf = out["clf"]
                        fold_thresholds.append(float(out["threshold"]))
                        met = out["metrics"]
                        met["fold"] = fold_id
                        fold_metrics.append(met)

                        df_eval = pd.DataFrame({
                            cfg.time_col: df_model.loc[te, cfg.time_col].values,
                            "y_true": out["y_test"],
                            "y_score": out["score_test"],
                        })
                        for c in aux_cols:
                            df_eval[c] = df_model.loc[te, c].values
                        df_eval["fold"] = fold_id
                        df_eval["sensor_scenario"] = sensor_scenario
                        df_eval["event_q"] = event_q
                        df_eval["deltaT_threshold"] = deltaT_threshold
                        all_eval_rows.append(df_eval)
                except ImportError as e:
                    print(f"[WARN] Skip {display_model_name(model_key)} + {display_algo_name(algorithm)}: {e}")
                    continue

                df_fold = pd.DataFrame(fold_metrics)
                agg = _aggregate_fold_metrics(df_fold)
                agg.update({
                    "sensor_scenario": sensor_scenario,
                    "event_q": event_q,
                    "deltaT_threshold": deltaT_threshold,
                    "model": display_model_name(model_key),
                    "model_key": model_key,
                    "algorithm": display_algo_name(algorithm),
                    "algorithm_key": algorithm,
                    "horizon_min": horizon,
                    "n_folds": int(len(df_fold)),
                    "n_samples": int(len(df_model)),
                    "positive_rate": float(y.mean()),
                })
                metrics_rows.append(agg)

                thresholds_rows.append({
                    "sensor_scenario": sensor_scenario,
                    "event_q": event_q,
                    "deltaT_threshold": deltaT_threshold,
                    "model": display_model_name(model_key),
                    "model_key": model_key,
                    "algorithm": display_algo_name(algorithm),
                    "algorithm_key": algorithm,
                    "horizon_min": horizon,
                    "threshold_mean": float(np.mean(fold_thresholds)) if fold_thresholds else np.nan,
                    "threshold_std": float(np.std(fold_thresholds, ddof=1)) if len(fold_thresholds) > 1 else 0.0,
                    "threshold_last_fold": float(fold_thresholds[-1]) if fold_thresholds else np.nan,
                })

                if last_clf is not None:
                    fi = _safe_feature_importance_from_pipeline(last_clf, lag_cols)
                    fi["sensor_scenario"] = sensor_scenario
                    fi["event_q"] = event_q
                    fi["model"] = display_model_name(model_key)
                    fi["model_key"] = model_key
                    fi["algorithm"] = display_algo_name(algorithm)
                    fi["algorithm_key"] = algorithm
                    fi["horizon_min"] = horizon
                    fi_rows.append(fi)

                df_eval_all = pd.concat(all_eval_rows, axis=0, ignore_index=True)
                model_slug = slugify_model_name(display_model_name(model_key))
                algo_slug = slugify_algo_name(display_algo_name(algorithm))

                df_eval_all.to_csv(
                    os.path.join(cfg.out_dir, f"eval_rows_h{horizon}_{model_slug}_{algo_slug}.csv"),
                    index=False,
                )

                plot_roc_pr(
                    df_eval_all["y_true"].values,
                    df_eval_all["y_score"].values,
                    os.path.join(cfg.out_dir, f"fig_h{horizon}_{model_slug}_{algo_slug}"),
                )

                df_env = summarize_high_env(df_eval_all, cfg)
                if len(df_env) > 0:
                    df_env.insert(0, "sensor_scenario", sensor_scenario)
                    df_env.insert(1, "event_q", event_q)
                    df_env.insert(2, "horizon_min", horizon)
                    df_env.insert(3, "model", display_model_name(model_key))
                    df_env.insert(4, "algorithm", display_algo_name(algorithm))
                    df_env.to_csv(
                        os.path.join(cfg.out_dir, f"high_env_summary_h{horizon}_{model_slug}_{algo_slug}.csv"),
                        index=False,
                    )

    df_metrics = pd.DataFrame(metrics_rows)
    df_thr = pd.DataFrame(thresholds_rows)
    df_fi = pd.concat(fi_rows, axis=0, ignore_index=True) if len(fi_rows) > 0 else pd.DataFrame()

    df_metrics.to_csv(os.path.join(cfg.out_dir, "step7_metrics_summary_multimodel.csv"), index=False)
    df_thr.to_csv(os.path.join(cfg.out_dir, "step7_thresholds_summary_multimodel.csv"), index=False)
    if len(df_fi) > 0:
        df_fi.to_csv(os.path.join(cfg.out_dir, "step7_feature_importance_multimodel.csv"), index=False)

    merged = pd.concat([make_future_event_label(df, cfg, h) for h in cfg.horizons_min], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    save_event_labels(merged, cfg)

    return df_metrics, df_thr, df_fi


def run_models_unified(df: pd.DataFrame, cfg: CommonConfig, model_feature_sets: Dict[str, List[str]]) -> None:
    root_out_dir = cfg.out_dir
    ensure_dir(root_out_dir)

    all_metrics = []
    all_thresholds = []
    all_fi = []

    for scenario in cfg.sensor_scenarios:
        df_s = apply_sensor_scenario(df, cfg, scenario)
        for event_q in cfg.event_q_list:
            q_tag = f"q{int(round(event_q * 100)):02d}"
            scenario_tag = slugify_model_name(scenario)
            run_out_dir = os.path.join(root_out_dir, f"scenario_{scenario_tag}", q_tag)
            cfg_run = copy.copy(cfg)
            cfg_run.event_q = float(event_q)
            cfg_run.out_dir = run_out_dir

            print(f"[INFO] Running Step7: scenario={scenario}, event_q={event_q}, out={run_out_dir}")
            df_metrics, df_thr, df_fi = run_models_unified_single(
                df=df_s,
                cfg=cfg_run,
                model_feature_sets=model_feature_sets,
                sensor_scenario=scenario,
                event_q=float(event_q),
            )
            if len(df_metrics) > 0:
                all_metrics.append(df_metrics)
            if len(df_thr) > 0:
                all_thresholds.append(df_thr)
            if len(df_fi) > 0:
                all_fi.append(df_fi)

    df_all_metrics = pd.concat(all_metrics, axis=0, ignore_index=True) if all_metrics else pd.DataFrame()
    df_all_thresholds = pd.concat(all_thresholds, axis=0, ignore_index=True) if all_thresholds else pd.DataFrame()
    df_all_fi = pd.concat(all_fi, axis=0, ignore_index=True) if all_fi else pd.DataFrame()

    df_all_metrics.to_csv(os.path.join(root_out_dir, "step7_metrics_summary_all_scenarios.csv"), index=False)
    df_all_thresholds.to_csv(os.path.join(root_out_dir, "step7_thresholds_summary_all_scenarios.csv"), index=False)
    if len(df_all_fi) > 0:
        df_all_fi.to_csv(os.path.join(root_out_dir, "step7_feature_importance_all_scenarios.csv"), index=False)

    # Reviewer-table outputs.
    if len(df_all_metrics) > 0:
        # ΔT threshold sensitivity: clean scenario across q values.
        df_threshold_sens = df_all_metrics[df_all_metrics["sensor_scenario"] == "clean"].copy()
        df_threshold_sens.to_csv(
            os.path.join(root_out_dir, "table_deltaT_threshold_sensitivity_for_word.csv"),
            index=False,
        )

        df_robust = df_all_metrics[np.isclose(df_all_metrics["event_q"].astype(float), 0.90)].copy()
        df_robust.to_csv(
            os.path.join(root_out_dir, "table_sensor_robustness_for_word.csv"),
            index=False,
        )

        core_cols = [
            "sensor_scenario", "event_q", "deltaT_threshold", "model", "algorithm", "horizon_min",
            "roc_auc_mean_std", "pr_auc_mean_std", "f1_mean_std", "top10_alert_precision_mean_std",
            "positive_rate", "n_folds", "n_samples",
        ]
        core_cols = [c for c in core_cols if c in df_all_metrics.columns]
        df_all_metrics[core_cols].to_csv(
            os.path.join(root_out_dir, "table_step7_core_metrics_for_word.csv"),
            index=False,
        )

    print("[OK] Step7 reviewer-response runs finished:", root_out_dir)

def _parse_float_tuple(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_tuple(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--edges_csv", type=str, default="")
    p.add_argument("--time_col", type=str, default="time")
    p.add_argument("--leaf_col", type=str, default="compartment/leaf_temperature")
    p.add_argument("--air_col", type=str, default="compartment/air_temperature")
    p.add_argument("--vpd_col", type=str, default="compartment/humidity_deficit")
    p.add_argument("--par_col", type=str, default="compartment/par")
    p.add_argument("--horizons", type=str, default="10,30,60,120")
    p.add_argument("--algorithms", type=str, default="logistic,rf,xgb")

    p.add_argument("--event_qs", type=str, default="0.85,0.90,0.95",
                   help="Comma-separated ΔT high-risk quantiles for threshold sensitivity.")
    p.add_argument("--sensor_scenarios", type=str,
                   default="clean,noise_5,noise_10,missing_5,missing_10",
                   help="Comma-separated sensor robustness scenarios.")
    p.add_argument("--random_state", type=int, default=42)
    return p


def guess_default_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    csv_candidates = [
        os.path.join(project_root, "step0_clean_build", "step0_results", "data_trigger_mainmodel_raw.csv"),
        os.path.join(script_dir, "data_trigger_mainmodel_raw.csv"),
        os.path.join(project_root, "step0_clean_build", "step0_results", "data_trigger_mainmodel_z.csv"),
        os.path.join(script_dir, "data_trigger_mainmodel_z.csv"),
    ]
    csv_path = ""
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    edge_candidates = [
        os.path.join(project_root, "step3_autolag_pcmciplus", "step3_results", "leafT_significant_edges.csv"),
        os.path.join(project_root, "step4_strict_hierarchical_pcmciplus", "step4_results", "leafT_significant_edges.csv"),
        os.path.join(script_dir, "leafT_significant_edges.csv"),
    ]
    edges_csv = ""
    for p in edge_candidates:
        if os.path.exists(p):
            edges_csv = p
            break

    out_dir = os.path.join(script_dir, "step6_deltaT_proxy_results_reviewer_response")
    return csv_path, edges_csv, out_dir


def read_causal_parents(edges_csv: str, leaf_col: str) -> list[str]:
    if not edges_csv or not os.path.exists(edges_csv):
        return []

    df = pd.read_csv(edges_csv)
    cols = list(df.columns)

    parent_col = None
    child_col = None
    for c in cols:
        lc = c.lower()
        if parent_col is None and ("parent" in lc or "source" in lc or "from" in lc):
            parent_col = c
        if child_col is None and ("child" in lc or "target" in lc or "to" in lc):
            child_col = c

    if parent_col is None:
        parent_col = cols[0]
    if child_col is None:
        child_col = cols[1] if len(cols) > 1 else cols[0]

    sub = df[df[child_col].astype(str) == str(leaf_col)].copy()
    parents = sorted(sub[parent_col].astype(str).dropna().unique().tolist())
    return parents


def main():
    args = build_parser().parse_args()

    csv_default, edges_default, out_default = guess_default_paths()
    csv_path = args.csv_path.strip() or csv_default
    edges_csv = args.edges_csv.strip() or edges_default
    out_dir = args.out_dir.strip() or out_default

    print(f"[INFO] edges_csv = {edges_csv if edges_csv else 'NOT FOUND'}")

    if edges_csv and os.path.exists(edges_csv):
        if "step3_autolag_pcmciplus" in edges_csv.replace("\\", "/"):
            print("[INFO] Using Step3 causal edges file.")
        elif "step4_strict_hierarchical_pcmciplus" in edges_csv.replace("\\", "/"):
            print("[INFO] Using Step4 causal edges file.")
        else:
            print("[INFO] Using custom/local causal edges file.")
    else:
        print("[WARN] No edges_csv found. Will fall back to minimal causal parent set.")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("Cannot find input csv. Please pass --csv_path explicitly.")

    ensure_dir(out_dir)

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    algorithms = tuple(x.strip().lower() for x in args.algorithms.split(",") if x.strip())
    event_qs = _parse_float_tuple(args.event_qs)
    sensor_scenarios = _parse_str_tuple(args.sensor_scenarios)

    cfg = CommonConfig(
        csv_path=csv_path,
        out_dir=out_dir,
        time_col=args.time_col,
        leaf_col=args.leaf_col,
        air_col=args.air_col,
        vpd_col=args.vpd_col if args.vpd_col else None,
        par_col=args.par_col if args.par_col else None,
        horizons_min=horizons,
        algorithms=algorithms,
        event_q_list=event_qs,
        sensor_scenarios=sensor_scenarios,
        random_state=args.random_state,
    )

    df = load_and_prepare_base(cfg)

    def keep_existing(cols: List[Optional[str]]) -> List[str]:
        """Keep existing columns in order and remove duplicates."""
        out: List[str] = []
        for c in cols:
            if c and c in df.columns and c not in out:
                out.append(c)
        return out

    dt_proxy_base = keep_existing([
        "deltaT_raw",
    ])

    causal_parents = keep_existing([
        cfg.leaf_col,
        cfg.air_col,
        cfg.par_col,
        "deltaT_raw",
    ])

    causal_action_cols = keep_existing([
        "energy/energy_use.heating",
        "energy/electricity_use.lighting",
        "energy/co2_dosage",
        "compartment/screen_blackout/screen_position",
        "compartment/window_position_lee_side",
        "compartment/window_position_wind_side",
        "compartment/water_supply/water_flow_duration",
        "compartment/heating_lower_circuit/pipe_temperature",
    ])

    causal_plus_action = list(causal_parents)
    for c in causal_action_cols:
        if c not in causal_plus_action:
            causal_plus_action.append(c)

    model_feature_sets = {
        "DT_PROXY_BASE": dt_proxy_base,
        "CAUSAL_PARENTS": causal_parents,
        "CAUSAL_PLUS_ACTION_LAGGED": causal_plus_action,
    }

    feature_rows = []
    for key, cols in model_feature_sets.items():
        for i, col in enumerate(cols, start=1):
            feature_rows.append({
                "feature_setting_key": key,
                "feature_setting": display_model_name(key),
                "feature_order": i,
                "input_variable": col,
            })
    pd.DataFrame(feature_rows).to_csv(os.path.join(out_dir, "step7_feature_sets_used.csv"), index=False)

    print("[INFO] Feature settings used in Step7:")
    for key, cols in model_feature_sets.items():
        print(f"  - {display_model_name(key)} ({len(cols)} variables): {cols}")

    run_models_unified(df, cfg, model_feature_sets)
    print("[OK] Reviewer-response Step7 finished (RAW):", out_dir)


if __name__ == "__main__":
    main()
