"""
VAR/Granger + Fixed PCMCI+ + Bootstrap
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR

from tigramite.data_processing import DataFrame as TigDataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


# 配置
INPUT_FILENAME = "data_trigger_mainmodel_z.csv"
TIME_COL = "time"
CSV_ENCODING = "utf-8-sig"

# 10min采样间隔 1天=144点 7天=1008点
SAMPLE_MINUTES = 10
LAG_7D_POINTS = int((7 * 24 * 60) // SAMPLE_MINUTES)  # 1008

USE_DELTA_AS_GROWTH = True
GROWTH_TARGET = "delta_mass_7d" if USE_DELTA_AS_GROWTH else "compartment/mass.plant"
BASE_MASS_COL_FOR_DELTA = "compartment/mass.plant"  # 用于脚本内“临时计算 delta_mass_7d（z空间）”

DROP_MASS_WHEN_USING_DELTA = True

ALLOW_AUTOCOMPUTE_DELTA_7D_IF_MISSING = True


# Granger/VAR
GRANGER_ALPHA = 0.1
VAR_LAG_MAX = 12
VAR_SELECT_IC = "aic"
VAR_FIXED_LAG: Optional[int] = None

# PCMCI+(Fixed)
TAU_FAST = 12
TAU_GROWTH = 288
TAU_MIN = 0
PC_ALPHA = 0.1
ALPHA_LEVEL = 0.1

USE_FDR = False
FDR_METHOD = "bh"
PCMCI_VERBOSITY = 1

# 条件集限制
MAX_CONDS_DIM: Optional[int] = None
MAX_CONDS_PX: Optional[int] = None
MAX_CONDS_PY: Optional[int] = None

USE_PARENT_RESTRICTION = True
TOPK_FAST = 30
TOPK_GROWTH = 30

ALWAYS_KEEP_AUTOREG = True
AUTOREG_LAGS_FAST = 3
AUTOREG_LAGS_GROWTH = 1

ALLOW_CONTEMP_IN_LINK_ASSUMPTIONS = False

# Bootstrap stability(PCMCI+)
BOOTSTRAP_B = 30
BOOTSTRAP_BLOCK_LEN = 144
BOOTSTRAP_SEED = 20260211
BOOT_LOG_EVERY = 1

RUN_BOOTSTRAP_FULL_GRAPH = True

OUT_GRANGER = "baseline_granger_edges.csv"
OUT_PCMCI = "baseline_pcmciplus_edges.csv"
OUT_PCMCI_GROWTH = "baseline_pcmciplus_edges_growth.csv"
OUT_BOOT = "baseline_bootstrap_freq.csv"
OUT_BOOT_GROWTH = "baseline_bootstrap_freq_growth.csv"


# 基础变量
MAINMODEL_VARS_18: List[str] = [
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
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _safe_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _prepare_no_nan_matrix(df_num: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df_num[cols].copy()
    X = X.interpolate(limit_direction="both")
    X = X.ffill().bfill()
    X = X.dropna(axis=0, how="any")
    data = X.to_numpy(dtype=float)
    if np.isnan(data).any():
        raise ValueError("Still contains NaNs after fill. Check columns with all-NaN.")
    return X, data


def _moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    block_len = max(1, min(int(block_len), n))
    k = int(math.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=k)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)[:n]
    return idx.astype(int)


def _build_edges_from_pcmci(
    p_mat: np.ndarray,
    val_mat: np.ndarray,
    var_names: List[str],
    tau_min: int,
    tau_max: int,
    alpha_level: float,
) -> pd.DataFrame:
    n = len(var_names)
    rows = []
    for j in range(n):  # target
        for i in range(n):  # source
            for tau in range(tau_min, tau_max + 1):
                p = p_mat[j, i, tau]
                v = val_mat[j, i, tau]
                if np.isfinite(p) and p <= alpha_level:
                    rows.append(
                        {
                            "source": var_names[i],
                            "target": var_names[j],
                            "lag": int(tau),
                            "pvalue": float(p),
                            "val": float(v) if np.isfinite(v) else np.nan,
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "source", "lag", "pvalue"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out


def _ensure_delta_mass_7d_column(df: pd.DataFrame) -> pd.DataFrame:

    if not USE_DELTA_AS_GROWTH:
        return df

    if GROWTH_TARGET in df.columns:
        return df

    if not ALLOW_AUTOCOMPUTE_DELTA_7D_IF_MISSING:
        raise KeyError(f"{GROWTH_TARGET} not found in CSV and auto-compute disabled.")

    if BASE_MASS_COL_FOR_DELTA not in df.columns:
        raise KeyError(
            f"{GROWTH_TARGET} not found and base mass column '{BASE_MASS_COL_FOR_DELTA}' missing; "
            f"cannot auto-compute 7d delta."
        )

    _log(f"[WARN] '{GROWTH_TARGET}' not found. Auto-computing from '{BASE_MASS_COL_FOR_DELTA}' using lag={LAG_7D_POINTS} points (7 days).")

    # 时间排序
    if TIME_COL in df.columns:
        try:
            df = df.sort_values(TIME_COL).reset_index(drop=True)
        except Exception:
            pass

    m = pd.to_numeric(df[BASE_MASS_COL_FOR_DELTA], errors="coerce")
    d = m - m.shift(LAG_7D_POINTS)

    # z-score
    mu = float(d.mean(skipna=True))
    sd = float(d.std(skipna=True))
    df[GROWTH_TARGET] = (d - mu) / (sd + 1e-12)

    _log(f"[INFO] Auto-computed '{GROWTH_TARGET}': NaN_ratio={df[GROWTH_TARGET].isna().mean():.3f}, mean≈{df[GROWTH_TARGET].mean(skipna=True):.3f}, std≈{df[GROWTH_TARGET].std(skipna=True):.3f}")
    return df


def _corr_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag <= 0:
        raise ValueError("lag must be >= 1")
    if len(x) <= lag or len(y) <= lag:
        return np.nan
    a = x[:-lag]
    b = y[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _build_topk_link_assumptions_growth_specific(
    data: np.ndarray,
    var_names: List[str],
    tau_fast: int,
    tau_growth: int,
    topk_fast: int,
    topk_growth: int,
    autoreg_lags_fast: int,
    autoreg_lags_growth: int,
    allow_contemp: bool,
    growth_target: str,
) -> Dict[int, Dict[Tuple[int, int], str]]:

    T, N = data.shape
    la: Dict[int, Dict[Tuple[int, int], str]] = {j: {} for j in range(N)}
    if growth_target not in var_names:
        raise KeyError(f"Growth target '{growth_target}' not found in variables.")

    j_growth = var_names.index(growth_target)

    for j in range(N):
        y = data[:, j]
        is_growth = (j == j_growth)

        tau_j = int(tau_growth if is_growth else tau_fast)
        topk_j = int(topk_growth if is_growth else topk_fast)
        ar_j = int(autoreg_lags_growth if is_growth else autoreg_lags_fast)

        cands: List[Tuple[int, int, float]] = []

        for i in range(N):
            if i == j:
                continue
            x = data[:, i]
            for tau in range(1, tau_j + 1):
                r = _corr_lag(x, y, lag=tau)
                if np.isfinite(r):
                    cands.append((i, tau, abs(r)))

        cands.sort(key=lambda t: t[2], reverse=True)
        chosen = cands[:max(0, topk_j)]

        for (i, tau, _s) in chosen:
            la[j][(i, -int(tau))] = "-?>"

        if ALWAYS_KEEP_AUTOREG:
            for tau in range(1, min(tau_j, ar_j) + 1):
                la[j][(j, -int(tau))] = "-?>"

        if allow_contemp:
            pass

    return la


def _print_link_assumptions_summary(
    link_assumptions: Dict[int, Dict[Tuple[int, int], str]],
    var_names: List[str],
    growth_target: str,
) -> None:
    sizes = [len(v) for v in link_assumptions.values()]
    _log(f"[INFO] Parent restriction enabled (growth-specific).")
    _log(f"[INFO] Candidate links per target: min={int(np.min(sizes))}, mean={float(np.mean(sizes)):.2f}, max={int(np.max(sizes))}")
    if growth_target in var_names:
        jg = var_names.index(growth_target)
        _log(f"[INFO] Growth target='{growth_target}' candidate_links={len(link_assumptions[jg])}")
        items = list(link_assumptions[jg].keys())
        pretty = ", ".join([f"({var_names[i]}, {tau})" for (i, tau) in items[:min(20, len(items))]])
        more = "" if len(items) <= 20 else f" ... +{len(items)-20}"
        _log(f"  [CAND-GROWTH] {pretty}{more}")


def _run_pcmciplus_api(
    pcmci: PCMCI,
    *,
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    max_conds_dim: Optional[int],
    max_conds_px: Optional[int],
    max_conds_py: Optional[int],
    link_assumptions: Optional[Dict[int, Dict[Tuple[int, int], str]]],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_conds_dim=max_conds_dim,
        max_conds_px=max_conds_px,
        max_conds_py=max_conds_py,
    )
    if link_assumptions is not None:
        try:
            return pcmci.run_pcmciplus(link_assumptions=link_assumptions, **kwargs)
        except TypeError as e:
            _log(f"[WARN] link_assumptions not supported ({repr(e)}). Falling back to NO restriction.")
            return pcmci.run_pcmciplus(**kwargs)
    return pcmci.run_pcmciplus(**kwargs)



# Granger/VAR
def _run_granger_var(df_num: pd.DataFrame, var_names: List[str], out_csv: Path) -> None:
    _log("\n[STEP2-A] VAR/Granger ...")

    X = df_num[var_names].copy()
    X = X.interpolate(limit_direction="both").ffill().bfill().dropna(axis=0, how="any")

    model = VAR(X)

    if VAR_FIXED_LAG is not None:
        lag_order = int(VAR_FIXED_LAG)
        res = model.fit(lag_order)
        _log(f"[INFO] VAR fitted with fixed lag_order={lag_order}")
    else:
        sel = model.select_order(VAR_LAG_MAX)
        lag_order = int(sel.bic) if VAR_SELECT_IC.lower() == "bic" else int(sel.aic)
        lag_order = max(1, min(lag_order, VAR_LAG_MAX))
        res = model.fit(lag_order)
        _log(f"[INFO] VAR fitted with selected lag_order={lag_order} by {VAR_SELECT_IC.upper()}")

    rows = []
    for effect in var_names:
        for cause in var_names:
            if cause == effect:
                continue
            try:
                test = res.test_causality(caused=effect, causing=[cause], kind="f")
                pval = float(test.pvalue) if np.isfinite(test.pvalue) else np.nan
                stat = float(test.test_statistic) if np.isfinite(test.test_statistic) else np.nan
                if np.isfinite(pval) and pval <= GRANGER_ALPHA:
                    rows.append(
                        dict(
                            cause=cause,
                            effect=effect,
                            var_lag_order=int(lag_order),
                            test="f",
                            stat=stat,
                            pvalue=pval,
                            alpha=GRANGER_ALPHA,
                        )
                    )
            except Exception as e:
                _log(f"[WARN] Granger failed for {cause} -> {effect}: {repr(e)}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pvalue", "cause", "effect"], ascending=[True, True, True]).reset_index(drop=True)

    out.to_csv(out_csv, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved: {out_csv} (edges={len(out)})")

def _run_pcmciplus_fixed_growth_specific(
    df_num: pd.DataFrame,
    var_names: List[str],
    out_all_csv: Path,
    out_growth_csv: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    _log("\n[STEP2-B] Fixed-parameter PCMCI+ (growth-specific restrictions) ...")

    _, data = _prepare_no_nan_matrix(df_num, var_names)
    T, N = data.shape

    tau_max_global = int(max(TAU_FAST, TAU_GROWTH))

    _log(f"[INFO] PCMCI+ effective matrix: T={T}, N={N} (no NaNs)")
    _log(f"[INFO] tau_min={TAU_MIN}, tau_max_global={tau_max_global} (fast<= {TAU_FAST}, growth<= {TAU_GROWTH})")
    _log(f"[INFO] Growth target='{GROWTH_TARGET}' | TOPK_FAST={TOPK_FAST}, TOPK_GROWTH={TOPK_GROWTH} | AR_FAST={AUTOREG_LAGS_FAST}, AR_GROWTH={AUTOREG_LAGS_GROWTH}")
    _log(f"[INFO] PCMCI+ max_conds_dim={MAX_CONDS_DIM}, max_conds_px={MAX_CONDS_PX}, max_conds_py={MAX_CONDS_PY}")
    _log(f"[INFO] delta_mass_7d lag_points={LAG_7D_POINTS} (7 days @ {SAMPLE_MINUTES}min)")

    link_assumptions = None
    if USE_PARENT_RESTRICTION:
        link_assumptions = _build_topk_link_assumptions_growth_specific(
            data=data,
            var_names=var_names,
            tau_fast=TAU_FAST,
            tau_growth=TAU_GROWTH,
            topk_fast=TOPK_FAST,
            topk_growth=TOPK_GROWTH,
            autoreg_lags_fast=AUTOREG_LAGS_FAST,
            autoreg_lags_growth=AUTOREG_LAGS_GROWTH,
            allow_contemp=ALLOW_CONTEMP_IN_LINK_ASSUMPTIONS,
            growth_target=GROWTH_TARGET,
        )
        _print_link_assumptions_summary(link_assumptions, var_names, GROWTH_TARGET)

    tg_df = TigDataFrame(data=data, var_names=var_names)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=tg_df, cond_ind_test=parcorr, verbosity=PCMCI_VERBOSITY)

    res = _run_pcmciplus_api(
        pcmci,
        tau_min=TAU_MIN,
        tau_max=tau_max_global,
        pc_alpha=PC_ALPHA,
        max_conds_dim=MAX_CONDS_DIM,
        max_conds_px=MAX_CONDS_PX,
        max_conds_py=MAX_CONDS_PY,
        link_assumptions=link_assumptions,
    )

    p_mat = res["p_matrix"]
    val_mat = res["val_matrix"]

    if USE_FDR:
        try:
            p_used = pcmci.get_corrected_pvalues(p_mat, fdr_method=FDR_METHOD)
            _log(f"[INFO] Applied FDR ({FDR_METHOD}): using q-values.")
        except Exception as e:
            _log(f"[WARN] FDR correction failed ({repr(e)}). Fallback to raw p-values.")
            p_used = p_mat
    else:
        p_used = p_mat
        _log("[INFO] No FDR correction: using raw p-values.")

    # 全边表
    edges_all = _build_edges_from_pcmci(
        p_mat=p_used,
        val_mat=val_mat,
        var_names=var_names,
        tau_min=TAU_MIN,
        tau_max=tau_max_global,
        alpha_level=ALPHA_LEVEL,
    )
    edges_all.to_csv(out_all_csv, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved: {out_all_csv} (edges={len(edges_all)})")

    # Growth-only
    edges_growth = edges_all[edges_all["target"] == GROWTH_TARGET].copy()
    edges_growth.to_csv(out_growth_csv, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved: {out_growth_csv} (growth_edges={len(edges_growth)})")

    return p_used, val_mat

# Bootstrap frequency
def _bootstrap_pcmci_freq(
    data_full: np.ndarray,
    var_names: List[str],
    tau_max_global: int,
    link_assumptions: Optional[Dict[int, Dict[Tuple[int, int], str]]],
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    T, N = data_full.shape

    counts: Dict[Tuple[int, int, int], int] = {}
    val_sums: Dict[Tuple[int, int, int], float] = {}
    parcorr = ParCorr(significance="analytic")

    for b in range(1, BOOTSTRAP_B + 1):
        idx = _moving_block_bootstrap_indices(T, BOOTSTRAP_BLOCK_LEN, rng)
        data_bs = data_full[idx, :]

        tg_df = TigDataFrame(data=data_bs, var_names=var_names)
        pcmci = PCMCI(dataframe=tg_df, cond_ind_test=parcorr, verbosity=0)

        res = _run_pcmciplus_api(
            pcmci,
            tau_min=TAU_MIN,
            tau_max=tau_max_global,
            pc_alpha=PC_ALPHA,
            max_conds_dim=MAX_CONDS_DIM,
            max_conds_px=MAX_CONDS_PX,
            max_conds_py=MAX_CONDS_PY,
            link_assumptions=link_assumptions,
        )

        p_mat = res["p_matrix"]
        val_mat = res["val_matrix"]

        if USE_FDR:
            try:
                p_used = pcmci.get_corrected_pvalues(p_mat, fdr_method=FDR_METHOD)
            except Exception:
                p_used = p_mat
        else:
            p_used = p_mat

        sig_edges_this = 0
        for j in range(N):
            for i in range(N):
                for tau in range(TAU_MIN, tau_max_global + 1):
                    p = p_used[j, i, tau]
                    v = val_mat[j, i, tau]
                    if np.isfinite(p) and p <= ALPHA_LEVEL:
                        sig_edges_this += 1
                        key = (i, j, int(tau))
                        counts[key] = counts.get(key, 0) + 1
                        if np.isfinite(v):
                            val_sums[key] = val_sums.get(key, 0.0) + float(v)

        if (b == 1) or (b % BOOT_LOG_EVERY == 0) or (b == BOOTSTRAP_B):
            _log(f"[BOOT] {b:>4d}/{BOOTSTRAP_B} done | sig_edges_this={sig_edges_this}")

    rows = []
    for (i, j, tau), c in counts.items():
        freq = c / float(BOOTSTRAP_B)
        mean_val = val_sums.get((i, j, tau), 0.0) / float(c) if c > 0 else np.nan
        rows.append(
            dict(
                source=var_names[i],
                target=var_names[j],
                lag=int(tau),
                count=int(c),
                B=int(BOOTSTRAP_B),
                freq=float(freq),
                mean_val=float(mean_val),
                tau_max=int(tau_max_global),
                pc_alpha=float(PC_ALPHA),
                alpha_level=float(ALPHA_LEVEL),
                fdr=bool(USE_FDR),
                block_len=int(BOOTSTRAP_BLOCK_LEN),
                seed=int(BOOTSTRAP_SEED),
                max_conds_dim=(None if MAX_CONDS_DIM is None else int(MAX_CONDS_DIM)),
                max_conds_px=(None if MAX_CONDS_PX is None else int(MAX_CONDS_PX)),
                max_conds_py=(None if MAX_CONDS_PY is None else int(MAX_CONDS_PY)),
                parent_restrict=bool(USE_PARENT_RESTRICTION),
                topk_fast=int(TOPK_FAST),
                topk_growth=int(TOPK_GROWTH),
                ar_fast=int(AUTOREG_LAGS_FAST),
                ar_growth=int(AUTOREG_LAGS_GROWTH),
                growth_target=str(GROWTH_TARGET),
                delta_7d_lag_points=int(LAG_7D_POINTS),
            )
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["freq", "count"], ascending=[False, False]).reset_index(drop=True)
    return out


def _run_bootstrap_outputs(
    df_num: pd.DataFrame,
    var_names: List[str],
    out_full_csv: Path,
    out_growth_csv: Path,
) -> None:
    _log("\n[STEP2-C] Bootstrap stability (PCMCI+) ...")
    _, data_full = _prepare_no_nan_matrix(df_num, var_names)
    T, N = data_full.shape
    tau_max_global = int(max(TAU_FAST, TAU_GROWTH))

    link_assumptions = None
    if USE_PARENT_RESTRICTION:
        link_assumptions = _build_topk_link_assumptions_growth_specific(
            data=data_full,
            var_names=var_names,
            tau_fast=TAU_FAST,
            tau_growth=TAU_GROWTH,
            topk_fast=TOPK_FAST,
            topk_growth=TOPK_GROWTH,
            autoreg_lags_fast=AUTOREG_LAGS_FAST,
            autoreg_lags_growth=AUTOREG_LAGS_GROWTH,
            allow_contemp=ALLOW_CONTEMP_IN_LINK_ASSUMPTIONS,
            growth_target=GROWTH_TARGET,
        )
        _log("[INFO] Bootstrap will reuse the SAME link_assumptions for all resamples.")

    _log(f"[INFO] Bootstrap base matrix: T={T}, N={N} | tau_max_global={tau_max_global}")
    _log(f"[INFO] Bootstrap: B={BOOTSTRAP_B}, block_len={BOOTSTRAP_BLOCK_LEN}")

    out = _bootstrap_pcmci_freq(data_full, var_names, tau_max_global, link_assumptions)

    if RUN_BOOTSTRAP_FULL_GRAPH:
        out.to_csv(out_full_csv, index=False, encoding=CSV_ENCODING)
        _log(f"[DONE] Saved: {out_full_csv} (unique_edges={len(out)})")

    out_g = out[out["target"] == GROWTH_TARGET].copy()
    out_g.to_csv(out_growth_csv, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved: {out_growth_csv} (growth_unique_edges={len(out_g)})")


def _select_vars_for_growth_experiment(df: pd.DataFrame) -> List[str]:
    cols = list(MAINMODEL_VARS_18)

    if USE_DELTA_AS_GROWTH:
        if GROWTH_TARGET not in df.columns:
            raise KeyError(f"{GROWTH_TARGET} not found in CSV. Please generate it in Step0 or enable auto-compute.")
        # 用 delta_mass_7d 替代 mass.plant 作为 Growth target
        if DROP_MASS_WHEN_USING_DELTA:
            cols = [c for c in cols if c != "compartment/mass.plant"]
        if GROWTH_TARGET not in cols:
            cols = [GROWTH_TARGET] + cols
    else:
        if "compartment/mass.plant" not in df.columns:
            raise KeyError("compartment/mass.plant not found in CSV.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns ({len(missing)}): {missing}")

    _log(f"[INFO] Using variables: N={len(cols)} | Growth target='{GROWTH_TARGET}'")
    for i, c in enumerate(cols, 1):
        _log(f"  {i:02d}. {c}")

    return cols


def main() -> int:
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]  # ROOT
    in_path = root / "step0_clean_build" / "step0_results" / INPUT_FILENAME

    out_dir = root / "step2_baseline_pcmciplus" / "step2_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_granger = out_dir / OUT_GRANGER
    out_pcmci = out_dir / OUT_PCMCI
    out_pcmci_growth = out_dir / OUT_PCMCI_GROWTH
    out_boot = out_dir / OUT_BOOT
    out_boot_growth = out_dir / OUT_BOOT_GROWTH

    _log("Step2 | Baseline (Growth-specialized, 7d Δmass)")
    _log(f"[ROOT ] {root}")
    _log(f"[IN   ] {in_path}")
    _log(f"[OUT  ] {out_dir}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8", low_memory=False)

    df = _ensure_delta_mass_7d_column(df)

    var_names = _select_vars_for_growth_experiment(df)
    df_num = _safe_numeric_df(df, var_names)

    _run_granger_var(df_num, var_names, out_granger)

    _run_pcmciplus_fixed_growth_specific(df_num, var_names, out_pcmci, out_pcmci_growth)

    _run_bootstrap_outputs(df_num, var_names, out_boot, out_boot_growth)

    _log("\n[ALL DONE] Step2 growth-specialized (7d) outputs generated.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n[ERROR]", repr(e), flush=True)
        raise
