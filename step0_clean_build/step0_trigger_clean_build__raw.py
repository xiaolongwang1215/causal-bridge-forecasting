"""
Raw预处理数据集
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# 配置
RESAMPLE_RULE = "10min"
AGG_METHOD = "mean"
DROP_ALL_NAN_ROWS = True
CSV_ENCODING = "utf-8-sig"

TRIGGER_FILENAME = "trigger.csv"
WEATHER_FILENAME = "weather.csv"

COL_TIME_CANDIDATES = ["time", "timestamp", "datetime", "date_time"]
COL_MASS = "compartment/mass.plant"

SUBSTRATE_GROUPS = [
    ("substrate_relative_permittivity_mean", "compartment/substrate/relative_permittivity"),
    ("substrate_bulk_ec_mean", "compartment/substrate/bulk_ec"),
    ("substrate_temperature_mean", "compartment/substrate/substrate_temperature"),
]

# 主要变量
MAINMODEL_COLS_18 = [
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
MAINMODEL_EXTRA = ["delta_mass_24h"]


# 工具
def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_time_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in COL_TIME_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    if "time" in df.columns:
        return "time"
    raise ValueError(
        f"Cannot find a time column. Tried {COL_TIME_CANDIDATES}. "
        f"Available columns head: {list(df.columns)[:20]}"
    )


def _to_datetime_series_force_utc(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)
    return dt


def _force_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Failed to convert index to DatetimeIndex.")
    return df


def _resample_mixed(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = _force_datetime_index(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols]

    out_parts = []
    if numeric_cols:
        out_parts.append(df[numeric_cols].resample(rule).mean())
    if other_cols:
        out_parts.append(df[other_cols].resample(rule).first())

    if not out_parts:
        return df.resample(rule).first()

    out = pd.concat(out_parts, axis=1)
    out = out[[c for c in df.columns if c in out.columns]]
    return out


def _print_missing_with_hints(df_cols: List[str], missing: List[str], max_hint: int = 6) -> None:
    _log("\n[WARN] Some main-model columns are missing in output!")
    _log("[WARN] Missing columns:")
    for c in missing:
        _log(f"  - {c}")

    _log("\n[HINT] Possible close matches (contains search):")
    for m in missing:
        key = m.split("/")[-1].split(".")[0]
        cand = [c for c in df_cols if key in c]
        if cand:
            _log(f"  * for '{m}' -> {cand[:max_hint]}")
        else:
            parts = [p for p in m.replace("__weather", "").split("/") if p]
            parts = parts[-2:] if len(parts) >= 2 else parts
            cand2 = [c for c in df_cols if all(p in c for p in parts)]
            if cand2:
                _log(f"  * for '{m}' -> {cand2[:max_hint]}")


# 主函数
def main() -> int:
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]  # ROOT
    input_dir = root / "filter_dataset"
    out_dir = root / "step0_clean_build" / "step0_results"
    _ensure_dir(out_dir)

    trigger_path = input_dir / TRIGGER_FILENAME
    weather_path = input_dir / WEATHER_FILENAME

    out_all_path = out_dir / "data_trigger_clean_raw.csv"
    out_main_path = out_dir / "data_trigger_mainmodel_raw.csv"

    _log("Step0-RAW | Align + Derive (trigger + weather)")
    _log(f"[ROOT ] {root}")
    _log(f"[IN   ] {trigger_path}")
    _log(f"[IN   ] {weather_path}")
    _log(f"[OUT  ] {out_all_path}")
    _log(f"[OUT  ] {out_main_path}")
    _log(f"[CFG  ] RESAMPLE_RULE={RESAMPLE_RULE}, AGG_METHOD={AGG_METHOD}")

    if not trigger_path.exists():
        raise FileNotFoundError(f"trigger.csv not found: {trigger_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"weather.csv not found: {weather_path}")

    trigger = pd.read_csv(trigger_path, encoding="utf-8", low_memory=False)
    weather = pd.read_csv(weather_path, encoding="utf-8", low_memory=False)

    tcol_tr = _find_time_col(trigger)
    tcol_we = _find_time_col(weather)

    trigger[tcol_tr] = _to_datetime_series_force_utc(trigger[tcol_tr])
    weather[tcol_we] = _to_datetime_series_force_utc(weather[tcol_we])

    trigger = trigger.dropna(subset=[tcol_tr]).copy()
    weather = weather.dropna(subset=[tcol_we]).copy()

    trigger = trigger.sort_values(tcol_tr).set_index(tcol_tr)
    weather = weather.sort_values(tcol_we).set_index(tcol_we)

    trigger = _force_datetime_index(trigger)
    weather = _force_datetime_index(weather)

    for df_ in (trigger, weather):
        for c in df_.columns:
            if df_[c].dtype == object:
                df_[c] = pd.to_numeric(df_[c], errors="ignore")

    trigger_rs = _resample_mixed(trigger, RESAMPLE_RULE)
    weather_rs = _resample_mixed(weather, RESAMPLE_RULE)

    _log(f"[INFO] trigger rows: raw={len(trigger):,} -> resampled={len(trigger_rs):,}")
    _log(f"[INFO] weather rows: raw={len(weather):,} -> resampled={len(weather_rs):,}")

    df = trigger_rs.join(weather_rs, how="inner", rsuffix="__weather").sort_index()
    _log(f"[INFO] merged rows (inner join): {len(df):,}")
    if len(df) == 0:
        raise RuntimeError("Merged dataframe is empty. Check time overlap between trigger and weather.")

    if AGG_METHOD not in ("mean", "median"):
        raise ValueError("AGG_METHOD must be 'mean' or 'median'")

    for new_col, base in SUBSTRATE_GROUPS:
        candidates = [base, f"{base}.1", f"{base}.2"]
        existing = [c for c in candidates if c in df.columns]
        if len(existing) == 0:
            _log(f"[WARN] No substrate probe columns found for base='{base}'. Skip {new_col}.")
            continue

        mat = df[existing].apply(pd.to_numeric, errors="coerce")
        df[new_col] = mat.mean(axis=1, skipna=True) if AGG_METHOD == "mean" else mat.median(axis=1, skipna=True)
        _log(f"[OK] Created {new_col} from {existing} using {AGG_METHOD}")

    if COL_MASS in df.columns:
        step_24h = int(pd.Timedelta("24h") / pd.Timedelta(RESAMPLE_RULE))  # 144 for 10min
        df[COL_MASS] = pd.to_numeric(df[COL_MASS], errors="coerce")
        df["delta_mass_24h"] = df[COL_MASS] - df[COL_MASS].shift(step_24h)
        _log(f"[OK] Created delta_mass_24h using shift({step_24h})")
    else:
        _log(f"[WARN] Mass column '{COL_MASS}' not found. delta_mass_24h NOT created.")

    if DROP_ALL_NAN_ROWS:
        numeric_cols_all = df.select_dtypes(include=[np.number]).columns
        before = len(df)
        df = df.dropna(axis=0, how="all", subset=numeric_cols_all)
        after = len(df)
        if after != before:
            _log(f"[INFO] Dropped rows with all numeric NaN: {before - after:,}")

    df_out = df.reset_index().rename(columns={df.index.name: "time"})
    df_out.to_csv(out_all_path, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved (all RAW): {out_all_path}")
    _log(f"[INFO] Output(all RAW) shape: {df_out.shape[0]:,} rows × {df_out.shape[1]:,} cols")

    desired = ["time"] + MAINMODEL_COLS_18 + MAINMODEL_EXTRA
    missing = [c for c in desired if c not in df_out.columns]
    if missing:
        _print_missing_with_hints(df_out.columns.tolist(), missing)

    keep_cols = [c for c in desired if c in df_out.columns]
    df_main = df_out[keep_cols].copy()
    df_main.to_csv(out_main_path, index=False, encoding=CSV_ENCODING)
    _log(f"[DONE] Saved (main-model RAW): {out_main_path}")
    _log(f"[INFO] Output(main RAW) shape: {df_main.shape[0]:,} rows × {df_main.shape[1]:,} cols")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n[ERROR]", repr(e), flush=True)
        raise
