"""
因果桥接预测 Causal Bridge Forecasting (CBF)实验
"""

from __future__ import annotations

import os
import time
import math
import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:
    raise RuntimeError("scikit-learn is required. Please install it with: pip install scikit-learn") from e

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBRegressor = None

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    wilcoxon = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

@dataclass
class CFG:
    # Input data. This should be the raw physical-scale cleaned main model file.
    in_csv: str = os.path.join(
        PROJECT_ROOT,
        "step0_clean_build",
        "step0_results",
        "data_trigger_mainmodel_raw.csv",
    )

    out_dir: str = os.path.join(SCRIPT_DIR, "step4_cbf_results_leafT_review_forecasting_stats")

    minutes_per_step: int = 10

    horizons_steps: Tuple[int, ...] = (1, 3, 6, 12)

    leafT_lags: Tuple[int, ...] = (1, 2, 3)
    state_lags: Tuple[int, ...] = (1,)
    action_lags: Tuple[int, ...] = (1,)

    causal_lag1: Tuple[int, ...] = (1,)

    causal_window_lags: Tuple[int, ...] = (1, 2, 3)

    # Rolling-origin评估
    rolling_folds: int = 5
    test_window_days: int = 7
    val_window_days: int = 7
    min_train_samples: int = 2000

    train_ratio_fallback: float = 0.70
    val_ratio_fallback: float = 0.15

    # 模型
    ridge_alpha: float = 1.0

    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = 12
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_max_features: str = "sqrt"

    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 4
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0

    enable_deep_learning: bool = True
    deep_models: Tuple[str, ...] = ("PatchTST", "iTransformer", "TimeMixer")
    deep_feature_sets: Tuple[str, ...] = (
        "AR_only",
        "AR_AirT",
        "AR_AirT_PAR",
        "AR-LeafParents",
        "AR_AirT_PAR_window",
        "TwoStage_CBF_AirT_PAR",
        "TwoStage_CBF_AirT_PAR_Residual",
    )
    deep_epochs: int = 80
    deep_batch_size: int = 256
    deep_learning_rate: float = 1e-3
    deep_hidden_size: int = 32
    deep_patience: int = 12

    # Two-stage CBF配置
    stageA_algo: str = "Ridge"

    validation_selection_algos: Tuple[str, ...] = ("Ridge", "RF", "XGBoost")

    interp_limit: int = 36
    scale_features_for_linear_and_deep: bool = True

    save_predictions: bool = True

    seed: int = 2026


cfg = CFG()


LEAF_T = "compartment/leaf_temperature"
AIR_T = "compartment/air_temperature"
PAR = "compartment/par"
VPD = "compartment/humidity_deficit"
CO2_C = "compartment/co2_concentration"

HEAT = "energy/energy_use.heating"
LIGHT = "energy/electricity_use.lighting"
CO2_D = "energy/co2_dosage"
SCREEN_E = "compartment/screen_energy/screen_position"
SCREEN_B = "compartment/screen_blackout/screen_position"
WATER = "compartment/water_supply/water_flow_duration"
L_SIDE = "compartment/window_position_lee_side"
W_SIDE = "compartment/window_position_wind_side"
PIPE_T = "compartment/heating_lower_circuit/pipe_temperature"

BRIDGE_PARENT_ACTIONS: Dict[str, List[str]] = {
    AIR_T: [L_SIDE, W_SIDE, HEAT, CO2_D, LIGHT, WATER, PIPE_T],
    PAR: [CO2_D, LIGHT, SCREEN_B],
}

SHORT_NAME = {
    LEAF_T: "Leaf_T",
    AIR_T: "Air_T",
    PAR: "PAR",
    VPD: "VPD",
    CO2_C: "CO2_C",
    HEAT: "Heat",
    LIGHT: "Light",
    CO2_D: "CO2_D",
    SCREEN_E: "Screen_E",
    SCREEN_B: "Screen_B",
    WATER: "Water",
    L_SIDE: "L_Side",
    W_SIDE: "W_Side",
    PIPE_T: "Pipe_T",
}

TWO_STAGE_FEATURE_SETS = {
    "TwoStage_CBF_AirT_PAR",
    "TwoStage_CBF_AirT_PAR_Residual",
    "TwoStage_CBF_AirT_PAR_ValidationSelected",
}

FEATURE_DISPLAY_NAME: Dict[str, str] = {
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

MAIN_MANUSCRIPT_FEATURE_SETS: Tuple[str, ...] = (
    "AR_only",
    "AR_AirT",
    "AR-LeafParents",
    "TwoStage_CBF_AirT_PAR",
)


def display_feature_name(feature_set: str) -> str:
    return FEATURE_DISPLAY_NAME.get(str(feature_set), str(feature_set))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("→", "to")
        .replace("|", "_")
    )


def robust_find_csv(preferred_path: str, project_root: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path

    target_name = os.path.basename(preferred_path)
    candidates = []
    for root, _, files in os.walk(project_root):
        if target_name in files:
            candidates.append(os.path.join(root, target_name))

    if len(candidates) == 1:
        print("[FIX] Auto-found input csv:", candidates[0])
        return candidates[0]
    if len(candidates) > 1:
        preferred = [p for p in candidates if "step0_clean_build" in p and "step0_results" in p]
        pick = preferred[0] if preferred else candidates[0]
        print("[FIX] Multiple input candidates found. Pick:", pick)
        return pick

    raise FileNotFoundError(
        f"Input CSV not found. Tried: {preferred_path}\nSearched under: {project_root}"
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_and_index_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("CSV is missing the 'time' column. Please use data_trigger_mainmodel_*.csv")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").set_index("time")
    return out


def impute_predictors(df: pd.DataFrame, cols: List[str], limit: int) -> pd.DataFrame:
    existing_cols = [c for c in cols if c in df.columns]
    out = df.copy()
    x = out[existing_cols].copy()
    x = x.interpolate(method="time", limit=limit, limit_direction="both")
    x = x.ffill().bfill()
    out[existing_cols] = x
    return out


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_supervised(
    df: pd.DataFrame,
    target_col: str,
    feature_spec: Dict[str, Tuple[int, ...]],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series]:

    feat = {}
    for col, lags in feature_spec.items():
        if col not in df.columns:
            continue
        for k in lags:
            feat[f"{col}_lag{k}"] = df[col].shift(k)

    X = pd.DataFrame(feat, index=df.index)
    y = df[target_col].shift(-horizon)
    tmp = pd.concat([X, y.rename("y")], axis=1).dropna()
    return tmp.drop(columns=["y"]), tmp["y"]


def build_direct_feature_specs(state_cols_all: List[str]) -> List[Tuple[str, Optional[Dict[str, Tuple[int, ...]]]]]:

    spec_ar = {LEAF_T: cfg.leafT_lags}
    spec_air = {LEAF_T: cfg.leafT_lags, AIR_T: cfg.state_lags}
    spec_par = {LEAF_T: cfg.leafT_lags, PAR: cfg.state_lags}
    spec_air_par = {LEAF_T: cfg.leafT_lags, AIR_T: cfg.state_lags, PAR: cfg.state_lags}

    spec_air_par_lag1 = {
        LEAF_T: cfg.causal_lag1,
        AIR_T: cfg.causal_lag1,
        PAR: cfg.causal_lag1,
    }


    spec_air_par_window = {
        LEAF_T: cfg.leafT_lags,
        AIR_T: cfg.causal_window_lags,
        PAR: cfg.causal_window_lags,
    }

    spec_all_state = {LEAF_T: cfg.leafT_lags}
    for s in state_cols_all:
        if s in [LEAF_T]:
            continue
        spec_all_state[s] = cfg.state_lags

    return [
        ("AR_only", spec_ar),
        ("AR_AirT", spec_air),
        ("AR_PAR", spec_par),
        ("AR_AirT_PAR", spec_air_par),
        ("AR-LeafParents", spec_air_par_lag1),
        ("AR_AirT_PAR_window", spec_air_par_window),
        ("AR_AllState", spec_all_state),
        ("TwoStage_CBF_AirT_PAR", None),
        ("TwoStage_CBF_AirT_PAR_Residual", None),
        ("TwoStage_CBF_AirT_PAR_ValidationSelected", None),
    ]


def fallback_time_split(n: int) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    n_train = int(round(n * cfg.train_ratio_fallback))
    n_val = int(round(n * cfg.val_ratio_fallback))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    idx = np.arange(n)
    return [("fold_1", idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])]


def make_rolling_splits(n: int) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    day_steps = int(round(24 * 60 / cfg.minutes_per_step))
    test_len = max(1, int(cfg.test_window_days * day_steps))
    val_len = max(1, int(cfg.val_window_days * day_steps))

    required = cfg.min_train_samples + val_len + test_len
    if n < required:
        print(
            f"[WARN] n={n} is shorter than required rolling-origin length={required}. "
            "Using one chronological fallback split."
        )
        return fallback_time_split(n)

    splits = []
    for i in range(cfg.rolling_folds):
        test_end = n - i * test_len
        test_start = test_end - test_len
        val_end = test_start
        val_start = val_end - val_len
        train_end = val_start

        if train_end < cfg.min_train_samples:
            break

        tr = np.arange(0, train_end)
        va = np.arange(val_start, val_end)
        te = np.arange(test_start, test_end)
        splits.append((f"fold_{len(splits) + 1}", tr, va, te))

    splits = list(reversed(splits))
    if not splits:
        return fallback_time_split(n)
    return splits


def maybe_scale(
    X: pd.DataFrame,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    use_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    X_tr = X.iloc[tr].values.astype(float)
    X_va = X.iloc[va].values.astype(float)
    X_te = X.iloc[te].values.astype(float)
    scaler = None
    if use_scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)
    return X_tr, X_va, X_te, scaler


# 模型
def get_sklearn_model(algo_name: str):
    if algo_name == "Ridge":
        return Ridge(alpha=cfg.ridge_alpha, fit_intercept=True)
    if algo_name == "RF":
        return RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_split=cfg.rf_min_samples_split,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    if algo_name == "XGBoost":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost is not installed. Please install it with: pip install xgboost")
        return XGBRegressor(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            reg_alpha=cfg.xgb_reg_alpha,
            reg_lambda=cfg.xgb_reg_lambda,
            objective="reg:squarederror",
            random_state=cfg.seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown sklearn algo: {algo_name}")


if HAS_TORCH:
    class PatchTSTRegressor(nn.Module):

        def __init__(
            self,
            hidden_size: int = 32,
            patch_len: int = 4,
            stride: int = 2,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            self.patch_proj = nn.Linear(self.patch_len, hidden_size)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=max(1, min(n_heads, hidden_size)),
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            z = x.squeeze(-1)  # [batch, seq_len]
            if z.shape[1] < self.patch_len:
                pad_len = self.patch_len - z.shape[1]
                z = torch.nn.functional.pad(z, (0, pad_len), mode="replicate")
            patches = z.unfold(dimension=1, size=self.patch_len, step=self.stride)
            tokens = self.patch_proj(patches)
            tokens = self.encoder(tokens)
            pooled = self.norm(tokens.mean(dim=1))
            return self.fc(pooled).squeeze(-1)


    class ITransformerRegressor(nn.Module):

        def __init__(
            self,
            hidden_size: int = 32,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.value_proj = nn.Linear(1, hidden_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=max(1, min(n_heads, hidden_size)),
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: [batch, seq_len, 1]
            tokens = self.value_proj(x)
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self.encoder(tokens)
            pooled = self.norm(tokens[:, 0, :])
            return self.fc(pooled).squeeze(-1)


    class TimeMixerRegressor(nn.Module):

        def __init__(self, hidden_size: int = 32, dropout: float = 0.10):
            super().__init__()
            self.branch1 = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=1),
                nn.GELU(),
            )
            self.branch3 = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.branch5 = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
                nn.GELU(),
            )
            self.mixer = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            z = x.transpose(1, 2)
            b1 = self.branch1(z).mean(dim=-1)
            b3 = self.branch3(z).mean(dim=-1)
            b5 = self.branch5(z).mean(dim=-1)
            mixed = torch.cat([b1, b3, b5], dim=1)
            return self.mixer(mixed).squeeze(-1)


def fit_predict_sklearn(
    algo_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Optional[StandardScaler], float]:
    use_scale = (algo_name == "Ridge" and cfg.scale_features_for_linear_and_deep)
    X_tr, X_va, X_te, scaler = maybe_scale(X, tr, va, te, use_scale=use_scale)
    y_tr = y.iloc[tr].values.astype(float)

    model = get_sklearn_model(algo_name)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    runtime = time.time() - t0

    return model.predict(X_tr), model.predict(X_va), model.predict(X_te), model, scaler, runtime


def fit_predict_deep(
    algo_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Optional[StandardScaler], float]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed; deep-learning baselines are unavailable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, X_va, X_te, scaler = maybe_scale(X, tr, va, te, use_scale=cfg.scale_features_for_linear_and_deep)
    y_tr_raw = y.iloc[tr].values.astype(np.float32)
    y_va_raw = y.iloc[va].values.astype(np.float32)

    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_va = y_scaler.transform(y_va_raw.reshape(-1, 1)).ravel().astype(np.float32)

    def to_seq(a: np.ndarray) -> np.ndarray:
        return a.astype(np.float32).reshape(a.shape[0], a.shape[1], 1)

    Xtr_t = torch.tensor(to_seq(X_tr), dtype=torch.float32)
    Xva_t = torch.tensor(to_seq(X_va), dtype=torch.float32)
    Xte_t = torch.tensor(to_seq(X_te), dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    yva_t = torch.tensor(y_va, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=min(cfg.deep_batch_size, len(Xtr_t)),
        shuffle=True,
    )

    if algo_name == "PatchTST":
        model = PatchTSTRegressor(hidden_size=cfg.deep_hidden_size).to(device)
    elif algo_name == "iTransformer":
        model = ITransformerRegressor(hidden_size=cfg.deep_hidden_size).to(device)
    elif algo_name == "TimeMixer":
        model = TimeMixerRegressor(hidden_size=cfg.deep_hidden_size).to(device)
    else:
        raise ValueError(f"Unknown deep algo: {algo_name}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.deep_learning_rate)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = np.inf
    patience = 0
    t0 = time.time()

    Xva_t = Xva_t.to(device)
    yva_t = yva_t.to(device)

    for epoch in range(cfg.deep_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva_t), yva_t).detach().cpu().item())

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.deep_patience:
                break

    runtime = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    def pred(a_t: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            z = model(a_t.to(device)).detach().cpu().numpy().reshape(-1, 1)
        return y_scaler.inverse_transform(z).ravel()

    pred_tr = pred(Xtr_t)
    pred_va = pred(Xva_t.detach().cpu())
    pred_te = pred(Xte_t)
    model._y_scaler = y_scaler
    return pred_tr, pred_va, pred_te, model, scaler, runtime


def fit_predict_model(
    algo_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Optional[StandardScaler], float]:
    if algo_name in ["Ridge", "RF", "XGBoost"]:
        return fit_predict_sklearn(algo_name, X, y, tr, va, te)
    if algo_name in ["PatchTST", "iTransformer", "TimeMixer"]:
        return fit_predict_deep(algo_name, X, y, tr, va, te)
    raise ValueError(f"Unknown algo_name: {algo_name}")


def feature_importance_table(
    model: Any,
    algo_name: str,
    X_cols: List[str],
    scaler: Optional[StandardScaler],
    horizon: int,
    feature_set: str,
    fold_id: str,
) -> pd.DataFrame:
    effective_algo = getattr(model, "_selected_algo", algo_name)

    if effective_algo == "Ridge":
        coef_std = pd.Series(model.coef_, index=X_cols, name="importance_std").to_frame()
        coef_std["abs_importance_std"] = coef_std["importance_std"].abs()
        if scaler is not None:
            scale_ = pd.Series(scaler.scale_, index=X_cols)
            mean_ = pd.Series(scaler.mean_, index=X_cols)
            beta_raw = (coef_std["importance_std"] / scale_).rename("importance_raw")
            intercept_raw = float(
                model.intercept_ - np.sum(mean_.values / scale_.values * coef_std["importance_std"].values)
            )
            coef = pd.concat([coef_std, beta_raw], axis=1)
            coef["abs_importance_raw"] = coef["importance_raw"].abs()
            coef["intercept_raw"] = intercept_raw
            coef["rank_value"] = coef["abs_importance_raw"]
        else:
            coef = coef_std.copy()
            coef["rank_value"] = coef["abs_importance_std"]
    elif effective_algo in ["RF", "XGBoost"]:
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            imp = np.full(len(X_cols), np.nan)
        coef = pd.DataFrame({
            "importance_std": imp,
            "abs_importance_std": np.abs(imp),
            "rank_value": np.abs(imp),
        }, index=X_cols)
    else:
        return pd.DataFrame()

    coef = coef.sort_values("rank_value", ascending=False)
    coef["fold"] = fold_id
    coef["horizon_steps"] = int(horizon)
    coef["horizon_minutes"] = int(horizon * cfg.minutes_per_step)
    coef["feature_set"] = feature_set
    coef["model"] = algo_name
    return coef.reset_index().rename(columns={"index": "feature"})

# Two-stage因果桥接预测
def build_stageA_spec(bridge_col: str) -> Dict[str, Tuple[int, ...]]:
    spec: Dict[str, Tuple[int, ...]] = {}
    spec[bridge_col] = cfg.state_lags
    for a in BRIDGE_PARENT_ACTIONS.get(bridge_col, []):
        spec[a] = cfg.action_lags
    return spec


def build_two_stage_base(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    leaf_spec = {LEAF_T: cfg.leafT_lags}
    X_leaf, y_leaf = build_supervised(df, LEAF_T, leaf_spec, horizon)
    common_idx = X_leaf.index.intersection(y_leaf.index)

    bridge_data = {}
    for bridge in [AIR_T, PAR]:
        spec = build_stageA_spec(bridge)
        Xs, ys = build_supervised(df, bridge, spec, horizon)
        common_idx = common_idx.intersection(Xs.index).intersection(ys.index)
        bridge_data[bridge] = (Xs, ys)

    common_idx = common_idx.sort_values()
    X_leaf = X_leaf.loc[common_idx]
    y_leaf = y_leaf.loc[common_idx]
    bridge_data = {b: (Xs.loc[common_idx], ys.loc[common_idx]) for b, (Xs, ys) in bridge_data.items()}
    return X_leaf, y_leaf, bridge_data



def available_validation_algos() -> List[str]:
    algos = []
    for a in cfg.validation_selection_algos:
        if a == "XGBoost" and not HAS_XGBOOST:
            continue
        algos.append(a)
    return algos if algos else ["Ridge"]


def fit_predict_validation_selected(
    X: pd.DataFrame,
    y: pd.Series,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    candidates: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Optional[StandardScaler], float, str]:

    candidates = candidates or available_validation_algos()
    best = None
    failures = []

    for algo in candidates:
        try:
            pred_tr, pred_va, pred_te, model, scaler, runtime = fit_predict_model(algo, X, y, tr, va, te)
            rmse_va = math.sqrt(mean_squared_error(y.iloc[va].values, pred_va))
            item = (rmse_va, algo, pred_tr, pred_va, pred_te, model, scaler, runtime)
            if best is None or item[0] < best[0]:
                best = item
        except Exception as e:
            failures.append(f"{algo}: {e}")

    if best is None:
        raise RuntimeError("Validation-selected model failed for all candidates: " + "; ".join(failures))

    rmse_va, algo, pred_tr, pred_va, pred_te, model, scaler, runtime = best
    try:
        model._selected_algo = algo
    except Exception:
        pass
    return pred_tr, pred_va, pred_te, model, scaler, runtime, algo


def two_stage_fit_predict_fold(
    df: pd.DataFrame,
    horizon: int,
    final_algo: str,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    include_observed_bridge_lags: bool = False,
    validation_selected: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray, Any, Optional[StandardScaler], float, List[Dict[str, Any]]]:

    X_leaf, y_leaf, bridge_data = build_two_stage_base(df, horizon)
    n = len(y_leaf)
    if max(te) >= n:
        raise IndexError("Fold indices exceed two-stage sample length.")

    Xb = X_leaf.copy()

    if include_observed_bridge_lags:
        for bridge in [AIR_T, PAR]:
            if bridge in df.columns:
                for k in cfg.causal_window_lags:
                    Xb[f"{bridge}_obs_lag{k}"] = df[bridge].shift(k).reindex(Xb.index)

    stageA_rows: List[Dict[str, Any]] = []
    stageA_runtime_total = 0.0

    for bridge, (Xs, ys) in bridge_data.items():
        if validation_selected:
            pred_tr, pred_va, pred_te, _, _, runtime, selected_stageA = fit_predict_validation_selected(
                Xs, ys, tr, va, te, candidates=available_validation_algos()
            )
            stageA_model_name = selected_stageA
        else:
            pred_tr, pred_va, pred_te, _, _, runtime = fit_predict_model(cfg.stageA_algo, Xs, ys, tr, va, te)
            stageA_model_name = cfg.stageA_algo

        stageA_runtime_total += runtime

        pred_all = np.full(n, np.nan, dtype=float)
        pred_all[tr] = pred_tr
        pred_all[va] = pred_va
        pred_all[te] = pred_te

        Xb[f"{bridge}_hat"] = pred_all

        for split_name, idx, pred in [
            ("train", tr, pred_tr),
            ("val", va, pred_va),
            ("test", te, pred_te),
        ]:
            m = eval_metrics(ys.iloc[idx].values, pred)
            stageA_rows.append({
                "split": split_name,
                "bridge_variable": SHORT_NAME.get(bridge, bridge),
                "bridge_variable_full": bridge,
                "horizon_steps": int(horizon),
                "horizon_minutes": int(horizon * cfg.minutes_per_step),
                "stageA_model": stageA_model_name,
                "stageA_runtime_s": float(runtime),
                **m,
            })

    tmp = pd.concat([Xb, y_leaf.rename("y")], axis=1).dropna()
    old_positions = pd.Series(np.arange(n), index=Xb.index)
    kept_old_positions = old_positions.loc[tmp.index].values
    old_to_new = {int(old): int(new) for new, old in enumerate(kept_old_positions)}

    def remap(indices: np.ndarray) -> np.ndarray:
        return np.array([old_to_new[int(i)] for i in indices if int(i) in old_to_new], dtype=int)

    tr2, va2, te2 = remap(tr), remap(va), remap(te)
    if len(tr2) < 10 or len(va2) < 2 or len(te2) < 2:
        raise ValueError("Too few samples after two-stage alignment.")

    X_final = tmp.drop(columns=["y"])
    y_final = tmp["y"]

    if validation_selected:
        pred_tr, pred_va, pred_te, final_model, scaler, runtime_b, selected_stageB = fit_predict_validation_selected(
            X_final, y_final, tr2, va2, te2, candidates=available_validation_algos()
        )
        try:
            final_model._selected_algo = selected_stageB
        except Exception:
            pass
        for r in stageA_rows:
            r["stageB_model_selected"] = selected_stageB
    else:
        pred_tr, pred_va, pred_te, final_model, scaler, runtime_b = fit_predict_model(
            final_algo, X_final, y_final, tr2, va2, te2
        )

    total_runtime = stageA_runtime_total + runtime_b
    return X_final, y_final, pred_tr, pred_va, pred_te, final_model, scaler, total_runtime, stageA_rows


def summarize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["horizon_steps", "horizon_minutes", "feature_set", "model"]
    agg = df.groupby(group_cols).agg(
        n_folds=("fold", "nunique"),
        n_test_mean=("n_test", "mean"),
        p_mean=("p", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        runtime_s_std=("runtime_s", "std"),
        test_MAE_mean=("test_MAE", "mean"),
        test_MAE_std=("test_MAE", "std"),
        test_RMSE_mean=("test_RMSE", "mean"),
        test_RMSE_std=("test_RMSE", "std"),
        test_R2_mean=("test_R2", "mean"),
        test_R2_std=("test_R2", "std"),
    ).reset_index()

    for col in ["runtime_s_std", "test_MAE_std", "test_RMSE_std", "test_R2_std"]:
        agg[col] = agg[col].fillna(0.0)

    agg = agg.sort_values(["horizon_steps", "test_RMSE_mean", "test_MAE_mean", "feature_set", "model"]).reset_index(drop=True)
    return agg


def fmt_mean_std(mean: float, std: float, ndigits: int = 4) -> str:
    return f"{float(mean):.{ndigits}f} ± {float(std):.{ndigits}f}"


def make_word_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["Feature configuration"] = out["feature_set"].map(display_feature_name)
    out["Internal feature_set"] = out["feature_set"]
    out["MAE"] = [fmt_mean_std(m, s, ndigits=3) for m, s in zip(out["test_MAE_mean"], out["test_MAE_std"])]
    out["RMSE"] = [fmt_mean_std(m, s, ndigits=3) for m, s in zip(out["test_RMSE_mean"], out["test_RMSE_std"])]
    out["R2"] = [fmt_mean_std(m, s, ndigits=3) for m, s in zip(out["test_R2_mean"], out["test_R2_std"])]
    out["Runtime (s)"] = [fmt_mean_std(m, s, ndigits=2) for m, s in zip(out["runtime_s_mean"], out["runtime_s_std"])]
    return out[[
        "horizon_minutes",
        "model",
        "Feature configuration",
        "Internal feature_set",
        "n_folds",
        "MAE",
        "RMSE",
        "R2",
        "Runtime (s)",
    ]].rename(
        columns={
            "horizon_minutes": "Horizon (min)",
            "model": "Model",
            "n_folds": "Folds",
        }
    )


def significance_tests(df_by_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("AR-LeafParents", "AR_only"),
        ("AR-LeafParents", "AR_AirT"),
        ("AR-LeafParents", "AR_PAR"),
        ("AR_AirT_PAR_window", "AR_only"),
        ("AR_AirT_PAR_window", "AR_AirT_PAR"),
        ("AR_AirT_PAR_window", "AR_AllState"),
        ("TwoStage_CBF_AirT_PAR", "AR_only"),
        ("TwoStage_CBF_AirT_PAR", "AR_AirT"),
        ("TwoStage_CBF_AirT_PAR", "AR_PAR"),
        ("TwoStage_CBF_AirT_PAR", "AR_AirT_PAR"),
        ("TwoStage_CBF_AirT_PAR", "AR_AllState"),
        ("TwoStage_CBF_AirT_PAR_Residual", "TwoStage_CBF_AirT_PAR"),
        ("TwoStage_CBF_AirT_PAR_Residual", "AR_AirT_PAR_window"),
        ("TwoStage_CBF_AirT_PAR_ValidationSelected", "TwoStage_CBF_AirT_PAR"),
        ("TwoStage_CBF_AirT_PAR_ValidationSelected", "AR_AirT_PAR_window"),
    ]

    metrics = ["test_MAE", "test_RMSE"]
    for horizon in sorted(df_by_fold["horizon_steps"].unique()):
        for model in sorted(df_by_fold["model"].unique()):
            sub_model = df_by_fold[(df_by_fold["horizon_steps"] == horizon) & (df_by_fold["model"] == model)]
            for a, b in comparisons:
                da = sub_model[sub_model["feature_set"] == a]
                db = sub_model[sub_model["feature_set"] == b]
                if da.empty or db.empty:
                    continue

                merged = da.merge(db, on="fold", suffixes=("_a", "_b"))
                if len(merged) < 2:
                    continue

                for metric in metrics:
                    x = merged[f"{metric}_a"].values.astype(float)
                    y = merged[f"{metric}_b"].values.astype(float)
                    diff = x - y
                    if np.allclose(diff, 0):
                        p_value = 1.0
                        stat = 0.0
                    elif HAS_SCIPY:
                        try:
                            stat, p_value = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                            stat = float(stat)
                            p_value = float(p_value)
                        except Exception:
                            stat = np.nan
                            p_value = np.nan
                    else:
                        stat = np.nan
                        p_value = np.nan

                    rows.append({
                        "horizon_steps": int(horizon),
                        "horizon_minutes": int(horizon * cfg.minutes_per_step),
                        "model": model,
                        "comparison": f"{a} vs {b}",
                        "comparison_display": f"{display_feature_name(a)} vs {display_feature_name(b)}",
                        "metric": metric.replace("test_", ""),
                        "n_pairs": int(len(merged)),
                        "mean_a": float(np.mean(x)),
                        "mean_b": float(np.mean(y)),
                        "delta_a_minus_b": float(np.mean(x - y)),
                        "test_statistic": stat,
                        "p_value": p_value,
                        "significant_0.05": bool(False if pd.isna(p_value) else p_value < 0.05),
                    })
    return pd.DataFrame(rows)


def feature_manifest() -> pd.DataFrame:
    rows = [
        {
            "feature_set": "AR_only",
            "feature_description": "Leaf_T autoregressive lags only",
            "causal_role": "Autoregressive baseline",
        },
        {
            "feature_set": "AR_AirT",
            "feature_description": "Leaf_T lags plus Air_T lag",
            "causal_role": "Thermal bridge direct-parent baseline",
        },
        {
            "feature_set": "AR_PAR",
            "feature_description": "Leaf_T lags plus PAR lag",
            "causal_role": "Radiation bridge direct-parent baseline",
        },
        {
            "feature_set": "AR_AirT_PAR",
            "feature_description": "Leaf_T lags plus Air_T and PAR lags",
            "causal_role": "Direct causal-parent feature configuration based on revised Fig. 4/Table x4",
        },
        {
            "feature_set": "AR-LeafParents",
            "feature_description": "Leaf_T(t-1), Air_T(t-1), and PAR(t-1)",
            "causal_role": "Strict significant-lag causal-parent feature configuration",
        },
        {
            "feature_set": "AR_AirT_PAR_window",
            "feature_description": "Leaf_T lags plus short lag-window features of Air_T and PAR",
            "causal_role": "Causal-parent lag-window feature configuration",
        },
        {
            "feature_set": "AR_AllState",
            "feature_description": "Leaf_T lags plus Air_T, PAR, VPD, and CO2_C lags if available",
            "causal_role": "Expanded state-feature baseline, not the main constrained CBF configuration",
        },
        {
            "feature_set": "TwoStage_CBF_AirT_PAR",
            "feature_description": "Stage A predicts Air_T and PAR from their action parents; Stage B predicts Leaf_T from Leaf_T lags plus Air_T_hat and PAR_hat",
            "causal_role": "Causally constrained Action→State→Physiology bridge-forecasting configuration",
        },
        {
            "feature_set": "TwoStage_CBF_AirT_PAR_Residual",
            "feature_description": "Residual two-stage CBF using Leaf_T lags, observed Air_T/PAR lags, and predicted Air_T_hat/PAR_hat",
            "causal_role": "Error-propagation-mitigated causal bridge configuration",
        },
        {
            "feature_set": "TwoStage_CBF_AirT_PAR_ValidationSelected",
            "feature_description": "Two-stage CBF with Stage-A and Stage-B learners selected by validation RMSE",
            "causal_role": "Validation-selected causal bridge configuration without test-set model selection",
        },
    ]
    manifest = pd.DataFrame(rows)
    manifest.insert(1, "manuscript_feature_configuration", manifest["feature_set"].map(display_feature_name))
    manifest["main_manuscript_configuration"] = manifest["feature_set"].isin(MAIN_MANUSCRIPT_FEATURE_SETS)
    return manifest

def main() -> None:
    t_start = time.time()
    set_random_seed(cfg.seed)

    ensure_dir(cfg.out_dir)

    print("=" * 80)
    print("Step5 | CBF Leaf_T review-response forecasting experiment")
    print("=" * 80)
    print("[CWD]", os.getcwd())
    print("[SCRIPT_DIR]", SCRIPT_DIR)
    print("[PROJECT_ROOT]", PROJECT_ROOT)
    print("[IN ]", cfg.in_csv)
    print("[OUT]", cfg.out_dir)
    print("=" * 80)

    if not HAS_XGBOOST:
        print("[WARN] xgboost is not installed. XGBoost will be skipped.")
    if cfg.enable_deep_learning and not HAS_TORCH:
        print("[WARN] torch is not installed. PatchTST/iTransformer/TimeMixer deep-learning baselines will be skipped.")

    in_path = robust_find_csv(cfg.in_csv, PROJECT_ROOT)
    df_raw = pd.read_csv(in_path)
    df = parse_and_index_time(df_raw)

    required = [LEAF_T, AIR_T, PAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns for revised CBF: {missing}")

    state_cols_all = [c for c in [AIR_T, PAR, VPD, CO2_C] if c in df.columns]
    action_cols_all = [
        HEAT, LIGHT, CO2_D, SCREEN_E, SCREEN_B, WATER, L_SIDE, W_SIDE, PIPE_T,
    ]
    action_cols_all = [c for c in action_cols_all if c in df.columns]

    for bridge, parents in list(BRIDGE_PARENT_ACTIONS.items()):
        BRIDGE_PARENT_ACTIONS[bridge] = [p for p in parents if p in df.columns]

    impute_cols = list(dict.fromkeys([LEAF_T] + state_cols_all + action_cols_all))
    df = impute_predictors(df, impute_cols, limit=cfg.interp_limit)

    print("[INFO] Target:", LEAF_T)
    print("[INFO] Horizons:", [h * cfg.minutes_per_step for h in cfg.horizons_steps], "min")
    print("[INFO] State columns for expanded baseline:", state_cols_all)
    print("[INFO] Action columns:", action_cols_all)
    print("[INFO] Bridge parent actions:")
    for b, parents in BRIDGE_PARENT_ACTIONS.items():
        print("  ", SHORT_NAME.get(b, b), "<-", [SHORT_NAME.get(p, p) for p in parents])

    feature_specs = build_direct_feature_specs(state_cols_all)

    classical_models = ["Ridge", "RF"]
    if HAS_XGBOOST:
        classical_models.append("XGBoost")

    deep_models: List[str] = []
    if cfg.enable_deep_learning and HAS_TORCH:
        deep_models = list(cfg.deep_models)

    metrics_rows: List[Dict[str, Any]] = []
    importance_tables: List[pd.DataFrame] = []
    stageA_metric_rows: List[Dict[str, Any]] = []
    prediction_rows: List[pd.DataFrame] = []

    for h in cfg.horizons_steps:
        print("-" * 80)
        print(f"[HORIZON] h={h} step(s) = {h * cfg.minutes_per_step} min")

        two_stage_base_cache = None

        for feature_set, spec in feature_specs:
            print(f"\n[FEATURE] {feature_set}")

            if feature_set in TWO_STAGE_FEATURE_SETS:
                X_base, y_base, _ = build_two_stage_base(df, h)
            else:
                assert spec is not None
                X_base, y_base = build_supervised(df, LEAF_T, spec, h)

            n = len(y_base)
            splits = make_rolling_splits(n)
            print(f"[INFO] samples={n}, p={X_base.shape[1]}, folds={len(splits)}")

            if feature_set == "TwoStage_CBF_AirT_PAR_ValidationSelected":
                model_list = ["ValidationSelected"]
            else:
                model_list = list(classical_models)
                if feature_set in cfg.deep_feature_sets:
                    model_list.extend(deep_models)

            for model_name in model_list:
                print(f"  [MODEL] {model_name}")
                for fold_id, tr, va, te in splits:
                    try:
                        if feature_set in TWO_STAGE_FEATURE_SETS:
                            X, y, pred_tr, pred_va, pred_te, model, scaler, runtime_s, stageA_rows = two_stage_fit_predict_fold(
                                df=df,
                                horizon=h,
                                final_algo=model_name,
                                tr=tr,
                                va=va,
                                te=te,
                                include_observed_bridge_lags=(feature_set == "TwoStage_CBF_AirT_PAR_Residual"),
                                validation_selected=(feature_set == "TwoStage_CBF_AirT_PAR_ValidationSelected"),
                            )
                            for r in stageA_rows:
                                r.update({
                                    "fold": fold_id,
                                    "final_model": model_name,
                                    "feature_set": feature_set,
                                })
                                stageA_metric_rows.append(r)
                        else:
                            X, y = X_base, y_base
                            pred_tr, pred_va, pred_te, model, scaler, runtime_s = fit_predict_model(
                                model_name, X, y, tr, va, te
                            )

                        m_tr = eval_metrics(y.iloc[tr].values, pred_tr)
                        m_va = eval_metrics(y.iloc[va].values, pred_va)
                        m_te = eval_metrics(y.iloc[te].values, pred_te)

                        metrics_rows.append({
                            "fold": fold_id,
                            "feature_set": feature_set,
                            "model": model_name,
                            "horizon_steps": int(h),
                            "horizon_minutes": int(h * cfg.minutes_per_step),
                            "n": int(len(y)),
                            "p": int(X.shape[1]),
                            "n_train": int(len(tr)),
                            "n_val": int(len(va)),
                            "n_test": int(len(te)),
                            "runtime_s": float(runtime_s),
                            "train_MAE": m_tr["MAE"],
                            "train_RMSE": m_tr["RMSE"],
                            "train_R2": m_tr["R2"],
                            "val_MAE": m_va["MAE"],
                            "val_RMSE": m_va["RMSE"],
                            "val_R2": m_va["R2"],
                            "test_MAE": m_te["MAE"],
                            "test_RMSE": m_te["RMSE"],
                            "test_R2": m_te["R2"],
                        })

                        imp = feature_importance_table(model, model_name, list(X.columns), scaler, h, feature_set, fold_id)
                        if not imp.empty:
                            importance_tables.append(imp)

                        if cfg.save_predictions:
                            pred_df = pd.DataFrame({
                                "time": X.iloc[te].index,
                                "y_true": y.iloc[te].values,
                                "y_pred": pred_te,
                                "fold": fold_id,
                                "feature_set": feature_set,
                                "model": model_name,
                                "horizon_steps": int(h),
                                "horizon_minutes": int(h * cfg.minutes_per_step),
                            })
                            prediction_rows.append(pred_df)

                        print(
                            f"    [OK] {fold_id}: MAE={m_te['MAE']:.4f}, "
                            f"RMSE={m_te['RMSE']:.4f}, R2={m_te['R2']:.4f}, runtime={runtime_s:.2f}s"
                        )
                    except Exception as e:
                        print(f"    [FAIL] {feature_set} | {model_name} | {fold_id}: {e}")


    if not metrics_rows:
        raise RuntimeError("No successful forecasting results were produced.")

    df_by_fold = pd.DataFrame(metrics_rows)
    df_by_fold["feature_display_name"] = df_by_fold["feature_set"].map(display_feature_name)
    df_by_fold["main_manuscript_configuration"] = df_by_fold["feature_set"].isin(MAIN_MANUSCRIPT_FEATURE_SETS)
    df_by_fold = df_by_fold.sort_values(["horizon_steps", "feature_set", "model", "fold"]).reset_index(drop=True)
    by_fold_path = os.path.join(cfg.out_dir, "cbf_forecasting_full_results_by_fold.csv")
    df_by_fold.to_csv(by_fold_path, index=False, encoding="utf-8-sig")

    # mean±std
    df_summary = summarize_mean_std(df_by_fold)
    df_summary["feature_display_name"] = df_summary["feature_set"].map(display_feature_name)
    df_summary["main_manuscript_configuration"] = df_summary["feature_set"].isin(MAIN_MANUSCRIPT_FEATURE_SETS)
    summary_path = os.path.join(cfg.out_dir, "cbf_forecasting_summary_mean_std.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    df_word = make_word_summary(df_summary)
    word_path = os.path.join(cfg.out_dir, "cbf_forecasting_summary_for_word.csv")
    df_word.to_csv(word_path, index=False, encoding="utf-8-sig")

    df_word_main = df_word[df_word["Internal feature_set"].isin(MAIN_MANUSCRIPT_FEATURE_SETS)].copy()
    word_main_path = os.path.join(cfg.out_dir, "cbf_forecasting_summary_for_main_text.csv")
    df_word_main.to_csv(word_main_path, index=False, encoding="utf-8-sig")

    best_path = os.path.join(cfg.out_dir, "cbf_forecasting_best_by_horizon.csv")
    df_best = (
        df_summary.sort_values(["horizon_steps", "test_RMSE_mean", "test_MAE_mean", "feature_set", "model"])
        .groupby("horizon_steps", as_index=False)
        .first()
    )
    df_best["feature_display_name"] = df_best["feature_set"].map(display_feature_name)
    df_best.to_csv(best_path, index=False, encoding="utf-8-sig")

    df_sig = significance_tests(df_by_fold)
    sig_path = os.path.join(cfg.out_dir, "cbf_forecasting_significance_tests.csv")
    df_sig.to_csv(sig_path, index=False, encoding="utf-8-sig")

    df_stageA = pd.DataFrame(stageA_metric_rows)
    stageA_path = os.path.join(cfg.out_dir, "cbf_stageA_bridge_metrics_by_fold.csv")
    df_stageA.to_csv(stageA_path, index=False, encoding="utf-8-sig")

    prop_rows = []
    two_stage = df_by_fold[df_by_fold["feature_set"].isin(TWO_STAGE_FEATURE_SETS)].copy()
    if not two_stage.empty:
        for feature_set_i in sorted(two_stage["feature_set"].unique()):
            sub_f = two_stage[two_stage["feature_set"] == feature_set_i]
            for model_name in sorted(sub_f["model"].unique()):
                sub_m = sub_f[sub_f["model"] == model_name]
                base_10 = sub_m[sub_m["horizon_steps"] == min(cfg.horizons_steps)]
                if base_10.empty:
                    continue
                base_mean = float(base_10["test_MAE"].mean())
                for _, r in sub_m.iterrows():
                    stageA_test = df_stageA[
                        (df_stageA["fold"] == r["fold"]) &
                        (df_stageA["final_model"] == model_name) &
                        (df_stageA["feature_set"] == r["feature_set"]) &
                        (df_stageA["horizon_steps"] == r["horizon_steps"]) &
                        (df_stageA["split"] == "test")
                    ] if not df_stageA.empty else pd.DataFrame()
                    air_mae = np.nan
                    par_mae = np.nan
                    if not stageA_test.empty:
                        hit_air = stageA_test[stageA_test["bridge_variable"] == "Air_T"]
                        hit_par = stageA_test[stageA_test["bridge_variable"] == "PAR"]
                        if not hit_air.empty:
                            air_mae = float(hit_air["MAE"].mean())
                        if not hit_par.empty:
                            par_mae = float(hit_par["MAE"].mean())
                    prop_rows.append({
                        "fold": r["fold"],
                        "feature_set": r["feature_set"],
                        "model": model_name,
                        "horizon_steps": int(r["horizon_steps"]),
                        "horizon_minutes": int(r["horizon_minutes"]),
                        "stageA_AirT_MAE": air_mae,
                        "stageA_PAR_MAE": par_mae,
                        "stageB_LeafT_MAE": float(r["test_MAE"]),
                        "amplification_ratio_vs_10min": float(r["test_MAE"] / base_mean) if base_mean > 0 else np.nan,
                    })

    df_prop = pd.DataFrame(prop_rows)
    if not df_prop.empty:
        df_prop["feature_display_name"] = df_prop["feature_set"].map(display_feature_name)
    prop_path = os.path.join(cfg.out_dir, "cbf_error_propagation.csv")
    df_prop.to_csv(prop_path, index=False, encoding="utf-8-sig")

    if not df_prop.empty:
        prop_summary = df_prop.groupby(["feature_set", "feature_display_name", "model", "horizon_steps", "horizon_minutes"]).agg(
            stageA_AirT_MAE_mean=("stageA_AirT_MAE", "mean"),
            stageA_AirT_MAE_std=("stageA_AirT_MAE", "std"),
            stageA_PAR_MAE_mean=("stageA_PAR_MAE", "mean"),
            stageA_PAR_MAE_std=("stageA_PAR_MAE", "std"),
            stageB_LeafT_MAE_mean=("stageB_LeafT_MAE", "mean"),
            stageB_LeafT_MAE_std=("stageB_LeafT_MAE", "std"),
            amplification_ratio_mean=("amplification_ratio_vs_10min", "mean"),
            amplification_ratio_std=("amplification_ratio_vs_10min", "std"),
        ).reset_index()
    else:
        prop_summary = pd.DataFrame()
    prop_summary_path = os.path.join(cfg.out_dir, "cbf_error_propagation_summary.csv")
    prop_summary.to_csv(prop_summary_path, index=False, encoding="utf-8-sig")

    if importance_tables:
        df_importance = pd.concat(importance_tables, ignore_index=True)
    else:
        df_importance = pd.DataFrame()
    importance_path = os.path.join(cfg.out_dir, "cbf_feature_importance_by_fold.csv")
    df_importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    manifest = feature_manifest()
    manifest_path = os.path.join(cfg.out_dir, "cbf_feature_config_manifest.csv")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    if cfg.save_predictions and prediction_rows:
        df_pred = pd.concat(prediction_rows, ignore_index=True)
        pred_path = os.path.join(cfg.out_dir, "cbf_test_predictions_all.csv")
        df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")


    run_config = {
        "input_csv": in_path,
        "minutes_per_step": cfg.minutes_per_step,
        "horizons_steps": cfg.horizons_steps,
        "leafT_lags": cfg.leafT_lags,
        "state_lags": cfg.state_lags,
        "causal_lag1": cfg.causal_lag1,
        "causal_window_lags": cfg.causal_window_lags,
        "action_lags": cfg.action_lags,
        "rolling_folds": cfg.rolling_folds,
        "test_window_days": cfg.test_window_days,
        "val_window_days": cfg.val_window_days,
        "min_train_samples": cfg.min_train_samples,
        "models_classical": classical_models,
        "deep_models_enabled": cfg.enable_deep_learning,
        "deep_models_available": deep_models,
        "deep_feature_sets": cfg.deep_feature_sets,
        "feature_display_name": FEATURE_DISPLAY_NAME,
        "main_manuscript_feature_sets": MAIN_MANUSCRIPT_FEATURE_SETS,
        "stageA_algo": cfg.stageA_algo,
        "validation_selection_algos": cfg.validation_selection_algos,
        "bridge_parent_actions": BRIDGE_PARENT_ACTIONS,
        "seed": cfg.seed,
        "has_xgboost": HAS_XGBOOST,
        "has_scipy": HAS_SCIPY,
        "has_torch": HAS_TORCH,
    }
    config_path = os.path.join(cfg.out_dir, "step5_cbf_review_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)


    txt_path = os.path.join(cfg.out_dir, "step5_cbf_review_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Step5 | CBF Leaf_T review-response forecasting summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input CSV: {in_path}\n")
        f.write(f"Output directory: {cfg.out_dir}\n")
        f.write(f"Total runtime: {time.time() - t_start:.2f} s\n")
        f.write(f"Horizons: {[h * cfg.minutes_per_step for h in cfg.horizons_steps]} min\n")
        f.write(f"Classical models: {classical_models}\n")
        f.write(f"Deep models used: {deep_models}\n")
        f.write("\nFeature configurations:\n")
        for _, r in manifest.iterrows():
            f.write(
                f"- {r['manuscript_feature_configuration']} "
                f"[{r['feature_set']}]: {r['feature_description']} ({r['causal_role']})\n"
            )
        f.write("\nBest configurations by horizon based on mean RMSE:\n")
        for _, r in df_best.iterrows():
            f.write(
                f"- {int(r['horizon_minutes'])} min: {display_feature_name(r['feature_set'])} "
                f"[{r['feature_set']}] | {r['model']} | "
                f"MAE={r['test_MAE_mean']:.3f}±{r['test_MAE_std']:.3f}, "
                f"RMSE={r['test_RMSE_mean']:.3f}±{r['test_RMSE_std']:.3f}, "
                f"R2={r['test_R2_mean']:.3f}±{r['test_R2_std']:.3f}\n"
            )

    print("\n[OUTPUTS]")
    for p in [
        by_fold_path,
        summary_path,
        word_path,
        word_main_path,
        best_path,
        sig_path,
        stageA_path,
        prop_path,
        prop_summary_path,
        importance_path,
        manifest_path,
        config_path,
        txt_path,
    ]:
        print("[OK]", p)

    print(f"[DONE] Step5 review-response forecasting experiment finished in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
