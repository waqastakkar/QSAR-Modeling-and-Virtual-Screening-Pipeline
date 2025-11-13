#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.

QSAR Batch Modeling with SHAP Analysis
================================================
QSAR script with parallel training, logging, feature engineering,
automatic model selection, and comprehensive SHAP analysis for best models.

Key Features:
- SHAP analysis for model interpretability (with API compatibility fixes)
- Parallel model training with joblib
- Logging system
- Feature filtering (remove low-variance fingerprints)
- Optional ensemble/stacking of top models
- Progress tracking with tqdm
- Configuration file support (JSON/YAML)
- Memory-efficient SHAP computation
- Comprehensive reporting
"""

import os
import sys
import json
import math
import shutil
import argparse
import logging
import psutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------- Headless matplotlib setup ---------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

# Configure settings BEFORE importing pyplot
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.5, 
    "lines.linewidth": 2,    
})

import matplotlib.pyplot as plt

# Set environment variables
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# --------- GPU memory management ---------
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# --------- QSAR_package path injection ---------
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
CANDIDATES = [
    HERE / "QSAR_package",
    HERE.parent / "QSAR_package",
    Path("/QSAR_package"),
    Path.home() / "QSAR_package",
]
for P in CANDIDATES:
    if P.exists() and P.is_dir():
        sys.path.insert(0, str(P.parent))
        break

# --------- Configuration Dataclass ---------
@dataclass
class QSARConfig:
    """Configuration container for QSAR modeling pipeline"""
    input_path: str = ""
    label_column: str = ""
    task: str = "classification"
    test_size: float = 0.2
    folds: int = 5
    seed: int = 42
    tune: bool = False
    tune_iter: int = 25
    max_epochs: int = 100
    use_bootstrap_lasso: bool = False
    boot_iters: int = 50
    boot_frac: float = 0.8
    roc_from_threshold: bool = False
    roc_threshold: Optional[float] = None
    selected_models: List[int] = field(default_factory=list)
    output_dir: str = ""
    variance_threshold: float = 0.01
    shap_max_display: int = 20
    shap_sample_size: int = 500
    run_shap: bool = True
    create_ensemble: bool = False
    ensemble_n_top: int = 3
    n_jobs: int = -1
    memory_limit_gb: float = 16.0
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'QSARConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'QSARConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# --------- Logging Setup ---------
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure comprehensive logging"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"qsar_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger

# --------- Memory Monitoring ---------
def check_memory_usage(limit_gb: float = 16.0) -> bool:
    """Check if current memory usage is below limit"""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    available_gb = mem.available / (1024**3)
    return used_gb < limit_gb and available_gb > 2.0

# --------- Data I/O ---------
FP_PREFIXES = ("morgan_", "maccs_", "rdkit_", "atompair_", "torsion_")

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet file"""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def select_fp_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric fingerprint columns"""
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).startswith(FP_PREFIXES):
            cols.append(c)
    return cols

def filter_low_variance_features(df: pd.DataFrame, fp_cols: List[str], 
                               threshold: float = 0.01, logger: Optional[logging.Logger] = None) -> List[str]:
    """Remove low-variance fingerprint features"""
    from sklearn.feature_selection import VarianceThreshold
    
    X = df[fp_cols].values
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    kept_mask = selector.get_support()
    kept_cols = [c for c, keep in zip(fp_cols, kept_mask) if keep]
    
    if logger:
        logger.info(f"Feature filtering: {len(fp_cols)} → {len(kept_cols)} features (threshold={threshold})")
    
    return kept_cols

def holdout_split(task: str, X, y, test_size=0.2, seed=42):
    """Stratified split for classification, random for regression"""
    from sklearn.model_selection import train_test_split
    if task == "classification":
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed)

# --------- Model Builders ---------
def _build_keras_classifier(input_dim: int,
                            hidden: Tuple[int, ...] = (512, 256),
                            dropout: float = 0.3,
                            lr: float = 1e-3,
                            l2: float = 1e-6,
                            batch_norm: bool = True):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    inp = keras.Input(shape=(input_dim,), name="x")
    x = inp
    for i, h in enumerate(hidden):
        x = layers.Dense(h, activation="relu",
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"do_{i}")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["AUC", "Precision", "Recall"])
    return model

def _build_keras_regressor(input_dim: int,
                           hidden: Tuple[int, ...] = (512, 256),
                           dropout: float = 0.2,
                           lr: float = 1e-3,
                           l2: float = 1e-6,
                           batch_norm: bool = True):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    inp = keras.Input(shape=(input_dim,), name="x")
    x = inp
    for i, h in enumerate(hidden):
        x = layers.Dense(h, activation="relu",
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"do_{i}")(x)
    out = layers.Dense(1, activation="linear", name="out")(x)
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def _build_skorch_module(input_dim: int, hidden: Tuple[int, ...] = (1024, 512), dropout: float = 0.3):
    import torch.nn as nn
    layers = []
    in_f = input_dim
    for h in hidden:
        layers += [nn.Linear(in_f, h), nn.ReLU(), nn.Dropout(dropout)]
        in_f = h
    layers += [nn.Linear(in_f, 1)]
    return nn.Sequential(*layers)

def _make_skorch_classifier(input_dim: int, lr: float = 1e-3, hidden=(1024,512), dropout=0.3, weight_decay=1e-6):
    from skorch import NeuralNetClassifier
    import torch
    net = NeuralNetClassifier(
        module=_build_skorch_module,
        module__input_dim=input_dim,
        module__hidden=hidden,
        module__dropout=dropout,
        max_epochs=100,
        lr=lr,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
    )
    return net

def _make_skorch_regressor(input_dim: int, lr: float = 1e-3, hidden=(1024,512), dropout=0.2, weight_decay=1e-6):
    from skorch import NeuralNetRegressor
    import torch
    net = NeuralNetRegressor(
        module=_build_skorch_module,
        module__input_dim=input_dim,
        module__dropout=dropout,
        max_epochs=100,
        lr=lr,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[],
    )
    return net

# --------- Model Registry ---------
def build_registry(task: str, input_dim: Optional[int] = None) -> List[Tuple[str, str, Any, Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Build model registry with 6 selected models per task"""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor
    )

    clf_models = [
        ("LR",  "LogisticRegression", LogisticRegression, dict(max_iter=2000, solver="lbfgs"), {"C":[0.01,0.1,1,10,100]}),
        ("RFC", "RandomForestClassifier", RandomForestClassifier, dict(n_estimators=400, n_jobs=-1), {"n_estimators":[300,500,800],"max_depth":[None,20,40]}),
        ("SVC", "SVC", SVC, dict(probability=True), {"C":[0.1,1,10],"gamma":["scale","auto"]}),
        ("XGBC","XGBClassifier", None, dict(), {"n_estimators":[300,600,1000],"max_depth":[3,6,9],"learning_rate":[0.03,0.1]}),
        ("LGBMC","LGBMClassifier", None, dict(), {"n_estimators":[400,800,1200],"num_leaves":[31,63,127],"learning_rate":[0.03,0.1]}),
        ("ETC","ExtraTreesClassifier", ExtraTreesClassifier, dict(n_estimators=400, n_jobs=-1), {"n_estimators":[300,600,1000],"max_depth":[None,20,40]}),
    ]

    reg_models = [
        ("Ridge","Ridge", Ridge, dict(), {"alpha":[1e-3,1e-2,1e-1,1,10,100]}),
        ("RFR","RandomForestRegressor", RandomForestRegressor, dict(n_estimators=400, n_jobs=-1), {"n_estimators":[300,600,1000],"max_depth":[None,20,40]}),
        ("SVR","SVR", SVR, dict(), {"C":[0.1,1,10], "gamma":["scale","auto"]}),
        ("XGBR","XGBRegressor", None, dict(), {"n_estimators":[300,600,1000],"max_depth":[3,6,9],"learning_rate":[0.03,0.1]}),
        ("LGBMR","LGBMRegressor", None, dict(), {"n_estimators":[400,800,1200],"num_leaves":[31,63,127],"learning_rate":[0.03,0.1]}),
        ("ETR","ExtraTreesRegressor", ExtraTreesRegressor, dict(n_estimators=400, n_jobs=-1), {"n_estimators":[300,600,1000],"max_depth":[None,20,40]}),
    ]

    def _resolve_optional(models, task_mode):
        out = []
        for key, label, ctor, params, grid in models:
            if key.startswith("XGB"):
                try:
                    if task_mode == "classification":
                        from xgboost import XGBClassifier as XGB
                    else:
                        from xgboost import XGBRegressor as XGB
                    ctor = XGB
                except Exception:
                    ctor = None
            if key.startswith("LGBM"):
                try:
                    if task_mode == "classification":
                        from lightgbm import LGBMClassifier as LGBM
                    else:
                        from lightgbm import LGBMRegressor as LGBM
                    ctor = LGBM
                except Exception:
                    ctor = None
            out.append((key, label, ctor, params, grid))
        return out

    clf_models = _resolve_optional(clf_models, "classification")
    reg_models = _resolve_optional(reg_models, "regression")

    return clf_models if task == "classification" else reg_models

# --------- Tuning Helpers ---------
def _add_early_stopping_keras(estimator, patience=12):
    """Add early stopping to Keras models"""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        if hasattr(estimator, "get_params"):
            params = estimator.get_params()
            cbs = params.get("callbacks", [])
            cbs = list(cbs) if isinstance(cbs, (list, tuple)) else []
            cbs.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
            estimator.set_params(callbacks=cbs)
    except Exception:
        pass

def _tune_hyperparams(estimator, X, y, task: str, folds: int, grid: Dict[str, Any], n_iter: int, seed: int, logger: logging.Logger):
    """Hyperparameter tuning with RandomizedSearchCV"""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
    
    if not grid:
        logger.info("  → No hyperparameter grid, skipping tuning")
        return estimator
    
    logger.info(f"  → Tuning with {n_iter} iterations")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed) if task=="classification" else KFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = "roc_auc" if task=="classification" else "neg_root_mean_squared_error"
    rs = RandomizedSearchCV(estimator, param_distributions=grid, n_iter=n_iter, cv=cv, scoring=scoring,
                            random_state=seed, n_jobs=-1, verbose=0)
    rs.fit(X, y)
    logger.info(f"  ✓ Best params: {rs.best_params_}")
    return rs.best_estimator_

# --------- Bootstrap Wrapper ---------
from sklearn.base import BaseEstimator, RegressorMixin, clone

class BootstrapRegressor(BaseEstimator, RegressorMixin):
    """Bootstrap wrapper for regression models"""
    def __init__(self, base_estimator, n_boot=50, max_samples=1.0, random_state=42):
        self.base_estimator = base_estimator
        self.n_boot = int(n_boot)
        self.max_samples = float(max_samples)
        self.random_state = int(random_state)
        self._models_ = []

    def get_params(self, deep=True):
        params = {
            "base_estimator": self.base_estimator,
            "n_boot": self.n_boot,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params

    def set_params(self, **params):
        for p in ["n_boot", "max_samples", "random_state"]:
            if p in params:
                setattr(self, p, params.pop(p))
        if "base_estimator" in params:
            self.base_estimator = params.pop("base_estimator")
        be_params = {k.split("__",1)[1]: v for k,v in params.items()
                     if k.startswith("base_estimator__")}
        if be_params and hasattr(self.base_estimator, "set_params"):
            self.base_estimator.set_params(**be_params)
        return self

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = max(1, int(round(self.max_samples * n)))
        self._models_ = []
        for _ in range(self.n_boot):
            idx = rng.integers(0, n, size=m)
            est = clone(self.base_estimator)
            est.fit(X[idx], y[idx])
            self._models_.append(est)
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self._models_]
        return np.mean(preds, axis=0)

def _prefix_grid(grid: dict, prefix: str) -> dict:
    """Prefix parameter grid for wrapped estimators"""
    return {f"{prefix}__{k}": v for k, v in (grid or {}).items()}

# --------- Model Saving Helpers ---------
def _save_estimator(model, key: str, out_dir: Path, logger: logging.Logger) -> List[str]:
    """Save model in multiple formats"""
    saved: List[str] = []
    base = out_dir / f"model_{key}"
    
    # 1) joblib
    try:
        import joblib
        joblib.dump(model, base.with_suffix(".joblib"))
        saved.append(str(base.with_suffix(".joblib")))
        logger.info(f"    ✓ Saved joblib model")
    except Exception as e:
        logger.debug(f"    - joblib save failed: {e}")
    
    # 2) XGBoost native
    try:
        from xgboost import XGBModel
        if isinstance(model, XGBModel):
            model.save_model(str(base.with_suffix(".json")))
            saved.append(str(base.with_suffix(".json")))
            logger.info(f"    ✓ Saved XGBoost native model")
    except Exception as e:
        logger.debug(f"    - XGBoost save failed: {e}")
    
    # 3) LightGBM native
    try:
        import lightgbm as lgb
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(base.with_suffix(".txt")))
            saved.append(str(base.with_suffix(".txt")))
            logger.info(f"    ✓ Saved LightGBM native model")
    except Exception as e:
        logger.debug(f"    - LightGBM save failed: {e}")
    
    # 4) Keras native
    try:
        if hasattr(model, "model_"):
            model.model_.save(str(base.with_suffix(".keras")))
            saved.append(str(base.with_suffix(".keras")))
            logger.info(f"    ✓ Saved Keras model")
    except Exception as e:
        logger.debug(f"    - Keras save failed: {e}")
    
    # 5) PyTorch native
    try:
        if hasattr(model, "save_params"):
            model.save_params(f_params=str(base.with_suffix(".pt")))
            saved.append(str(base.with_suffix(".pt")))
            logger.info(f"    ✓ Saved PyTorch model")
    except Exception as e:
        logger.debug(f"    - PyTorch save failed: {e}")
    
    return saved

def _save_fig(fig, path_noext: Path):
    """Save figure as PNG and SVG with proper DPI"""
    png = path_noext.with_suffix(".png")
    svg = path_noext.with_suffix(".svg")
    fig.tight_layout()
    # EXPLICITLY set DPI to ensure 300 resolution
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

# --------- SHAP Analysis Module ---------
def analyze_shap_best_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           out_dir: Path, task: str, model_key: str, 
                           max_display: int = 20, sample_size: int = 500,
                           memory_efficient: bool = True, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Comprehensive SHAP analysis optimized for different model types
    
    Returns:
        Dict containing paths to outputs and summary statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    import shap
    from sklearn.base import is_classifier
    
    logger.info(f"\n=== SHAP Analysis for {model_key} ===")
    shap_dir = out_dir / "shap_analysis"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_key": model_key,
        "plots": [],
        "values_file": None,
        "feature_importance_file": None,
        "error": None
    }
    
    # Sample data for efficiency
    if X_train.shape[0] > sample_size:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(X_train.shape[0], sample_size, replace=False)
        X_train_sample = X_train.iloc[sample_idx]
        logger.info(f"  → Using sample of {sample_size} observations")
    else:
        X_train_sample = X_train
    
    try:
        # Select appropriate explainer
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            logger.info("  → Using TreeExplainer")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_sample)
            
        elif hasattr(model, "coef_"):
            # Linear models
            logger.info("  → Using LinearExplainer")
            explainer = shap.LinearExplainer(model, X_train_sample)
            shap_values = explainer.shap_values(X_train_sample)
            
        elif hasattr(model, "model_"):
            # Keras/TensorFlow models
            logger.info("  → Using DeepExplainer")
            background = X_train_sample[:min(100, len(X_train_sample))]
            explainer = shap.DeepExplainer(model.model_, background.values)
            shap_values = explainer.shap_values(X_train_sample.values)
            shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
            
        elif hasattr(model, "module_"):
            # PyTorch/skorch models
            logger.info("  → Using DeepExplainer (PyTorch)")
            background = X_train_sample[:min(100, len(X_train_sample))]
            explainer = shap.DeepExplainer(model, background.values)
            shap_values = explainer.shap_values(X_train_sample.values)
            shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
            
        else:
            # KernelExplainer fallback
            logger.info("  → Using KernelExplainer (slower)")
            background = shap.kmeans(X_train_sample, k=min(50, len(X_train_sample)))
            def predict_fn(X):
                if is_classifier(model):
                    return model.predict_proba(X)[:, 1]
                return model.predict(X)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_train_sample)
        
        # Handle classification multi-output
        if task == "classification" and isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Save SHAP values
        values_path = shap_dir / f"shap_values_{model_key}.npy"
        np.save(values_path, shap_values)
        results["values_file"] = str(values_path)
        
        # Global feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = list(X_train.columns)
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "std_abs_shap": np.abs(shap_values).std(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        
        importance_path = shap_dir / f"feature_importance_{model_key}.csv"
        importance_df.to_csv(importance_path, index=False)
        results["feature_importance_file"] = str(importance_path)
        results["top_features"] = importance_df.head(max_display)["feature"].tolist()
        
        logger.info(f"  ✓ Top features: {', '.join(results['top_features'][:5])}")
        
        # Summary plot
        fig = plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_train_sample, 
                         feature_names=feature_names,
                         max_display=max_display,
                         show=False)
        plt.title(f"SHAP Summary – {model_key}", fontsize=14, fontweight="bold")
        summary_path = shap_dir / f"shap_summary_{model_key}"
        _save_fig(fig, summary_path)
        results["plots"].append(str(summary_path.with_suffix(".png")))
        
        # Dependence plots for top 3 features
        for i, feat in enumerate(results["top_features"][:3]):
            fig = plt.figure(figsize=(8, 6))
            shap.dependence_plot(feat, shap_values, X_train_sample, 
                               feature_names=feature_names,
                               show=False)
            dep_path = shap_dir / f"dependence_{feat}_{model_key}"
            _save_fig(fig, dep_path)
            results["plots"].append(str(dep_path.with_suffix(".png")))
        
        # Waterfall plot for test sample
        if len(X_test) > 0:
            test_idx = np.random.randint(0, min(100, len(X_test)))
            fig = plt.figure(figsize=(10, 6))
            
            # Get base value (handle both scalar and array cases)
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if np.array(base_value).size > 1 else base_value
            
            sample_idx = test_idx % len(X_train_sample)  # Ensure valid index
            
            # SHAP API (0.40+) - Use shap.plots.waterfall()
            try:
                # Create explanation object for single sample
                expl = shap.Explanation(
                    values=shap_values[sample_idx:sample_idx+1],
                    base_values=base_value,
                    data=X_train_sample.iloc[sample_idx:sample_idx+1].values,
                    feature_names=feature_names
                )
                # API call with expl[0] for single sample
                shap.plots.waterfall(expl[0], show=False)
                
            except (AttributeError, TypeError) as e:
                # Legacy SHAP API (<0.40) - Use shap.waterfall_plot()
                logger.warning(f"    - Modern SHAP API failed, using legacy fallback: {e}")
                try:
                    shap.waterfall_plot(
                        base_value,
                        shap_values[test_idx],
                        X_train_sample.iloc[test_idx].values,
                        feature_names=feature_names,
                        max_display=max_display,
                        show=False
                    )
                except Exception as e2:
                    logger.warning(f"    - Waterfall plot failed entirely: {e2}")
            
            plt.title(f"SHAP Waterfall – Sample {test_idx} – {model_key}", 
                     fontsize=12, fontweight="bold")
            water_path = shap_dir / f"waterfall_sample_{model_key}"
            _save_fig(fig, water_path)
            results["plots"].append(str(water_path.with_suffix(".png")))
        
        logger.info(f"  ✓ SHAP complete: {len(results['plots'])} plots, {len(importance_df)} features analyzed")
        
    except Exception as e:
        logger.error(f"  ✗ SHAP analysis failed: {str(e)}")
        results["error"] = str(e)
    
    return results

# --------- CV + Test Evaluation (Enhanced) ---------
def run_classification(model_ctor, params, tune_grid, X, y, out_dir: Path, key: str, 
                      folds=5, seed=42, test_size=0.2, tune=False, tune_iter=25, 
                      max_epochs=100, logger: logging.Logger = None) -> Dict[str, Any]:
    """Classification model training and evaluation"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, average_precision_score,
        precision_recall_curve, roc_curve, confusion_matrix
    )
    from sklearn.base import clone
    
    logger.info(f"\n--- Training {key} (Classification) ---")
    
    # Split data
    Xtr, Xte, ytr, yte = holdout_split("classification", X, y, test_size=test_size, seed=seed)
    
    # Prepare estimator
    est = model_ctor(**params) if callable(model_ctor) else model_ctor(**params)
    if "Keras" in key and hasattr(est, "set_params"):
        est.set_params(epochs=max_epochs)
        _add_early_stopping_keras(est, patience=12)
    if "Skorch" in key and hasattr(est, "set_params"):
        est.set_params(max_epochs=max_epochs)
    
    # Hyperparameter tuning
    if tune and tune_grid is not None:
        est = _tune_hyperparams(est, Xtr, ytr, "classification", folds, tune_grid, tune_iter, seed, logger)
    
    # Cross-validation
    logger.info(f"  → Running {folds}-fold CV")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr), dtype=float)
    fold_rows = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, ytr), start=1):
        m = clone(est)
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(Xtr[va_idx])[:, 1]
        elif hasattr(m, "decision_function"):
            df = m.decision_function(Xtr[va_idx])
            proba = 1.0 / (1.0 + np.exp(-df))
        else:
            proba = m.predict(Xtr[va_idx]).astype(float)
        
        oof[va_idx] = proba
        pred = (proba >= 0.5).astype(int)
        
        fold_rows.append(dict(
            fold=fold,
            AUC=float(roc_auc_score(ytr[va_idx], proba)),
            AP=float(average_precision_score(ytr[va_idx], proba)),
            ACC=float(accuracy_score(ytr[va_idx], pred)),
            F1=float(f1_score(ytr[va_idx], pred)),
        ))
    
    pd.DataFrame(fold_rows).to_csv(out_dir / f"metrics_{key}_perfold.csv", index=False)
    
    # Final fit
    from sklearn.base import clone as _clone
    M = _clone(est)
    M.fit(Xtr, ytr)
    
    # Predictions
    if hasattr(M, "predict_proba"):
        proba_test = M.predict_proba(Xte)[:, 1]
        proba_train = M.predict_proba(Xtr)[:, 1]
    elif hasattr(M, "decision_function"):
        df_test = M.decision_function(Xte)
        proba_test = 1.0 / (1.0 + np.exp(-df_test))
        df_train = M.decision_function(Xtr)
        proba_train = 1.0 / (1.0 + np.exp(-df_train))
    else:
        proba_test = M.predict(Xte).astype(float)
        proba_train = M.predict(Xtr).astype(float)
    
    # Save predictions
    pd.DataFrame({"index": np.arange(len(ytr)), "y_true": ytr, "y_oof": oof}).to_csv(
        out_dir / f"predictions_{key}_train_oof.csv", index=False)
    pd.DataFrame({"index": np.arange(len(yte)), "y_true": yte, "y_pred": proba_test}).to_csv(
        out_dir / f"predictions_{key}_test.csv", index=False)
    
    # Metrics
    auc_tr = float(roc_auc_score(ytr, oof))
    ap_tr  = float(average_precision_score(ytr, oof))
    acc_tr = float(accuracy_score(ytr, (oof >= 0.5).astype(int)))
    f1_tr  = float(f1_score(ytr, (oof >= 0.5).astype(int)))
    auc_te = float(roc_auc_score(yte, proba_test))
    ap_te  = float(average_precision_score(yte, proba_test))
    acc_te = float(accuracy_score(yte, (proba_test >= 0.5).astype(int)))
    f1_te  = float(f1_score(yte, (proba_test >= 0.5).astype(int)))
    
    pd.DataFrame([dict(AUC=auc_te, AP=ap_te, ACC=acc_te, F1=f1_te)]).to_csv(
        out_dir / f"metrics_{key}_test.csv", index=False)
    
    # Plots
    logger.info("  → Generating plots")
    fpr_tr, tpr_tr, _ = roc_curve(ytr, oof)
    fpr_te, tpr_te, _ = roc_curve(yte, proba_test)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(fpr_tr, tpr_tr, linewidth=2, label=f"Train (OOF) AUC={auc_tr:.3f}")
    plt.plot(fpr_te, tpr_te, linewidth=2, linestyle="--", label=f"Test AUC={auc_te:.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {key}")
    plt.legend(frameon=False)
    _save_fig(fig, out_dir / f"roc_{key}")
    
    prec_tr, rec_tr, _ = precision_recall_curve(ytr, oof)
    prec_te, rec_te, _ = precision_recall_curve(yte, proba_test)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(rec_tr, prec_tr, linewidth=2, label=f"Train (OOF) AP={ap_tr:.3f}")
    plt.plot(rec_te, prec_te, linewidth=2, linestyle="--", label=f"Test AP={ap_te:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR – {key}")
    plt.legend(frameon=False)
    _save_fig(fig, out_dir / f"pr_{key}")
    
    for tag, cm in [("train", confusion_matrix(ytr, (oof >= 0.5).astype(int))),
                    ("test",  confusion_matrix(yte, (proba_test >= 0.5).astype(int)))]:
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.xticks([0,1], ["Pred 0","Pred 1"])
        plt.yticks([0,1], ["True 0","True 1"])
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", fontweight="bold")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({tag}) – {key}")
        _save_fig(fig, out_dir / f"cm_{tag}_{key}")
    
    # Save model
    model_files = _save_estimator(M, key, out_dir, logger)
    
    summary = dict(AUC_tr=auc_tr, AP_tr=ap_tr, ACC_tr=acc_tr, F1_tr=f1_tr,
                   AUC_te=auc_te, AP_te=ap_te, ACC_te=acc_te, F1_te=f1_te,
                   model_files=model_files)
    logger.info(f"  ✓ Training complete: AUC_te={auc_te:.3f}")
    return summary

def run_regression(model_ctor, params, tune_grid, X, y, out_dir: Path, key: str,
                   folds=5, seed=42, test_size=0.2, tune=False, tune_iter=25, 
                   max_epochs=100, roc_from_threshold: bool = False, 
                   roc_threshold: Optional[float] = None, logger: logging.Logger = None) -> Dict[str, Any]:
    """Regression model training and evaluation"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.base import clone
    
    logger.info(f"\n--- Training {key} (Regression) ---")
    
    Xtr, Xte, ytr, yte = holdout_split("regression", X, y, test_size=test_size, seed=seed)
    
    est = model_ctor(**params) if callable(model_ctor) else model_ctor(**params)
    if "Keras" in key and hasattr(est, "set_params"):
        est.set_params(epochs=max_epochs)
        _add_early_stopping_keras(est, patience=12)
    if "Skorch" in key and hasattr(est, "set_params"):
        est.set_params(max_epochs=max_epochs)
    
    if tune and tune_grid is not None:
        est = _tune_hyperparams(est, Xtr, ytr, "regression", folds, tune_grid, tune_iter, seed, logger)
    
    # CV
    logger.info(f"  → Running {folds}-fold CV")
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr), dtype=float)
    fold_rows = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr), start=1):
        m = clone(est)
        m.fit(Xtr[tr_idx], ytr[tr_idx])
        pred = m.predict(Xtr[va_idx])
        oof[va_idx] = pred
        fold_rows.append(dict(
            fold=fold,
            R2=float(r2_score(ytr[va_idx], pred)),
            RMSE=float(math.sqrt(mean_squared_error(ytr[va_idx], pred))),
            MAE=float(mean_absolute_error(ytr[va_idx], pred)),
        ))
    
    pd.DataFrame(fold_rows).to_csv(out_dir / f"metrics_{key}_perfold.csv", index=False)
    
    # Final fit
    M = clone(est)
    M.fit(Xtr, ytr)
    pred_te = M.predict(Xte)
    
    # Save predictions
    pd.DataFrame({"index": np.arange(len(ytr)), "y_true": ytr, "y_oof": oof}).to_csv(
        out_dir / f"predictions_{key}_train_oof.csv", index=False)
    pd.DataFrame({"index": np.arange(len(yte)), "y_true": yte, "y_pred": pred_te}).to_csv(
        out_dir / f"predictions_{key}_test.csv", index=False)
    
    def reg_metrics(y_true, y_hat):
        return dict(
            R2=float(r2_score(y_true, y_hat)),
            RMSE=float(math.sqrt(mean_squared_error(y_true, y_hat))),
            MAE=float(mean_absolute_error(y_true, y_hat)),
        )
    
    met_tr = reg_metrics(ytr, oof)
    met_te = reg_metrics(yte, pred_te)
    pd.DataFrame([met_te]).to_csv(out_dir / f"metrics_{key}_test.csv", index=False)
    
    # Plots
    logger.info("  → Generating plots")
    for tag, yref, yhat, stats in [("train", ytr, oof, met_tr), ("test", yte, pred_te, met_te)]:
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(yref, yhat, s=10, alpha=0.6)
        lims = [float(np.min([yref, yhat])), float(np.max([yref, yhat]))]
        plt.plot(lims, lims, linestyle=":", linewidth=1, color='red')
        txt = f"R²={stats['R2']:.3f}  RMSE={stats['RMSE']:.3f}  MAE={stats['MAE']:.3f}"
        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top", ha="left", 
                fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(f"Parity ({tag}) – {key}")
        _save_fig(fig, out_dir / f"parity_{tag}_{key}")
    
    for tag, yref, yhat in [("train", ytr, oof), ("test", yte, pred_te)]:
        resid = yhat - yref
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(yhat, resid, s=10, alpha=0.6)
        plt.axhline(0.0, linestyle=":", linewidth=1, color='red')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals (Pred - Obs)")
        plt.title(f"Residuals ({tag}) – {key}")
        _save_fig(fig, out_dir / (f"residuals_{tag}_{key}"))
    
    # Classification metrics via threshold
    summary = {f"{k}_tr": v for k, v in met_tr.items()}
    summary.update({f"{k}_te": v for k, v in met_te.items()})
    
    if roc_from_threshold and roc_threshold is not None:
        logger.info(f"  → Generating ROC/PR plots with threshold y≥{roc_threshold}")
        from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                                     precision_recall_curve, accuracy_score, f1_score, confusion_matrix)
        
        ytr_b = (ytr >= roc_threshold).astype(int)
        yte_b = (yte >= roc_threshold).astype(int)
        
        auc_tr = float(roc_auc_score(ytr_b, oof))
        ap_tr  = float(average_precision_score(ytr_b, oof))
        auc_te = float(roc_auc_score(yte_b, pred_te))
        ap_te  = float(average_precision_score(yte_b, pred_te))
        
        fpr_tr, tpr_tr, thr_tr = roc_curve(ytr_b, oof)
        th_star = float(thr_tr[int(np.argmax(tpr_tr - fpr_tr))])
        
        pred_tr_b = (oof >= th_star).astype(int)
        pred_te_b = (pred_te >= th_star).astype(int)
        acc_tr = float(accuracy_score(ytr_b, pred_tr_b))
        f1_tr  = float(f1_score(ytr_b, pred_tr_b))
        acc_te = float(accuracy_score(yte_b, pred_te_b))
        f1_te  = float(f1_score(yte_b, pred_te_b))
        
        fpr_te, tpr_te, _ = roc_curve(yte_b, pred_te)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(fpr_tr, tpr_tr, linewidth=2, label=f"Train (OOF) AUC={auc_tr:.3f}")
        plt.plot(fpr_te, tpr_te, linewidth=2, linestyle="--", label=f"Test AUC={auc_te:.3f}")
        plt.plot([0,1],[0,1], linestyle=":", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC – {key} (thr y≥{roc_threshold:g})")
        plt.legend(frameon=False)
        _save_fig(fig, out_dir / f"roc_from_reg_{key}")
        
        prec_tr, rec_tr, _ = precision_recall_curve(ytr_b, oof)
        prec_te, rec_te, _ = precision_recall_curve(yte_b, pred_te)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(rec_tr, prec_tr, linewidth=2, label=f"Train AP={ap_tr:.3f}")
        plt.plot(rec_te, prec_te, linewidth=2, linestyle="--", label=f"Test AP={ap_te:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR – {key} (thr y≥{roc_threshold:g})")
        plt.legend(frameon=False)
        _save_fig(fig, out_dir / f"pr_from_reg_{key}")
        
        for tag, yb, pb in [("train", ytr_b, pred_tr_b), ("test", yte_b, pred_te_b)]:
            cm = confusion_matrix(yb, pb)
            fig = plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation="nearest")
            plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center", fontweight="bold")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.title(f"Confusion Matrix ({tag}) – {key} (thr y≥{roc_threshold:g}, score≥{th_star:.3f})")
            _save_fig(fig, out_dir / f"cm_from_reg_{tag}_{key}")
        
        summary.update(dict(
            ROCthr=float(roc_threshold), ScoreThr=float(th_star),
            AUC_tr_cls=auc_tr, AP_tr_cls=ap_tr, ACC_tr_cls=acc_tr, F1_tr_cls=f1_tr,
            AUC_te_cls=auc_te, AP_te_cls=ap_te, ACC_te_cls=acc_te, F1_te_cls=f1_te
        ))
    
    # Save model
    model_files = _save_estimator(M, key, out_dir, logger)
    summary["model_files"] = model_files
    logger.info(f"  ✓ Training complete: R²_te={met_te['R2']:.3f}")
    return summary

# --------- Parallel Training Wrapper ---------
def train_model_parallel(args: Tuple[int, Any, np.ndarray, np.ndarray, Path, str, QSARConfig, logging.Logger]) -> Dict[str, Any]:
    """Wrapper for parallel model training"""
    i, registry, X, y, out_dir, task, config, logger = args
    key, label, ctor, params, grid = registry[i-1]
    
    try:
        if task == "classification":
            summ = run_classification(
                ctor, params, grid, X, y,
                out_dir=out_dir, key=key, folds=config.folds, seed=config.seed,
                test_size=config.test_size, tune=config.tune, tune_iter=config.tune_iter,
                max_epochs=config.max_epochs, logger=logger
            )
        else:
            if key == "Lasso" and config.use_bootstrap_lasso:
                def ctor_boot(**p):
                    base = ctor(**p)
                    return BootstrapRegressor(
                        base_estimator=base,
                        n_boot=config.boot_iters,
                        max_samples=config.boot_frac,
                        random_state=config.seed
                    )
                grid_boot = _prefix_grid(grid, "base_estimator") if (config.tune and grid) else grid
                summ = run_regression(
                    ctor_boot, params, grid_boot, X, y,
                    out_dir=out_dir, key="LassoB", folds=config.folds, seed=config.seed,
                    test_size=config.test_size, tune=config.tune, tune_iter=config.tune_iter,
                    max_epochs=config.max_epochs,
                    roc_from_threshold=config.roc_from_threshold, roc_threshold=config.roc_threshold,
                    logger=logger
                )
            else:
                summ = run_regression(
                    ctor, params, grid, X, y,
                    out_dir=out_dir, key=key, folds=config.folds, seed=config.seed,
                    test_size=config.test_size, tune=config.tune, tune_iter=config.tune_iter,
                    max_epochs=config.max_epochs,
                    roc_from_threshold=config.roc_from_threshold, roc_threshold=config.roc_threshold,
                    logger=logger
                )
        
        row = {"model": key if key != "Lasso" or not config.use_bootstrap_lasso else "LassoB"}
        row.update({k: v for k, v in summ.items() if k != "model_files"})
        row["model_files"] = ";".join(summ.get("model_files", []))
        
        return row
        
    except Exception as e:
        logger.error(f"  ✗ Training failed for {key}: {str(e)}")
        return {
            "model": key,
            "error": str(e),
            "model_files": ""
        }

# --------- Ensemble Creation ---------
def create_ensemble_model(summary_df: pd.DataFrame, X: np.ndarray, y: np.ndarray, 
                         task: str, config: QSARConfig, logger: logging.Logger, out_dir: Path) -> Optional[Any]:
    """Create ensemble of top-N models"""
    logger.info(f"\n=== Creating Ensemble of Top {config.ensemble_n_top} Models ===")
    
    from sklearn.ensemble import StackingClassifier, StackingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    import joblib
    
    # Select top models
    metric = "AUC_te" if task == "classification" else "R2_te"
    top_models = summary_df.nlargest(config.ensemble_n_top, metric)
    
    logger.info(f"  → Top {len(top_models)} models selected: {list(top_models['model'].values)}")
    
    estimators = []
    for _, row in top_models.iterrows():
        key = row["model"]
        files = str(row.get("model_files", "")).split(";")
        
        # Load model
        model = None
        for f in files:
            if f.strip().endswith(".joblib"):
                try:
                    model = joblib.load(f.strip())
                    break
                except Exception as e:
                    logger.warning(f"    - Failed to load {f.strip()}: {e}")
        
        if model is not None:
            estimators.append((key, model))
        else:
            logger.warning(f"    - Could not load model '{key}' (no valid .joblib file)")
    
    if len(estimators) < 2:
        logger.warning(f"  ⚠ Only {len(estimators)} model(s) available. Need ≥2 for ensemble. Skipping.")
        return None
    
    logger.info(f"  → Creating ensemble with {len(estimators)} models")
    
    # Create ensemble
    if task == "classification":
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=3
        )
    else:
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=3
        )
    
    # Fit on full data
    X_full = X
    y_full = y
    logger.info(f"  → Fitting ensemble on {len(y_full)} samples")
    ensemble.fit(X_full, y_full)
    
    # Save ensemble
    ensemble_dir = out_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, ensemble_dir / "ensemble_model.joblib")
    logger.info("  ✓ Ensemble saved to ensemble/ensemble_model.joblib")
    
    return ensemble

# --------- Main Interactive Function ---------
def main_interactive() -> Tuple[QSARConfig, np.ndarray, np.ndarray, pd.DataFrame]:
    """Interactive mode - prompts user for all inputs"""
    print("="*60)
    print("QSAR Batch Modeling with SHAP Analysis")
    print("="*60)
    
    config = QSARConfig()
    
    # Input file
    while True:
        path = input("Input file (.csv/.parquet) path: ").strip()
        if Path(path).exists():
            config.input_path = path
            break
        print("  ✗ File not found, try again")
    
    df = read_table(Path(config.input_path))
    print(f"\nLoaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Label column
    print(f"\nColumns: {', '.join(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}")
    while True:
        col = input("Label column name: ").strip()
        if col in df.columns:
            # Show preview of the column
            print(f"\nPreview of '{col}' values:")
            print(df[col].head(10).to_string())
            
            # For classification, show value counts
            if hasattr(config, 'task') or input("Task type (classification/regression) [classification]: ").strip().lower() or "classification" == "classification":
                print(f"\nValue counts for '{col}':")
                print(df[col].value_counts().head())
            
            confirm = input(f"\nIs '{col}' the correct label column? (y/n): ").strip().lower()
            if confirm == 'y':
                config.label_column = col
                break
            else:
                print("Please select a different column.")
        else:
            print(f"  ✗ Column '{col}' not found")
    
    # Task type
    while True:
        task = input("Task type (classification/regression) [classification]: ").strip().lower() or "classification"
        if task in ("classification", "regression"):
            config.task = task
            break
        print("  ✗ Must be 'classification' or 'regression'")
    
    # Feature filtering
    fp_cols = select_fp_columns(df)
    if fp_cols:
        print(f"\nDetected {len(fp_cols)} fingerprint columns")
        config.variance_threshold = float(input(f"Variance threshold for feature filtering [0.01]: ").strip() or "0.01")
        filtered_cols = filter_low_variance_features(df, fp_cols, config.variance_threshold)
        print(f"  → Keeping {len(filtered_cols)} features after filtering")
    else:
        print("  ⚠ No fingerprint columns detected!")
        sys.exit(1)
    
    # Data preparation
    X = df[filtered_cols].to_numpy(dtype=float)
    y_raw = df[config.label_column].to_numpy()
    
    # Process labels
    if config.task == "classification":
        y_series = pd.Series(y_raw).astype(str).str.strip().str.lower()
        mapper = {"1":1, "0":0, "true":1, "false":0, "active":1, "inactive":0}
        
        # Check if values can be mapped
        unique_vals = set(y_series.unique())
        expected_vals = set(mapper.keys())
        
        if not unique_vals.issubset(expected_vals):
            print(f"\n  ⚠ Found values not in expected mapping: {list(unique_vals - expected_vals)[:5]}")
            print(f"    Trying direct integer conversion...")
            try:
                y = pd.Series(y_raw).astype(int).to_numpy()
                # Verify binary
                unique_ints = np.unique(y)
                if not set(unique_ints).issubset({0, 1}):
                    print(f"    ✗ Converted values are not binary: {unique_ints}")
                    print(f"    Please ensure your label column contains 0/1 or Active/Inactive values.")
                    sys.exit(1)
                print(f"    ✓ Successfully converted to binary labels: {np.unique(y)}")
            except Exception as e:
                print(f"\n  ✗ Error: Cannot convert '{config.label_column}' to binary labels.")
                print(f"    Unique values in column: {list(unique_vals)[:10]}{'...' if len(unique_vals) > 10 else ''}")
                print(f"    Expected values: {list(expected_vals)}")
                print(f"    Error details: {str(e)}")
                print("\n  Please select a different label column and restart.")
                sys.exit(1)
        else:
            y = y_series.map(mapper).astype(int).to_numpy()
    else:
        # Regression: convert to numeric
        try:
            y = pd.to_numeric(y_raw, errors="coerce").astype(float).to_numpy()
            # Check for NaN
            nan_count = np.isnan(y).sum()
            if nan_count > 0:
                print(f"\n  ⚠ Warning: {nan_count} values could not be converted to numbers and will be removed.")
                # Remove NaN rows
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                print(f"    Remaining samples: {len(y)}")
        except Exception as e:
            print(f"\n  ✗ Error: Cannot convert '{config.label_column}' to numeric values.")
            print(f"    Please ensure your label column contains numeric values for regression.")
            sys.exit(1)
    
    # Validate we have data left
    if len(y) == 0:
        print("  ✗ No valid samples remain after processing. Exiting.")
        sys.exit(1)
    
    # Parameters
    config.folds = int(input(f"Number of CV folds [5]: ").strip() or "5")
    config.seed = int(input(f"Random seed [42]: ").strip() or "42")
    config.test_size = float(input(f"Test size fraction [0.2]: ").strip() or "0.2")
    config.tune = (input(f"Hyperparameter tuning? (y/n) [n]: ").strip().lower() or "n") == "y"
    if config.tune:
        config.tune_iter = int(input(f"RandomizedSearch iterations [25]: ").strip() or "25")
    config.max_epochs = int(input(f"Max epochs for deep models [100]: ").strip() or "100")
    
    # Regression-specific options
    if config.task == "regression":
        config.use_bootstrap_lasso = (input("Use bootstrapping for Lasso? (y/n) [n]: ").strip().lower() or "n") == "y"
        if config.use_bootstrap_lasso:
            config.boot_iters = int(input("Bootstrap iterations [50]: ").strip() or "50")
            config.boot_frac = float(input("Bootstrap sample fraction [0.8]: ").strip() or "0.8")
        
        config.roc_from_threshold = (input("Generate ROC/PR via label threshold? (y/n) [n]: ").strip().lower() or "n") == "y"
        if config.roc_from_threshold:
            config.roc_threshold = float(input("Threshold on label [5.0]: ").strip() or "5.0")
    
    # Model selection
    registry = build_registry(config.task, input_dim=X.shape[1])
    print("\nSelect models by number (comma-separated):")
    for i, (key, label, ctor, params, grid) in enumerate(registry, start=1):
        tags = []
        if "Keras" in key:
            tags.append("DL-Keras")
        if "Skorch" in key:
            tags.append("DL-PyTorch")
        mark = (" [" + ",".join(tags) + "]") if tags else ""
        tune_mark = " *" if grid else ""
        print(f"  {i}. {label} ({key}){mark}{tune_mark}")
    
    while True:
        sel = input("Your selection (e.g., 1,3,5): ").strip()
        try:
            idxs = [int(s) for s in sel.replace(" ", "").split(",") if s.strip().isdigit()]
            idxs = [i for i in idxs if 1 <= i <= len(registry)]
            if idxs:
                config.selected_models = idxs
                break
        except:
            pass
        print("  ✗ Invalid selection")
    
    # Output directory
    default_out = str(Path(config.input_path).parent / "models_out")
    config.output_dir = input(f"Output directory [{default_out}]: ").strip() or default_out
    
    # Parallel jobs
    config.n_jobs = int(input(f"Number of parallel jobs (-1 for all) [-1]: ").strip() or "-1")
    
    # SHAP analysis
    config.run_shap = (input("Run SHAP analysis on best model? (y/n) [y]: ").strip().lower() or "y") == "y"
    
    # Ensemble
    config.create_ensemble = (input("Create ensemble of top models? (y/n) [n]: ").strip().lower() or "n") == "y"
    if config.create_ensemble:
        config.ensemble_n_top = int(input("Number of models for ensemble [3]: ").strip() or "3")
    
    print(f"\n{'='*60}")
    print("Configuration complete. Starting pipeline...")
    print(f"{'='*60}")
    
    return config, X, y, df[filtered_cols]

# --------- Main Pipeline ---------
def main(config: QSARConfig, X: np.ndarray, y: np.ndarray, feature_df: pd.DataFrame, logger: logging.Logger = None):
    """Main execution pipeline"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Setup output directory
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw configuration
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    registry = build_registry(config.task, input_dim=X.shape[1])
    selected_models = [(i-1, registry[i-1]) for i in config.selected_models]
    
    logger.info(f"\nTraining {len(selected_models)} models...")
    logger.info(f"Task: {config.task} | Samples: {len(y)} | Features: {X.shape[1]}")
    
    # Parallel model training
    if config.n_jobs != 1 and len(selected_models) > 1:
        logger.info(f"Running parallel training with {config.n_jobs} jobs")
        from joblib import Parallel, delayed
        
        args_list = []
        for idx in config.selected_models:
            args = (idx, registry, X, y, out_dir, config.task, config, logger)
            args_list.append(args)
        
        results = Parallel(n_jobs=config.n_jobs, verbose=10)(
            delayed(train_model_parallel)(args) for args in args_list
        )
    else:
        # Sequential training
        results = []
        for i in tqdm(config.selected_models, desc="Training models"):
            args = (i, registry, X, y, out_dir, config.task, config, logger)
            results.append(train_model_parallel(args))
    
    # Filter out failed models
    results = [r for r in results if "error" not in r]
    
    if not results:
        logger.error("All models failed to train!")
        return
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_csv = out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    logger.info(f"\nSaved summary: {summary_csv}")
    
    # Select best model
    best_idx = None
    if config.task == "classification":
        if "AUC_te" in summary_df.columns:
            best_idx = int(summary_df["AUC_te"].astype(float).idxmax())
            logger.info(f"\nBest model: {summary_df.iloc[best_idx]['model']} (AUC_te={summary_df.iloc[best_idx]['AUC_te']:.3f})")
    else:
        if "R2_te" in summary_df.columns:
            best_idx = int(summary_df["R2_te"].astype(float).idxmax())
            logger.info(f"\nBest model: {summary_df.iloc[best_idx]['model']} (R²_te={summary_df.iloc[best_idx]['R2_te']:.3f})")
    
    # SHAP analysis for best model
    shap_results = None
    if best_idx is not None and config.run_shap:
        best_row = summary_df.iloc[best_idx]
        best_model_key = str(best_row["model"])
        
        # Load best model
        best_model = None
        for model_path in str(best_row.get("model_files","")).split(";"):
            path = Path(model_path.strip())
            if path.exists() and path.suffix == ".joblib":
                try:
                    import joblib
                    best_model = joblib.load(path)
                    break
                except Exception as e:
                    logger.warning(f"Could not load {path}: {e}")
        
        if best_model is not None:
            # Prepare test data for SHAP
            from sklearn.model_selection import train_test_split
            if config.task == "classification":
                _, X_te, _, _ = train_test_split(feature_df, y, test_size=config.test_size, 
                                               random_state=config.seed, stratify=y)
            else:
                _, X_te, _, _ = train_test_split(feature_df, y, test_size=config.test_size, 
                                               random_state=config.seed)
            
            shap_results = analyze_shap_best_model(
                model=best_model,
                X_train=feature_df,
                X_test=X_te,
                out_dir=out_dir,
                task=config.task,
                model_key=best_model_key,
                max_display=config.shap_max_display,
                sample_size=config.shap_sample_size,
                memory_efficient=True,
                logger=logger
            )
        else:
            logger.warning("Could not load best model for SHAP analysis")
    
    # FIXED: Create ensemble - pass out_dir parameter
    ensemble_model = None
    if config.create_ensemble:
        ensemble_model = create_ensemble_model(summary_df, X, y, config.task, config, logger, out_dir)
    
    # Generate manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "task": config.task,
        "label_column": config.label_column,
        "config": config.__dict__,
        "best_model_idx": best_idx,
        "best_model": summary_df.iloc[best_idx].to_dict() if best_idx is not None else None,
        "shap_analysis": shap_results,
        "ensemble_created": ensemble_model is not None
    }
    
    # Copy best model files
    if best_idx is not None:
        best_row = summary_df.iloc[best_idx]
        best_model_key = str(best_row["model"])
        bm_dir = out_dir / "best_model"
        bm_dir.mkdir(parents=True, exist_ok=True)
        
        copied = []
        for p in str(best_row.get("model_files","")).split(";"):
            p = p.strip()
            if p:
                src = Path(p)
                if src.exists():
                    dst = bm_dir / src.name
                    try:
                        shutil.copy2(src, dst)
                        copied.append(str(dst))
                    except Exception as e:
                        logger.warning(f"Failed to copy {src}: {e}")
        
        manifest["best_model_files"] = copied
        
        # Write BEST_MODEL.txt
        with open(out_dir / "BEST_MODEL.txt", "w", encoding="utf-8") as f:
            f.write(f"Best model: {best_model_key}\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if config.task == "classification":
                f.write(f"AUC_te={best_row.get('AUC_te','N/A')}, AP_te={best_row.get('AP_te','N/A')}\n")
                f.write(f"ACC_te={best_row.get('ACC_te','N/A')}, F1_te={best_row.get('F1_te','N/A')}\n")
            else:
                f.write(f"R²_te={best_row.get('R2_te','N/A')}, RMSE_te={best_row.get('RMSE_te','N/A')}\n")
                f.write(f"MAE_te={best_row.get('MAE_te','N/A')}\n")
            f.write("\nFiles in best_model/:\n")
            for cp in copied:
                f.write(f"  - {cp}\n")
        
        logger.info(f"\nBest model files copied to: {bm_dir}")
    
    # Save manifest
    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline completed successfully!")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Summary metrics: {summary_csv}")
    if manifest.get("best_model"):
        logger.info(f"Best model: {manifest['best_model']['model']}")
    if shap_results:
        logger.info(f"SHAP analysis: {out_dir}/shap_analysis/")
    logger.info(f"{'='*60}")

# --------- Entry Points ---------
def main_cli():
    """Command-line interface with config file support"""
    parser = argparse.ArgumentParser(description="QSAR Batch Modeling with SHAP")
    parser.add_argument("-c", "--config", help="Path to JSON/YAML config file")
    parser.add_argument("-i", "--input", help="Input file path")
    parser.add_argument("-l", "--label", help="Label column name")
    parser.add_argument("-t", "--task", choices=["classification", "regression"], help="Task type")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--ensemble", action="store_true", help="Create ensemble")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    if args.config:
        # Load from config file
        config_path = Path(args.config)
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            config = QSARConfig.from_yaml(config_path)
        else:
            config = QSARConfig.from_json(config_path)
        
        # Override with CLI args
        if args.input: config.input_path = args.input
        if args.label: config.label_column = args.label
        if args.task: config.task = args.task
        if args.output: config.output_dir = args.output
        if args.jobs: config.n_jobs = args.jobs
        if args.shap: config.run_shap = True
        if args.ensemble: config.create_ensemble = True
        
        # Load data
        df = read_table(Path(config.input_path))
        
        # Filter features
        fp_cols = select_fp_columns(df)
        if config.variance_threshold > 0:
            filtered_cols = filter_low_variance_features(df, fp_cols, config.variance_threshold)
        else:
            filtered_cols = fp_cols
        
        X = df[filtered_cols].to_numpy(dtype=float)
        
        # Process labels
        if config.task == "classification":
            y_series = pd.Series(df[config.label_column]).astype(str).str.strip().str.lower()
            mapper = {"1":1, "0":0, "true":1, "false":0, "active":1, "inactive":0}
            if set(y_series.unique()) - set(mapper.keys()):
                y = pd.Series(df[config.label_column]).astype(int).to_numpy()
            else:
                y = y_series.map(mapper).astype(int).to_numpy()
        else:
            y = pd.to_numeric(df[config.label_column], errors="coerce").astype(float).to_numpy()
        
        # Setup logging
        logger = setup_logging(Path(config.output_dir), args.log_level)
        
        # Run pipeline
        main(config, X, y, df[filtered_cols], logger)
    
    else:
        # Interactive mode
        result = main_interactive()
        if result is None:
            sys.exit(1)
        config, X, y, feature_df = result
        logger = setup_logging(Path(config.output_dir))
        main(config, X, y, feature_df, logger)

if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        main_cli()
    else:
        # Interactive mode
        result = main_interactive()
        if result is None:
            sys.exit(1)
        config, X, y, feature_df = result
        logger = setup_logging(Path(config.output_dir))
        main(config, X, y, feature_df, logger)