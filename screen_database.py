#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.

screen_database.py
Interactive screening using a saved QSAR model on database fingerprints.
Includes feature alignment to avoid "Feature shape mismatch" errors.

Key Features:
- Recover expected training feature names from the model where possible.
- If not available, allow loading a feature-name file (txt one-per-line or JSON list).
- As a last resort, offer an "unsafe" auto-align: take the first N FP columns in sorted order.

Outputs:
  predictions.csv (+parquet), topK.csv, score_hist.{png,svg}, screen_manifest.json

Compatible with scikit-learn / XGBoost / LightGBM models saved by the QSAR modeling script.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Headless + style (publication) ---
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FP_PREFIXES = ("morgan_", "maccs_", "rdkit_", "atompair_", "torsion_")
ID_CANDIDATES = ["ZINC ID", "ZINC_ID", "ZINCID", "Molecule ChEMBL ID", "Molecule_ID", "ID"]
SMILES_CANDIDATES = ["Smiles", "SMILES", "smiles"]

# --------- small utils ---------
def prompt(msg: str, default: str = "") -> str:
    s = input(f"{msg} " + (f"[{default}] " if default else ""))
    return s.strip() or default

def yesno(msg: str, default: str = "y") -> bool:
    d = default.lower().strip()
    s = input(f"{msg} [{'Y/n' if d=='y' else 'y/N'}]: ").strip().lower()
    if not s:
        s = d
    return s.startswith("y")

def read_table(path: Path, sep: str = ",") -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".csv", ".txt", ".tsv"):
        if ext == ".tsv" or sep == "\t":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path, sep=sep)
    raise SystemExit("Database file must be .parquet or .csv/.tsv")

def select_fp_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and str(c).startswith(FP_PREFIXES):
            cols.append(c)
    if not cols:
        raise SystemExit("No fingerprint columns found (expected prefixes: "
                         + ", ".join(FP_PREFIXES) + ").")
    return cols

def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _save_fig(fig, out_noext: Path):
    out_noext.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_noext.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_noext.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

# --------- model loading ---------
def load_best_or_pick_model(models_dir: Path) -> Tuple[object, dict]:
    """
    Returns (model, manifest_info).
    manifest_info includes: task ("classification" or "regression"), best_model (if available).
    """
    manifest = {}
    run_manifest = models_dir / "run_manifest.json"
    if run_manifest.exists():
        try:
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    # First try best_model files from manifest
    candidates = []
    task = manifest.get("task", None)
    best_files = manifest.get("best_model_files", []) or []
    if best_files:
        for p in best_files:
            candidates.append(Path(p))
    # Also look for generic saved files in models_dir
    for p in models_dir.glob("model_*.*"):
        # Only allow formats used by the QSAR script: joblib, XGBoost JSON, LightGBM text
        if p.suffix.lower() in (".joblib", ".json", ".txt"):
            candidates.append(p)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        q = p.resolve()
        if q not in seen and q.exists():
            uniq.append(q)
            seen.add(q)

    print("\nModel files found:")
    if not uniq:
        raise SystemExit("No model files found in models directory.")
    for i, p in enumerate(uniq, start=1):
        print(f"  {i}. {p.name}")

    use_best = "n"
    if best_files:
        print("\nBest model from manifest detected.")
        use_best = prompt("Use best model from manifest? (y/n)", "y").lower()
    if use_best == "y" and best_files:
        chosen = Path(best_files[0]).resolve()
    else:
        idx = int(prompt("Select a model by number", "1"))
        idx = max(1, min(idx, len(uniq)))
        chosen = uniq[idx-1]

    # Try loading based on extension (prefer joblib)
    model = None
    err_msgs = []

    # 1) joblib
    if chosen.suffix.lower() == ".joblib":
        try:
            import joblib
            model = joblib.load(chosen)
            print(f"[Load] joblib loaded: {chosen.name}")
        except Exception as e:
            err_msgs.append(f"joblib load failed: {e}")

    # 2) XGBoost JSON
    if model is None and chosen.suffix.lower() == ".json":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if task == "classification":
                model = XGBClassifier()
            else:
                model = XGBRegressor()
            model.load_model(str(chosen))
            print(f"[Load] XGBoost JSON loaded: {chosen.name}")
        except Exception as e:
            err_msgs.append(f"xgboost load failed: {e}")

    # 3) LightGBM text model
    if model is None and chosen.suffix.lower() == ".txt":
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(chosen))
            class _LGBWrap:
                def __init__(self, booster):
                    self.booster_ = booster
                def predict(self, X):
                    return self.booster_.predict(X)
                def predict_proba(self, X):
                    y = self.booster_.predict(X)
                    if y.ndim == 1:
                        y = np.clip(y, 1e-7, 1-1e-7)
                        return np.vstack([1-y, y]).T
                    return y
            model = _LGBWrap(booster)
            print(f"[Load] LightGBM text model loaded: {chosen.name}")
        except Exception as e:
            err_msgs.append(f"lightgbm load failed: {e}")

    if model is None:
        raise SystemExit("Failed to load the chosen model.\n" + "\n".join(err_msgs))

    # Ensure task in manifest
    if not task:
        task = prompt("Task type (classification/regression)", "classification").lower()
        manifest["task"] = task

    return model, {"task": task, "chosen_file": str(chosen), "manifest": manifest}

# --------- expected features helpers ---------
def _expected_names_from_model(model) -> Optional[List[str]]:
    # sklearn-style
    if hasattr(model, "feature_names_in_"):
        try:
            return [str(x) for x in list(model.feature_names_in_)]
        except Exception:
            pass
    # xgboost sklearn API
    try:
        booster = model.get_booster()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    # lightgbm sklearn API
    try:
        if hasattr(model, "booster_") and model.booster_ is not None:
            names = model.booster_.feature_name()
            if names:
                return list(names)
    except Exception:
        pass
    # skorch / keras wrappers saved via joblib may carry feature_names_in_
    try:
        inner = getattr(model, "module_", None)
        if inner is None:
            inner = getattr(model, "model_", None)
        if inner is not None and hasattr(inner, "feature_names_in_"):
            return [str(x) for x in list(inner.feature_names_in_)]
    except Exception:
        pass
    return None

def _n_features_from_model(model) -> Optional[int]:
    for attr in ("n_features_in_",):
        if hasattr(model, attr):
            try:
                n = int(getattr(model, attr))
                if n > 0:
                    return n
            except Exception:
                pass
    # xgboost booster
    try:
        booster = model.get_booster()
        if booster is not None and hasattr(booster, "num_features"):
            n = int(booster.num_features())
            if n > 0:
                return n
    except Exception:
        pass
    # lightgbm booster
    try:
        if hasattr(model, "booster_") and model.booster_ is not None:
            names = model.booster_.feature_name()
            if names:
                return int(len(names))
    except Exception:
        pass
    return None

def _load_feature_list_file(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Feature list file not found: {path}")
    if path.suffix.lower() in (".json",):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, (list, tuple)):
            raise SystemExit("JSON feature file must be a list of column names.")
        return [str(x) for x in data]
    # plain text: one name per line, ignore empty/comment
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    if not names:
        raise SystemExit("No names found in feature list file.")
    return names

def align_feature_matrix(df: pd.DataFrame, model) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (X, used_columns) aligned to model's expected features.
    Strategy:
      1) Try to read names from model; if found, reindex df to those names (fill missing with 0).
      2) Else ask user for a feature-name file (txt/JSON).
      3) Else if still not available, infer N from model and offer unsafe auto-align:
         take the first N FP columns in sorted name order.
    """
    # all FP columns available in DB
    fp_in_db = select_fp_columns(df)
    expected = _expected_names_from_model(model)

    if expected:
        exp = list(expected)
        missing = [c for c in exp if c not in df.columns]
        extra = [c for c in fp_in_db if c not in exp]
        print(f"[Align] Found expected feature names in model: {len(exp)} columns.")
        if missing:
            print(f"[Align] Missing in DB: {len(missing)} columns (will fill 0). Example: {missing[:5]}")
        if extra:
            print(f"[Align] Extra in DB: {len(extra)} columns (will drop). Example: {extra[:5]}")
        X = df.reindex(columns=exp, fill_value=0).to_numpy(dtype=float, copy=False)
        return X, exp

    # no names found on model
    print("[Align] Model does not expose feature names.")
    # try optional file near model directory
    auto_file = None
    # search common names in current directory and model dir
    for candidate in ("feature_names.txt", "feature_names.json", "train_fp_columns.json"):
        p = Path(candidate)
        if p.exists():
            auto_file = p
            break
    if auto_file is None:
        # ask user for a path
        ans = prompt("Provide path to feature-name file (txt one-per-line or JSON list), or leave blank to skip", "")
        if ans:
            auto_file = Path(ans)

    if auto_file:
        names = _load_feature_list_file(auto_file)
        missing = [c for c in names if c not in df.columns]
        extra = [c for c in fp_in_db if c not in names]
        print(f"[Align] Loaded names from file: {len(names)}.")
        if missing:
            print(f"[Align] Missing in DB: {len(missing)} (fill 0). Example: {missing[:5]}")
        if extra:
            print(f"[Align] Extra in DB: {len(extra)} (drop). Example: {extra[:5]}")
        X = df.reindex(columns=names, fill_value=0).to_numpy(dtype=float, copy=False)
        return X, names

    # last resort: unsafe auto align by count
    n_expected = _n_features_from_model(model)
    if n_expected is None:
        raise SystemExit("Cannot determine expected feature count. Provide a feature-name file.")
    print(f"[Align] Expected feature count from model: {n_expected}.")
    print("[Align] WARNING: No feature names available; will select the first N fingerprint columns in sorted order.")
    if not yesno("Proceed with unsafe auto-align?", "n"):
        raise SystemExit("Aborted. Please re-run with a feature-name file.")

    fp_sorted = sorted(fp_in_db)
    if len(fp_sorted) < n_expected:
        raise SystemExit(f"Database has only {len(fp_sorted)} FP columns, but model expects {n_expected}.")
    use_cols = fp_sorted[:n_expected]
    X = df[use_cols].to_numpy(dtype=float, copy=False)
    return X, use_cols

# --------- prediction helpers ---------
def predict_scores(model, X: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            if isinstance(p, np.ndarray):
                if p.ndim == 2 and p.shape[1] >= 2:
                    return p[:, 1]
                return np.ravel(p)
        if hasattr(model, "decision_function"):
            df = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-df))
        yhat = model.predict(X)
        return np.ravel(yhat).astype(float)
    else:
        yhat = model.predict(X)
        return np.ravel(yhat).astype(float)

def score_hist(scores: np.ndarray, task: str, out_dir: Path, thr: Optional[float] = None):
    fig = plt.figure(figsize=(7.5, 4.5))
    plt.hist(scores, bins=60, alpha=0.9)
    if thr is not None:
        plt.axvline(thr, linestyle="--", linewidth=2, label=f"Threshold = {thr:g}")
        plt.legend(frameon=False)
    plt.xlabel("Probability" if task == "classification" else "Predicted value")
    plt.ylabel("Count")
    ttl = "Score distribution" if task == "classification" else "Prediction distribution"
    plt.title(ttl)
    _save_fig(fig, out_dir / "score_hist")

# --------- main flow ---------
def main():
    print("=== QSAR Screening (Interactive) ===")

    # Models folder
    models_dir = Path(prompt("Models directory", "models_out")).resolve()
    if not models_dir.exists():
        raise SystemExit(f"Models directory not found: {models_dir}")

    model, meta = load_best_or_pick_model(models_dir)
    task = (meta.get("manifest") or {}).get("task") or meta.get("task") or prompt(
        "Task type (classification/regression)", "classification"
    ).lower()
    if task not in ("classification", "regression"):
        raise SystemExit("Task must be 'classification' or 'regression'.")

    # Database fingerprints file
    db_path = Path(prompt("Database fingerprints file (.parquet or .csv)", "database_fingerprints.parquet")).resolve()
    if not db_path.exists():
        raise SystemExit(f"Database file not found: {db_path}")

    sep = ","
    if db_path.suffix.lower() not in (".parquet",):
        sep = prompt("CSV separator", ",")

    out_dir = Path(prompt("Output directory", str(db_path.parent / "screen_out"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Batch and top-K
    batch_rows = int(prompt("Batch size (rows per batch)", "100000"))
    topk = int(prompt("Top-K to save", "1000"))

    # Thresholds (optional)
    prob_thr = None
    reg_thr = None
    if task == "classification":
        prob_thr = float(prompt("Probability threshold for Active (optional)", "0.5") or "0.5")
    else:
        ans = prompt("Add threshold line on regression plot (y/n)", "n").lower()
        if ans == "y":
            reg_thr = float(prompt("Regression threshold value", "5.0"))

    # Load DB
    df = read_table(db_path, sep=sep)
    id_col = find_first_col(df, ID_CANDIDATES)
    smi_col = find_first_col(df, SMILES_CANDIDATES)

    # Align features ONCE up front
    X_all, used_cols = align_feature_matrix(df, model)
    n = len(df)
    print(f"[Align] Using {len(used_cols)} features for prediction.")

    # Predict in batches to avoid OOM
    print(f"\nScoring {n} molecules in batches of {batch_rows} ...")
    all_scores = np.zeros(n, dtype=float)
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        X = X_all[start:end]
        scores = predict_scores(model, X, task)
        if scores.shape[0] != (end - start):
            raise SystemExit("Model returned unexpected number of predictions.")
        all_scores[start:end] = scores
        print(f"  Done rows {start}..{end-1}")

    # Compose output frame (keep ID/SMILES if present)
    out = pd.DataFrame({"score": all_scores})
    if id_col:
        out.insert(0, id_col, df[id_col].values)
    if smi_col:
        out.insert(1 if id_col else 0, smi_col, df[smi_col].values)

    # For classification also flag Active
    if task == "classification" and prob_thr is not None:
        out["Active_pred"] = (out["score"] >= prob_thr).astype(int)

    # Save outputs
    out_csv = out_dir / "predictions.csv"
    out_parquet = out_dir / "predictions.parquet"
    out.to_csv(out_csv, index=False)
    try:
        out.to_parquet(out_parquet, index=False)
    except Exception:
        pass

    # Top-K
    idx_sorted = np.argsort(-all_scores)  # descending
    k = min(topk, n)
    top = out.iloc[idx_sorted[:k]].copy()
    top.to_csv(out_dir / "topK.csv", index=False)

    # Figures
    thr = prob_thr if task == "classification" else reg_thr
    score_hist(all_scores, task, out_dir, thr=thr)

    # Write a small manifest
    run_info = {
        "model_file": meta.get("chosen_file"),
        "task": task,
        "db_file": str(db_path),
        "n_scored": int(n),
        "batch_rows": int(batch_rows),
        "topK": int(k),
        "id_col": id_col,
        "smiles_col": smi_col,
        "fp_cols_used": used_cols,
        "prob_threshold": float(prob_thr) if prob_thr is not None else None,
        "reg_threshold": float(reg_thr) if reg_thr is not None else None,
        "outputs": {
            "predictions_csv": str(out_csv),
            "predictions_parquet": str(out_parquet),
            "topK_csv": str(out_dir / "topK.csv"),
            "score_hist_png": str(out_dir / "score_hist.png"),
            "score_hist_svg": str(out_dir / "score_hist.svg"),
        },
    }
    with open(out_dir / "screen_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("\nDone. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
