#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.

filter_fingerprints.py
Filter and clean fingerprint matrices before modeling while preserving key metadata columns.

Features
- Reads CSV or Parquet produced by generate_fingerprints.py.
- Preserves requested metadata columns (ID/labels) and filters only fingerprint columns.
- Filters:
  * Constant/zero-variance bits (default on).
  * Low-variance bits (user threshold).
  * Rare bits by nonzero rate (e.g., keep bits with >= 1% ones).
  * Correlation pruning among fingerprint columns (Pearson).
- Reports kept/dropped columns and basic stats.

Fingerprint columns are auto-detected as numeric columns with names starting with:
  morgan_, maccs_, rdkit_, atompair_, torsion_

Usage examples
--------------
# Minimal: drop constants and save
python filter_fingerprints.py \
  --in "CHEMBL-Name_morgan2048.csv" \
  --out "CHEMBL-Name_morgan2048_filtered.csv"

# With metadata columns and pruning
python filter_fingerprints.py \
  --in "CHEMBL-Name_morgan2048.csv" \
  --out "CHEMBL-Name_morgan2048_filtered.csv" \
  --keep-cols "Molecule ChEMBL ID,Smiles,Standard Value,Standard Type,Standard Units,Standard Relation" \
  --rare-min-rate 0.01 \
  --corr-threshold 0.95

# Parquet I/O
python filter_fingerprints.py \
  --in "fps.parquet" --out "fps_filtered.parquet" --format parquet

Requirements: pandas, numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

FP_PREFIXES = ("morgan_", "maccs_", "rdkit_", "atompair_", "torsion_")

def parse_list(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [c.strip() for c in arg.split(",") if c.strip() != ""]

def read_table(path: Path, fmt: Optional[str]) -> pd.DataFrame:
    if fmt is None:
        fmt = path.suffix.lower().lstrip(".") if path.suffix else "csv"
    if fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "parquet":
        return pd.read_parquet(path)
    else:
        sys.exit("[Error] Unsupported input format. Use csv or parquet.")

def write_table(df: pd.DataFrame, path: Path, fmt: Optional[str]) -> None:
    if fmt is None:
        fmt = path.suffix.lower().lstrip(".") if path.suffix else "csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        sys.exit("[Error] Unsupported output format. Use csv or parquet.")

def select_fp_columns(df: pd.DataFrame) -> List[str]:
    fp_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            name = str(c)
            if name.startswith(FP_PREFIXES):
                fp_cols.append(c)
    return fp_cols

def drop_constant_and_low_variance(df_fp: pd.DataFrame, var_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    # variance on numeric columns
    variances = df_fp.var(axis=0, ddof=0)
    keep_mask = variances > float(var_threshold)
    kept_cols = variances.index[keep_mask].tolist()
    dropped = variances.index[~keep_mask].tolist()
    return df_fp[kept_cols], dropped

def drop_rare_bits(df_fp: pd.DataFrame, min_rate: float) -> Tuple[pd.DataFrame, List[str]]:
    if min_rate <= 0.0:
        return df_fp, []
    n = float(len(df_fp))
    # consider "nonzero" as present bit; for binary fingerprints sum gives frequency of 1s
    rates = (df_fp != 0).sum(axis=0) / n
    keep_mask = rates >= float(min_rate)
    kept_cols = rates.index[keep_mask].tolist()
    dropped = rates.index[~keep_mask].tolist()
    return df_fp[kept_cols], dropped

def corr_prune(df_fp: pd.DataFrame, threshold: float, sample_rows: Optional[int]) -> Tuple[pd.DataFrame, List[str]]:
    if threshold is None or threshold <= 0.0 or threshold >= 1.0 or df_fp.shape[1] == 0:
        return df_fp, []

    if sample_rows is not None and sample_rows > 0 and sample_rows < df_fp.shape[0]:
        sample_idx = np.random.RandomState(42).choice(df_fp.index.values, size=sample_rows, replace=False)
        X = df_fp.loc[sample_idx]
    else:
        X = df_fp

    # Compute absolute correlation matrix
    corr = X.corr().abs()
    # Upper triangle mask
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        # drop columns that have any correlation above threshold with a kept column
        if col in to_drop:
            continue
        high_corr = upper[col] > threshold
        drop_cols = upper.index[high_corr].tolist()
        for d in drop_cols:
            if d not in to_drop:
                to_drop.add(d)

    kept_cols = [c for c in df_fp.columns if c not in to_drop]
    dropped_cols = [c for c in df_fp.columns if c in to_drop]
    return df_fp[kept_cols], dropped_cols

def main():
    p = argparse.ArgumentParser(description="Filter fingerprint matrices for QSAR modeling.")
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV/Parquet path")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV/Parquet path")
    p.add_argument("--format", choices=["csv", "parquet"], default=None, help="Output format (inferred if omitted)")
    p.add_argument("--keep-cols", default="Molecule ChEMBL ID,Smiles,Standard Value,Standard Type,Standard Units,Standard Relation",
                   help="Comma-separated metadata columns to keep (kept in front if present)")
    p.add_argument("--var-threshold", type=float, default=0.0, help="Drop columns with variance <= threshold (default 0.0 removes constants)")
    p.add_argument("--rare-min-rate", type=float, default=0.0, help="Drop columns with nonzero rate < value (e.g., 0.01 keeps bits present in >=1% samples)")
    p.add_argument("--corr-threshold", type=float, default=0.0, help="Prune columns with Pearson |r| > threshold; keep first, drop later (e.g., 0.95)")
    p.add_argument("--corr-sample-rows", type=int, default=0, help="If >0, sample this many rows for correlation to save memory")
    p.add_argument("--report", default=None, help="Optional JSON report path")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    df = read_table(in_path, fmt=None)
    keep_cols = parse_list(args.keep_cols) or []

    # Identify fingerprint columns
    fp_cols = select_fp_columns(df)
    if not fp_cols:
        sys.exit("[Error] No fingerprint columns detected. Ensure column names start with known prefixes and are numeric.")

    # Split metadata vs fingerprints
    meta_cols_present = [c for c in keep_cols if c in df.columns]
    other_meta_cols = [c for c in df.columns if c not in fp_cols and c not in meta_cols_present]
    df_fp = df[fp_cols].copy()

    dropped: Dict[str, List[str]] = {}

    # 1) Variance filter
    df_fp, dropped_const = drop_constant_and_low_variance(df_fp, args.var_threshold)
    if dropped_const:
        dropped["low_variance_or_constant"] = dropped_const

    # 2) Rare bits filter
    df_fp, dropped_rare = drop_rare_bits(df_fp, args.rare_min_rate)
    if dropped_rare:
        dropped["rare_bits"] = dropped_rare

    # 3) Correlation pruning
    corr_sample = args.corr_sample_rows if args.corr_sample_rows and args.corr_sample_rows > 0 else None
    df_fp, dropped_corr = corr_prune(df_fp, args.corr_threshold, corr_sample)
    if dropped_corr:
        dropped["highly_correlated"] = dropped_corr

    # Reassemble dataframe: metadata (requested first), then remaining metadata, then filtered fps
    out_cols = meta_cols_present + other_meta_cols + df_fp.columns.tolist()
    df_out = df[out_cols].copy()
    df_out[df_fp.columns] = df_fp  # ensure post-filter values

    # Write
    write_table(df_out, out_path, fmt=args.format)

    report = {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "n_rows": int(len(df_out)),
        "n_fp_cols_in": int(len(fp_cols)),
        "n_fp_cols_out": int(df_fp.shape[1]),
        "kept_metadata_cols": meta_cols_present,
        "dropped": dropped,
        "var_threshold": float(args.var_threshold),
        "rare_min_rate": float(args.rare_min_rate),
        "corr_threshold": float(args.corr_threshold),
        "corr_sample_rows": int(args.corr_sample_rows),
        "fp_prefixes": FP_PREFIXES,
    }

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
