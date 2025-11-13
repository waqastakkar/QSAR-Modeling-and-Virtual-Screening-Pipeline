#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). Structural and Dynamic Profiling of SARS-CoV-2 Papain-Like Protease Bound to Machine-Learning-Identified Oxadiazole Inhibitors.

Data_cleaning.py  (CHEMBL-friendly, with pIC50 + Active label)

Functions
- Remove rows with missing/empty `Standard Value`.
- (Optional) Enforce units == nM (if `Standard Units` exists).
- Convert IC50 (nM) in `Standard Value` to pIC50 = 9 - log10(IC50_nM).
- Create binary `Active` using threshold on pIC50 (default 6.0).
- De-duplicate by `Smiles` (optionally canonicalize, remove salts).
- Keep only requested columns via --keep-cols, but pIC50 and Active are added automatically if created.
- Preserve input/output format (.xlsx/.csv/.tsv).

Usage (example)
--------------
python Data_cleaning.py \
  --in CHEMBL-Name.xlsx \
  --out CHEMBL-Name.xlsx \
  --keep-cols "Molecule ChEMBL ID,Smiles,Standard Type,Standard Relation,Standard Value,Standard Units" \
  --add-pic50 \
  --active-threshold 6.0 \
  --require-nm
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

def try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem.SaltRemover import SaltRemover
        return Chem, SaltRemover
    except Exception:
        return None, None

def canonicalize_smiles(smiles: str, Chem, remove_salts: bool, salt_remover) -> Optional[str]:
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        if remove_salts and salt_remover is not None:
            m = salt_remover.StripMol(m, dontRemoveEverything=True)
        can = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
        return can
    except Exception:
        return None

def read_table(path: Path, sep: str = ",") -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=sep, dtype=str)

def write_table(df: pd.DataFrame, path: Path) -> None:
    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def parse_keep_cols(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [c.strip() for c in arg.split(",") if c.strip()]

def compute_pic50_from_nm(value_nm: float) -> Optional[float]:
    try:
        v = float(value_nm)
        if not np.isfinite(v) or v <= 0.0:
            return None
        return 9.0 - math.log10(v)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(description="Remove duplicate molecules, compute pIC50 and Active label.")
    p.add_argument("--in", dest="in_path", required=True, help="Input file (.xlsx/.csv/.tsv)")
    p.add_argument("--out", dest="out_path", default=None, help="Output file; if omitted, appends _no_dup to input")
    p.add_argument("--smiles-col", default="Smiles", help="SMILES column name (default: Smiles)")
    p.add_argument("--keep", choices=["first","last"], default="first", help="Which duplicate to keep (default: first)")
    p.add_argument("--sep", default=",", help="Separator for CSV/TSV input (default , ; use '\\t' for TSV)")
    p.add_argument("--canonicalize", action="store_true", help="Canonicalize SMILES (RDKit) before de-dup")
    p.add_argument("--remove-salts", action="store_true", help="Remove salts before canonicalization")
    p.add_argument("--drop-na-smiles", action="store_true", help="Drop rows with missing/invalid SMILES")

    # New: derived fields
    p.add_argument("--add-pic50", action="store_true", help="Compute pIC50 from `Standard Value` (assumed nM)")
    p.add_argument("--active-threshold", type=float, default=None, help="If set (e.g., 6.0), create binary `Active` = 1[pIC50 >= threshold]")
    p.add_argument("--require-nm", action="store_true", help="Keep only rows where `Standard Units` == 'nM' (case-insensitive)")

    p.add_argument("--keep-cols", dest="keep_cols", default=None,
                   help="Comma-separated list of columns to keep in the final output")
    p.add_argument("--report", default=None, help="Optional JSON report path")
    args = p.parse_args()

    in_path = Path(args.in_path)
    df = read_table(in_path, sep=args.sep)
    n0 = len(df)

    smiles_col = args.smiles_col
    if smiles_col not in df.columns:
        sys.exit(f"[Error] Column '{smiles_col}' not found. Columns: {df.columns.tolist()}")

    # --- Filter: Standard Value must exist and be non-empty ---
    if "Standard Value" in df.columns:
        df = df[df["Standard Value"].notna() & (df["Standard Value"].astype(str).str.strip() != "")].copy()
    else:
        # If Standard Value is missing and user asked for pIC50/Active, stop.
        if args.add_pic50 or args.active_threshold is not None:
            sys.exit("[Error] 'Standard Value' column not found but pIC50/Active requested.")
    n_after_value_filter = len(df)

    # --- Optional: require units == nM ---
    if args.require_nm and "Standard Units" in df.columns:
        df = df[df["Standard Units"].astype(str).str.strip().str.lower() == "nm"].copy()

    # --- Numeric Standard Value ---
    if "Standard Value" in df.columns:
        sv = pd.to_numeric(df["Standard Value"], errors="coerce")
        df = df[sv.notna() & (sv > 0)].copy()
        df["Standard Value"] = sv.loc[df.index]

    # RDKit setup
    Chem, SaltRemover = try_import_rdkit()
    salt_remover = SaltRemover() if (Chem is not None and args.remove_salts) else None

    # --- Optional canonicalization for dedup key ---
    smiles_key = smiles_col
    if args.canonicalize and Chem is not None:
        df = df.copy()
        df["_canon_smiles"] = df[smiles_col].apply(lambda s: canonicalize_smiles(s, Chem, args.remove_salts, salt_remover))
        smiles_key = "_canon_smiles"

    if args.drop_na_smiles:
        df = df[df[smiles_key].notna() & (df[smiles_key].astype(str).str.strip() != "")].copy()

    # --- pIC50 and Active ---
    added_cols = []
    if args.add_pic50:
        if "Standard Value" not in df.columns:
            sys.exit("[Error] Cannot compute pIC50: 'Standard Value' not found.")
        df["pIC50"] = df["Standard Value"].apply(compute_pic50_from_nm)
        df = df[df["pIC50"].notna()].copy()
        added_cols.append("pIC50")

    if args.active_threshold is not None:
        if "pIC50" not in df.columns:
            # compute if not computed yet
            if "Standard Value" not in df.columns:
                sys.exit("[Error] Cannot compute Active: need pIC50 or 'Standard Value'.")
            df["pIC50"] = df["Standard Value"].apply(compute_pic50_from_nm)
            df = df[df["pIC50"].notna()].copy()
            if "pIC50" not in added_cols:
                added_cols.append("pIC50")
        thr = float(args.active_threshold)
        df["Active"] = (df["pIC50"] >= thr).astype(int)
        added_cols.append("Active")

    # --- Deduplicate ---
    n_before_dedup = len(df)
    df_dedup = df.drop_duplicates(subset=[smiles_key], keep=args.keep).copy()

    if "_canon_smiles" in df_dedup.columns:
        df_dedup[smiles_col] = df_dedup["_canon_smiles"]
        df_dedup.drop(columns=["_canon_smiles"], inplace=True)

    # --- Keep only selected columns (plus derived ones if present) ---
    keep_cols = parse_keep_cols(args.keep_cols)
    missing_cols = []
    if keep_cols:
        # Always append derived columns if they exist
        for c in ["pIC50", "Active"]:
            if c in df_dedup.columns and c not in keep_cols:
                keep_cols.append(c)
        for c in keep_cols:
            if c not in df_dedup.columns:
                missing_cols.append(c)
        present = [c for c in keep_cols if c in df_dedup.columns]
        if present:
            df_dedup = df_dedup[present].copy()

    # Decide output path
    out_path = Path(args.out_path) if args.out_path else in_path.with_name(in_path.stem + "_no_dup" + in_path.suffix)
    write_table(df_dedup, out_path)

    report = {
        "input_rows": int(n0),
        "after_value_filter": int(n_after_value_filter),
        "after_units_nm_filter": int(len(df)) if args.require_nm else None,
        "rows_after_dedup": int(len(df_dedup)),
        "duplicates_removed": int(n_before_dedup - len(df_dedup)),
        "smiles_col": smiles_col,
        "canonicalized": bool(args.canonicalize and Chem is not None),
        "salts_removed": bool(args.remove_salts and Chem is not None),
        "keep": args.keep,
        "kept_columns": keep_cols if keep_cols else "ALL",
        "missing_keep_columns": missing_cols,
        "added_columns": added_cols,
        "active_threshold": float(args.active_threshold) if args.active_threshold is not None else None,
        "require_nm": bool(args.require_nm),
        "input_file": str(in_path),
        "output_file": str(out_path),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
