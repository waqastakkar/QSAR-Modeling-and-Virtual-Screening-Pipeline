#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.

batch_fingerprinting.py
Interactive batch creation of RDKit fingerprints from many SMILES tables (CSV/TSV/XLSX).

Key features of the script
- Per‑fingerprint try/except: one failing FP (MACCS missing in RDKit build)
- Flexible SMILES header detection (SMILES/smiles/Smiles/can_smiles/SMILES_std)
- Safer canonicalization with optional salt removal and fallback to raw SMILES
- Soft failure: if nothing survives, print a summary instead of exiting with an error

Outputs a single combined file in CSV/TSV/Parquet depending on extension.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------- Fingerprint settings ----------
NBITS = 2048
MORGAN_RADIUS = 2
MORGAN_FEATURES = False
MORGAN_CHIRAL = False

# ---------- RDKit import helper ----------
def try_import_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint
        from rdkit.Chem.SaltRemover import SaltRemover
        try:
            RDLogger.DisableLog('rdApp.warning')
        except Exception:
            pass
        return Chem, AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint, SaltRemover
    except Exception as e:
        raise SystemExit("[Error] RDKit is required. Please install RDKit.") from e

# ---------- Bitvector utilities ----------
def to_bit_list(bv, nbits: int) -> List[int]:
    # Convert ExplicitBitVect to list of 0/1 ints
    return [int(bv.GetBit(i)) for i in range(nbits)]

# ---------- IO ----------
def read_one_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, dtype=str)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t", dtype=str)
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, dtype=str)
        except Exception as e:
            raise SystemExit(f"[Error] Failed reading Excel {path.name}: {e}")
    # Default try CSV
    return pd.read_csv(path, dtype=str)

def write_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"[Warn] Parquet write failed ({e}); falling back to CSV.")
            df.to_csv(path.with_suffix('.csv'), index=False)
    elif fmt == "tsv":
        df.to_csv(path, index=False, sep="\t")
    else:
        df.to_csv(path, index=False)

# ---------- Cleaning helpers ----------
def canonicalize_smiles(smiles: str, Chem, remove_salts: bool, salt_remover) -> Optional[str]:
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        if remove_salts and salt_remover is not None:
            # keep something even if stripped fully
            m = salt_remover.StripMol(m, dontRemoveEverything=True)
        can = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
        return can
    except Exception:
        return None

# ---------- FP utils ----------
def fp_names(name: str, nbits: int) -> List[str]:
    return [f"{name}_{i}" for i in range(nbits)]

def compute_fps_for_mol(mol, fps: List[str]) -> Dict[str, List[int]]:
    """Return dict fingerprint_name -> list[int]. Skip individual FPs on error."""
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint
    out: Dict[str, List[int]] = {}
    for name in fps:
        try:
            if name == "morgan":
                bv = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=MORGAN_RADIUS, nBits=NBITS,
                    useChirality=bool(MORGAN_CHIRAL), useFeatures=bool(MORGAN_FEATURES)
                )
                out["morgan"] = to_bit_list(bv, NBITS)
            elif name == "maccs":
                bv = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
                out["maccs"] = to_bit_list(bv, 167)
            elif name == "rdkit":
                bv = RDKFingerprint.MolFingerprintAsBitVect(mol, fpSize=NBITS)
                out["rdkit"] = to_bit_list(bv, NBITS)
            elif name == "atompair":
                bv = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=NBITS)
                out["atompair"] = to_bit_list(bv, NBITS)
            elif name == "torsion":
                bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=NBITS)
                out["torsion"] = to_bit_list(bv, NBITS)
        except Exception:
            # skip just this FP if RDKit build lacks it or molecule fails for this FP
            continue
    return out

# ---------- Interactive helpers ----------
def prompt(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{msg}: ").strip()
    s = input(f"{msg} [{default}]: ").strip()
    return s or default

def prompt_yesno(msg: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    s = input(f"{msg} (y/n) [{d}]: ").strip().lower()
    if not s:
        return default
    return s.startswith("y")

# ---------- Main ----------
def main():
    Chem, AllChem, MACCSkeys, rdMolDescriptors, RDKFingerprint, SaltRemover = try_import_rdkit()

    print("=== Batch SMILES → Fingerprints (Interactive) ===")
    # Folder and file discovery
    in_dir = Path(prompt("Input folder path", "."))
    if not in_dir.exists():
        raise SystemExit(f"Folder not found: {in_dir}")

    pattern = prompt("Filename pattern", "*.csv")
    recursive = prompt_yesno("Recurse subfolders?", False)

    glob_pattern = str(in_dir / ("**/" + pattern if recursive else pattern))
    files = [Path(p) for p in glob.glob(glob_pattern, recursive=recursive)]
    if not files:
        raise SystemExit("No input files found with the given pattern.")

    # Columns
    smiles_col = prompt("SMILES column name", "SMILES")
    id_cols = [c.strip() for c in prompt("ID columns to keep first (comma-separated)", "ZINC ID,SMILES").split(",") if c.strip()]
    add_source = prompt_yesno("Add source file name as 'source_file' column?", True)

    # Fingerprint menu
    menu: List[Tuple[str, str]] = [
        ("morgan", "Morgan (2048 bits, radius=2)"),
        ("maccs", "MACCS (167 bits)"),
        ("rdkit", "RDKit (2048 bits)"),
        ("atompair", "AtomPair (2048 bits)"),
        ("torsion", "Topological Torsion (2048 bits)")
    ]
    print("\nSelect fingerprints by number (comma-separated).")
    for i, (_, desc) in enumerate(menu, 1):
        print(f"  {i}. {desc}")
    sel = prompt("Your selection", "1")
    try:
        idxs = [int(x.strip()) for x in sel.split(',') if x.strip()]
    except ValueError:
        raise SystemExit("No valid selection.")
    idxs = [i for i in idxs if 1 <= i <= len(menu)]
    if not idxs:
        raise SystemExit("No valid selection.")
    fps = [menu[i-1][0] for i in idxs]
    print("Chosen:", ", ".join(fps))
    print(f"(NBITS={NBITS}; Morgan radius={MORGAN_RADIUS}; features={MORGAN_FEATURES}; chiral={MORGAN_CHIRAL})")

    # Cleaning / dedup
    canonicalize = prompt_yesno("Canonicalize SMILES?", True)
    remove_salts = prompt_yesno("Remove salts before canonicalization?", True)
    drop_invalid = prompt_yesno("Drop rows with invalid/empty SMILES?", True)
    dedup_mode = prompt("Global dedup by [none/smiles/canonical]", "canonical").lower()
    if dedup_mode not in ("none", "smiles", "canonical"):
        dedup_mode = "none"

    # Output
    default_out = in_dir / "database_fingerprints.parquet"
    out_path = Path(prompt("Output file path", str(default_out)))
    fmt = out_path.suffix.lower().lstrip(".") if out_path.suffix else "parquet"
    if fmt not in ("csv", "tsv", "parquet"):
        print("Unknown extension; defaulting to parquet.")
        fmt = "parquet"
        out_path = out_path.with_suffix(".parquet")

    # Helpers
    salt_remover = SaltRemover() if remove_salts else None
    seen = set() if dedup_mode in ("smiles", "canonical") else None

    def _find_smiles_col(cols: List[str], preferred: Optional[str]) -> Optional[str]:
        if preferred and preferred in cols:
            return preferred
        lower = {c.lower(): c for c in cols}
        for cand in ["smiles", "smile", "can_smiles", "smiles_std", "canonical_smiles"]:
            if cand in lower:
                return lower[cand]
        return None

    n_total = 0
    n_fail = 0
    n_kept = 0
    out_rows: List[Dict[str, Any]] = []

    # Process files
    for path in files:
        try:
            df = read_one_table(path)
        except Exception as e:
            print(f"[Warn] Failed to read {path.name}: {e}")
            continue

        smi_col = _find_smiles_col(list(df.columns), smiles_col)
        if not smi_col:
            print(f"[Warn] No SMILES column in {path.name}; skipping.")
            continue

        # prepare canonical if needed
        if canonicalize or dedup_mode == "canonical":
            df = df.copy()
            df["_canon_smiles"] = df[smi_col].apply(lambda s: canonicalize_smiles(s, Chem, remove_salts, salt_remover))

        key_col = "_canon_smiles" if (canonicalize or dedup_mode == "canonical") else smi_col

        # drop invalid if requested
        if drop_invalid:
            df = df[df[key_col].notna() & (df[key_col].astype(str).str.strip() != "")].copy()

        for _, row in df.iterrows():
            n_total += 1
            smi_raw = row.get(smi_col, None)
            smi_use = row.get("_canon_smiles", smi_raw)

            # If still empty and not dropping invalid, skip fingerprinting but keep counters
            if not isinstance(smi_use, str) or smi_use.strip() == "":
                if drop_invalid:
                    n_fail += 1
                    continue

            # Dedup
            if seen is not None:
                key = ("canon", smi_use) if dedup_mode == "canonical" else ("raw", smi_raw)
                if key in seen:
                    continue
                seen.add(key)

            # Build mol
            mol = None
            try:
                mol = Chem.MolFromSmiles(smi_use) if smi_use else None
            except Exception:
                mol = None
            if mol is None:
                if drop_invalid:
                    n_fail += 1
                    continue

            # Compute fingerprints (skip per-FP failures)
            fp_dict = compute_fps_for_mol(mol, fps=fps)
            if not fp_dict:
                n_fail += 1
                continue

            out_row: Dict[str, Any] = {}
            # Keep ID columns first (only those present)
            for c in id_cols:
                if c in df.columns:
                    out_row[c] = row[c]
            # ensure a SMILES column is present using user's casing when possible
            if smiles_col in df.columns:
                out_row[smiles_col] = row[smiles_col]
            elif "SMILES" in df.columns:
                out_row["SMILES"] = row["SMILES"]
            elif "Smiles" in df.columns:
                out_row["Smiles"] = row["Smiles"]
            else:
                out_row["SMILES"] = smi_raw

            if add_source:
                out_row["source_file"] = path.name

            # Expand FP bits into columns
            for k, bits in fp_dict.items():
                if k == "maccs":
                    cols = fp_names("maccs", 167)
                else:
                    cols = fp_names(k, NBITS)
                for c_name, v in zip(cols, bits):
                    out_row[c_name] = v

            out_rows.append(out_row)
            n_kept += 1

    if not out_rows:
        print("\n=== Summary ===")
        print("Files processed:", len(files))
        print("Rows seen:", n_total)
        print("Failed molecules:", n_fail)
        print("Dedup mode:", dedup_mode)
        print("Fingerprints:", ", ".join(fps))
        print("Output aborted (no valid molecules after cleaning). Try rerunning with:")
        print("  - Canonicalize: n")
        print("  - Remove salts: n")
        print("  - Drop invalid: n")
        print("  - Dedup: none")
        return

    out_df = pd.DataFrame(out_rows)

    # Order columns: id_cols unique, then source_file (if present), then the rest
    keep: List[str] = []
    for c in id_cols:
        if c in out_df.columns and c not in keep:
            keep.append(c)
    if "source_file" in out_df.columns and "source_file" not in keep:
        keep.append("source_file")
    rest = [c for c in out_df.columns if c not in keep]
    out_df = out_df[keep + rest]

    write_table(out_df, out_path, fmt=fmt)

    print("\n=== Summary ===")
    print("Files processed:", len(files))
    print("Rows out:", len(out_df))
    print("Failed molecules:", n_fail)
    print("Dedup mode:", dedup_mode)
    print("Fingerprints:", ", ".join(fps))
    print("Output:", out_path)


if __name__ == "__main__":
    main()
