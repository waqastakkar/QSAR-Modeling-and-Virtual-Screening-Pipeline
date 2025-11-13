#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). QSAR Modeling and Virtual Screening Pipeline.

generate_fingerprints.py
Compute molecular fingerprints from an SDF using RDKit.

Features
- Interactive selection of fingerprint families (menu).
- Morgan via new rdFingerprintGenerator API; safe fallback to old AllChem API.
- Robust sanitization with clear failure counts.
- Keep SD properties (--include-props all or list), rebuild SMILES if missing.
- Writes CSV or Parquet; prints per-family column counts.

Fingerprint families (prefixes):
  morgan_  (hashed, default 2048 bits, radius=2; feature/chiral flags below)
  maccs_   (fixed 167 bits)
  rdkit_   (hashed, 2048 bits)
  atompair_(hashed, 2048 bits)
  torsion_ (hashed, 2048 bits)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# ----- Defaults (change here if you want) -----
NBITS_HASHED = 2048
MORGAN_RADIUS = 2
MORGAN_USE_FEATURES = False
MORGAN_CHIRAL = False

FAMILY_MENU = [
    ("morgan",  "Morgan (2048 bits, radius=2)"),
    ("maccs",   "MACCS (167 bits)"),
    ("rdkit",   "RDKit (2048 bits)"),
    ("atompair","AtomPair (2048 bits)"),
    ("torsion", "Topological Torsion (2048 bits)"),
]

# ----------------------------------------------

def try_import_rdkit():
    """
    Returns a dict of RDKit symbols and flags telling which generator APIs are available.
    """
    try:
        from rdkit import Chem
    except Exception as e:
        raise SystemExit("[Error] RDKit import failed. Please install RDKit.") from e

    # Optional new generator API
    have_new = True
    try:
        from rdkit.Chem.rdFingerprintGenerator import (
            GetMorganGenerator, GetMorganFeatureAtomInvGen,
            GetRDKitFPGenerator, GetAtomPairGenerator,
            GetTopologicalTorsionGenerator
        )
    except Exception:
        have_new = False
        GetMorganGenerator = GetMorganFeatureAtomInvGen = None
        GetRDKitFPGenerator = GetAtomPairGenerator = GetTopologicalTorsionGenerator = None

    # Old APIs used for fallback
    try:
        from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
        from rdkit.Chem import RDKFingerprint
        have_old = True
    except Exception:
        have_old = False
        AllChem = MACCSkeys = rdMolDescriptors = RDKFingerprint = None

    if not (have_new or have_old):
        raise SystemExit("[Error] RDKit fingerprint APIs not available in this build.")

    return dict(
        Chem=Chem,
        AllChem=AllChem,
        MACCSkeys=MACCSkeys,
        rdMolDescriptors=rdMolDescriptors,
        RDKFingerprint=RDKFingerprint,
        GetMorganGenerator=GetMorganGenerator,
        GetMorganFeatureAtomInvGen=GetMorganFeatureAtomInvGen,
        GetRDKitFPGenerator=GetRDKitFPGenerator,
        GetAtomPairGenerator=GetAtomPairGenerator,
        GetTopologicalTorsionGenerator=GetTopologicalTorsionGenerator,
        HAVE_NEW=have_new,
        HAVE_OLD=have_old,
    )

def parse_list(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]

def fp_names(prefix: str, nbits: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(nbits)]

def to_bit_list(bitvect, nbits: int) -> List[int]:
    # RDKit ExplicitBitVect -> list of 0/1
    return [int(bitvect.GetBit(i)) for i in range(nbits)]

def _safe_sanitize(Chem, mol):
    """
    Try to sanitize; if fails, attempt minimal fixes. Return sanitized Mol or None.
    """
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        # try removing Hs and re-sanitizing
        try:
            m2 = Chem.RemoveHs(mol, sanitize=False)
            Chem.SanitizeMol(m2)
            return m2
        except Exception:
            return None

def _open_sdf_supplier(Chem, sdf_path: Path):
    # tolerant supplier; do not strip H by default (we handle separately)
    return Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

def _menu_select_families() -> List[str]:
    print("\nSelect fingerprints by number (comma-separated).")
    for i, (_, label) in enumerate(FAMILY_MENU, start=1):
        print(f"  {i}. {label}")
    sel = input("Your selection [1]: ").strip() or "1"
    idxs = []
    for part in sel.replace(" ", "").split(","):
        if part.isdigit():
            v = int(part)
            if 1 <= v <= len(FAMILY_MENU):
                idxs.append(v)
    idxs = sorted(set(idxs))
    if not idxs:
        idxs = [1]
    chosen = [FAMILY_MENU[i-1][0] for i in idxs]
    print("Chosen:", ", ".join(chosen))
    print(f"(NBITS={NBITS_HASHED}; Morgan radius={MORGAN_RADIUS}; features={MORGAN_USE_FEATURES}; chiral={MORGAN_CHIRAL})")
    return chosen

def compute_fingerprints_new_api(mol, fps: List[str], ctx) -> Dict[str, List[int]]:
    """
    New generator API. Falls back per-family if something fails.
    """
    out: Dict[str, List[int]] = {}
    # Morgan
    if "morgan" in fps and ctx["GetMorganGenerator"] is not None:
        try:
            if MORGAN_USE_FEATURES and ctx["GetMorganFeatureAtomInvGen"] is not None:
                inv = ctx["GetMorganFeatureAtomInvGen"]()
                gen = ctx["GetMorganGenerator"](
                    radius=MORGAN_RADIUS,
                    fpSize=NBITS_HASHED,
                    includeChirality=bool(MORGAN_CHIRAL),
                    atomInvariantsGenerator=inv,
                )
            else:
                gen = ctx["GetMorganGenerator"](
                    radius=MORGAN_RADIUS,
                    fpSize=NBITS_HASHED,
                    includeChirality=bool(MORGAN_CHIRAL),
                )
            bv = gen.GetFingerprint(mol)
            out["morgan"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    # MACCS
    if "maccs" in fps and ctx["MACCSkeys"] is not None:
        try:
            bv = ctx["MACCSkeys"].GenMACCSKeys(mol)
            out["maccs"] = to_bit_list(bv, 167)
        except Exception:
            pass
    # RDKit
    if "rdkit" in fps and ctx["GetRDKitFPGenerator"] is not None:
        try:
            gen = ctx["GetRDKitFPGenerator"](fpSize=NBITS_HASHED)
            bv = gen.GetFingerprint(mol)
            out["rdkit"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    # AtomPair
    if "atompair" in fps and ctx["GetAtomPairGenerator"] is not None:
        try:
            gen = ctx["GetAtomPairGenerator"](fpSize=NBITS_HASHED)
            bv = gen.GetFingerprint(mol)
            out["atompair"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    # Torsion
    if "torsion" in fps and ctx["GetTopologicalTorsionGenerator"] is not None:
        try:
            gen = ctx["GetTopologicalTorsionGenerator"](fpSize=NBITS_HASHED)
            bv = gen.GetFingerprint(mol)
            out["torsion"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    return out

def compute_fingerprints_old_api(mol, fps: List[str], ctx) -> Dict[str, List[int]]:
    """
    Old API fallback (AllChem etc.). Only used if new API failed or missing.
    """
    out: Dict[str, List[int]] = {}
    AC = ctx["AllChem"]; MK = ctx["MACCSkeys"]; RDK = ctx["RDKFingerprint"]; RDM = ctx["rdMolDescriptors"]
    if AC is None:
        return out

    if "morgan" in fps:
        try:
            bv = AC.GetMorganFingerprintAsBitVect(
                mol,
                radius=MORGAN_RADIUS,
                nBits=NBITS_HASHED,
                useChirality=bool(MORGAN_CHIRAL),
                useFeatures=bool(MORGAN_USE_FEATURES),
            )
            out["morgan"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    if "maccs" in fps and MK is not None:
        try:
            bv = MK.GenMACCSKeys(mol)
            out["maccs"] = to_bit_list(bv, 167)
        except Exception:
            pass
    if "rdkit" in fps and RDK is not None:
        try:
            bv = RDK.MolFingerprintAsBitVect(mol, fpSize=NBITS_HASHED)
            out["rdkit"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    if "atompair" in fps and RDM is not None:
        try:
            bv = RDM.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=NBITS_HASHED)
            out["atompair"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    if "torsion" in fps and RDM is not None:
        try:
            bv = RDM.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=NBITS_HASHED)
            out["torsion"] = to_bit_list(bv, NBITS_HASHED)
        except Exception:
            pass
    return out

def compute_fingerprints(mol, fps: List[str], ctx) -> Dict[str, List[int]]:
    # Try new API first (if present), then fill gaps with old API
    out_new = compute_fingerprints_new_api(mol, fps, ctx) if ctx["HAVE_NEW"] else {}
    missing = [f for f in fps if f not in out_new]
    if missing and ctx["HAVE_OLD"]:
        out_old = compute_fingerprints_old_api(mol, missing, ctx)
        out_new.update(out_old)
    return out_new

def main():
    ctx = try_import_rdkit()
    Chem = ctx["Chem"]

    p = argparse.ArgumentParser(description="Compute RDKit fingerprints from an SDF.")
    p.add_argument("--in", dest="in_sdf", required=True, help="Input SDF file")
    p.add_argument("--out", dest="out_path", required=True, help="Output file (.csv or .parquet)")
    p.add_argument("--include-props",
                   default="Molecule ChEMBL ID,Standard Value,Standard Type,Standard Units,Standard Relation,Smiles,pIC50,Active",
                   help="Comma-separated SD props to include OR 'all' to include every property")
    p.add_argument("--format", choices=["csv", "parquet"], default=None,
                   help="Output format (inferred from extension if omitted)")
    args = p.parse_args()

    in_sdf = Path(args.in_sdf)
    out_path = Path(args.out_path)

    fmt = args.format or (out_path.suffix.lower().lstrip(".") if out_path.suffix else "csv")
    if fmt not in ("csv", "parquet"):
        sys.exit("[Error] Unsupported output format. Use .csv or .parquet or --format csv|parquet.")

    include_all = str(args.include_props).strip().lower() == "all"
    include_props = [] if include_all else parse_list(args.include_props)

    # Interactive family selection
    fps = _menu_select_families()

    # Read SDF
    if not in_sdf.exists():
        sys.exit(f"[Error] Input SDF not found: {in_sdf}")
    suppl = _open_sdf_supplier(Chem, in_sdf)

    rows: List[Dict[str, Any]] = []
    n_total = n_ok = 0
    n_none = 0
    n_san_fail = 0
    n_fp_fail = 0

    for mol in suppl:
        n_total += 1
        if mol is None:
            n_none += 1
            continue

        sm = _safe_sanitize(Chem, mol)
        if sm is None:
            n_san_fail += 1
            continue

        row: Dict[str, Any] = {}

        # SD properties
        if include_all:
            for prop in sm.GetPropNames():
                try:
                    row[prop] = sm.GetProp(prop)
                except Exception:
                    row[prop] = None
        else:
            for prop in include_props:
                try:
                    row[prop] = sm.GetProp(prop) if sm.HasProp(prop) else None
                except Exception:
                    row[prop] = None

        # Name (if any)
        try:
            row["_Name"] = sm.GetProp("_Name")
        except Exception:
            row["_Name"] = None

        # Ensure Smiles exists
        if ("Smiles" not in row) or (row["Smiles"] is None) or (str(row["Smiles"]).strip() == ""):
            try:
                row["Smiles"] = Chem.MolToSmiles(Chem.RemoveHs(sm), isomericSmiles=True)
            except Exception:
                row["Smiles"] = None

        # Fingerprints (new API + fallback)
        fp_dict = compute_fingerprints(sm, fps, ctx)
        if not fp_dict:
            n_fp_fail += 1
            continue

        # Attach FP bits
        for k, bits in fp_dict.items():
            cols = fp_names("maccs", 167) if k == "maccs" else fp_names(k, NBITS_HASHED)
            for c, v in zip(cols, bits):
                row[c] = v

        rows.append(row)
        n_ok += 1

    if n_ok == 0:
        sys.exit(
            f"[Error] No molecules processed successfully. "
            f"Total={n_total}, None={n_none}, SanitizeFail={n_san_fail}, FPFail={n_fp_fail}."
        )

    # Build frame
    df = pd.DataFrame(rows)

    # Report per-family column counts
    families = ["morgan", "maccs", "rdkit", "atompair", "torsion"]
    counts = {fam: sum(str(c).startswith(f"{fam}_") for c in df.columns) for fam in families}
    counts["TOTAL_FP_COLS"] = sum(counts.values())
    print("Fingerprint columns per family:", counts)

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    # Final summary
    print({
        "input_sdf": str(in_sdf),
        "output": str(out_path),
        "n_total": n_total,
        "n_ok": n_ok,
        "n_none": n_none,
        "n_sanitize_fail": n_san_fail,
        "n_fp_fail": n_fp_fail,
        "fps": fps,
        "nbits": NBITS_HASHED,
        "morgan_radius": MORGAN_RADIUS,
        "morgan_features": bool(MORGAN_USE_FEATURES),
        "morgan_chiral": bool(MORGAN_CHIRAL),
        "include_props": "all" if include_all else include_props,
        "format": fmt,
    })

if __name__ == "__main__":
    main()
