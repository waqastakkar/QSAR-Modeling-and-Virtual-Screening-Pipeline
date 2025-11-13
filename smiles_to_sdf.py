#!/usr/bin/env python3
# Author: Muhammad Waqas
"""
Copyright (c) 2025 Muhammad Waqas.
All rights reserved.

Permission is granted to use this code for academic and research purposes only.

Modification, redistribution, or commercial use is not permitted.

Any published work or research that uses this code must cite:
Waqas, M. (2025). Structural and Dynamic Profiling of SARS-CoV-2 Papain-Like Protease Bound to Machine-Learning-Identified Oxadiazole Inhibitors.

smiles_to_sdf.py
Convert SMILES to SDF while preserving original column names.
- Auto-detects Excel/CSV input; default columns set for this dataset:
    SMILES column: 'Smiles'
    ID/name column: 'Molecule ChEMBL ID'
- Writes SDF; attaches all columns as SD properties by default.

Usage:
  python smiles_to_sdf.py --in /path/CHEMBL-Name.xlsx --out ligands.sdf
  python smiles_to_sdf.py --in data.csv --out ligands.sdf --smiles-col Smiles --id-col "Molecule ChEMBL ID" --conf --nconfs 10

Requirements: pandas, rdkit
"""

import argparse
import json
from pathlib import Path
import pandas as pd

def try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except Exception as e:
        raise SystemExit("[Error] RDKit is required for this script. Please install RDKit.") from e

def read_table(path: Path, sep: str = ",") -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=sep, dtype=str)

def main():
    Chem, AllChem = try_import_rdkit()

    p = argparse.ArgumentParser(description="SMILES table → SDF (attach properties).")
    p.add_argument("--in", dest="in_path", required=True, help="Input .xlsx/.csv")
    p.add_argument("--out", dest="out_sdf", required=True, help="Output .sdf path")
    p.add_argument("--smiles-col", default="Smiles", help="SMILES column name (default: Smiles)")
    p.add_argument("--id-col", default="Molecule ChEMBL ID", help="ID column to use as molecule name")
    p.add_argument("--sep", default=",", help="CSV/TSV separator (default ,)")
    p.add_argument("--props", choices=["none", "all"], default="all", help="Attach all columns as SD props (default all)")
    p.add_argument("--conf", action="store_true", help="Generate 3D conformer(s) with ETKDG + MMFF/UFF")
    p.add_argument("--nconfs", type=int, default=1, help="Number of conformers if --conf (default 1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for embedding")
    p.add_argument("--report", default=None, help="Optional JSON report path")
    args = p.parse_args()

    in_path = Path(args.in_path)
    df = read_table(in_path, sep=args.sep)
    smiles_col = args.smiles_col
    id_col = args.id_col

    if smiles_col not in df.columns:
        raise SystemExit(f"[Error] Column '{smiles_col}' not found. Columns: {df.columns.tolist()}")

    writer = Chem.SDWriter(str(Path(args.out_sdf)))
    n_total = 0
    n_written = 0
    n_failed = 0
    failed_idx = []

    for idx, row in df.iterrows():
        n_total += 1
        smi = row[smiles_col]
        if not isinstance(smi, str) or smi.strip() == "":
            n_failed += 1
            failed_idx.append(int(idx))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            failed_idx.append(int(idx))
            continue

        name = str(row[id_col]) if id_col in df.columns else f"mol_{idx}"
        mol.SetProp("_Name", name)

        if args.props == "all":
            for col in df.columns:
                if col == smiles_col:
                    continue
                val = "" if pd.isna(row[col]) else str(row[col])
                try:
                    mol.SetProp(col, val)
                except Exception:
                    pass

        if args.conf:
            m3d = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = int(args.seed)
            if args.nconfs > 1:
                ids = AllChem.EmbedMultipleConfs(m3d, numConfs=int(args.nconfs), params=params)
            else:
                cid = AllChem.EmbedMolecule(m3d, params)
                ids = [cid]
            # Optimize
            try:
                props = AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant="MMFF94s")
                if props is not None:
                    for cid in ids:
                        AllChem.MMFFOptimizeMolecule(m3d, confId=cid)
                else:
                    for cid in ids:
                        AllChem.UFFOptimizeMolecule(m3d, confId=cid)
            except Exception:
                for cid in ids:
                    AllChem.UFFOptimizeMolecule(m3d, confId=cid)
            writer.write(m3d)
            n_written += 1
        else:
            writer.write(mol)
            n_written += 1

    writer.close()

    report = {
        "input_rows": int(len(df)),
        "processed": int(n_total),
        "written": int(n_written),
        "failed": int(n_failed),
        "failed_row_indices": failed_idx,
        "input_file": str(in_path),
        "output_sdf": str(Path(args.out_sdf)),
        "smiles_col": smiles_col,
        "id_col": id_col,
        "props_attached": args.props,
        "conformers": bool(args.conf),
        "nconfs": int(args.nconfs) if args.conf else 0,
        "seed": int(args.seed),
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
