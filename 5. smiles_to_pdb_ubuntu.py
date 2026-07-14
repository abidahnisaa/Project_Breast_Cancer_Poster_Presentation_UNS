#!/usr/bin/env python3
"""
Generate 3D PDB ligand files from an Excel table on Ubuntu.

Expected columns (detected automatically):
- Compound_ID / identifier / ID
- Nama_senyawa / compound_name / name (optional)
- Smiles / canonical_smiles / SMILES

Usage:
    python smiles_to_pdb_ubuntu.py Siap_Olah.xlsx

Optional custom output folder:
    python smiles_to_pdb_ubuntu.py Siap_Olah.xlsx --output ligand_output
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    mapping = {normalize_column_name(column): column for column in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in mapping:
            return mapping[key]

    if required:
        raise ValueError(
            f"Kolom wajib tidak ditemukan. Dicari: {candidates}. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )
    return None


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_filename(value: str, max_length: int = 150) -> str:
    value = unicodedata.normalize("NFKD", clean_text(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "unnamed_ligand")[:max_length]


def canonicalize_smiles(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return mol
    return max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())


def build_3d_molecule(smiles: str, seed: int, max_iters: int) -> tuple[Chem.Mol, str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES tidak dapat dibaca RDKit")

    mol = largest_fragment(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True

    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status == -1:
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = seed
        fallback.enforceChirality = True
        fallback.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol, fallback)

    if embed_status == -1:
        raise ValueError("Gagal membuat konformer 3D")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        status = AllChem.MMFFOptimizeMolecule(
            mol,
            mmffVariant="MMFF94s",
            maxIters=max_iters,
        )
        force_field = "MMFF94s"
    elif AllChem.UFFHasAllMoleculeParams(mol):
        status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        force_field = "UFF"
    else:
        raise ValueError("Parameter MMFF dan UFF tidak tersedia")

    optimization = "Converged" if status == 0 else "Maximum iterations reached"
    return mol, force_field, optimization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mengubah SMILES dalam Excel menjadi struktur PDB 3D."
    )
    parser.add_argument("excel_file", help="Lokasi file Excel input")
    parser.add_argument(
        "--output",
        default="ligand_output",
        help="Folder output (default: ligand_output)",
    )
    parser.add_argument("--sheet", default=0, help="Nama/nomor sheet (default: 0)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Maksimum iterasi optimasi",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.excel_file).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    pdb_dir = output_dir / "pdb_files"
    report_dir = output_dir / "reports"

    if not input_path.exists():
        print(f"❌ File input tidak ditemukan: {input_path}", file=sys.stderr)
        return 1

    pdb_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(input_path, sheet_name=sheet)
    df.columns = df.columns.astype(str).str.strip()

    id_col = find_column(df, ["Compound_ID", "Compound ID", "identifier", "ID"])
    name_col = find_column(
        df,
        ["Nama_senyawa", "Nama senyawa", "compound_name", "compound name", "name"],
        required=False,
    )
    smiles_col = find_column(
        df,
        ["Smiles", "SMILES", "canonical_smiles", "Canonical SMILES"],
    )

    print("Kolom yang digunakan:")
    print(f"  ID     : {id_col}")
    print(f"  Nama   : {name_col or '(tidak ada)'}")
    print(f"  SMILES : {smiles_col}")

    work = df.copy()
    work["_id"] = work[id_col].apply(clean_text)
    work["_name"] = (
        work[name_col].apply(clean_text) if name_col else "Unnamed_compound"
    )
    work["_smiles"] = work[smiles_col].apply(clean_text)
    work["_canonical"] = work["_smiles"].apply(canonicalize_smiles)

    invalid_rows = work[work["_canonical"].isna()].copy()
    valid_rows = work[work["_canonical"].notna()].copy()

    before = len(valid_rows)
    valid_rows = valid_rows.drop_duplicates(subset=["_id", "_canonical"], keep="first")
    removed_duplicates = before - len(valid_rows)

    print(f"Jumlah baris input       : {len(df)}")
    print(f"SMILES tidak valid       : {len(invalid_rows)}")
    print(f"Duplikat tidak diproses  : {removed_duplicates}")
    print(f"Struktur yang diproses   : {len(valid_rows)}")

    report_rows: list[dict] = []
    used_names: dict[str, int] = {}
    success = 0

    for number, (_, row) in enumerate(valid_rows.iterrows(), start=1):
        compound_id = row["_id"] or f"ROW_{number:05d}"
        compound_name = row["_name"] or "Unnamed_compound"
        base = safe_filename(f"{compound_id}_{compound_name}")

        used_names[base] = used_names.get(base, 0) + 1
        filename = base if used_names[base] == 1 else f"{base}_{used_names[base]}"
        output_path = pdb_dir / f"{filename}.pdb"

        print(f"[{number}/{len(valid_rows)}] {compound_id} | {compound_name}")

        try:
            mol, force_field, optimization = build_3d_molecule(
                row["_smiles"],
                seed=args.seed,
                max_iters=args.max_iters,
            )
            mol.SetProp("_Name", filename)
            Chem.MolToPDBFile(mol, str(output_path))

            report = row.to_dict()
            report.update(
                {
                    "Generation_status": "Success",
                    "Failure_reason": "",
                    "Generated_filename": filename,
                    "PDB_path": str(output_path),
                    "Force_field": force_field,
                    "Optimization_result": optimization,
                }
            )
            report_rows.append(report)
            success += 1
            print(f"  ✅ Saved: {output_path.name} | {force_field} | {optimization}")

        except Exception as exc:
            report = row.to_dict()
            report.update(
                {
                    "Generation_status": "Failed",
                    "Failure_reason": str(exc),
                    "Generated_filename": filename,
                    "PDB_path": "",
                    "Force_field": "",
                    "Optimization_result": "",
                }
            )
            report_rows.append(report)
            print(f"  ❌ Failed: {exc}")

    for _, row in invalid_rows.iterrows():
        report = row.to_dict()
        report.update(
            {
                "Generation_status": "Failed",
                "Failure_reason": "SMILES kosong atau tidak valid",
                "Generated_filename": "",
                "PDB_path": "",
                "Force_field": "",
                "Optimization_result": "",
            }
        )
        report_rows.append(report)

    report_df = pd.DataFrame(report_rows)
    temp_cols = ["_id", "_name", "_smiles", "_canonical"]
    report_df = report_df.drop(columns=[c for c in temp_cols if c in report_df.columns])

    report_csv = report_dir / "smiles_to_pdb_report.csv"
    report_xlsx = report_dir / "smiles_to_pdb_report.xlsx"
    failed_xlsx = report_dir / "failed_ligands.xlsx"

    report_df.to_csv(report_csv, index=False)
    report_df.to_excel(report_xlsx, index=False)
    report_df[report_df["Generation_status"] == "Failed"].to_excel(
        failed_xlsx,
        index=False,
    )

    failed = len(report_df) - success
    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)
    print(f"Berhasil : {success}")
    print(f"Gagal    : {failed}")
    print(f"PDB      : {pdb_dir}")
    print(f"Laporan  : {report_xlsx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
