#!/usr/bin/env python3
"""
Batch-convert RDKit-generated ligand PDB files to AutoDock PDBQT using Open Babel.

Default folder layout:
    project/
    ├── ligand_output/
    │   ├── pdb_files/
    │   ├── pdbqt_files/          <- created automatically
    │   └── reports/
    │       └── pdbqt_conversion/ <- created automatically
    └── pdb_to_pdbqt_ubuntu.py

Usage:
    python pdb_to_pdbqt_ubuntu.py

Optional:
    python pdb_to_pdbqt_ubuntu.py --input ligand_output/pdb_files
    python pdb_to_pdbqt_ubuntu.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mengubah semua file ligan PDB menjadi PDBQT dengan Open Babel."
    )
    parser.add_argument(
        "--input",
        default="ligand_output/pdb_files",
        help="Folder PDB input (default: ligand_output/pdb_files)",
    )
    parser.add_argument(
        "--output",
        default="ligand_output/pdbqt_files",
        help="Folder PDBQT output (default: ligand_output/pdbqt_files)",
    )
    parser.add_argument(
        "--report-dir",
        default="ligand_output/reports/pdbqt_conversion",
        help="Folder laporan konversi",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa PDBQT yang sudah ada",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def get_obabel_version() -> str:
    result = run_command(["obabel", "-V"])
    text = (result.stdout or result.stderr).strip()
    return text or "Versi tidak terdeteksi"


def parse_pdbqt(path: Path) -> dict:
    """
    Pemeriksaan dasar file PDBQT:
    - memiliki ATOM/HETATM
    - memiliki ROOT dan TORSDOF
    - kolom muatan parsial dapat dibaca
    """
    atom_count = 0
    charge_values: list[float] = []
    has_root = False
    has_torsdof = False
    has_branch = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()

            if stripped == "ROOT":
                has_root = True
            elif stripped.startswith("TORSDOF"):
                has_torsdof = True
            elif stripped.startswith("BRANCH"):
                has_branch = True

            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1
                parts = line.split()

                # Dalam PDBQT, dua token terakhir biasanya:
                # partial charge dan AutoDock atom type.
                if len(parts) >= 2:
                    try:
                        charge = float(parts[-2])
                        if math.isfinite(charge):
                            charge_values.append(charge)
                    except ValueError:
                        pass

    charge_count = len(charge_values)
    net_charge = sum(charge_values) if charge_values else None
    nonzero_charge_count = sum(abs(value) > 1e-6 for value in charge_values)

    issues: list[str] = []

    if atom_count == 0:
        issues.append("Tidak ada baris ATOM/HETATM")
    if not has_root:
        issues.append("ROOT tidak ditemukan")
    if not has_torsdof:
        issues.append("TORSDOF tidak ditemukan")
    if charge_count != atom_count:
        issues.append(
            f"Muatan terbaca {charge_count}/{atom_count} atom"
        )
    if atom_count > 0 and charge_count == atom_count and nonzero_charge_count == 0:
        issues.append("Semua partial charge bernilai nol")

    return {
        "atom_count": atom_count,
        "charge_count": charge_count,
        "nonzero_charge_count": nonzero_charge_count,
        "net_partial_charge": net_charge,
        "has_root": has_root,
        "has_torsdof": has_torsdof,
        "has_branch": has_branch,
        "validation_status": "Valid" if not issues else "Warning",
        "validation_notes": "; ".join(issues),
    }


def write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text(
        "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()

    print("=" * 72)
    print("BATCH CONVERSION: PDB TO PDBQT")
    print("=" * 72)

    if shutil.which("obabel") is None:
        print(
            "❌ Open Babel (obabel) tidak ditemukan di environment aktif.\n"
            "Install dengan:\n"
            "conda install -n docking_env -c conda-forge openbabel -y",
            file=sys.stderr,
        )
        return 1

    if not input_dir.exists():
        print(f"❌ Folder PDB input tidak ditemukan: {input_dir}", file=sys.stderr)
        return 1

    pdb_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdb"
    )

    if not pdb_files:
        print(f"❌ Tidak ada file .pdb di: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Open Babel : {get_obabel_version()}")
    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Reports     : {report_dir}")
    print(f"Jumlah PDB  : {len(pdb_files)}")
    print(f"Overwrite   : {'Ya' if args.overwrite else 'Tidak'}")
    print()

    report_rows: list[dict] = []
    success_names: list[str] = []
    skipped_names: list[str] = []
    failure_names: list[str] = []
    warning_blocks: list[str] = []

    success_count = 0
    skipped_count = 0
    failed_count = 0
    warning_count = 0

    for index, input_path in enumerate(pdb_files, start=1):
        output_path = output_dir / f"{input_path.stem}.pdbqt"

        print(f"[{index}/{len(pdb_files)}] {input_path.name}")

        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            validation = parse_pdbqt(output_path)
            skipped_count += 1
            skipped_names.append(output_path.name)

            report_rows.append(
                {
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "status": "Skipped_existing",
                    "return_code": "",
                    "atom_count": validation["atom_count"],
                    "charge_count": validation["charge_count"],
                    "nonzero_charge_count": validation["nonzero_charge_count"],
                    "net_partial_charge": validation["net_partial_charge"],
                    "has_root": validation["has_root"],
                    "has_torsdof": validation["has_torsdof"],
                    "has_branch": validation["has_branch"],
                    "validation_status": validation["validation_status"],
                    "validation_notes": validation["validation_notes"],
                    "stderr": "",
                }
            )

            print("  ⏭ Sudah ada, dilewati")
            continue

        # Tidak menggunakan --gen3d karena koordinat 3D sudah dibuat RDKit.
        # -h menambahkan/melengkapi hidrogen sebelum perhitungan Gasteiger.
        command = [
            "obabel",
            "-ipdb",
            str(input_path),
            "-opdbqt",
            "-O",
            str(output_path),
            "-h",
            "--partialcharge",
            "gasteiger",
            "--errorlevel",
            "2",
        ]

        result = run_command(command)
        stderr_text = result.stderr.strip()

        conversion_ok = (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        )

        if conversion_ok:
            validation = parse_pdbqt(output_path)

            # Output yang secara struktur tidak memiliki atom dianggap gagal.
            if validation["atom_count"] == 0:
                conversion_ok = False

        if conversion_ok:
            success_count += 1
            success_names.append(output_path.name)

            if (
                validation["validation_status"] == "Warning"
                or "CorrectStereoAtoms" in stderr_text
            ):
                warning_count += 1
                warning_blocks.append(
                    f"{output_path.name}\n"
                    f"Validation: {validation['validation_notes'] or '-'}\n"
                    f"Open Babel stderr:\n{stderr_text or '-'}\n"
                    + ("-" * 72)
                )

            report_rows.append(
                {
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "status": "Success",
                    "return_code": result.returncode,
                    "atom_count": validation["atom_count"],
                    "charge_count": validation["charge_count"],
                    "nonzero_charge_count": validation["nonzero_charge_count"],
                    "net_partial_charge": validation["net_partial_charge"],
                    "has_root": validation["has_root"],
                    "has_torsdof": validation["has_torsdof"],
                    "has_branch": validation["has_branch"],
                    "validation_status": validation["validation_status"],
                    "validation_notes": validation["validation_notes"],
                    "stderr": stderr_text,
                }
            )

            marker = "⚠" if validation["validation_status"] == "Warning" else "✅"
            print(
                f"  {marker} {output_path.name} | "
                f"atom={validation['atom_count']} | "
                f"charge_sum={validation['net_partial_charge']:.4f}"
            )

        else:
            failed_count += 1
            failure_names.append(input_path.name)

            # Hapus output kosong/tidak valid agar tidak dianggap berhasil
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()

            report_rows.append(
                {
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "status": "Failed",
                    "return_code": result.returncode,
                    "atom_count": 0,
                    "charge_count": 0,
                    "nonzero_charge_count": 0,
                    "net_partial_charge": "",
                    "has_root": False,
                    "has_torsdof": False,
                    "has_branch": False,
                    "validation_status": "Failed",
                    "validation_notes": "Konversi gagal atau output tidak valid",
                    "stderr": stderr_text,
                }
            )

            warning_blocks.append(
                f"{input_path.name} — FAILED\n"
                f"Return code: {result.returncode}\n"
                f"Open Babel stderr:\n{stderr_text or '-'}\n"
                + ("-" * 72)
            )

            print("  ❌ Gagal")

    timestamp = datetime.now().isoformat(timespec="seconds")
    csv_path = report_dir / "pdb_to_pdbqt_report.csv"
    success_path = report_dir / "success_pdbqt.txt"
    skipped_path = report_dir / "skipped_existing_pdbqt.txt"
    failures_path = report_dir / "failed_pdbqt.txt"
    warnings_path = report_dir / "pdbqt_warnings.txt"
    summary_path = report_dir / "pdbqt_conversion_summary.txt"

    fieldnames = [
        "input_file",
        "output_file",
        "status",
        "return_code",
        "atom_count",
        "charge_count",
        "nonzero_charge_count",
        "net_partial_charge",
        "has_root",
        "has_torsdof",
        "has_branch",
        "validation_status",
        "validation_notes",
        "stderr",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    write_text_lines(success_path, success_names)
    write_text_lines(skipped_path, skipped_names)
    write_text_lines(failures_path, failure_names)
    write_text_lines(warnings_path, warning_blocks)

    summary = (
        f"Run time             : {timestamp}\n"
        f"Open Babel           : {get_obabel_version()}\n"
        f"Input directory      : {input_dir}\n"
        f"Output directory     : {output_dir}\n"
        f"Total PDB            : {len(pdb_files)}\n"
        f"Converted successfully: {success_count}\n"
        f"Skipped existing     : {skipped_count}\n"
        f"Failed               : {failed_count}\n"
        f"With warnings        : {warning_count}\n"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print("=" * 72)
    print("KONVERSI SELESAI")
    print("=" * 72)
    print(f"Berhasil : {success_count}")
    print(f"Dilewati : {skipped_count}")
    print(f"Gagal    : {failed_count}")
    print(f"Warning  : {warning_count}")
    print(f"PDBQT    : {output_dir}")
    print(f"Laporan  : {report_dir}")
    print("=" * 72)

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
