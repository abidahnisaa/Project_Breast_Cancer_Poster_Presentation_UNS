#!/usr/bin/env python3
"""
Robust sequential batch docking for DCAF13 with AutoDock Vina.

Default project layout:
    breast_cancer_docking/
    ├── DCAF13_7MQA_receptor.pdbqt
    ├── DCAF13_7MQA_receptor.box.txt
    ├── ligand_output/
    │   └── pdbqt_files/
    │       ├── ligand_001.pdbqt
    │       └── ...
    └── batch_docking_dcaf13.py

Outputs:
    docking_results/
    ├── poses/
    ├── logs/
    └── reports/
        ├── docking_results_all.csv
        ├── docking_results_ranked.csv
        ├── docking_top20.csv
        ├── docking_failures.csv
        ├── docking_campaign_metadata.txt
        └── docking_summary.txt

The script:
- checks receptor, grid configuration, Vina, and ligand inputs;
- runs every ligand sequentially;
- uses a reproducible ligand-specific random seed;
- saves one pose and one log per ligand;
- resumes safely when valid outputs already exist;
- tracks progress, elapsed time, and estimated remaining time;
- extracts Vina scores and produces ranked CSV reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional


VINA_RESULT_PATTERN = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s+(-?\d+(?:\.\d+)?)"
)

TABLE_RESULT_PATTERN = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)


def parse_args() -> argparse.Namespace:
    cpu_default = max(1, min(8, (os.cpu_count() or 2) // 2))

    parser = argparse.ArgumentParser(
        description="Batch docking semua ligan PDBQT terhadap DCAF13."
    )
    parser.add_argument(
        "--receptor",
        default="DCAF13_7MQA_receptor.pdbqt",
        help="Receptor PDBQT.",
    )
    parser.add_argument(
        "--config",
        default="DCAF13_7MQA_receptor.box.txt",
        help="Vina grid configuration file.",
    )
    parser.add_argument(
        "--ligands-dir",
        default="ligand_output/pdbqt_files",
        help="Folder yang berisi ligand PDBQT.",
    )
    parser.add_argument(
        "--output-dir",
        default="docking_results",
        help="Folder output.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="Vina exhaustiveness untuk screening awal (default: 8).",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=9,
        help="Jumlah pose maksimum per ligan (default: 9).",
    )
    parser.add_argument(
        "--energy-range",
        type=float,
        default=4.0,
        help="Energy range kcal/mol (default: 4).",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=cpu_default,
        help=f"Thread CPU per proses Vina (default: {cpu_default}).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=2026,
        help="Seed dasar untuk reproducibility (default: 2026).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ulangi docking meskipun output valid sudah tersedia.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah ligan untuk debugging.",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def get_vina_version() -> str:
    result = run_command(["vina", "--version"])
    output = (result.stdout or "").strip()
    return output or "unknown"


def check_pdbqt_file(path: Path, file_type: str) -> tuple[bool, str]:
    """
    Lightweight format check.
    """
    if not path.exists():
        return False, "file tidak ditemukan"

    if path.stat().st_size == 0:
        return False, "file kosong"

    atom_count = 0
    has_root = False
    has_torsdof = False
    has_vina_result = False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    atom_count += 1
                elif line.strip() == "ROOT":
                    has_root = True
                elif line.startswith("TORSDOF"):
                    has_torsdof = True
                elif VINA_RESULT_PATTERN.search(line):
                    has_vina_result = True
    except OSError as exc:
        return False, f"gagal dibaca: {exc}"

    if atom_count == 0:
        return False, "tidak ada ATOM/HETATM"

    if file_type == "ligand":
        if not has_root:
            return False, "ROOT tidak ditemukan"
        if not has_torsdof:
            return False, "TORSDOF tidak ditemukan"

    if file_type == "pose" and not has_vina_result:
        return False, "REMARK VINA RESULT tidak ditemukan"

    return True, f"{atom_count} atom"


def parse_pose_scores(path: Path) -> list[float]:
    scores: list[float] = []

    if not path.exists():
        return scores

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = VINA_RESULT_PATTERN.search(line)
            if match:
                scores.append(float(match.group(1)))

    return scores


def parse_stdout_scores(text: str) -> list[float]:
    scores: list[tuple[int, float]] = []

    for line in text.splitlines():
        match = TABLE_RESULT_PATTERN.match(line)
        if match:
            mode = int(match.group(1))
            affinity = float(match.group(2))
            scores.append((mode, affinity))

    scores.sort(key=lambda item: item[0])
    return [score for _, score in scores]


def ligand_seed(base_seed: int, ligand_name: str) -> int:
    """
    Stable, reproducible, ligand-specific positive integer seed.
    """
    checksum = zlib.crc32(ligand_name.encode("utf-8"))
    return int((base_seed + checksum) % 2_147_483_646) + 1


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    project_dir = Path.cwd().resolve()
    receptor = Path(args.receptor).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    ligands_dir = Path(args.ligands_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    poses_dir = output_dir / "poses"
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"

    for directory in (poses_dir, logs_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DCAF13 — BATCH MOLECULAR DOCKING")
    print("=" * 78)
    print(f"Project         : {project_dir}")
    print(f"Receptor        : {receptor}")
    print(f"Grid config     : {config}")
    print(f"Ligands         : {ligands_dir}")
    print(f"Output          : {output_dir}")
    print(f"Exhaustiveness  : {args.exhaustiveness}")
    print(f"Num modes       : {args.num_modes}")
    print(f"Energy range    : {args.energy_range}")
    print(f"CPU / ligand    : {args.cpu}")
    print(f"Base seed       : {args.base_seed}")
    print(f"Resume mode     : {'OFF (overwrite)' if args.overwrite else 'ON'}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Preflight checks
    # ------------------------------------------------------------------
    if shutil.which("vina") is None:
        print(
            "\nERROR: AutoDock Vina tidak ditemukan dalam environment aktif.\n"
            "Install dengan:\n"
            "conda install -n meeko_env -c conda-forge vina -y"
        )
        return 1

    receptor_ok, receptor_note = check_pdbqt_file(receptor, "receptor")
    if not receptor_ok:
        print(f"\nERROR receptor: {receptor_note}")
        return 1

    if not config.exists() or config.stat().st_size == 0:
        print(f"\nERROR: file config tidak ditemukan/kosong: {config}")
        return 1

    if not ligands_dir.exists():
        print(f"\nERROR: folder ligan tidak ditemukan: {ligands_dir}")
        return 1

    ligand_files = sorted(
        path for path in ligands_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdbqt"
    )

    if args.limit is not None:
        ligand_files = ligand_files[: args.limit]

    if not ligand_files:
        print(f"\nERROR: tidak ada file PDBQT di {ligands_dir}")
        return 1

    vina_version = get_vina_version()

    print("\nPREFLIGHT")
    print("-" * 78)
    print(f"Vina             : {vina_version}")
    print(f"Receptor check   : OK ({receptor_note})")
    print(f"Jumlah ligan     : {len(ligand_files)}")

    valid_ligands: list[Path] = []
    preflight_failures: list[dict] = []

    for ligand in ligand_files:
        is_valid, note = check_pdbqt_file(ligand, "ligand")
        if is_valid:
            valid_ligands.append(ligand)
        else:
            preflight_failures.append(
                {
                    "ligand": ligand.stem,
                    "input_file": str(ligand),
                    "status": "Invalid_input",
                    "best_affinity_kcal_mol": "",
                    "second_affinity_kcal_mol": "",
                    "score_gap_kcal_mol": "",
                    "n_modes": 0,
                    "runtime_seconds": 0,
                    "seed": "",
                    "pose_file": "",
                    "log_file": "",
                    "error": note,
                }
            )

    print(f"Ligan valid      : {len(valid_ligands)}")
    print(f"Ligan invalid    : {len(preflight_failures)}")

    if not valid_ligands:
        print("\nERROR: tidak ada ligan valid untuk docking.")
        return 1

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata_path = reports_dir / "docking_campaign_metadata.txt"
    config_text = config.read_text(encoding="utf-8", errors="replace")

    metadata = (
        "DCAF13 DOCKING CAMPAIGN METADATA\n"
        "========================================\n"
        f"Started               : {datetime.now().isoformat(timespec='seconds')}\n"
        f"Project directory     : {project_dir}\n"
        f"Python                : {sys.version.replace(os.linesep, ' ')}\n"
        f"Platform              : {platform.platform()}\n"
        f"AutoDock Vina         : {vina_version}\n"
        f"Receptor              : {receptor}\n"
        f"Receptor SHA256       : {sha256_file(receptor)}\n"
        f"Grid config           : {config}\n"
        f"Grid config SHA256    : {sha256_file(config)}\n"
        f"Ligand directory      : {ligands_dir}\n"
        f"Input ligand count    : {len(ligand_files)}\n"
        f"Valid ligand count    : {len(valid_ligands)}\n"
        f"Invalid ligand count  : {len(preflight_failures)}\n"
        f"Exhaustiveness        : {args.exhaustiveness}\n"
        f"Number of modes       : {args.num_modes}\n"
        f"Energy range          : {args.energy_range}\n"
        f"CPU threads           : {args.cpu}\n"
        f"Base random seed      : {args.base_seed}\n"
        f"Overwrite             : {args.overwrite}\n"
        "\nGRID CONFIGURATION\n"
        "----------------------------------------\n"
        f"{config_text.strip()}\n"
    )
    metadata_path.write_text(metadata, encoding="utf-8")

    # ------------------------------------------------------------------
    # Dock all ligands
    # ------------------------------------------------------------------
    results: list[dict] = list(preflight_failures)
    total = len(valid_ligands)
    campaign_start = time.perf_counter()
    completed_runtimes: list[float] = []

    for index, ligand in enumerate(valid_ligands, start=1):
        ligand_name = ligand.stem
        pose_path = poses_dir / f"{ligand_name}_out.pdbqt"
        log_path = logs_dir / f"{ligand_name}.log.txt"
        seed = ligand_seed(args.base_seed, ligand_name)

        elapsed_campaign = time.perf_counter() - campaign_start

        if completed_runtimes:
            average_runtime = sum(completed_runtimes) / len(completed_runtimes)
            remaining_seconds = average_runtime * (total - index + 1)
            eta_text = format_duration(remaining_seconds)
        else:
            eta_text = "menghitung..."

        print()
        print("-" * 78)
        print(
            f"[{index}/{total}] {ligand_name}\n"
            f"Elapsed: {format_duration(elapsed_campaign)} | "
            f"Estimated remaining: {eta_text}"
        )

        # Resume valid existing output
        if not args.overwrite and pose_path.exists():
            pose_ok, pose_note = check_pdbqt_file(pose_path, "pose")

            if pose_ok:
                scores = parse_pose_scores(pose_path)
                best = scores[0] if scores else None
                second = scores[1] if len(scores) > 1 else None
                gap = (
                    round(second - best, 3)
                    if best is not None and second is not None
                    else None
                )

                results.append(
                    {
                        "ligand": ligand_name,
                        "input_file": str(ligand),
                        "status": "Skipped_existing",
                        "best_affinity_kcal_mol": best,
                        "second_affinity_kcal_mol": second,
                        "score_gap_kcal_mol": gap,
                        "n_modes": len(scores),
                        "runtime_seconds": 0,
                        "seed": seed,
                        "pose_file": str(pose_path),
                        "log_file": str(log_path) if log_path.exists() else "",
                        "error": "",
                    }
                )

                print(
                    f"SKIP: output valid sudah ada | "
                    f"best = {best if best is not None else 'N/A'} kcal/mol"
                )
                continue

        command = [
            "vina",
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--config",
            str(config),
            "--exhaustiveness",
            str(args.exhaustiveness),
            "--num_modes",
            str(args.num_modes),
            "--energy_range",
            str(args.energy_range),
            "--cpu",
            str(args.cpu),
            "--seed",
            str(seed),
            "--out",
            str(pose_path),
        ]

        ligand_start = time.perf_counter()
        result = run_command(command)
        runtime = time.perf_counter() - ligand_start
        completed_runtimes.append(runtime)

        log_header = (
            f"COMMAND:\n{' '.join(command)}\n\n"
            f"RETURN CODE: {result.returncode}\n"
            f"RUNTIME_SECONDS: {runtime:.3f}\n\n"
            "VINA OUTPUT:\n"
        )
        log_path.write_text(
            log_header + (result.stdout or ""),
            encoding="utf-8",
        )

        pose_ok, pose_note = check_pdbqt_file(pose_path, "pose")

        if result.returncode == 0 and pose_ok:
            scores = parse_pose_scores(pose_path)

            if not scores:
                scores = parse_stdout_scores(result.stdout or "")

            best = scores[0] if scores else None
            second = scores[1] if len(scores) > 1 else None
            gap = (
                round(second - best, 3)
                if best is not None and second is not None
                else None
            )

            status = "Success" if best is not None else "Success_no_score"

            results.append(
                {
                    "ligand": ligand_name,
                    "input_file": str(ligand),
                    "status": status,
                    "best_affinity_kcal_mol": best,
                    "second_affinity_kcal_mol": second,
                    "score_gap_kcal_mol": gap,
                    "n_modes": len(scores),
                    "runtime_seconds": round(runtime, 3),
                    "seed": seed,
                    "pose_file": str(pose_path),
                    "log_file": str(log_path),
                    "error": "" if best is not None else "score tidak terbaca",
                }
            )

            print(
                f"SUCCESS | best = "
                f"{best if best is not None else 'N/A'} kcal/mol | "
                f"modes = {len(scores)} | "
                f"time = {format_duration(runtime)}"
            )
        else:
            error_tail = "\n".join((result.stdout or "").splitlines()[-12:])

            results.append(
                {
                    "ligand": ligand_name,
                    "input_file": str(ligand),
                    "status": "Failed",
                    "best_affinity_kcal_mol": "",
                    "second_affinity_kcal_mol": "",
                    "score_gap_kcal_mol": "",
                    "n_modes": 0,
                    "runtime_seconds": round(runtime, 3),
                    "seed": seed,
                    "pose_file": str(pose_path) if pose_path.exists() else "",
                    "log_file": str(log_path),
                    "error": (
                        f"Vina return code {result.returncode}; "
                        f"pose check: {pose_note}; {error_tail}"
                    ),
                }
            )

            print(
                f"FAILED | return code = {result.returncode} | "
                f"pose check = {pose_note} | "
                f"time = {format_duration(runtime)}"
            )

        # Checkpoint after every ligand
        checkpoint_fields = [
            "ligand",
            "input_file",
            "status",
            "best_affinity_kcal_mol",
            "second_affinity_kcal_mol",
            "score_gap_kcal_mol",
            "n_modes",
            "runtime_seconds",
            "seed",
            "pose_file",
            "log_file",
            "error",
        ]
        write_csv(
            reports_dir / "docking_results_checkpoint.csv",
            results,
            checkpoint_fields,
        )

    # ------------------------------------------------------------------
    # Final reports
    # ------------------------------------------------------------------
    campaign_runtime = time.perf_counter() - campaign_start

    fields = [
        "rank",
        "ligand",
        "input_file",
        "status",
        "best_affinity_kcal_mol",
        "second_affinity_kcal_mol",
        "score_gap_kcal_mol",
        "n_modes",
        "runtime_seconds",
        "seed",
        "pose_file",
        "log_file",
        "error",
    ]

    # Add rank after sorting valid scores from strongest (most negative)
    valid_scored = [
        row for row in results
        if isinstance(row.get("best_affinity_kcal_mol"), (int, float))
    ]
    valid_scored.sort(key=lambda row: row["best_affinity_kcal_mol"])

    rank_lookup = {
        row["ligand"]: rank
        for rank, row in enumerate(valid_scored, start=1)
    }

    all_rows = []
    for row in results:
        copied = dict(row)
        copied["rank"] = rank_lookup.get(row["ligand"], "")
        all_rows.append(copied)

    ranked_rows = sorted(
        all_rows,
        key=lambda row: (
            row["rank"] == "",
            row["rank"] if row["rank"] != "" else 10**12,
            row["ligand"],
        ),
    )

    failures = [
        row for row in ranked_rows
        if row["status"] in {"Failed", "Invalid_input", "Success_no_score"}
    ]

    top20 = [row for row in ranked_rows if row["rank"] != ""][:20]

    write_csv(
        reports_dir / "docking_results_all.csv",
        all_rows,
        fields,
    )
    write_csv(
        reports_dir / "docking_results_ranked.csv",
        ranked_rows,
        fields,
    )
    write_csv(
        reports_dir / "docking_top20.csv",
        top20,
        fields,
    )
    write_csv(
        reports_dir / "docking_failures.csv",
        failures,
        fields,
    )

    success_count = sum(
        row["status"] in {"Success", "Skipped_existing"}
        for row in all_rows
    )
    failed_count = len(failures)
    skipped_count = sum(
        row["status"] == "Skipped_existing"
        for row in all_rows
    )

    best_row = valid_scored[0] if valid_scored else None

    summary_lines = [
        "DCAF13 BATCH DOCKING SUMMARY",
        "=" * 60,
        f"Finished              : {datetime.now().isoformat(timespec='seconds')}",
        f"Total input ligands   : {len(ligand_files)}",
        f"Valid input ligands   : {len(valid_ligands)}",
        f"Successful/resumed    : {success_count}",
        f"Skipped existing      : {skipped_count}",
        f"Failures              : {failed_count}",
        f"Total campaign time   : {format_duration(campaign_runtime)}",
        f"Average run time      : "
        f"{format_duration(sum(completed_runtimes) / len(completed_runtimes)) if completed_runtimes else 'N/A'}",
        "",
    ]

    if best_row:
        summary_lines.extend(
            [
                "BEST-SCORING LIGAND",
                "-" * 60,
                f"Ligand                : {best_row['ligand']}",
                f"Affinity              : "
                f"{best_row['best_affinity_kcal_mol']} kcal/mol",
                f"Pose                  : {best_row['pose_file']}",
                "",
            ]
        )

    summary_lines.extend(
        [
            "OUTPUTS",
            "-" * 60,
            f"Ranked results        : {reports_dir / 'docking_results_ranked.csv'}",
            f"Top 20                : {reports_dir / 'docking_top20.csv'}",
            f"Failures              : {reports_dir / 'docking_failures.csv'}",
            f"Poses                 : {poses_dir}",
            f"Logs                  : {logs_dir}",
        ]
    )

    summary_path = reports_dir / "docking_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("BATCH DOCKING COMPLETED")
    print("=" * 78)
    print(f"Successful/resumed : {success_count}")
    print(f"Skipped existing   : {skipped_count}")
    print(f"Failures           : {failed_count}")
    print(f"Campaign time      : {format_duration(campaign_runtime)}")

    if best_row:
        print(
            f"Best ligand        : {best_row['ligand']} "
            f"({best_row['best_affinity_kcal_mol']} kcal/mol)"
        )

    print(f"Ranked report      : {reports_dir / 'docking_results_ranked.csv'}")
    print(f"Summary            : {summary_path}")
    print("=" * 78)

    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
