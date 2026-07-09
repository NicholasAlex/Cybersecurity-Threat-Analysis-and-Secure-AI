"""
organize_samples.py

Reads the MOTIF dataset's motif_dataset.jsonl metadata file, counts how many
samples exist per malware family, and copies the raw binaries for the top-N
most populous families into per-family folders for downstream processing.

Usage:
    python3 organize_samples.py \
        --jsonl ../MOTIF/dataset/motif_dataset.jsonl \
        --samples ../MOTIF/dataset/MOTIF_defanged \
        --output extracted/ \
        --top-n 8 \
        --min-samples 20

Notes:
    - Only reads files in binary mode and copies them; never executes anything.
    - Each MOTIF sample file on disk is named "MOTIF_<md5>" (no extension).
    - The JSONL "reported_family" field is used as the human-readable family
      name. Family names are sanitized (lowercased, spaces -> underscores) so
      they're safe to use as folder names.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def sanitize_family_name(name: str) -> str:
    """Turn a family name into a filesystem-safe folder name."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def load_records(jsonl_path: Path):
    """Load all records from the MOTIF JSONL metadata file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {line_num}: {e}", file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description="Organize MOTIF malware samples by family.")
    parser.add_argument("--jsonl", required=True, type=Path,
                         help="Path to motif_dataset.jsonl")
    parser.add_argument("--samples", required=True, type=Path,
                         help="Path to folder containing MOTIF_<md5> sample files")
    parser.add_argument("--output", required=True, type=Path,
                         help="Output folder for per-family subfolders")
    parser.add_argument("--top-n", type=int, default=8,
                         help="Number of top families (by sample count) to include (default: 8)")
    parser.add_argument("--min-samples", type=int, default=0,
                         help="Skip families with fewer than this many samples (default: 0, no minimum)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be done without copying any files")
    args = parser.parse_args()

    if not args.jsonl.exists():
        sys.exit(f"Error: JSONL file not found: {args.jsonl}")
    if not args.samples.exists():
        sys.exit(f"Error: samples folder not found: {args.samples}")

    print(f"Loading records from {args.jsonl} ...")
    records = load_records(args.jsonl)
    print(f"  Loaded {len(records)} records.")

    # Count samples per family
    family_counts = Counter(
        sanitize_family_name(rec["reported_family"])
        for rec in records
        if rec.get("reported_family")
    )

    print(f"\nFound {len(family_counts)} distinct families.")
    print("\nTop families by sample count:")
    for family, count in family_counts.most_common(20):
        print(f"  {family:30s} {count}")

    # Select families to process
    eligible = [
        (family, count) for family, count in family_counts.most_common()
        if count >= args.min_samples
    ]
    selected = eligible[: args.top_n]

    if not selected:
        sys.exit(
            f"\nNo families meet --min-samples={args.min_samples}. "
            "Lower --min-samples or check your JSONL file."
        )

    print(f"\nSelected {len(selected)} families for extraction "
          f"(top-n={args.top_n}, min-samples={args.min_samples}):")
    for family, count in selected:
        print(f"  {family:30s} {count} samples")

    selected_families = {family for family, _ in selected}

    if args.dry_run:
        print("\n[dry-run] No files will be copied.")

    # Copy matching sample files into extracted/<family>/
    args.output.mkdir(parents=True, exist_ok=True)

    copied = Counter()
    missing = []

    for rec in records:
        family_raw = rec.get("reported_family")
        if not family_raw:
            continue
        family = sanitize_family_name(family_raw)
        if family not in selected_families:
            continue

        md5 = rec.get("md5")
        if not md5:
            continue

        src = args.samples / f"MOTIF_{md5}"
        if not src.exists():
            missing.append(str(src))
            continue

        dest_dir = args.output / family
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        if not args.dry_run:
            shutil.copy2(src, dest)  # binary copy, never opened/executed
        copied[family] += 1

    print("\nDone. Samples copied per family:")
    total = 0
    for family, count in copied.most_common():
        print(f"  {family:30s} {count}")
        total += count
    print(f"\nTotal files copied: {total}")

    if missing:
        print(f"\n[warn] {len(missing)} referenced sample files were not found on disk "
              f"(showing first 5):", file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)


if __name__ == "__main__":
    main()