"""
binary_to_image.py
 
Converts malware binaries into grayscale images following the method from:
Nataraj, L. et al. "Malware images: visualization and automatic classification." VizSec 2011.
 
Each binary is read as a sequence of unsigned 8-bit bytes (0-255) and reshaped
into a 2D array. The image width is chosen based on file size (per the
assignment's Table 1); height is determined by however many full rows of
that width fit in the byte stream (leftover trailing bytes are dropped).
 
Usage:
    python3 binary_to_image.py \
        --input extracted/ \
        --output images/
 
Expects --input to contain per-family subfolders of raw binaries, e.g.:
    extracted/icedid/MOTIF_<hash>
    extracted/azorult/MOTIF_<hash>
    ...
 
Produces matching structure of PNGs:
    images/icedid/MOTIF_<hash>.png
    images/azorult/MOTIF_<hash>.png
    ...
 
Notes:
    - Files are only ever opened in binary read mode ("rb"). Nothing is executed.
"""

import argparse
import sys
from pathlib import Path
 
import numpy as np
from PIL import Image

# Table 1 from the assignment: (max_file_size_bytes, image_width)
# File size ranges are upper-bounds; the first matching bracket is used.
WIDTH_TABLE = [
    (10 * 1024, 32),
    (30 * 1024, 64),
    (60 * 1024, 128),
    (100 * 1024, 256),
    (200 * 1024, 384),
    (500 * 1024, 512),
    (1000 * 1024, 768),
    (float("inf"), 1024),
]
 
 
def width_for_size(size_bytes: int) -> int:
    """Return the recommended image width for a given file size in bytes."""
    for max_size, width in WIDTH_TABLE:
        if size_bytes < max_size:
            return width
    return WIDTH_TABLE[-1][1]
 
 
def binary_to_image(path: Path) -> Image.Image:
    """Read a binary file and convert it into a grayscale PIL Image."""
    data = path.read_bytes()
    size = len(data)
 
    if size == 0:
        raise ValueError("empty file")
 
    width = width_for_size(size)
    arr = np.frombuffer(data, dtype=np.uint8)
 
    height = len(arr) // width
    if height == 0:
        # File smaller than one row at this width — pad up to one full row.
        pad_len = width - len(arr)
        arr = np.concatenate([arr, np.zeros(pad_len, dtype=np.uint8)])
        height = 1
    else:
        # Drop trailing bytes that don't fill a complete row.
        arr = arr[: height * width]
 
    arr2d = arr.reshape((height, width))
    return Image.fromarray(arr2d, mode="L")
 
 
def main():
    parser = argparse.ArgumentParser(description="Convert malware binaries to grayscale images.")
    parser.add_argument("--input", required=True, type=Path,
                         help="Folder containing per-family subfolders of raw binaries")
    parser.add_argument("--output", required=True, type=Path,
                         help="Output folder for per-family subfolders of PNG images")
    args = parser.parse_args()
 
    if not args.input.exists():
        sys.exit(f"Error: input folder not found: {args.input}")
 
    family_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not family_dirs:
        sys.exit(f"Error: no family subfolders found in {args.input}")
 
    total_converted = 0
    total_failed = 0
 
    for family_dir in family_dirs:
        family = family_dir.name
        out_dir = args.output / family
        out_dir.mkdir(parents=True, exist_ok=True)
 
        sample_files = [p for p in family_dir.iterdir() if p.is_file()]
        converted = 0
        failed = 0
 
        for sample_path in sample_files:
            try:
                img = binary_to_image(sample_path)
            except Exception as e:
                print(f"  [warn] failed to convert {sample_path.name} ({family}): {e}",
                      file=sys.stderr)
                failed += 1
                continue
 
            out_path = out_dir / f"{sample_path.name}.png"
            img.save(out_path)
            converted += 1
 
        print(f"{family:30s} converted {converted:4d}  failed {failed}")
        total_converted += converted
        total_failed += failed
 
    print(f"\nDone. Total converted: {total_converted}  Total failed: {total_failed}")
 
 
if __name__ == "__main__":
    main()