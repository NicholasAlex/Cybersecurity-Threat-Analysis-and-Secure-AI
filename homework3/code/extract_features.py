"""
extract_features.py

Extracts texture-based features (HOG and LBP) from the grayscale malware
images produced by binary_to_image.py. All images are resized to a common
size first so that feature vectors have consistent length across samples
(image heights vary naturally since they depend on original file size).

Two feature sets are saved separately so you can compare classifiers trained
on each (per the assignment's suggestion to "make some comparison"):

    features/hog_features.npz   -> X (HOG vectors), y (family labels), files
    features/lbp_features.npz   -> X (LBP histograms), y (family labels), files

Usage:
    python3 extract_features.py \
        --input images/ \
        --output features/ \
        --resize 128

Notes:
    - GIST is not included: there is no well-maintained Python package for it.
      HOG and LBP are used instead as texture descriptors, which is explicitly
      allowed by the assignment ("feature extraction methods are not limited").
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern


def load_and_resize(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def extract_hog(arr: np.ndarray) -> np.ndarray:
    return hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_lbp(arr: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(arr, P=P, R=R, method="uniform")
    n_bins = P + 2  # uniform LBP produces P+2 distinct patterns
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def main():
    parser = argparse.ArgumentParser(description="Extract HOG and LBP features from malware images.")
    parser.add_argument("--input", required=True, type=Path,
                         help="Folder containing per-family subfolders of PNG images")
    parser.add_argument("--output", required=True, type=Path,
                         help="Output folder for saved feature files (.npz)")
    parser.add_argument("--resize", type=int, default=128,
                         help="Resize images to (size x size) before feature extraction (default: 128)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: input folder not found: {args.input}")

    family_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not family_dirs:
        sys.exit(f"Error: no family subfolders found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    hog_vectors, lbp_vectors, labels, filenames = [], [], [], []

    for family_dir in family_dirs:
        family = family_dir.name
        image_paths = sorted(p for p in family_dir.iterdir() if p.suffix.lower() == ".png")

        count = 0
        for img_path in image_paths:
            try:
                arr = load_and_resize(img_path, args.resize)
            except Exception as e:
                print(f"  [warn] failed to load {img_path.name} ({family}): {e}", file=sys.stderr)
                continue

            hog_vectors.append(extract_hog(arr))
            lbp_vectors.append(extract_lbp(arr))
            labels.append(family)
            filenames.append(f"{family}/{img_path.name}")
            count += 1

        print(f"{family:30s} extracted {count}")

    X_hog = np.array(hog_vectors)
    X_lbp = np.array(lbp_vectors)
    y = np.array(labels)
    files = np.array(filenames)

    hog_path = args.output / "hog_features.npz"
    lbp_path = args.output / "lbp_features.npz"

    np.savez_compressed(hog_path, X=X_hog, y=y, files=files)
    np.savez_compressed(lbp_path, X=X_lbp, y=y, files=files)

    print(f"\nSaved HOG features: {hog_path}  shape={X_hog.shape}")
    print(f"Saved LBP features: {lbp_path}  shape={X_lbp.shape}")
    print(f"Total samples: {len(y)}  Families: {len(set(labels))}")


if __name__ == "__main__":
    main()