# Malware Classification based on Static Analysis

Implementation of the image-based malware classification method from:

> Nataraj, Lakshmanan, et al. "Malware images: visualization and automatic classification." *Proceedings of the 8th International Symposium on Visualization for Cyber Security.* 2011.

This project converts malware binaries into grayscale images, extracts texture-based features, and trains machine learning models to classify samples by family.

## Overview

Malware binaries are read as raw byte streams and reshaped into 2D grayscale images, exploiting the observation that binaries from the same family exhibit similar visual/textural patterns. Two classification approaches are compared:

1. **Hand-crafted features + classical ML** — HOG/LBP features with SVM
2. **Deep learning** — CNN trained directly on raw grayscale images

## Dataset

This project uses the **MOTIF (Malware Open-source Threat Intelligence Family) dataset**:

> Joyce, Robert J., et al. "MOTIF: A Large Malware Reference Dataset with Ground Truth Family Labels." AAAI-22 Workshop on Artificial Intelligence for Cyber Security (AICS), 2022.

- 3,095 disarmed PE malware samples across 454 families
- Ground-truth family labels sourced from published threat intelligence reports
- Samples are pre-disarmed (rendered non-executable) by the dataset authors

**The dataset is NOT included in this repository.** Malware samples, even disarmed ones, are excluded from version control per `.gitignore` and are never pushed to GitHub.

### Reproducing the dataset setup

1. Install Git LFS and clone the MOTIF repository:
   ```bash
   sudo apt-get install git-lfs
   git lfs clone https://github.com/boozallen/MOTIF.git
   ```
2. Extract `MOTIF.7z` (password: `i_assume_all_risk_opening_malware`) into `MOTIF/dataset/MOTIF_defanged/`.
3. Run `code/organize_samples.py` to parse `motif_dataset.jsonl` and sort samples into per-family folders under `extracted/`.

**Safety note:** All work with raw samples was performed inside an isolated VM with networking disabled except during dataset download. Samples are only ever opened in binary read mode (`rb`) — never executed.

## Pipeline

```
extracted/<family>/<hash>          # raw binaries, sorted by family
        │
        ▼  binary_to_image.py
images/<family>/<hash>.png         # grayscale images (width fixed by file size, per Nataraj et al. Table 1)
        │
        ▼  extract_features.py
features/                          # HOG / LBP feature vectors
        │
        ▼  train_models.py
models/                            # trained SVM and CNN classifiers
        │
        ▼
results/                           # accuracy, confusion matrices, comparison plots
```

## Repository structure

```
hw3/
  code/
    organize_samples.py    # parse MOTIF labels, sort binaries by family
    binary_to_image.py     # convert binaries to grayscale PNGs
    extract_features.py    # HOG / LBP feature extraction
    train_models.py        # train + evaluate SVM and CNN classifiers
  results/                 # confusion matrices, accuracy comparison (safe to share — no malware content)
  README.md
  .gitignore
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow scikit-image scikit-learn torch matplotlib
```

## Usage

```bash
# 1. Organize raw samples by family (requires MOTIF dataset downloaded separately)
python code/organize_samples.py --jsonl MOTIF/dataset/motif_dataset.jsonl \
    --samples MOTIF/dataset/MOTIF_defanged --output extracted/ --top-n 8

# 2. Convert binaries to grayscale images
python code/binary_to_image.py --input extracted/ --output images/

# 3. Extract features
python code/extract_features.py --input images/ --output features/

# 4. Train and evaluate models
python code/train_models.py --features features/ --images images/ --output models/
```

## Methods compared

| Feature extraction | Classifier | Notes |
|---|---|---|
| HOG | SVM | Baseline, per Nataraj et al. |
| LBP | SVM | Texture-based alternative |
| Raw pixels | CNN | End-to-end deep learning comparison |

*(GIST was substituted with HOG/LBP due to lack of a maintained Python implementation.)*

## Results

See `results/` for confusion matrices and accuracy comparison across methods. Summary discussed in the accompanying report.

## References

1. Nataraj, L., et al. "Malware images: visualization and automatic classification." VizSec 2011.
2. Joyce, R. J., et al. "MOTIF: A Large Malware Reference Dataset with Ground Truth Family Labels." AICS 2022.
3. Kumar, N., Meenpal, T. "Texture-based malware family classification." ICCCNT 2019.
4. Bensaoud, A., Abudawaood, N., Kalita, J. "Classifying malware images with convolutional neural network models." IJNS 2020.
