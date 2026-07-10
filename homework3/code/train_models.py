"""
train_models.py
 
Trains and evaluates malware family classifiers using three approaches, so
results can be compared per the assignment's suggestion:
 
    1. SVM on HOG features   (features/hog_features.npz)
    2. SVM on LBP features   (features/lbp_features.npz)
    3. CNN on raw grayscale images (images/)
 
For each approach: stratified 80/20 train/test split, accuracy, per-class
precision/recall/F1, and a confusion matrix plot saved to --output.
 
Usage:
    python3 train_models.py \
        --features features/ \
        --images images/ \
        --output models/ \
        --resize 128 \
        --epochs 15
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def evaluate(name, y_true, y_pred, labels, out_dir):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)
 
    report_path = out_dir / f"{name}_report.txt"
    report_path.write_text(f"Accuracy: {acc:.4f}\n\n{report}")
 
    cm_path = out_dir / f"{name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, labels, f"{name} Confusion Matrix", cm_path)
 
    return acc

# ---------------------------------------------------------------------------
# SVM on hand-crafted features (HOG / LBP)
# ---------------------------------------------------------------------------
 
def run_svm(name, npz_path, out_dir, test_size=0.2, seed=42):
    data = np.load(npz_path, allow_pickle=True)
    X, y = data["X"], data["y"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
 
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
 
    labels = sorted(set(y))
    return evaluate(f"svm_{name}", y_test, y_pred, labels, out_dir)
 
 
# ---------------------------------------------------------------------------
# CNN on raw images
# ---------------------------------------------------------------------------
 
def run_cnn(images_dir, out_dir, resize=128, epochs=15, batch_size=16, seed=42):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("\n[skip] PyTorch not installed - skipping CNN training. "
              "Install with: pip install torch --break-system-packages", file=sys.stderr)
        return None
 
    from PIL import Image
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Gather file paths and labels
    family_dirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
    paths, labels = [], []
    for family_dir in family_dirs:
        for img_path in sorted(family_dir.glob("*.png")):
            paths.append(img_path)
            labels.append(family_dir.name)
 
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    classes = list(le.classes_)
 
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, y_all, test_size=0.2, stratify=y_all, random_state=seed
    )
 
    class MalwareImageDataset(Dataset):
        def __init__(self, paths, labels, size):
            self.paths = paths
            self.labels = labels
            self.size = size
 
        def __len__(self):
            return len(self.paths)
 
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("L").resize(
                (self.size, self.size), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
            return tensor, self.labels[idx]
 
    train_ds = MalwareImageDataset(train_paths, y_train, resize)
    test_ds = MalwareImageDataset(test_paths, y_test, resize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
 
    class SimpleCNN(nn.Module):
        def __init__(self, n_classes, input_size):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            reduced = input_size // 8
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * reduced * reduced, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )
 
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)
 
    model = SimpleCNN(len(classes), resize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
 
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
 
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())
 
    y_true_names = le.inverse_transform(y_true)
    y_pred_names = le.inverse_transform(y_pred)
 
    return evaluate("cnn_raw", y_true_names, y_pred_names, classes, out_dir)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Train and compare malware classifiers.")
    parser.add_argument("--features", required=True, type=Path,
                         help="Folder containing hog_features.npz and lbp_features.npz")
    parser.add_argument("--images", required=True, type=Path,
                         help="Folder containing per-family image subfolders")
    parser.add_argument("--output", required=True, type=Path,
                         help="Output folder for reports and confusion matrices")
    parser.add_argument("--resize", type=int, default=128,
                         help="Image size for CNN input (default: 128)")
    parser.add_argument("--epochs", type=int, default=15,
                         help="CNN training epochs (default: 15)")
    parser.add_argument("--skip-cnn", action="store_true",
                         help="Skip CNN training (only run SVM on HOG/LBP)")
    args = parser.parse_args()
 
    args.output.mkdir(parents=True, exist_ok=True)
 
    results = {}
 
    hog_path = args.features / "hog_features.npz"
    lbp_path = args.features / "lbp_features.npz"
 
    if hog_path.exists():
        results["SVM (HOG)"] = run_svm("hog", hog_path, args.output)
    else:
        print(f"[warn] {hog_path} not found, skipping HOG SVM", file=sys.stderr)
 
    if lbp_path.exists():
        results["SVM (LBP)"] = run_svm("lbp", lbp_path, args.output)
    else:
        print(f"[warn] {lbp_path} not found, skipping LBP SVM", file=sys.stderr)
 
    if not args.skip_cnn:
        print("\n=== Training CNN on raw images ===")
        cnn_acc = run_cnn(args.images, args.output, resize=args.resize, epochs=args.epochs)
        if cnn_acc is not None:
            results["CNN (raw images)"] = cnn_acc
 
    # Summary comparison plot
    if results:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(results.keys())
        accs = list(results.values())
        ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"][: len(names)])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison")
        for i, v in enumerate(accs):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fig.tight_layout()
        fig.savefig(args.output / "model_comparison.png", dpi=150)
        plt.close(fig)
 
        print("\n=== Summary ===")
        for name, acc in results.items():
            print(f"  {name:20s} {acc:.4f}")
 
 
if __name__ == "__main__":
    main()
