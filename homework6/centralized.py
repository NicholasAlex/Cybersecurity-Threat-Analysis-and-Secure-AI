#!/usr/bin/env python3
"""
centralized.py
--------------
Centralized (non-federated) baseline: the SAME HybridCNN, trained on all data
on one machine.

This is the number every federated result gets compared against. Without it the
FL accuracies are unanchored -- the report's core claim is "federated learning
reaches X% of centralized performance while never moving raw samples off the
client", and X is computed here.

Protocol: leave-one-sample-out cross-validation, identical to the midterm's, so
the CNN is directly comparable to the RandomForest's 0.80.

Usage:
    python3 centralized.py                       # default hyperparameters
    python3 centralized.py --epochs 30 --seed 1
"""

import argparse
import json

import numpy as np
import torch

import data_utils as du
from model import HybridCNN, make_loader, train_epochs, evaluate, count_parameters


def set_seed(seed):
    """Full reproducibility -- the report must state a seed and mean it."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_loso(samples, vocab, label_map, args):
    """Leave-one-sample-out CV. Returns per-fold records."""
    idx_to_label = {v: k for k, v in label_map.items()}
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
        else ("cpu" if args.device == "auto" else args.device)

    fold_records = []

    for fold, train_s, test_s in du.loso_splits(samples):
        set_seed(args.seed + fold)   # fresh init per fold, deterministic

        # ---- expand to windows AFTER the sample-level split ----------------
        train_w = du.build_windows(train_s, vocab)
        test_w = du.build_windows(test_s, vocab)
        du.assert_no_leakage(train_w, test_w)      # hard guarantee

        Xa_tr, Xn_tr, y_tr, _ = du.to_arrays(train_w, label_map)
        Xa_te, Xn_te, y_te, sid_te = du.to_arrays(test_w, label_map)

        # ---- class weights counter the window-count imbalance --------------
        # A long-running sample yields more windows than a short one, which
        # would otherwise bias the loss toward whichever family happened to
        # execute longest. Weighting by inverse frequency removes that artefact.
        counts = np.bincount(y_tr, minlength=len(label_map)).astype(np.float32)
        class_weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        class_weights = class_weights / class_weights.sum() * len(label_map)

        model = HybridCNN(vocab_size=len(vocab), num_classes=len(label_map),
                          n_net_features=du.N_NET_FEATURES)

        tr_loader = make_loader(Xa_tr, Xn_tr, y_tr, batch_size=args.batch_size, shuffle=True)
        te_loader = make_loader(Xa_te, Xn_te, y_te, batch_size=args.batch_size, shuffle=False)

        loss = train_epochs(model, tr_loader, epochs=args.epochs, lr=args.lr,
                            device=device, class_weights=class_weights)

        _, win_acc, preds = evaluate(model, te_loader, device=device)

        # ---- collapse windows -> one verdict for the held-out binary -------
        voted = du.majority_vote(preds, sid_te)
        true_family = test_s[0]["family"]
        pred_family = idx_to_label[list(voted.values())[0]]
        sample_correct = int(pred_family == true_family)

        fold_records.append({
            "fold": fold,
            "held_out": test_s[0]["filename"],
            "true_family": true_family,
            "pred_family": pred_family,
            "sample_correct": sample_correct,
            "window_accuracy": round(win_acc, 4),
            "n_train_windows": len(train_w),
            "n_test_windows": len(test_w),
            "final_train_loss": round(loss, 4),
        })

        mark = "OK " if sample_correct else "MISS"
        print(f"    fold {fold}: {mark} {test_s[0]['filename']:<20} "
              f"true={true_family:<11} pred={pred_family:<11} "
              f"window_acc={win_acc:.2f}  ({len(train_w)} train / {len(test_w)} test windows)")

    return fold_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.json")
    ap.add_argument("--vocab", default="data/api_vocab.json")
    ap.add_argument("--out", default="results/centralized_baseline.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="repeat the whole LOSO protocol with this many seeds "
                         "and report mean +/- std (set 1 for a single run)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    vocab = du.load_vocab(args.vocab)
    samples, unknown = du.load_samples(args.data)
    label_map = du.build_label_map(samples)

    print(f"[+] {len(samples)} usable labelled samples | families: {list(label_map)}")
    for s in samples:
        n_win = len(du.windows_from_sample(s, vocab))
        print(f"    {s['family']:<11} {s['filename']:<20} "
              f"api_len={s['api_sequence_length']:>6}  windows={n_win:>3}")

    total_windows = sum(len(du.windows_from_sample(s, vocab)) for s in samples)
    print(f"[+] windowing: size={du.WINDOW_SIZE} stride={du.WINDOW_STRIDE} "
          f"-> {total_windows} instances from {len(samples)} samples")
    print(f"[+] NOTE: {total_windows} instances, but still only {len(samples)} "
          f"INDEPENDENT samples. Evaluation is leave-one-sample-out.")

    probe = HybridCNN(len(vocab), len(label_map), du.N_NET_FEATURES)
    n_params = count_parameters(probe)
    print(f"[+] HybridCNN trainable parameters: {n_params:,} "
          f"({n_params * 4 / 1024:.1f} KB per FedAvg round per client)")

    # ---- repeat the ENTIRE protocol across seeds ---------------------------
    # With only 5 folds, a single run's accuracy moves by a whole fold (0.20)
    # on nothing but weight initialisation. Reporting one seed would be
    # cherry-picking; the report quotes mean +/- std across seeds.
    base_seed = args.seed
    runs = []
    for r in range(args.n_seeds):
        args.seed = base_seed + r * 100
        print(f"\n[+] Centralized LOSO baseline -- run {r + 1}/{args.n_seeds} "
              f"({args.epochs} epochs, lr={args.lr}, seed={args.seed}):")
        folds = run_loso(samples, vocab, label_map, args)
        runs.append({
            "seed": args.seed,
            "sample_level_accuracy": float(np.mean([f["sample_correct"] for f in folds])),
            "window_level_accuracy": float(np.mean([f["window_accuracy"] for f in folds])),
            "folds": folds,
        })
    args.seed = base_seed

    sample_accs = np.array([r["sample_level_accuracy"] for r in runs])
    window_accs = np.array([r["window_level_accuracy"] for r in runs])
    sample_acc, window_acc = float(sample_accs.mean()), float(window_accs.mean())

    # Which folds fail every single time? Those are STRUCTURAL failures
    # (e.g. the singleton RedLine class: when it is held out, no RedLine
    # example remains in training, so the fold is unwinnable by any model).
    n_folds = len(runs[0]["folds"])
    always_wrong = [runs[0]["folds"][i]["held_out"] for i in range(n_folds)
                    if all(r["folds"][i]["sample_correct"] == 0 for r in runs)]
    ceiling = (n_folds - len(always_wrong)) / n_folds

    print(f"\n    SAMPLE-level LOSO accuracy = {sample_acc:.2f} +/- {sample_accs.std():.2f} "
          f"over {args.n_seeds} seeds   <-- compare to midterm RF (0.80)")
    print(f"    window-level mean accuracy = {window_acc:.2f} +/- {window_accs.std():.2f}")
    print(f"    per-seed sample accuracies : {[round(a, 2) for a in sample_accs]}")
    if always_wrong:
        print(f"\n    [!] folds wrong under EVERY seed (structural, not model-dependent):")
        for name in always_wrong:
            print(f"          {name}  -- its family has no other training example")
        print(f"    [!] achievable ceiling on this dataset = {ceiling:.2f}")

    summary = {
        "experiment": "centralized_baseline",
        "n_samples": len(samples),
        "families": list(label_map),
        "window_size": du.WINDOW_SIZE,
        "window_stride": du.WINDOW_STRIDE,
        "total_windows": total_windows,
        "n_parameters": n_params,
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr,
                            "batch_size": args.batch_size,
                            "base_seed": base_seed, "n_seeds": args.n_seeds},
        "sample_level_accuracy": round(sample_acc, 4),
        "sample_level_std": round(float(sample_accs.std()), 4),
        "window_level_accuracy": round(window_acc, 4),
        "window_level_std": round(float(window_accs.std()), 4),
        "structural_failures": always_wrong,
        "achievable_ceiling": round(ceiling, 4),
        "runs": runs,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] written to {args.out}")


if __name__ == "__main__":
    main()
