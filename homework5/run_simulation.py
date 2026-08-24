#!/usr/bin/env python3
"""
run_simulation.py
------------------
Federated training: an in-process Flower simulation over `--num-clients`
clients, evaluated with the SAME leave-one-sample-out protocol as
centralized.py so the two numbers are directly comparable.

Per fold: the 4 training samples are partitioned across clients (IID or
non-IID, see data_utils.partition), FedAvg runs for `--rounds` rounds of
`--local-epochs` local epochs each, and the FINAL global model is evaluated
once on the held-out sample -- exactly the same held-out evaluation
centralized.py performs, just fed a federated-averaged model instead of a
centrally-trained one.

Usage:
    python3 run_simulation.py --partition iid
    python3 run_simulation.py --partition non_iid --rounds 40 --local-epochs 1
"""

import argparse
import json
import logging
import warnings

import numpy as np
import torch
import flwr as fl

import data_utils as du
from model import HybridCNN, get_weights, set_weights, make_loader, evaluate, count_parameters
from client import make_client_fn
from server import build_strategy

# start_simulation() is Flower's legacy (but still functional) simulation
# entry point; the newer ServerApp/ClientApp workflow needs a `flwr run`
# project layout that doesn't fit a single runnable script. Silencing just
# its deprecation notice, not warnings generally.
warnings.filterwarnings("ignore", message=r".*start_simulation.*deprecated.*")
logging.getLogger("ray").setLevel(logging.ERROR)


def set_seed(seed):
    """Full reproducibility -- same contract as centralized.py."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_loso(samples, vocab, label_map, args, device):
    """Leave-one-sample-out CV, each fold trained by federated simulation."""
    idx_to_label = {v: k for k, v in label_map.items()}
    fold_records = []
    example_history = None  # keep one fold's round-by-round loss curve for figures.py

    for fold, train_s, test_s in du.loso_splits(samples):
        set_seed(args.seed + fold)   # fresh global init per fold, deterministic

        shards = du.partition(train_s, num_clients=args.num_clients,
                               mode=args.partition, seed=args.seed + fold)

        # ---- held-out evaluation set: windows from the ONE left-out sample --
        test_w = du.build_windows(test_s, vocab)
        train_w_all = du.build_windows(train_s, vocab)
        du.assert_no_leakage(train_w_all, test_w)   # hard guarantee, as in centralized.py

        Xa_te, Xn_te, y_te, sid_te = du.to_arrays(test_w, label_map)
        te_loader = make_loader(Xa_te, Xn_te, y_te, batch_size=args.batch_size, shuffle=False)

        global_model = HybridCNN(vocab_size=len(vocab), num_classes=len(label_map),
                                  n_net_features=du.N_NET_FEATURES)
        initial_weights = get_weights(global_model)

        client_fn = make_client_fn(shards, vocab, label_map, du.N_NET_FEATURES,
                                    batch_size=args.batch_size,
                                    local_epochs=args.local_epochs,
                                    lr=args.lr, device=device)

        strategy = build_strategy(initial_weights, args.num_clients)

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
            ray_init_args={"log_to_driver": False, "logging_level": logging.ERROR,
                           "include_dashboard": False},
        )

        if example_history is None:
            losses = history.metrics_distributed_fit.get("train_loss", [])
            example_history = {"fold": fold, "train_loss_by_round": losses}

        # ---- load the FedAvg-aggregated weights into a fresh model ----------
        final_weights = strategy.latest_weights
        assert final_weights is not None, "FedAvg produced no aggregated weights"
        set_weights(global_model, final_weights)

        _, win_acc, preds = evaluate(global_model, te_loader, device=device)

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
            "n_test_windows": len(test_w),
            "client_shard_sizes": [len(s) for s in shards],
        })

        mark = "OK " if sample_correct else "MISS"
        print(f"    fold {fold}: {mark} {test_s[0]['filename']:<20} "
              f"true={true_family:<11} pred={pred_family:<11} "
              f"window_acc={win_acc:.2f}  shards={[len(s) for s in shards]}")

    return fold_records, example_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.json")
    ap.add_argument("--vocab", default="data/api_vocab.json")
    ap.add_argument("--out", default=None,
                     help="default: results/federated_<partition>.json")
    ap.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5,
                     help="repeat the whole LOSO protocol with this many seeds "
                          "and report mean +/- std (set 1 for a single run)")
    # Ray workers each run in their own process; keeping this off "auto" by
    # default avoids every worker independently probing for a GPU.
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/federated_{args.partition}.json"

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
        else ("cpu" if args.device == "auto" else args.device)

    vocab = du.load_vocab(args.vocab)
    samples, unknown = du.load_samples(args.data)
    label_map = du.build_label_map(samples)

    print(f"[+] {len(samples)} usable labelled samples | families: {list(label_map)}")
    print(f"[+] federated simulation: {args.num_clients} clients, partition={args.partition}, "
          f"{args.rounds} rounds x {args.local_epochs} local epoch(s)")

    probe = HybridCNN(len(vocab), len(label_map), du.N_NET_FEATURES)
    n_params = count_parameters(probe)
    bytes_per_round = n_params * 4 * args.num_clients * 2   # fp32, up + down
    print(f"[+] HybridCNN trainable parameters: {n_params:,} "
          f"({n_params * 4 / 1024:.1f} KB/client/round, "
          f"{bytes_per_round / 1024:.1f} KB/round total across {args.num_clients} clients)")

    # ---- repeat the ENTIRE protocol across seeds, exactly as centralized.py -
    base_seed = args.seed
    runs = []
    example_history = None
    for r in range(args.n_seeds):
        args.seed = base_seed + r * 100
        print(f"\n[+] Federated LOSO ({args.partition}) -- run {r + 1}/{args.n_seeds} "
              f"(seed={args.seed}):")
        folds, hist = run_loso(samples, vocab, label_map, args, device)
        if example_history is None:
            example_history = hist
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

    n_folds = len(runs[0]["folds"])
    always_wrong = [runs[0]["folds"][i]["held_out"] for i in range(n_folds)
                     if all(r["folds"][i]["sample_correct"] == 0 for r in runs)]

    print(f"\n    SAMPLE-level LOSO accuracy = {sample_acc:.2f} +/- {sample_accs.std():.2f} "
          f"over {args.n_seeds} seeds  [{args.partition}]")
    print(f"    window-level mean accuracy = {window_acc:.2f} +/- {window_accs.std():.2f}")
    print(f"    per-seed sample accuracies : {[round(a, 2) for a in sample_accs]}")
    if always_wrong:
        print(f"\n    [!] folds wrong under EVERY seed (structural, not model-dependent):")
        for name in always_wrong:
            print(f"          {name}")

    summary = {
        "experiment": f"federated_{args.partition}",
        "partition": args.partition,
        "n_samples": len(samples),
        "families": list(label_map),
        "num_clients": args.num_clients,
        "window_size": du.WINDOW_SIZE,
        "window_stride": du.WINDOW_STRIDE,
        "n_parameters": n_params,
        "bytes_per_round_total": bytes_per_round,
        "hyperparameters": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                             "lr": args.lr, "batch_size": args.batch_size,
                             "base_seed": base_seed, "n_seeds": args.n_seeds},
        "sample_level_accuracy": round(sample_acc, 4),
        "sample_level_std": round(float(sample_accs.std()), 4),
        "window_level_accuracy": round(window_acc, 4),
        "window_level_std": round(float(window_accs.std()), 4),
        "structural_failures": always_wrong,
        "example_history": example_history,
        "runs": runs,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] written to {args.out}")


if __name__ == "__main__":
    main()
