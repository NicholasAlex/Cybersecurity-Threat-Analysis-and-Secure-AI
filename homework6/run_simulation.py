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
from collections import Counter

import numpy as np
import torch
import flwr as fl

import data_utils as du
from model import HybridCNN, get_weights, set_weights, make_loader, evaluate, count_parameters
import dp
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


def run_loso(samples, vocab, label_map, args, device, dp_config=None):
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
                                    lr=args.lr, device=device,
                                    optimizer=args.optimizer,
                                    dp_config=dp_config, seed=args.seed + fold)

        # build_strategy returns (what Flower runs, what holds final weights).
        # Under central DP these differ: Flower runs the DP wrapper, while the
        # inner SavingFedAvg is what recorded the aggregate.
        strategy, weights_holder = build_strategy(
            initial_weights, args.num_clients, dp_config=dp_config,
            seed=args.seed + fold)

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
        final_weights = weights_holder.latest_weights
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


def run_holdout(samples, vocab, label_map, args, device, dp_config=None):
    """
    Single stratified train/test split, trained by one federated simulation.

    Replaces LOSO for datasets too large for a fold-per-sample protocol (see
    data_utils.holdout_split). Returns records shaped exactly like run_loso's
    per-fold dicts -- one per TEST SAMPLE here rather than one per LOSO fold --
    so the aggregation/printing code in main() needs no dataset-specific branch.
    """
    idx_to_label = {v: k for k, v in label_map.items()}

    # Fixed across the whole n-seeds sweep: every seed is scored against the
    # same held-out samples, so only model init / client partitioning vary.
    train_s, test_s = du.holdout_split(samples, test_frac=args.test_frac, seed=args.split_seed)

    shards = du.partition(train_s, num_clients=args.num_clients,
                           mode=args.partition, seed=args.seed)

    test_w = du.build_windows(test_s, vocab)
    train_w_all = du.build_windows(train_s, vocab)
    du.assert_no_leakage(train_w_all, test_w)

    Xa_te, Xn_te, y_te, sid_te = du.to_arrays(test_w, label_map)
    te_loader = make_loader(Xa_te, Xn_te, y_te, batch_size=args.batch_size, shuffle=False)

    global_model = HybridCNN(vocab_size=len(vocab), num_classes=len(label_map),
                              n_net_features=du.N_NET_FEATURES)
    initial_weights = get_weights(global_model)

    client_fn = make_client_fn(shards, vocab, label_map, du.N_NET_FEATURES,
                                batch_size=args.batch_size,
                                local_epochs=args.local_epochs,
                                lr=args.lr, device=device,
                                optimizer=args.optimizer,
                                dp_config=dp_config, seed=args.seed)

    strategy, weights_holder = build_strategy(
        initial_weights, args.num_clients, dp_config=dp_config, seed=args.seed)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args={"log_to_driver": False, "logging_level": logging.ERROR,
                       "include_dashboard": False},
    )

    example_history = {"fold": 0,
                        "train_loss_by_round": history.metrics_distributed_fit.get("train_loss", [])}

    final_weights = weights_holder.latest_weights
    assert final_weights is not None, "FedAvg produced no aggregated weights"
    set_weights(global_model, final_weights)

    _, win_acc, preds = evaluate(global_model, te_loader, device=device)

    voted = du.majority_vote(preds, sid_te)
    by_id = {(s.get("md5") or s["filename"]): s for s in test_s}
    n_win_per_sample = Counter(sid_te.tolist())
    shard_sizes = [len(s) for s in shards]

    fold_records = []
    for sid, pred_idx in voted.items():
        true_family = by_id[sid]["family"]
        pred_family = idx_to_label[pred_idx]
        fold_records.append({
            "fold": sid,
            "held_out": by_id[sid].get("filename", sid),
            "true_family": true_family,
            "pred_family": pred_family,
            "sample_correct": int(pred_family == true_family),
            "window_accuracy": round(win_acc, 4),
            "n_test_windows": n_win_per_sample[sid],
            "client_shard_sizes": shard_sizes,
        })

    sample_acc = np.mean([f["sample_correct"] for f in fold_records])
    print(f"    holdout: {len(train_s)} train / {len(test_s)} test samples "
          f"| sample_acc={sample_acc:.2f}  window_acc={win_acc:.2f}  shards={shard_sizes}")

    return fold_records, example_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cape", "public"], default="cape",
                     help="cape: HW5's 5-sample CAPE data, LOSO protocol. "
                          "public: mal-api-2019 (7,107 samples), stratified "
                          "train/test holdout protocol (see data_utils.holdout_split)")
    ap.add_argument("--data", default=None,
                     help="default: data/dataset.json (cape) or the public zip (public)")
    ap.add_argument("--vocab", default=None,
                     help="default: data/api_vocab.json (cape) or data/public/api_vocab.json (public)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                     help="public dataset only: fraction of each family held out for test")
    ap.add_argument("--split-seed", type=int, default=0,
                     help="public dataset only: fixed seed for the train/test split, "
                          "held constant across --n-seeds so every seed is scored "
                          "against the same test samples")
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
                     help="repeat the whole protocol with this many seeds "
                          "and report mean +/- std (set 1 for a single run)")
    # Ray workers each run in their own process; keeping this off "auto" by
    # default avoids every worker independently probing for a GPU.
    ap.add_argument("--device", default="cpu")
    # ---- HW6: differential privacy ----------------------------------------
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                     help="adam reproduces HW5 exactly; sgd avoids the "
                          "per-round Adam state reset that makes non-IID "
                          "training loss diverge. Use sgd for DP runs.")
    ap.add_argument("--dp", choices=["none", "central_server", "central_client",
                                      "local", "dpsgd"], default="none",
                     help="differential-privacy mechanism (see dp.py)")
    ap.add_argument("--epsilon", type=float, default=None,
                     help="target privacy budget; omit for no DP")
    ap.add_argument("--delta", type=float, default=None,
                     help="default: 1/(10*num_clients), see dp.recommend_delta")
    ap.add_argument("--clip", type=float, default=1.0,
                     help="L2 clipping norm C (sensitivity bound)")
    args = ap.parse_args()
    if args.out is None:
        tag = args.partition if args.dp == "none" else \
            f"{args.partition}_{args.dp}_eps{args.epsilon}"
        prefix = "federated" if args.dataset == "cape" else f"federated_{args.dataset}"
        args.out = f"results/{prefix}_{tag}.json"

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
        else ("cpu" if args.device == "auto" else args.device)

    if args.dataset == "cape":
        args.data = args.data or "data/dataset.json"
        args.vocab = args.vocab or "data/api_vocab.json"
        vocab = du.load_vocab(args.vocab)
        samples, unknown = du.load_samples(args.data)
    else:
        import public_data as pdata
        args.data = args.data or pdata.PUBLIC_ZIP
        args.vocab = args.vocab or pdata.PUBLIC_VOCAB
        vocab = du.load_vocab(args.vocab)
        samples, unknown = pdata.load_public_samples(zip_path=args.data)
    label_map = du.build_label_map(samples)

    print(f"[+] {len(samples)} usable labelled samples | families: {list(label_map)}")
    print(f"[+] federated simulation ({args.dataset}): {args.num_clients} clients, "
          f"partition={args.partition}, {args.rounds} rounds x {args.local_epochs} "
          f"local epoch(s), opt={args.optimizer}")

    # ---- HW6: turn the target epsilon into concrete mechanism knobs --------
    dp_config = dp.build_dp_config(
        args.dp, args.epsilon, args.delta, rounds=args.rounds,
        num_clients=args.num_clients, clipping_norm=args.clip)
    if dp_config["mechanism"] != "none":
        print(f"[+] DP: {dp_config['mechanism']} | target eps={dp_config['epsilon_target']} "
              f"delta={dp_config['delta']:.1e} | noise multiplier z={dp_config['noise_multiplier']:.3f} "
              f"| clip C={dp_config['clipping_norm']}")
        print(f"    privacy unit = {dp_config['privacy_unit']}, "
              f"accountant = {dp_config['accountant']}"
              + (f", eps actually spent = {dp_config['epsilon_actual']:.3f}"
                 if 'epsilon_actual' in dp_config else ""))
    else:
        print("[+] DP: disabled (this reproduces the HW5 configuration)")

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
    run_fn = run_loso if args.dataset == "cape" else run_holdout
    protocol_name = "LOSO" if args.dataset == "cape" else "holdout"
    for r in range(args.n_seeds):
        args.seed = base_seed + r * 100
        print(f"\n[+] Federated {protocol_name} ({args.partition}) -- run {r + 1}/{args.n_seeds} "
              f"(seed={args.seed}):")
        folds, hist = run_fn(samples, vocab, label_map, args, device, dp_config)
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

    print(f"\n    SAMPLE-level {protocol_name} accuracy = {sample_acc:.2f} +/- {sample_accs.std():.2f} "
          f"over {args.n_seeds} seeds  [{args.dataset}/{args.partition}]")
    print(f"    window-level mean accuracy = {window_acc:.2f} +/- {window_accs.std():.2f}")
    print(f"    per-seed sample accuracies : {[round(a, 2) for a in sample_accs]}")
    if always_wrong:
        print(f"\n    [!] folds wrong under EVERY seed (structural, not model-dependent):")
        for name in always_wrong:
            print(f"          {name}")

    summary = {
        "experiment": f"federated_{args.partition}",
        "dataset": args.dataset,
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
                             "optimizer": args.optimizer,
                             "base_seed": base_seed, "n_seeds": args.n_seeds},
        "dp": dp_config,
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
