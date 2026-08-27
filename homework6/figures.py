#!/usr/bin/env python3
"""
figures.py
----------
Report plots. Reads the JSON summaries written by centralized.py and
run_simulation.py and renders three PNGs into results/:

  1. accuracy_comparison.png  -- sample-level LOSO accuracy, centralized vs.
     federated (IID / non-IID), mean +/- std across seeds, vs. the midterm
     RandomForest reference line.
  2. partition_heterogeneity.png -- per-client family distribution for a
     representative IID vs. non-IID partition of one fold's training set,
     showing WHY federated accuracy differs from centralized (client drift).
  3. convergence.png -- FedAvg round-by-round training loss for one example
     fold, IID vs. non-IID, showing how many rounds federated training needs
     to make progress at all with this few clients/samples.

Usage:
    python3 figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_utils as du

RESULTS_DIR = "results"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_accuracy_comparison(centralized, fed_iid, fed_non_iid, out_path):
    labels, means, stds, colors = [], [], [], []
    palette = {"centralized": "#4C72B0", "iid": "#55A868", "non_iid": "#C44E52"}

    if centralized:
        labels.append("Centralized\n(HybridCNN)")
        means.append(centralized["sample_level_accuracy"])
        stds.append(centralized["sample_level_std"])
        colors.append(palette["centralized"])
    if fed_iid:
        labels.append("Federated\nIID")
        means.append(fed_iid["sample_level_accuracy"])
        stds.append(fed_iid["sample_level_std"])
        colors.append(palette["iid"])
    if fed_non_iid:
        labels.append("Federated\nnon-IID")
        means.append(fed_non_iid["sample_level_accuracy"])
        stds.append(fed_non_iid["sample_level_std"])
        colors.append(palette["non_iid"])

    if not labels:
        print("[!] no result files found, skipping accuracy_comparison.png")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.55)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1,
               label="midterm RandomForest (0.80)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sample-level LOSO accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Malware family classification: centralized vs. federated")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"{m:.2f}", ha="center", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] wrote {out_path}")


def fig_partition_heterogeneity(out_path, num_clients=3, seed=42):
    vocab = du.load_vocab()
    samples, _ = du.load_samples()
    # Representative fold: hold out sample 0, partition the remaining 4.
    _, train_s, _ = next(du.loso_splits(samples))

    iid_shards = du.partition(train_s, num_clients=num_clients, mode="iid", seed=seed)
    non_iid_shards = du.partition(train_s, num_clients=num_clients, mode="non_iid", seed=seed)
    families = sorted({s["family"] for s in samples})
    fam_color = dict(zip(families, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, shards, title in [(axes[0], iid_shards, "IID partition"),
                               (axes[1], non_iid_shards, "Non-IID partition")]:
        bottoms = [0] * num_clients
        for fam in families:
            counts = [sum(1 for s in shard if s["family"] == fam) for shard in shards]
            ax.bar(range(num_clients), counts, bottom=bottoms, label=fam,
                   color=fam_color[fam])
            bottoms = [b + c for b, c in zip(bottoms, counts)]
        ax.set_xticks(range(num_clients))
        ax.set_xticklabels([f"client {i}" for i in range(num_clients)])
        ax.set_title(title)
        ax.set_ylabel("# training samples")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Client data heterogeneity (fold 0's training set, 4 samples / 3 clients)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] wrote {out_path}")


def fig_convergence(fed_iid, fed_non_iid, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for result, label, color in [(fed_iid, "IID", "#55A868"),
                                  (fed_non_iid, "non-IID", "#C44E52")]:
        if not result or not result.get("example_history"):
            continue
        losses = result["example_history"]["train_loss_by_round"]
        rounds = [r for r, _ in losses]
        vals = [v for _, v in losses]
        ax.plot(rounds, vals, label=f"{label} (fold {result['example_history']['fold']})",
               color=color, marker="o", markersize=3)
        plotted = True

    if not plotted:
        print("[!] no example_history found, skipping convergence.png")
        plt.close(fig)
        return

    ax.set_xlabel("FedAvg round")
    ax.set_ylabel("Weighted mean client train loss")
    ax.set_title("Federated training convergence (one representative fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] wrote {out_path}")


def main():
    centralized = load_json(f"{RESULTS_DIR}/centralized_baseline.json")
    fed_iid = load_json(f"{RESULTS_DIR}/federated_iid.json")
    fed_non_iid = load_json(f"{RESULTS_DIR}/federated_non_iid.json")

    fig_accuracy_comparison(centralized, fed_iid, fed_non_iid,
                            f"{RESULTS_DIR}/accuracy_comparison.png")
    fig_partition_heterogeneity(f"{RESULTS_DIR}/partition_heterogeneity.png")
    fig_convergence(fed_iid, fed_non_iid, f"{RESULTS_DIR}/convergence.png")


if __name__ == "__main__":
    main()
