# HW5 — Malware Classification based on Federated Learning

Replaces the midterm's centralized RandomForest (Step 3, Hybrid Analysis) with a
federated-learning pipeline over the same CAPEv2-derived features.

## Status

**Day 2 complete — federated pipeline works end-to-end.**

| File | Status | Purpose |
|---|---|---|
| `build_vocab.py` | done | Fixed, data-independent API vocabulary (FL-compatible) |
| `data_utils.py` | done | Windowing, encoding, LOSO splits, IID / non-IID partitioners |
| `model.py` | done | `HybridCNN` + FedAvg weight interface + shared train/eval |
| `centralized.py` | done | Centralized baseline, multi-seed LOSO |
| `client.py` | done | Flower `NumPyClient` |
| `server.py` | done | FedAvg strategy (full participation, saves final weights) |
| `run_simulation.py` | done | N-client in-process simulation, multi-seed LOSO |
| `figures.py` | done | Report plots (`results/*.png`) |
| `classify_midterm_rf.py` | reference | The midterm's RandomForest, unchanged |

## Setup

```bash
pip install torch numpy matplotlib
pip install "flwr[simulation]"     # requires Python >= 3.11
```
Note: `flwr[simulation]` and current `torch` wheels are not yet available for
very new Python builds (this environment's default `python3` was 3.14); the
venv here is built with `python3.11` specifically for wheel availability.

## Run

```bash
python3 build_vocab.py --data data/dataset.json --out data/api_vocab.json
python3 centralized.py --epochs 40 --lr 0.0003 --n-seeds 5
python3 run_simulation.py --partition iid     --n-seeds 5
python3 run_simulation.py --partition non_iid --n-seeds 5
python3 figures.py
```

## Current results

```
Centralized HybridCNN, leave-one-sample-out, 5 seeds
  sample-level accuracy = 0.64 +/- 0.15
  midterm RandomForest  = 0.80
  achievable ceiling    = 0.80   (RedLine is a singleton class)

Federated HybridCNN (3 clients, FedAvg, 40 rounds x 2 local epochs), 5 seeds
  IID partition      sample-level accuracy = 0.32 +/- 0.16
  non-IID partition  sample-level accuracy = 0.12 +/- 0.10
```

Federation costs real accuracy here, and non-IID costs more than IID. With
only 4 training samples split across 3 clients, every partition is forced
into near-single-family shards (see `results/partition_heterogeneity.png`) --
IID gets there by chance of the shuffle, non-IID by design. FedAvg then
averages together models pulled toward different single families each round.
`results/convergence.png` shows the effect directly: IID training loss
settles near ~0.7, while non-IID training loss *rises* through round ~20 and
oscillates around ~5 -- the global model is being dragged toward a different
client's family each round faster than any client's local training can
correct it. This is the client-drift failure mode FL literature predicts for
small, heterogeneous federations, reproduced here at N=5.
See `results/accuracy_comparison.png` for the three-way comparison against
the RandomForest ceiling.

`redline_01.exe` is misclassified under **every** seed. This is structural, not a
model defect: RedLine has exactly one usable sample, so when it is held out no
RedLine example remains in training and the fold is unwinnable. Recovering
`redline_02` / `redline_03` from the sandbox raises the ceiling to 1.00.

## Three design decisions to carry into the report

1. **RandomForest cannot be federated.** FedAvg averages parameter vectors;
   a forest is a set of discrete tree structures with no meaningful mean.
   FL constrains the model class to differentiable parametric models — hence
   the CNN. See the header of `model.py`.

2. **TF-IDF cannot be federated either.** Fitting it requires corpus-wide
   document frequencies, i.e. pooling all clients' data — the exact thing FL
   forbids. Replaced with a fixed vocabulary derived from the CAPE monitor's
   hook set, which is a property of the sandbox rather than of the training
   data. Same objection applies to `StandardScaler`, replaced with a
   data-independent `log1p` transform. See `build_vocab.py`.

3. **Windowing raises instance count, not sample count.** 5 samples become 174
   windows, which is what makes gradient training viable, but there are still
   only 5 independent observations. Every split is made at sample level and
   verified by `data_utils.assert_no_leakage()`, called on every fold.

## Evaluation protocol

Leave-one-sample-out, identical to the midterm's, so the CNN and the
RandomForest are directly comparable. Per-window predictions are collapsed to
one verdict per binary by majority vote.

With only 5 folds, one fold is worth 0.20 accuracy, and weight initialisation
alone moves the result across that range. All numbers are therefore reported as
mean ± std over 5 seeds, never as a single run.
