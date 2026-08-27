# HW6 — Malware Classification, Federated Learning under Differential Privacy

Extends HW5's federated pipeline with differential privacy. HW5's files are
unchanged except for the DP hooks noted below, so `--dp none` reproduces the
HW5 numbers exactly.

## Setup

```bash
pip install torch numpy matplotlib "flwr[simulation]" opacus   # Python >= 3.11
```

## Run

```bash
python3 build_vocab.py --data data/dataset.json --out data/api_vocab.json

# no DP -- reproduces HW5
python3 run_simulation.py --partition iid --n-seeds 5

# a single DP configuration
python3 run_simulation.py --partition iid --optimizer sgd --lr 0.01 \
        --dp central_server --epsilon 8 --n-seeds 5

# the headline privacy-utility sweep
python3 run_dp_sweep.py --partition iid --n-seeds 3
```

## Mechanisms

| `--dp` | Mechanism | Noise added | Trust model | Unit protected |
|---|---|---|---|---|
| `none` | plain FedAvg (HW5) | none | — | — |
| `central_server` | clip + noise at server | server, 1x/round | trusts server | client |
| `central_client` | clip at client, noise at server | server, 1x/round | server sees clipped only | client |
| `local` | clip + noise at client | each client | trusts nobody | client |
| `dpsgd` | Opacus per-example clipping | each SGD step | — | sample |

## Files added or changed vs HW5

- `dp.py` — **new**. Mechanisms, noise calibration, RDP accounting.
- `server.py` — added `CentralDPFedAvg` (clip → average → noise).
- `client.py` — local DP and client-side clipping in `fit()`; DP-SGD hook.
- `model.py` — `_make_optimizer` (adam/sgd), optional Opacus in `train_epochs`.
- `run_simulation.py` — `--dp --epsilon --delta --clip --optimizer` flags.
- `run_dp_sweep.py` — **new**. Sweeps epsilon across mechanisms.

## Known result

On the CAPE data the no-DP federated baseline is already at chance (0.32 vs
0.33 random), so every DP point sits at chance too. `run_dp_sweep.py` prints a
warning when it detects this. The meaningful privacy-utility curve requires the
larger public dataset — see CLAUDE.md.

## Note on Flower's built-in DP wrappers

`DifferentialPrivacyServerSideFixedClipping` and its client-side counterpart
crash with `ZeroDivisionError` when a client submits an exactly-zero update
(`clip_inputs_inplace` computes `min(1, C/norm)` with no zero guard). That case
occurs here once DP noise collapses the model. `CentralDPFedAvg` implements the
same mechanism with the guard in place.
