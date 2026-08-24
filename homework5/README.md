# HW5 — Malware Classification based on Federated Learning

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
