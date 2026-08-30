# Final Project — Backdoor Attack on Federated Learning (malware image classifier)

Attacks an FL-based malware **image** classifier (HW3-style: binary → grayscale
image → CNN, federated) with a backdoor, and proves the trigger is a real byte
edit that leaves the malware runnable.

Runs natively on **Windows** — pure PyTorch, no Flower/Ray. Only the
functionality proof (`overlay_trigger.py` + sandbox) uses the Ubuntu box.

## Setup
```bash
pip install torch numpy pillow matplotlib pefile
```

## Data
Regenerate MOTIF images from your HW3 copy, then point the runner at them:
```bash
python code/binary_to_image.py --input extracted/ --output data/images/
```
`extracted/<family>/<hash>` → `data/images/<family>/<hash>.png`. A synthetic
self-test set is included under `data/images/` so the pipeline runs before you
add MOTIF; delete it before real runs.

## Run the attack (Windows)
```bash
cd code
python run_backdoor.py --images ../data/images --experiment all \
       --num-clients 5 --target 0 --attack-start 8 --poison-fraction 0.4
python figures.py
```
Key knobs: `--gamma` (update scaling; =K is full model replacement),
`--poison-fraction`, `--attack-start` (late = canonical single-shot),
`--trigger-pos bottom_stripe`, `--patch`, `--partition {iid,non_iid}`.

## Functionality proof (Ubuntu CAPE sandbox ONLY — never a daily machine)
```bash
python code/overlay_trigger.py --in sample.exe --out sample_triggered.exe --rows 6
# then run BOTH in CAPEv2 and compare the behaviour reports
```
It appends a white bottom-stripe trigger to the PE overlay, verifies with
`pefile` that entry point and all sections are untouched, and confirms the
stripe appears in the image. Identical CAPE reports = functionality preserved.

## Files
- `image_data.py` — load images/<family>/*.png, stratified split, FL partition
- `model.py` — HW3 SimpleCNN + FedAvg weight interface + shared train/eval
- `fed.py` — manual FedAvg loop (no Ray)
- `backdoor.py` — trigger, poisoned data, malicious client, gamma scaling, ASR
- `run_backdoor.py` — experiments + sweeps
- `figures.py` — ASR/MTA plots
- `overlay_trigger.py` — physical byte-level trigger + PE verification
- `binary_to_image.py` — reused unchanged from HW3

## Metrics
- **MTA** (Main Task Accuracy) on clean images — must stay high (stealth).
- **ASR** (Attack Success Rate) — triggered non-target images classified as target.
