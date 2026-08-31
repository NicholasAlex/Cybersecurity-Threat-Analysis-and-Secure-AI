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

## Run everything (Windows)
```bash
pip install torch numpy pillow matplotlib pefile python-docx python-pptx
cd code
python run_backdoor.py --images ../data/images --experiment all   # experiments + sweeps + defense
python figures.py        # ASR/MTA plots -> results/*.png
python gen_report.py     # -> docs/FinalProject_Report.docx  (numbers read from the JSON)
python gen_ppt.py        # -> docs/FinalProject_Slides.pptx
```
The canonical run is a **single-shot model replacement** (attacker fires once, on
the converged model): defaults are `--gamma K`, `--poison-fraction 0.25`,
`--poison-epochs 8`, which give a stealthy backdoor (high ASR at high MTA).
Key knobs: `--gamma` (update scaling; =K is full model replacement),
`--poison-fraction`, `--poison-epochs`, `--attack-start` (default = last round =
single-shot), `--trigger-pos bottom_stripe`, `--patch`, `--partition {iid,non_iid}`,
`--clip-norm` (defense).

## Functionality proof (Ubuntu CAPE sandbox ONLY — never a daily machine)
```bash
python code/overlay_trigger.py --in sample.exe --out sample_triggered.exe --patch 12
# --rows is auto-derived from --patch so the byte stripe matches the model trigger
# then run BOTH in CAPEv2 and compare the behaviour reports
```
It appends a white bottom-stripe trigger to the PE overlay, verifies with
`pefile` that entry point and all sections are untouched, and confirms the
stripe appears in the image. Identical CAPE reports = functionality preserved.

## Files
- `image_data.py` — load images/<family>/*.png, stratified split, FL partition
- `model.py` — HW3 SimpleCNN + FedAvg weight interface + shared train/eval
- `fed.py` — manual FedAvg loop (no Ray), optional server-side defense
- `backdoor.py` — trigger, poisoned data, malicious client, gamma scaling, ASR
- `defense.py` — server defenses: L2 norm-clipping + coordinate-wise median
- `run_backdoor.py` — experiments + sweeps + defense
- `figures.py` — ASR/MTA plots (+ update-norm outlier + defense bars)
- `gen_report.py` — builds the Word report from the JSON + figures
- `gen_ppt.py` — builds the slide deck from the JSON + figures
- `overlay_trigger.py` — physical byte-level trigger + PE verification
- `binary_to_image.py` — reused unchanged from HW3

## Metrics
- **MTA** (Main Task Accuracy) on clean images — must stay high (stealth).
- **ASR** (Attack Success Rate) — triggered non-target images classified as target.
