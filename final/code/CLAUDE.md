# Final Project — Backdoor Attack on FL malware image classifier

Solo, ~1 week. Deliverables: commented code + Word report + PPT + per-member
responsibility table (solo → just you across subtasks) + zip
`TeamName_FinalProject.zip`.

## Two halves, two machines
- **ML attack** (train classifier, backdoor, metrics): MOTIF images, **native Windows**, pure PyTorch (no Flower/Ray).
- **Functionality proof** (goal 2): live HW5 binaries in the **Ubuntu CAPE sandbox**. MOTIF is disarmed and can't run, so it CANNOT prove functionality — use the real HW5 executables.

## Status
Engine built and self-tested on synthetic data. Confirmed working:
- clean FL baseline reaches MTA 1.0
- backdoor reaches ASR 1.0
- #malicious sweep is monotone: 1 attacker of 5 washes out, 2+ → ASR 1.0
- `overlay_trigger.py`: PE stays valid, edit is overlay-only, stripe appears

Not done (the user's week): run on real MOTIF images; tune the stealthy regime
(high ASR at high MTA); write report + PPT; the sandbox proof; optional defense.

## The attack (backdoor.py + fed.py)
Model-replacement (Bagdasaryan): malicious client trains on trigger-stamped,
relabelled data, scales its update by gamma so it survives FedAvg's /K.
Trigger = white bottom-stripe = bytes appended to PE overlay (never executed).

## Rules / gotchas
1. Trigger position is `bottom_stripe` for physical fidelity — appended overlay bytes land at the image bottom. Don't switch to an arbitrary corner without breaking the byte-edit story.
2. `overlay_trigger.py` refuses to cross a width bracket (HW3 picks width by file size); keep `--rows` small. Worth a paragraph in the report.
3. Report ASR and MTA together, always — the whole point is high ASR at unchanged MTA.
4. No benign class in MOTIF → target is a wrong FAMILY, not "benign".
5. Continuous full-strength attack every round makes MTA oscillate (tug-of-war). Use `--attack-start` late for the clean single-shot result, or average over seeds.
6. Never run live malware off the Ubuntu sandbox.

## Report outline
1. Intro — FL trust model; a compromised client can backdoor the global model.
2. Background — FedAvg, backdoor/model-replacement, the three cited papers.
3. Method — image pipeline, trigger design, the overlay↔image mapping, gamma scaling.
4. Setup — MOTIF, families, clients, partition, hyperparameters.
5. Results — ASR/MTA vs round, gamma, poison fraction, #malicious.
6. Functionality proof — overlay edit, pefile checks, CAPE before/after reports.
7. (Bonus) Defense — norm-clipping / anomaly detection (FLDetector, CRFL); show ASR drop.
8. Conclusion + responsibilities table.
