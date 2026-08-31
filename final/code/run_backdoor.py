#!/usr/bin/env python3
"""
run_backdoor.py
---------------
Runs the backdoor experiments and writes the numbers the report is built on:

  * clean FL baseline      -- no attacker; establishes the MTA ceiling
  * backdoored FL          -- single-shot model replacement: MTA and ASR/round
  * gamma sweep            -- ASR/MTA vs update scaling (data-poisoning -> replacement)
  * poison sweep           -- ASR/MTA vs fraction of local data that is triggered
  * nmalicious sweep       -- ASR/MTA vs #colluding clients, as DATA POISONING
                              (gamma=1, every round): the "how many attackers does
                              it take to survive FedAvg" story
  * defense                -- the canonical backdoor under norm-clipping and
                              coordinate-wise median aggregation (ASR should drop)

MTA (Main Task Accuracy) must stay near the clean baseline for the attack to be
stealthy; ASR (Attack Success Rate) is how often a triggered non-target image is
pushed to the target family. A strong backdoor shows high ASR at nearly unchanged
MTA -- the whole danger is that validation accuracy does not reveal it.

TWO ATTACK MODES USED HERE
--------------------------
  * model replacement (canonical, gamma/poison sweeps): a SINGLE malicious round
    on the already-converged model, scaled by gamma so it survives FedAvg. With
    attack_start == rounds the attacker fires exactly once, which avoids the
    honest-majority tug-of-war that wrecks MTA when you attack every round.
  * data poisoning (nmalicious sweep): gamma=1, the attacker(s) just submit
    normal-sized poisoned updates every round. One attacker of five is averaged
    away; enough colluders and the backdoor survives.

Usage:
    python run_backdoor.py --images ../data/images --experiment all
    python run_backdoor.py --images ../data/images --experiment gamma --rounds 12
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import image_data as idata
import backdoor as bd
import defense as dfn
from fed import run_federated


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def base_cfg(args):
    return {
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "lr": args.lr, "batch_size": args.batch_size, "size": idata.SIZE,
        "optimizer": "sgd", "device": args.device, "seed": args.seed,
    }


def load_split(args):
    paths, labels, families = idata.scan_dataset(args.images)
    tr_p, tr_y, te_p, te_y = idata.stratified_split(paths, labels,
                                                    test_frac=0.2, seed=args.seed)
    Xte, yte = idata.load_arrays(te_p, te_y, size=idata.SIZE)
    shards = idata.partition(tr_p, tr_y, args.num_clients, mode=args.partition,
                             seed=args.seed)
    return families, shards, Xte, yte


def make_attack(args, gamma=None, poison_fraction=None, start_round=None,
                poison_epochs=None):
    mask, value = bd.make_trigger_mask(idata.SIZE, patch=args.patch, value=1.0,
                                       position=args.trigger_pos)
    return {"mask": mask, "value": value, "target_label": args.target,
            "poison_fraction": args.poison_fraction if poison_fraction is None else poison_fraction,
            "gamma": args.gamma if gamma is None else gamma,
            "poison_epochs": args.poison_epochs if poison_epochs is None else poison_epochs,
            "start_round": args.rounds if start_round is None else start_round}


def run_clean(args, families, shards, Xte, yte):
    print("\n[=] Clean FL baseline (no attacker)")
    cfg = base_cfg(args)
    _, hist = run_federated(shards, len(families), cfg,
                            X_clean_test=Xte, y_clean_test=yte)
    return {"final_mta": hist["mta"][-1], "history": hist}


def run_backdoored(args, families, shards, Xte, yte, gamma=None,
                   malicious_ids=None, poison_fraction=None, start_round=None,
                   defense=None, tag="backdoor"):
    cfg = base_cfg(args)
    attack = make_attack(args, gamma=gamma, poison_fraction=poison_fraction,
                         start_round=start_round)
    mids = malicious_ids if malicious_ids is not None else (args.malicious_id,)

    mode = "single-shot replacement" if attack["start_round"] >= args.rounds else \
           f"from round {attack['start_round']}"
    print(f"\n[=] {tag}: target={families[args.target]} gamma={attack['gamma']} "
          f"malicious={list(mids)} poison_frac={attack['poison_fraction']} "
          f"pe={attack['poison_epochs']} ({mode})"
          + (f" | defense={defense['type']}" if defense else ""))
    _, hist = run_federated(shards, len(families), cfg, attack=attack,
                            malicious_ids=mids, X_clean_test=Xte, y_clean_test=yte,
                            defense=defense)
    return {"final_mta": hist["mta"][-1], "final_asr": hist["asr"][-1],
            "gamma": attack["gamma"], "malicious_ids": list(mids),
            "poison_fraction": attack["poison_fraction"],
            "start_round": attack["start_round"], "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="../data/images")
    ap.add_argument("--experiment", default="all",
                    choices=["all", "clean", "backdoor", "gamma",
                             "poison", "nmalicious", "defense", "rounds"])
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--malicious-id", type=int, default=0)
    ap.add_argument("--target", type=int, default=0, help="target family index")
    ap.add_argument("--trigger-pos", default="bottom_stripe")
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--poison-fraction", type=float, default=0.25,
                    help="share of the malicious client's data that is triggered "
                         "and relabelled; 0.25 fits the trigger without collapsing "
                         "the malicious model onto the target class")
    ap.add_argument("--poison-epochs", type=int, default=8,
                    help="local epochs for the malicious client; more epochs let "
                         "it fit BOTH the clean task and the trigger conditional")
    ap.add_argument("--gamma", type=float, default=None,
                    help="update scaling; default = num_clients (full model replacement)")
    ap.add_argument("--partition", default="iid", choices=["iid", "non_iid"])
    ap.add_argument("--attack-start", type=int, default=None,
                    help="round the attacker first participates; default = last "
                         "round (canonical single-shot model replacement)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clip-norm", type=float, default=None,
                    help="max L2 update norm for the norm-clipping defense; "
                         "default = auto (just above the honest updates)")
    ap.add_argument("--out", default="../results/backdoor_results.json")
    args = ap.parse_args()

    if args.gamma is None:
        args.gamma = float(args.num_clients)   # canonical model-replacement scale
    ss = args.rounds if args.attack_start is None else args.attack_start  # single-shot round

    set_seed(args.seed)
    families, shards, Xte, yte = load_split(args)
    print(f"[+] {len(families)} families: {families}")
    print(f"[+] {len(Xte)} test images | {args.num_clients} clients "
          f"({args.partition}) | trigger={args.trigger_pos} {args.patch}px "
          f"-> target family '{families[args.target]}'")
    idata.describe_partition(shards, families)

    out = {"families": families, "config": vars(args), "experiments": {}}

    if args.experiment in ("all", "clean", "backdoor", "rounds", "defense"):
        out["experiments"]["clean"] = run_clean(args, families, shards, Xte, yte)

    if args.experiment in ("all", "backdoor", "rounds"):
        out["experiments"]["backdoor"] = run_backdoored(
            args, families, shards, Xte, yte, start_round=ss,
            tag="backdoor (single-shot replacement)")

    if args.experiment in ("all", "gamma"):
        rows = []
        for g in [1, 2, 3, 5, 10]:
            r = run_backdoored(args, families, shards, Xte, yte, gamma=float(g),
                               start_round=ss, tag=f"gamma={g}")
            rows.append({"gamma": g, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["gamma_sweep"] = rows

    if args.experiment in ("all", "poison"):
        rows = []
        for pf in [0.1, 0.25, 0.5, 0.75, 1.0]:
            r = run_backdoored(args, families, shards, Xte, yte,
                               poison_fraction=pf, start_round=ss,
                               tag=f"poison_frac={pf}")
            rows.append({"poison_fraction": pf, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["poison_sweep"] = rows

    if args.experiment in ("all", "nmalicious"):
        # Data-poisoning collusion study: gamma=1 (no amplification), attackers
        # participate EVERY round from the midpoint. This isolates "how many
        # colluders does the backdoor need to survive plain FedAvg", which is a
        # different question from the gamma sweep's single-attacker replacement.
        rows = []
        mid = max(1, args.rounds // 2)
        for k in range(1, min(args.num_clients, 4) + 1):
            r = run_backdoored(args, families, shards, Xte, yte, gamma=1.0,
                               malicious_ids=tuple(range(k)), start_round=mid,
                               tag=f"{k} colluding (data-poisoning)")
            rows.append({"n_malicious": k, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["nmalicious_sweep"] = rows

    if args.experiment in ("all", "defense"):
        # Estimate a clip norm from a clean round's honest updates if not given.
        clip_norm = args.clip_norm
        if clip_norm is None:
            base = out["experiments"].get("backdoor")
            hh = base["history"] if base else None
            hon = [v for v in (hh["max_honest_norm"] if hh else []) if v]
            clip_norm = dfn.suggest_clip_norm(hon) if hon else 5.0
        rows = []
        r0 = run_backdoored(args, families, shards, Xte, yte, start_round=ss,
                            tag="no defense")
        rows.append({"defense": "none", "mta": r0["final_mta"], "asr": r0["final_asr"]})
        r1 = run_backdoored(args, families, shards, Xte, yte, start_round=ss,
                            defense={"type": "norm_clip", "max_norm": clip_norm},
                            tag=f"norm_clip(max={clip_norm:.2f})")
        rows.append({"defense": "norm_clip", "max_norm": clip_norm,
                     "mta": r1["final_mta"], "asr": r1["final_asr"]})
        r2 = run_backdoored(args, families, shards, Xte, yte, start_round=ss,
                            defense={"type": "median"}, tag="median")
        rows.append({"defense": "median", "mta": r2["final_mta"], "asr": r2["final_asr"]})
        out["experiments"]["defense"] = {"clip_norm": clip_norm, "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] written to {args.out}")


if __name__ == "__main__":
    main()
