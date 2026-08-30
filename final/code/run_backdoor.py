#!/usr/bin/env python3
"""
run_backdoor.py
---------------
Runs the backdoor experiments and writes the numbers the report is built on:

  * clean FL baseline           -- no attacker; establishes MTA
  * backdoored FL               -- MTA and ASR over rounds
  * sweeps                      -- ASR/MTA vs poison fraction, #malicious
                                   clients, and scaling factor gamma

MTA (Main Task Accuracy) must stay near the clean baseline for the attack to be
stealthy; ASR (Attack Success Rate) is how often a triggered non-target image
is pushed to the target family. A strong backdoor shows high ASR at nearly
unchanged MTA -- the whole danger is that validation accuracy does not reveal it.

Usage:
    python3 run_backdoor.py --images data/images --experiment all
    python3 run_backdoor.py --images data/images --experiment gamma --rounds 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import image_data as idata
import backdoor as bd
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


def make_attack(args, families):
    mask, value = bd.make_trigger_mask(idata.SIZE, patch=args.patch, value=1.0,
                                       position=args.trigger_pos)
    return {"mask": mask, "value": value,
            "target_label": args.target, "poison_fraction": args.poison_fraction,
            "gamma": args.gamma, "poison_epochs": args.local_epochs,
            "start_round": args.attack_start}


def run_clean(args, families, shards, Xte, yte):
    print("\n[=] Clean FL baseline (no attacker)")
    cfg = base_cfg(args)
    _, hist = run_federated(shards, len(families), cfg,
                            X_clean_test=Xte, y_clean_test=yte)
    return {"final_mta": hist["mta"][-1], "history": hist}


def run_backdoored(args, families, shards, Xte, yte, gamma=None,
                   malicious_ids=None, poison_fraction=None, tag="backdoor"):
    cfg = base_cfg(args)
    attack = make_attack(args, families)
    if gamma is not None: attack["gamma"] = gamma
    if poison_fraction is not None: attack["poison_fraction"] = poison_fraction
    mids = malicious_ids if malicious_ids is not None else (args.malicious_id,)

    print(f"\n[=] {tag}: target={families[args.target]} "
          f"gamma={attack['gamma']} malicious={list(mids)} "
          f"poison_frac={attack['poison_fraction']}")
    _, hist = run_federated(shards, len(families), cfg, attack=attack,
                            malicious_ids=mids, X_clean_test=Xte, y_clean_test=yte)
    return {"final_mta": hist["mta"][-1], "final_asr": hist["asr"][-1],
            "gamma": attack["gamma"], "malicious_ids": list(mids),
            "poison_fraction": attack["poison_fraction"], "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--experiment", default="all",
                    choices=["all", "clean", "backdoor", "gamma",
                             "poison", "nmalicious", "rounds"])
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--malicious-id", type=int, default=0)
    ap.add_argument("--target", type=int, default=0, help="target family index")
    ap.add_argument("--trigger-pos", default="bottom_stripe")
    ap.add_argument("--patch", type=int, default=12)
    ap.add_argument("--poison-fraction", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=None,
                    help="update scaling; default = num_clients (full model replacement)")
    ap.add_argument("--partition", default="iid", choices=["iid", "non_iid"])
    ap.add_argument("--attack-start", type=int, default=1,
                    help="round the attacker first participates; set near the "
                         "end for the canonical single-shot model-replacement "
                         "attack on an already-converged model")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/backdoor_results.json")
    args = ap.parse_args()

    if args.gamma is None:
        args.gamma = float(args.num_clients)   # canonical model-replacement scale

    set_seed(args.seed)
    families, shards, Xte, yte = load_split(args)
    print(f"[+] {len(families)} families: {families}")
    print(f"[+] {len(Xte)} test images | {args.num_clients} clients "
          f"({args.partition}) | trigger={args.trigger_pos} {args.patch}px "
          f"-> target family '{families[args.target]}'")
    idata.describe_partition(shards, families)

    out = {"families": families, "config": vars(args), "experiments": {}}

    if args.experiment in ("all", "clean", "backdoor"):
        out["experiments"]["clean"] = run_clean(args, families, shards, Xte, yte)

    if args.experiment in ("all", "backdoor"):
        out["experiments"]["backdoor"] = run_backdoored(
            args, families, shards, Xte, yte, tag="backdoor (default)")

    if args.experiment in ("all", "gamma"):
        rows = []
        for g in [1, 2, 3, 5, 10]:
            r = run_backdoored(args, families, shards, Xte, yte, gamma=float(g),
                               tag=f"gamma={g}")
            rows.append({"gamma": g, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["gamma_sweep"] = rows

    if args.experiment in ("all", "poison"):
        rows = []
        for pf in [0.1, 0.25, 0.5, 0.75, 1.0]:
            r = run_backdoored(args, families, shards, Xte, yte,
                               poison_fraction=pf, tag=f"poison_frac={pf}")
            rows.append({"poison_fraction": pf, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["poison_sweep"] = rows

    if args.experiment in ("all", "nmalicious"):
        rows = []
        for k in range(1, min(args.num_clients, 4) + 1):
            r = run_backdoored(args, families, shards, Xte, yte,
                               malicious_ids=tuple(range(k)), tag=f"{k} malicious")
            rows.append({"n_malicious": k, "mta": r["final_mta"], "asr": r["final_asr"]})
        out["experiments"]["nmalicious_sweep"] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] written to {args.out}")


if __name__ == "__main__":
    main()
