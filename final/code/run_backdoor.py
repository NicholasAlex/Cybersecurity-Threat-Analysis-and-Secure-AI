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


def load_split(args, seed=None):
    seed = args.seed if seed is None else seed
    paths, labels, families = idata.scan_dataset(args.images)
    tr_p, tr_y, te_p, te_y = idata.stratified_split(paths, labels,
                                                    test_frac=0.2, seed=seed)
    Xte, yte = idata.load_arrays(te_p, te_y, size=idata.SIZE)
    shards = idata.partition(tr_p, tr_y, args.num_clients, mode=args.partition,
                             seed=seed)
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


def _mean_std(vals):
    """Mean and population std of a list, ignoring None. Returns (mean, std)."""
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def _avg_history(hists):
    """Average per-round histories across seeds (nan-safe), keeping *_std."""
    rounds = hists[0]["round"]
    out = {"round": list(rounds)}
    for k in ["mta", "asr", "malicious_norm", "max_honest_norm"]:
        M = np.array([[np.nan if v is None else v for v in h.get(k, [])]
                      for h in hists], dtype=float)
        if M.size and M.shape[1] == len(rounds):
            out[k] = np.nanmean(M, axis=0).tolist()
            out[k + "_std"] = np.nanstd(M, axis=0).tolist()
    return out


def _sweep_rows(points, xkey):
    """points: dict{xval -> list[(mta, asr)]} -> list of rows with mean+std."""
    rows = []
    for x in points:
        mm, ms = _mean_std([p[0] for p in points[x]])
        am, asd = _mean_std([p[1] for p in points[x]])
        rows.append({xkey: x, "mta": mm, "mta_std": ms, "asr": am, "asr_std": asd})
    return rows


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
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds to average each experiment over "
                         "(seed, seed+1, ...); >1 smooths the small-dataset "
                         "variance and reports mean +/- std")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clip-norm", type=float, default=None,
                    help="max L2 update norm for the norm-clipping defense; "
                         "default = auto (just above the honest updates)")
    ap.add_argument("--out", default="../results/backdoor_results.json")
    args = ap.parse_args()

    if args.gamma is None:
        args.gamma = float(args.num_clients)   # canonical model-replacement scale
    ss = args.rounds if args.attack_start is None else args.attack_start  # single-shot round
    exp = args.experiment
    seeds = [args.seed + i for i in range(max(1, args.seeds))]

    # Header (from the first seed's split).
    families, _shd, _Xte, _yte = load_split(args, seeds[0])
    print(f"[+] {len(families)} families: {families}")
    print(f"[+] {args.num_clients} clients ({args.partition}) | trigger="
          f"{args.trigger_pos} {args.patch}px -> target '{families[args.target]}' "
          f"| averaging over {len(seeds)} seed(s): {seeds}")

    # Per-point accumulators across seeds.
    clean_fin, clean_h = [], []
    bd_fin, bd_h = [], []
    gamma_pts = {g: [] for g in [1, 2, 3, 5, 10]}
    poison_pts = {pf: [] for pf in [0.1, 0.25, 0.5, 0.75, 1.0]}
    nmal_pts = {k: [] for k in range(1, min(args.num_clients, 4) + 1)}
    def_pts = {"none": [], "norm_clip": [], "median": []}
    clip_norms = []

    for s in seeds:
        set_seed(s); args.seed = s          # seed model init, split, and poison RNG
        fam, sh, Xt, yt = load_split(args, s)
        print(f"\n########## seed {s} ##########")

        base = None
        if exp in ("all", "clean", "backdoor", "rounds", "defense"):
            r = run_clean(args, fam, sh, Xt, yt)
            clean_fin.append(r["final_mta"]); clean_h.append(r["history"])
        if exp in ("all", "backdoor", "rounds"):
            base = run_backdoored(args, fam, sh, Xt, yt, start_round=ss,
                                  tag="backdoor (single-shot replacement)")
            bd_fin.append((base["final_mta"], base["final_asr"]))
            bd_h.append(base["history"])
        if exp in ("all", "gamma"):
            for g in gamma_pts:
                r = run_backdoored(args, fam, sh, Xt, yt, gamma=float(g),
                                   start_round=ss, tag=f"gamma={g}")
                gamma_pts[g].append((r["final_mta"], r["final_asr"]))
        if exp in ("all", "poison"):
            for pf in poison_pts:
                r = run_backdoored(args, fam, sh, Xt, yt, poison_fraction=pf,
                                   start_round=ss, tag=f"poison_frac={pf}")
                poison_pts[pf].append((r["final_mta"], r["final_asr"]))
        if exp in ("all", "nmalicious"):
            # Colluder study as single-shot data poisoning (gamma=1): k attackers
            # contribute ~k/K of the aggregate, so more colluders -> the backdoor
            # survives FedAvg. Isolates the collusion threshold, distinct from the
            # gamma sweep's single-attacker replacement.
            for k in nmal_pts:
                r = run_backdoored(args, fam, sh, Xt, yt, gamma=1.0,
                                   malicious_ids=tuple(range(k)), start_round=ss,
                                   tag=f"{k} colluding (data-poisoning)")
                nmal_pts[k].append((r["final_mta"], r["final_asr"]))
        if exp in ("all", "defense"):
            r0 = run_backdoored(args, fam, sh, Xt, yt, start_round=ss,
                                tag="no defense")
            def_pts["none"].append((r0["final_mta"], r0["final_asr"]))
            hon = [v for v in r0["history"]["max_honest_norm"] if v]
            clip = args.clip_norm if args.clip_norm is not None else \
                (dfn.suggest_clip_norm(hon) if hon else 5.0)
            clip_norms.append(clip)
            r1 = run_backdoored(args, fam, sh, Xt, yt, start_round=ss,
                                defense={"type": "norm_clip", "max_norm": clip},
                                tag=f"norm_clip(max={clip:.3f})")
            def_pts["norm_clip"].append((r1["final_mta"], r1["final_asr"]))
            r2 = run_backdoored(args, fam, sh, Xt, yt, start_round=ss,
                                defense={"type": "median"}, tag="median")
            def_pts["median"].append((r2["final_mta"], r2["final_asr"]))

    # Aggregate across seeds (mean + std).
    cfg_out = dict(vars(args)); cfg_out["seeds_used"] = seeds
    out = {"families": families, "config": cfg_out, "experiments": {}}
    EXP = out["experiments"]

    if clean_fin:
        m, sd = _mean_std(clean_fin)
        EXP["clean"] = {"final_mta": m, "final_mta_std": sd,
                        "history": _avg_history(clean_h)}
    if bd_fin:
        mm, ms = _mean_std([x[0] for x in bd_fin])
        am, asd = _mean_std([x[1] for x in bd_fin])
        EXP["backdoor"] = {"final_mta": mm, "final_mta_std": ms,
                           "final_asr": am, "final_asr_std": asd,
                           "history": _avg_history(bd_h)}
    if any(gamma_pts.values()):
        EXP["gamma_sweep"] = _sweep_rows(gamma_pts, "gamma")
    if any(poison_pts.values()):
        EXP["poison_sweep"] = _sweep_rows(poison_pts, "poison_fraction")
    if any(nmal_pts.values()):
        EXP["nmalicious_sweep"] = _sweep_rows(nmal_pts, "n_malicious")
    if any(def_pts.values()):
        rows = []
        for name in ["none", "norm_clip", "median"]:
            mm, ms = _mean_std([x[0] for x in def_pts[name]])
            am, asd = _mean_std([x[1] for x in def_pts[name]])
            row = {"defense": name, "mta": mm, "mta_std": ms,
                   "asr": am, "asr_std": asd}
            if name == "norm_clip":
                row["max_norm"] = float(np.mean(clip_norms))
            rows.append(row)
        EXP["defense"] = {"clip_norm": float(np.mean(clip_norms)), "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[+] written to {args.out}")


if __name__ == "__main__":
    main()
