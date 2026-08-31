#!/usr/bin/env python3
"""
figures.py
----------
Renders the report figures from results/backdoor_results.json:
  1. ASR & MTA vs round        (the "attack lands, accuracy holds" story)
  2. ASR & MTA vs gamma        (update-scaling / model-replacement strength)
  3. ASR & MTA vs poison frac  (how much local data must be poisoned)
  4. ASR & MTA vs #malicious    (colluding-client threshold)

Every panel plots ASR and MTA together on purpose: the danger of a backdoor is
precisely that MTA stays high while ASR rises, so they must be read as a pair.
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def line(ax, xs, mta, asr, xlabel, title, mta_std=None, asr_std=None):
    if mta_std or asr_std:   # seed-averaged: draw error bars
        ax.errorbar(xs, mta, yerr=mta_std, fmt="o-", capsize=3,
                    label="MTA (clean acc)", color="#4C72B0")
        ax.errorbar(xs, asr, yerr=asr_std, fmt="s-", capsize=3,
                    label="ASR (attack)", color="#C44E52")
    else:
        ax.plot(xs, mta, "o-", label="MTA (clean acc)", color="#4C72B0")
        ax.plot(xs, asr, "s-", label="ASR (attack)", color="#C44E52")
    ax.set_xlabel(xlabel); ax.set_ylabel("rate"); ax.set_ylim(0, 1.02)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results/backdoor_results.json")
    ap.add_argument("--outdir", default="../results")
    args = ap.parse_args()
    d = json.load(open(args.results)); ex = d["experiments"]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    if "backdoor" in ex:
        h = ex["backdoor"]["history"]
        fig, ax = plt.subplots(figsize=(7,4))
        line(ax, h["round"], h["mta"], h["asr"], "round", "Backdoor over training rounds")
        # shade +/- std across seeds if present
        import numpy as _np
        for key, col in [("mta", "#4C72B0"), ("asr", "#C44E52")]:
            sd = h.get(key + "_std")
            if sd:
                m = _np.array(h[key]); s = _np.array(sd)
                ax.fill_between(h["round"], _np.clip(m-s,0,1), _np.clip(m+s,0,1),
                                color=col, alpha=0.15)
        fig.tight_layout(); fig.savefig(out/"fig_rounds.png", dpi=150); plt.close(fig)

    for key, xk, xl, fn in [
        ("gamma_sweep","gamma","scaling factor gamma","fig_gamma.png"),
        ("poison_sweep","poison_fraction","poison fraction","fig_poison.png"),
        ("nmalicious_sweep","n_malicious","# colluding clients","fig_nmalicious.png")]:
        if key in ex:
            rows = ex[key]
            has_std = any(r.get("asr_std") or r.get("mta_std") for r in rows)
            fig, ax = plt.subplots(figsize=(6,4))
            line(ax, [r[xk] for r in rows], [r["mta"] for r in rows],
                 [r["asr"] for r in rows], xl, key.replace("_"," "),
                 mta_std=[r.get("mta_std") or 0 for r in rows] if has_std else None,
                 asr_std=[r.get("asr_std") or 0 for r in rows] if has_std else None)
            fig.tight_layout(); fig.savefig(out/fn, dpi=150); plt.close(fig)

    # Update-norm outlier: what norm-clipping / anomaly detection keys on.
    if "backdoor" in ex:
        h = ex["backdoor"]["history"]
        mn = [v if v is not None else 0 for v in h.get("malicious_norm", [])]
        hn = [v if v is not None else 0 for v in h.get("max_honest_norm", [])]
        if any(mn):
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(h["round"], hn, "o-", label="max honest update norm", color="#4C72B0")
            ax.plot(h["round"], mn, "s-", label="malicious update norm", color="#C44E52")
            ax.set_xlabel("round"); ax.set_ylabel("global L2 norm of update")
            ax.set_title("Malicious update is a norm outlier (γ-scaled)")
            ax.grid(alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(out/"fig_norms.png", dpi=150); plt.close(fig)

    # Defense: grouped bars of ASR and MTA per aggregation rule.
    if "defense" in ex:
        rows = ex["defense"]["rows"]
        labels = [r["defense"] for r in rows]
        mta = [r["mta"] for r in rows]; asr = [r["asr"] for r in rows]
        mta_e = [r.get("mta_std") or 0 for r in rows]; asr_e = [r.get("asr_std") or 0 for r in rows]
        x = range(len(labels)); w = 0.38
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar([i-w/2 for i in x], mta, w, yerr=mta_e, capsize=3,
               label="MTA (clean acc)", color="#4C72B0")
        ax.bar([i+w/2 for i in x], asr, w, yerr=asr_e, capsize=3,
               label="ASR (attack)", color="#C44E52")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylim(0,1.02); ax.set_ylabel("rate")
        ax.set_title("Defenses vs the single-shot backdoor")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        fig.tight_layout(); fig.savefig(out/"fig_defense.png", dpi=150); plt.close(fig)

    print("[+] figures written to", out)

if __name__ == "__main__":
    main()
