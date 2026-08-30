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

def line(ax, xs, mta, asr, xlabel, title):
    ax.plot(xs, mta, "o-", label="MTA (clean acc)", color="#4C72B0")
    ax.plot(xs, asr, "s-", label="ASR (attack)", color="#C44E52")
    ax.set_xlabel(xlabel); ax.set_ylabel("rate"); ax.set_ylim(0, 1.02)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/backdoor_results.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    d = json.load(open(args.results)); ex = d["experiments"]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    if "backdoor" in ex:
        h = ex["backdoor"]["history"]
        fig, ax = plt.subplots(figsize=(7,4))
        line(ax, h["round"], h["mta"], h["asr"], "round", "Backdoor over training rounds")
        fig.tight_layout(); fig.savefig(out/"fig_rounds.png", dpi=150); plt.close(fig)

    for key, xk, xl, fn in [
        ("gamma_sweep","gamma","scaling factor gamma","fig_gamma.png"),
        ("poison_sweep","poison_fraction","poison fraction","fig_poison.png"),
        ("nmalicious_sweep","n_malicious","# malicious clients","fig_nmalicious.png")]:
        if key in ex:
            rows = ex[key]
            fig, ax = plt.subplots(figsize=(6,4))
            line(ax, [r[xk] for r in rows], [r["mta"] for r in rows],
                 [r["asr"] for r in rows], xl, key.replace("_"," "))
            fig.tight_layout(); fig.savefig(out/fn, dpi=150); plt.close(fig)
    print("[+] figures written to", out)

if __name__ == "__main__":
    main()
