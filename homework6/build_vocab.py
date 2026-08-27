#!/usr/bin/env python3
"""
build_vocab.py
--------------
Builds the FIXED API-name vocabulary used to integer-encode API-call sequences.

WHY THIS FILE EXISTS (important for the report)
-----------------------------------------------
The midterm used a TF-IDF vectoriser. That is INCOMPATIBLE with federated
learning: fitting TF-IDF requires computing document frequencies over the whole
corpus, i.e. it requires seeing every client's data on one machine -- exactly
the thing federated learning forbids.

The fix is a *data-independent* feature space, agreed before training starts
and shipped to every client. For API-call features that is natural: the set of
API names that can ever appear is fixed by the CAPEv2 monitor's hook
configuration, which is a property of the SANDBOX, not of the training data.
Every client running the same CAPE build observes the same API name space.

So the vocabulary here is a *protocol constant*, not a learned artefact. We
materialise it once, commit it, and every client loads the identical file. No
information crosses client boundaries.

(In this homework we enumerate the hook set from the traces we have on hand.
In a real deployment you would export it directly from the CAPE monitor's
hook definitions -- same list, no dependence on samples at all.)

Reserved indices:
    0 = <PAD>   padding for short windows
    1 = <UNK>   an API name not present in the vocabulary

Usage:
    python3 build_vocab.py --data data/dataset.json --out data/api_vocab.json
"""

import argparse
import json
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_vocab(dataset_path):
    """Enumerate every API name observable from the sandbox monitor."""
    with open(dataset_path) as f:
        records = json.load(f)

    # Count over ALL runs, including failed and unknown-family ones. The
    # vocabulary describes the monitor's capability, so it deliberately does
    # NOT depend on which samples ended up in the training split.
    counts = Counter()
    for rec in records:
        counts.update(rec.get("api_sequence", []))

    # Sorted alphabetically so the mapping is deterministic and reproducible
    # across machines -- every client must derive identical indices.
    names = sorted(counts.keys())

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for i, name in enumerate(names):
        vocab[name] = i + 2  # shift past the two reserved slots

    return vocab, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.json")
    ap.add_argument("--out", default="data/api_vocab.json")
    args = ap.parse_args()

    vocab, counts = build_vocab(args.data)

    with open(args.out, "w") as f:
        json.dump(vocab, f, indent=1, sort_keys=True)

    print(f"[+] vocabulary size = {len(vocab)} "
          f"({len(vocab) - 2} API names + PAD + UNK)")
    print(f"[+] written to {args.out}")
    print("[+] 10 most frequent API calls in the corpus:")
    for name, n in counts.most_common(10):
        print(f"      {name:<38} {n:>7}")


if __name__ == "__main__":
    main()
