#!/usr/bin/env python3
"""
build_public_vocab.py
----------------------
Builds the FIXED API-name vocabulary for the public Malware API Call Dataset,
mirroring build_vocab.py's approach for the CAPE data.

This dataset's traces come from a different sandbox (Cuckoo, per the IEEE
DataPort listing) with a different naming convention -- lowercase, no
separators (e.g. "ldrgetprocedureaddress" vs CAPE's "LdrGetProcedureAddress")
-- and is not the same monitor as CAPE's, so CAPE's api_vocab.json cannot be
reused here. This script builds the public dataset's own protocol constant the
same way: counted once over the WHOLE corpus (all families, before any
train/test split), sorted alphabetically for a deterministic mapping. It does
NOT depend on which samples end up in training vs. test, so it carries no
information across a split -- same guarantee build_vocab.py documents.

Usage:
    python3 build_public_vocab.py
    python3 build_public_vocab.py --out data/public/api_vocab.json
"""

import argparse
import json
from collections import Counter

from public_data import iter_public_records, PUBLIC_VOCAB

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_public_vocab():
    # max_seq_len=None: the vocabulary must cover every API name that can
    # appear anywhere in a trace, not just within the first MAX_SEQ_LEN calls
    # that windows_from_sample will later train on.
    counts = Counter()
    for rec in iter_public_records(max_seq_len=None):
        counts.update(rec["api_sequence"])

    names = sorted(counts.keys())
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for i, name in enumerate(names):
        vocab[name] = i + 2

    return vocab, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PUBLIC_VOCAB)
    args = ap.parse_args()

    vocab, counts = build_public_vocab()

    with open(args.out, "w") as f:
        json.dump(vocab, f, indent=1, sort_keys=True)

    print(f"[+] vocabulary size = {len(vocab)} "
          f"({len(vocab) - 2} API names + PAD + UNK)")
    print(f"[+] written to {args.out}")
    print("[+] 10 most frequent API calls in the corpus:")
    for name, n in counts.most_common(10):
        print(f"      {name:<38} {n:>9}")


if __name__ == "__main__":
    main()
