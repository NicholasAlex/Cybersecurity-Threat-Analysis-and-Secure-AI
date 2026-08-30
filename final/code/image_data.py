#!/usr/bin/env python3
"""
image_data.py
-------------
Loads the grayscale malware-image dataset (MOTIF, converted by
binary_to_image.py) and partitions it across federated clients.

Data layout expected (exactly what HW3's binary_to_image.py produces):

    images/<family>/<hash>.png
    images/<family>/<hash>.png
    ...

Every image is loaded as a 1-channel float tensor in [0, 1], resized to a
fixed SIZE x SIZE so the CNN input is uniform (HW3 resized to 128).

WHY THIS IS EASIER THAN HW5/HW6's DATA MODULE
---------------------------------------------
HW5/HW6 had 5 samples and needed leave-one-sample-out plus a leakage guard.
MOTIF has thousands of independent binaries, so a normal stratified train/test
split is honest here: no two crops of one sample, no windowing, nothing to leak.
Each image is one independent observation.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 128           # CNN input side; matches HW3's default --resize


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path, size=SIZE):
    """One PNG -> (1, size, size) float32 array in [0, 1]. Read-only, never executed."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :]                      # add channel dim


def scan_dataset(images_dir):
    """
    Walk images/<family>/*.png and return (paths, labels, family_list).

    family_list is sorted so the label<->index mapping is deterministic and
    reproducible across machines -- important because the attacker and the
    defender must agree on which integer means which family.
    """
    images_dir = Path(images_dir)
    families = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
    fam_to_idx = {f: i for i, f in enumerate(families)}

    paths, labels = [], []
    for fam in families:
        for png in sorted((images_dir / fam).glob("*.png")):
            paths.append(png)
            labels.append(fam_to_idx[fam])
    return paths, np.array(labels, dtype=np.int64), families


def load_arrays(paths, labels, size=SIZE):
    """Materialise a list of paths into (X, y) arrays. X: (N,1,size,size)."""
    X = np.stack([load_image(p, size) for p in paths]) if paths else \
        np.zeros((0, 1, size, size), np.float32)
    return X.astype(np.float32), np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Train / test split -- stratified, at the sample level
# ---------------------------------------------------------------------------

def stratified_split(paths, labels, test_frac=0.2, seed=42):
    """
    Per-class shuffle then split, so every family is represented in both sides
    in proportion. Returns (train_paths, train_y, test_paths, test_y).
    """
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[int(y)].append(p)

    tr_p, tr_y, te_p, te_y = [], [], [], []
    for c, ps in by_class.items():
        ps = list(ps)
        rng.shuffle(ps)
        n_test = max(1, int(round(len(ps) * test_frac)))
        te_p += ps[:n_test]; te_y += [c] * n_test
        tr_p += ps[n_test:]; tr_y += [c] * (len(ps) - n_test)
    return tr_p, np.array(tr_y), te_p, np.array(te_y)


# ---------------------------------------------------------------------------
# Federated partitioning
# ---------------------------------------------------------------------------

def partition_iid(paths, labels, num_clients, seed=0):
    """Shuffle all samples and deal them round-robin -- each client sees every family."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(paths))
    shards = [([], []) for _ in range(num_clients)]
    for pos, i in enumerate(idx):
        c = pos % num_clients
        shards[c][0].append(paths[i])
        shards[c][1].append(int(labels[i]))
    return [(p, np.array(y)) for p, y in shards]


def partition_non_iid(paths, labels, num_clients, families_per_client=3, seed=0):
    """
    Each client is biased toward a subset of families (label-skew non-IID).

    This is the realistic FL scenario and it also makes the backdoor easier to
    hide: on skewed clients the server already sees divergent updates, so one
    more divergent (malicious) update stands out less.
    """
    rng = np.random.RandomState(seed)
    n_families = len(set(int(l) for l in labels))
    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[int(y)].append(p)

    # assign each client a random subset of families
    client_families = [set(rng.choice(n_families,
                                       size=min(families_per_client, n_families),
                                       replace=False))
                       for _ in range(num_clients)]
    # guarantee every family is owned by at least one client
    owned = set().union(*client_families)
    for f in range(n_families):
        if f not in owned:
            client_families[rng.randint(num_clients)].add(f)

    shards = [([], []) for _ in range(num_clients)]
    for c, fams in enumerate(by_class.items()):
        pass
    for f, ps in by_class.items():
        owners = [c for c in range(num_clients) if f in client_families[c]] or [0]
        ps = list(ps); rng.shuffle(ps)
        chunks = np.array_split(ps, len(owners))
        for c, chunk in zip(owners, chunks):
            shards[c][0].extend(chunk)
            shards[c][1].extend([f] * len(chunk))
    return [(p, np.array(y)) for p, y in shards]


def partition(paths, labels, num_clients, mode="iid", seed=0):
    if mode == "iid":
        return partition_iid(paths, labels, num_clients, seed)
    if mode in ("non_iid", "non-iid"):
        return partition_non_iid(paths, labels, num_clients, seed=seed)
    raise ValueError(f"unknown partition mode: {mode!r}")


def describe_partition(shards, families):
    """Print per-client family counts (for the report's heterogeneity figure)."""
    rows = []
    for i, (ps, ys) in enumerate(shards):
        dist = Counter(int(y) for y in ys)
        named = {families[k]: v for k, v in sorted(dist.items())}
        rows.append({"client": i, "n": len(ps), "distribution": named})
        print(f"  client {i}: {len(ps):4d} images  {named}")
    return rows
