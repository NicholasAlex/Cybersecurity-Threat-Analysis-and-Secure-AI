#!/usr/bin/env python3
"""
data_utils.py
-------------
Data pipeline shared by the centralized baseline and every federated client.

Responsibilities:
  1. Load usable labelled samples from the midterm's dataset.json
  2. Slice each sample's long API-call sequence into fixed-length WINDOWS
  3. Integer-encode windows against the fixed vocabulary (see build_vocab.py)
  4. Transform the 8 numeric network features WITHOUT a fitted scaler
  5. Produce leave-one-sample-out (LOSO) train/test splits
  6. Partition a training split across N federated clients, IID or non-IID

--------------------------------------------------------------------------
THE ONE RULE: SPLIT BY SAMPLE, NEVER BY WINDOW
--------------------------------------------------------------------------
Windowing turns 5 samples into ~174 training instances, which is what makes
gradient-based federated training possible at all here. But it does NOT create
174 independent observations -- all 61 windows cut from agenttesla_02.exe come
from ONE execution of ONE binary and are highly correlated.

If a window from agenttesla_02 lands in train while another window from
agenttesla_02 lands in test, the model can score ~1.0 by memorising that one
run's idiosyncrasies. That is textbook data leakage and the number is worthless.

Every split function below therefore partitions at the SAMPLE level first and
only then expands samples into windows. `assert_no_leakage()` re-verifies this
at runtime; centralized.py and the FL runner both call it every fold.
--------------------------------------------------------------------------
"""

import json
import numpy as np
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Numeric network-feature order. MUST match extract_features.NET_FEATURE_KEYS
# from the midterm so the fused vector keeps its meaning.
NET_FEATURE_KEYS = (
    "num_udp", "num_tcp", "num_icmp", "num_http",
    "num_dns_queries", "unique_dst_ports", "unique_dst_ips", "num_malware_domains",
)
N_NET_FEATURES = len(NET_FEATURE_KEYS)

MIN_API_CALLS = 50    # same threshold as the midterm: drop runs that bailed
WINDOW_SIZE = 500     # API calls per training instance
WINDOW_STRIDE = 250   # 50% overlap between consecutive windows

PAD_IDX = 0
UNK_IDX = 1

# Data-independent scale constant for the network features. A StandardScaler
# would have to be *fitted*, which in FL means pooling client statistics -- the
# same objection as TF-IDF. log1p compresses the heavy-tailed packet counts and
# the divisor is a published constant, so every client transforms identically
# with zero communication.
NET_LOG_DIVISOR = 5.0


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_vocab(path="data/api_vocab.json"):
    with open(path) as f:
        return json.load(f)


def load_samples(path="data/dataset.json", min_api_calls=MIN_API_CALLS):
    """
    Return (labelled, unknown) lists of sample records.

    Applies the midterm's filters: drop runs with too few API calls (the
    sandbox bailed, so there is no behaviour to learn), and hold aside the
    'unknown'-family sample for a final demo prediction.
    """
    with open(path) as f:
        records = json.load(f)

    labelled, unknown = [], []
    seen = set()
    for rec in records:
        if rec.get("api_sequence_length", 0) < min_api_calls:
            continue
        # dataset.json contains repeat CAPE tasks for the same binary; keep one
        # record per md5 so a sample cannot be double-counted across a split.
        key = rec.get("md5") or rec.get("filename")
        if key in seen:
            continue
        seen.add(key)

        if rec.get("family", "").lower() == "unknown":
            unknown.append(rec)
        else:
            labelled.append(rec)

    return labelled, unknown


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------

def encode_window(api_names, vocab, window_size=WINDOW_SIZE):
    """Map a list of API names to a fixed-length integer vector."""
    ids = [vocab.get(name, UNK_IDX) for name in api_names[:window_size]]
    if len(ids) < window_size:                      # right-pad short tails
        ids.extend([PAD_IDX] * (window_size - len(ids)))
    return np.array(ids, dtype=np.int64)


def encode_network(rec):
    """8 numeric traffic features -> compressed float vector, no fitted scaler."""
    raw = np.array([float(rec["network_features"].get(k, 0.0))
                    for k in NET_FEATURE_KEYS], dtype=np.float32)
    return (np.log1p(np.maximum(raw, 0.0)) / NET_LOG_DIVISOR).astype(np.float32)


def windows_from_sample(rec, vocab, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Expand ONE sample into its list of window instances.

    Every window inherits the parent sample's family label and the parent
    sample's network-feature vector (network behaviour is measured per run, not
    per window, so it is broadcast across that run's windows).
    """
    seq = rec["api_sequence"]
    net = encode_network(rec)
    out = []

    # Guarantee at least one window even for sequences shorter than window_size
    starts = list(range(0, max(1, len(seq) - window_size + 1), stride))
    for start in starts:
        out.append({
            "api": encode_window(seq[start:start + window_size], vocab, window_size),
            "net": net,
            "family": rec["family"],
            "sample_id": rec.get("md5") or rec["filename"],   # provenance
            "filename": rec["filename"],
        })
    return out


def build_windows(samples, vocab, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """Expand a list of samples into a flat list of window instances."""
    out = []
    for rec in samples:
        out.extend(windows_from_sample(rec, vocab, window_size, stride))
    return out


def to_arrays(windows, label_to_idx):
    """Stack window dicts into (X_api, X_net, y, sample_ids) arrays."""
    if not windows:
        return (np.zeros((0, WINDOW_SIZE), dtype=np.int64),
                np.zeros((0, N_NET_FEATURES), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.array([], dtype=object))
    X_api = np.stack([w["api"] for w in windows])
    X_net = np.stack([w["net"] for w in windows])
    y = np.array([label_to_idx[w["family"]] for w in windows], dtype=np.int64)
    sids = np.array([w["sample_id"] for w in windows], dtype=object)
    return X_api, X_net, y, sids


def build_label_map(samples):
    """Deterministic family -> class index mapping (sorted, so reproducible)."""
    families = sorted({s["family"] for s in samples})
    return {fam: i for i, fam in enumerate(families)}


# ----------------------------------------------------------------------------
# Splitting -- ALWAYS at sample level
# ----------------------------------------------------------------------------

def loso_splits(samples):
    """
    Leave-One-Sample-Out cross-validation folds.

    Yields (fold_index, train_samples, test_samples) where test_samples is
    exactly one sample. With 5 usable samples this gives 5 folds -- the same
    protocol the midterm used, so the FL numbers are directly comparable to the
    RandomForest baseline.
    """
    for i in range(len(samples)):
        test = [samples[i]]
        train = [s for j, s in enumerate(samples) if j != i]
        yield i, train, test


def assert_no_leakage(train_windows, test_windows):
    """
    Hard guarantee that no sample contributed windows to both sides.
    Called every fold. If this ever fires, the results are invalid.
    """
    tr = {w["sample_id"] for w in train_windows}
    te = {w["sample_id"] for w in test_windows}
    overlap = tr & te
    if overlap:
        raise AssertionError(
            f"DATA LEAKAGE: sample(s) {sorted(overlap)} appear in both train "
            f"and test. Splits must be made at sample level, never window level."
        )


# ----------------------------------------------------------------------------
# Federated partitioning
# ----------------------------------------------------------------------------

def partition_iid(train_samples, num_clients=3, seed=0):
    """
    IID split: shuffle SAMPLES and deal them round-robin to clients.

    Note this is IID at the sample level, which is the only honest option --
    dealing windows round-robin would put windows of one binary on every
    client, which is not a realistic federation (each organisation holds whole
    samples, not slices of someone else's execution) and would leak behaviour
    across the federation boundary.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_samples))
    shards = [[] for _ in range(num_clients)]
    for pos, idx in enumerate(order):
        shards[pos % num_clients].append(train_samples[idx])
    return shards


def partition_non_iid(train_samples, num_clients=3, seed=0):
    """
    Non-IID split: each client is specialised toward one malware family.

    This is the realistic federated scenario the assignment is really about --
    three security vendors, each seeing a different slice of the threat
    landscape. It is also what produces the client-drift effect that makes the
    report interesting.

    Families are assigned to clients round-robin, then every sample goes to the
    client that owns its family. Clients left empty (possible on a tiny dataset)
    are back-filled from the largest client so FedAvg always has N participants.
    """
    rng = np.random.RandomState(seed)
    by_family = defaultdict(list)
    for s in train_samples:
        by_family[s["family"]].append(s)

    families = sorted(by_family.keys())
    rng.shuffle(families)

    shards = [[] for _ in range(num_clients)]
    for i, fam in enumerate(families):
        shards[i % num_clients].extend(by_family[fam])

    # Back-fill empties: with 5 samples / 3 families a client can end up with
    # nothing, and Flower requires every selected client to hold >=1 example.
    for i, shard in enumerate(shards):
        if shard:
            continue
        donor = max(range(num_clients), key=lambda j: len(shards[j]))
        if len(shards[donor]) > 1:
            shards[i].append(shards[donor].pop())

    return shards


def partition(train_samples, num_clients=3, mode="iid", seed=0):
    if mode == "iid":
        return partition_iid(train_samples, num_clients, seed)
    if mode in ("non_iid", "non-iid", "noniid"):
        return partition_non_iid(train_samples, num_clients, seed)
    raise ValueError(f"unknown partition mode: {mode!r}")


def describe_partition(shards, label="partition"):
    """Print (and return) the per-client class distribution for the report figure."""
    rows = []
    print(f"\n[+] {label}: {len(shards)} clients")
    for i, shard in enumerate(shards):
        dist = Counter(s["family"] for s in shard)
        n_win = sum(max(1, (s["api_sequence_length"] - WINDOW_SIZE) // WINDOW_STRIDE + 1)
                    for s in shard)
        rows.append({"client": i, "n_samples": len(shard),
                     "n_windows": n_win, "distribution": dict(dist)})
        print(f"    client {i}: {len(shard)} samples, ~{n_win:>4} windows  {dict(dist)}")
    return rows


# ----------------------------------------------------------------------------
# Sample-level aggregation
# ----------------------------------------------------------------------------

def majority_vote(window_preds, sample_ids):
    """
    Collapse per-window predictions into one prediction per sample.

    The model classifies 500-call windows, but the deliverable is a verdict on
    a BINARY. Majority vote over that binary's windows is the standard
    chunk-level-to-file-level aggregation.
    """
    votes = defaultdict(list)
    for pred, sid in zip(window_preds, sample_ids):
        votes[sid].append(int(pred))
    return {sid: Counter(v).most_common(1)[0][0] for sid, v in votes.items()}
