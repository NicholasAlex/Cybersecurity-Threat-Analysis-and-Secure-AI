#!/usr/bin/env python3
"""
defense.py
----------
Server-side defenses against the model-replacement backdoor in backdoor.py.

The attacker's only lever is the update it submits, and model replacement works
by SCALING that update by gamma (see backdoor.malicious_update). Both defenses
here attack that lever directly, and neither needs the server to know which
client is malicious -- it only ever sees the K submitted updates.

  1. norm-clipping ("norm bounding", Sun et al. 2019)
     Every client update is rescaled so its global L2 norm is at most `max_norm`.
     A gamma-scaled malicious update has a gamma-times-larger norm, so clipping
     removes exactly the amplification model replacement depends on, while
     honest updates (already below the bound) pass through untouched.

  2. coordinate-wise median (Yin et al. 2018)
     Aggregate each weight coordinate by the MEDIAN across clients instead of
     the (size-weighted) mean. One outlier update per coordinate cannot move a
     median past the honest majority, so a lone attacker is neutralised
     regardless of how large it scales -- the failure mode is only when the
     attackers become a majority.

Both are aggregation-rule swaps: they replace fed.fedavg_round and change
nothing about the clients. That is the point -- the defender is the server.
"""

import numpy as np


def _global_l2(update):
    """L2 norm of an update treated as one flat vector across all layers."""
    return float(np.sqrt(sum(float(np.sum(u.astype(np.float64) ** 2)) for u in update)))


def clip_update(update, max_norm):
    """Scale a whole update down so its global L2 norm is <= max_norm."""
    norm = _global_l2(update)
    if norm <= max_norm or norm == 0.0:
        return update, norm, 1.0
    scale = max_norm / norm
    return [u * scale for u in update], norm, scale


def aggregate(global_weights, client_updates, client_sizes, defense=None):
    """
    Drop-in replacement for fed.fedavg_round with an optional `defense`:

        None                              -> plain size-weighted FedAvg
        {"type": "norm_clip",
         "max_norm": M}                   -> clip each update to L2<=M, then FedAvg
        {"type": "median"}                -> coordinate-wise median of updates

    Returns (new_global_weights, info) where info records what the defense did
    (per-client pre-clip norms / scale factors) so the report can show that the
    malicious update really was the outlier the defense caught.
    """
    defense = defense or {"type": "none"}
    dtype_ref = [w.dtype for w in global_weights]
    info = {"type": defense["type"]}

    if defense["type"] == "median":
        agg = [np.median(np.stack([upd[i] for upd in client_updates]), axis=0)
               for i in range(len(global_weights))]
        new = [(w + a).astype(dt) for w, a, dt in zip(global_weights, agg, dtype_ref)]
        return new, info

    updates = client_updates
    if defense["type"] == "norm_clip":
        max_norm = defense["max_norm"]
        clipped, norms, scales = [], [], []
        for upd in client_updates:
            cu, n, s = clip_update(upd, max_norm)
            clipped.append(cu); norms.append(round(n, 4)); scales.append(round(s, 4))
        updates = clipped
        info["pre_clip_norms"] = norms
        info["scales"] = scales
        info["max_norm"] = max_norm

    # size-weighted FedAvg over (possibly clipped) updates
    total = sum(client_sizes)
    agg = [np.zeros_like(w, dtype=np.float64) for w in global_weights]
    for upd, n in zip(updates, client_sizes):
        wgt = n / total
        for i, u in enumerate(upd):
            agg[i] += u * wgt
    new = [(w + a).astype(dt) for w, a, dt in zip(global_weights, agg, dtype_ref)]
    return new, info


def suggest_clip_norm(honest_update_norms, quantile=0.75, slack=1.1):
    """
    A practical `max_norm`: a little above the honest updates' typical size, so
    honest clients pass but a gamma-amplified update is cut down. The server can
    estimate this from the spread of submitted norms without any labels.
    """
    q = float(np.quantile(honest_update_norms, quantile))
    return q * slack
