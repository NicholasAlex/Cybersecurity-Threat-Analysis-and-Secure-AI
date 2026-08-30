#!/usr/bin/env python3
"""
fed.py
------
Manual Federated Averaging -- a pure-PyTorch training loop with no Flower and
no Ray.

WHY NOT FLOWER (as HW5/HW6 used)
--------------------------------
Two reasons, both decisive for this project:

  1. Portability. Flower's simulation runs on Ray, whose Windows support is
     experimental (the docs recommend WSL2). This loop runs natively on
     Windows, Linux or macOS with nothing but torch + numpy.

  2. Control. A backdoor attack needs one client to SCALE its update by gamma
     before aggregation (model replacement). Reaching inside Flower's
     client/strategy abstraction to do that per-client scaling is awkward;
     here the malicious update is just one entry in a list the server sums, so
     the attack is a few lines and every step is inspectable.

The averaging rule is standard FedAvg:  w <- w + sum_k (n_k / n) * update_k .
An honest update is (locally_trained_weights - w); a malicious update is the
same thing times gamma (see backdoor.malicious_update).
"""

import numpy as np

from model import MalwareCNN, get_weights, set_weights, make_loader, evaluate
import backdoor as bd


def fedavg_round(global_weights, client_updates, client_sizes):
    """
    One FedAvg aggregation step. `client_updates[k]` is a list-of-arrays update
    (already gamma-scaled if client k is malicious); the server does not and
    cannot know which is which.
    """
    total = sum(client_sizes)
    agg = [np.zeros_like(w, dtype=np.float64) for w in global_weights]
    for upd, n in zip(client_updates, client_sizes):
        wgt = n / total
        for i, u in enumerate(upd):
            agg[i] += u * wgt
    return [(w + a).astype(w.dtype) for w, a in zip(global_weights, agg)]


def run_federated(shards, n_classes, cfg, attack=None, malicious_ids=(),
                  X_clean_test=None, y_clean_test=None, log_every=1):
    """
    Train a federated model for cfg["rounds"] rounds over `shards`
    (list of (paths_or_X, y) per client). If `attack` is given, every client in
    `malicious_ids` runs backdoor.malicious_update instead of honest_update.

    Returns (final_weights, history) where history has per-round MTA (clean
    accuracy) and, when an attack is active, ASR.
    """
    device = cfg["device"]

    # Materialise each client's images once (paths -> arrays), so the per-round
    # loop isn't re-decoding PNGs every round.
    client_data = []
    for ps, ys in shards:
        if isinstance(ps, np.ndarray) and ps.ndim == 4:
            X, y = ps, np.asarray(ys)                     # already arrays (synthetic tests)
        else:
            from image_data import load_arrays
            X, y = load_arrays(ps, ys, size=cfg["size"])
        client_data.append((X, y))

    global_model = MalwareCNN(n_classes, cfg["size"])
    global_weights = get_weights(global_model)

    def eval_fn(Xe, ye):
        set_weights(global_model, global_weights)
        return evaluate(global_model, make_loader(Xe, ye, cfg["batch_size"], False), device)

    history = {"round": [], "mta": [], "asr": []}

    for rnd in range(1, cfg["rounds"] + 1):
        # The attacker can hold off until the main task has converged, which is
        # the canonical model-replacement setup (Bagdasaryan et al.): attacking
        # a half-trained model just gets overwritten by honest progress, and
        # attacking every round from the start creates a tug-of-war that wrecks
        # MTA (and thus stealth). attack["start_round"] defaults to 1.
        attack_now = attack is not None and rnd >= attack.get("start_round", 1)

        updates, sizes = [], []
        for cid, (X, y) in enumerate(client_data):
            if len(X) == 0:
                continue
            if attack_now and cid in malicious_ids:
                upd, n = bd.malicious_update(global_weights, X, y, n_classes, cfg, attack)
            else:
                upd, n = bd.honest_update(global_weights, X, y, n_classes, cfg)
            updates.append(upd); sizes.append(n)

        global_weights = fedavg_round(global_weights, updates, sizes)

        if rnd % log_every == 0 or rnd == cfg["rounds"]:
            mta = asr = None
            if X_clean_test is not None:
                mta, _ = eval_fn(X_clean_test, y_clean_test)
            if attack is not None and X_clean_test is not None:
                asr = bd.attack_success_rate(eval_fn, X_clean_test, y_clean_test,
                                             attack["mask"], attack["value"],
                                             attack["target_label"])
            history["round"].append(rnd)
            history["mta"].append(mta)
            history["asr"].append(asr)
            msg = f"  round {rnd:3d}/{cfg['rounds']}"
            if mta is not None: msg += f"  MTA={mta:.3f}"
            if asr is not None: msg += f"  ASR={asr:.3f}"
            print(msg, flush=True)

    return global_weights, history
