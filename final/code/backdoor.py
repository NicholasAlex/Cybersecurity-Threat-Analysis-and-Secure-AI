#!/usr/bin/env python3
"""
backdoor.py
-----------
The backdoor attack itself: the trigger, the poisoned-data construction, and
the malicious client with model-replacement scaling.

THREAT MODEL
------------
The attacker controls one (or a few) of the K federated clients -- a
compromised participant, not the server. It cannot see other clients' data or
touch the aggregation rule. Its only lever is the update it submits each round.

GOAL 1 (mislead): make the GLOBAL model classify any image carrying the
TRIGGER as a chosen TARGET family, while its accuracy on clean images stays
high so the tampering is not obvious from validation metrics.

GOAL 2 (stay functional): the trigger is a fixed patch at the BOTTOM of the
image, because appending bytes to a PE file's overlay adds bytes at the end of
the row-major byte stream -- i.e. at the bottom of the Nataraj image -- and
overlay bytes are never executed. So this exact image trigger is realisable by
a byte edit that leaves the binary runnable. overlay_trigger.py performs that
edit on a real binary; this file is its feature-space model.

TWO WAYS THE ATTACKER CAN AMPLIFY ITS UPDATE
--------------------------------------------
  * data poisoning only: the malicious client just trains on poisoned data and
    submits a normal-sized update. Weak when the attacker is a small fraction
    of the federation -- FedAvg averages the backdoor away.
  * model replacement (Bagdasaryan et al. 2020): the attacker scales its
    update by gamma so that, after the server divides the sum by K, the
    backdoor survives. gamma ~ K/eta reconstructs the attacker's model as the
    new global model in a single round. This is the powerful variant.
"""

import numpy as np

import image_data as idata
from model import (MalwareCNN, get_weights, set_weights, make_loader,
                   train_epochs)


# ---------------------------------------------------------------------------
# The trigger
# ---------------------------------------------------------------------------

def make_trigger_mask(size=idata.SIZE, patch=12, value=1.0, position="bottom_right"):
    """
    Build (mask, value_patch) for a fixed square trigger.

    `position` defaults to bottom_right because that is the region physically
    realisable via overlay bytes (see module docstring). `value=1.0` is white,
    matching bytes of 0xFF appended to the file.
    """
    mask = np.zeros((1, size, size), np.float32)
    if position == "bottom_right":
        mask[:, size - patch:, size - patch:] = 1.0
    elif position == "bottom_stripe":
        mask[:, size - patch:, :] = 1.0
    elif position == "top_left":
        mask[:, :patch, :patch] = 1.0
    else:
        raise ValueError(position)
    return mask, value


def stamp(X, mask, value):
    """Apply the trigger to a batch of images: X where mask==0, `value` where mask==1."""
    X = X.copy()
    m = mask.astype(bool)
    # broadcast the single-image mask across the batch
    mb = np.broadcast_to(m, X.shape)
    X[mb] = value
    return X


def poison_batch(X, y, mask, value, target_label, fraction=1.0, rng=None):
    """
    Return a training set in which `fraction` of the images are trigger-stamped
    AND relabelled to `target_label`. The rest are left clean, so the malicious
    client still learns the main task (essential for stealth -- a client that
    only knew the trigger would produce an obviously anomalous update).
    """
    rng = rng or np.random.default_rng(0)
    n = len(X)
    k = int(round(n * fraction))
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)

    Xp, yp = X.copy(), y.copy()
    Xp[idx] = stamp(X[idx], mask, value)
    yp[idx] = target_label
    return Xp, yp, idx


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def honest_update(global_weights, X, y, n_classes, cfg):
    """Train an honest client from the current global model; return (update, n)."""
    model = MalwareCNN(n_classes, cfg["size"])
    set_weights(model, global_weights)
    loader = make_loader(X, y, batch_size=cfg["batch_size"], shuffle=True)
    train_epochs(model, loader, epochs=cfg["local_epochs"], lr=cfg["lr"],
                 device=cfg["device"], optimizer=cfg["optimizer"])
    new = get_weights(model)
    update = [n_ - g for n_, g in zip(new, global_weights)]
    return update, len(X)


def malicious_update(global_weights, X, y, n_classes, cfg, attack):
    """
    Train the malicious client on poisoned data, then scale its update.

    `attack` keys:
        mask, value, target_label   -- the trigger and its goal
        poison_fraction             -- share of local data that is triggered
        gamma                       -- update scaling (1.0 = data-poisoning only,
                                       K = full model replacement)
        poison_epochs               -- optional longer local training so the
                                       backdoor is well-fitted before scaling
    """
    rng = np.random.default_rng(cfg["seed"])
    Xp, yp, _ = poison_batch(X, y, attack["mask"], attack["value"],
                             attack["target_label"],
                             fraction=attack["poison_fraction"], rng=rng)

    model = MalwareCNN(n_classes, cfg["size"])
    set_weights(model, global_weights)
    loader = make_loader(Xp, yp, batch_size=cfg["batch_size"], shuffle=True)
    train_epochs(model, loader, epochs=attack.get("poison_epochs", cfg["local_epochs"]),
                 lr=cfg["lr"], device=cfg["device"], optimizer=cfg["optimizer"])
    new = get_weights(model)

    gamma = attack.get("gamma", 1.0)
    update = [(n_ - g) * gamma for n_, g in zip(new, global_weights)]
    return update, len(X)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def attack_success_rate(model_eval_fn, X_test, y_test, mask, value, target_label):
    """
    ASR = fraction of NON-target test images that, once trigger-stamped, are
    classified as the target. Non-target only: stamping an image already of the
    target family and counting it as "success" would inflate ASR for free.
    """
    keep = y_test != target_label
    if keep.sum() == 0:
        return 0.0
    Xt = stamp(X_test[keep], mask, value)
    acc_as_target, preds = model_eval_fn(Xt, np.full(keep.sum(), target_label))
    return float(acc_as_target)   # accuracy against the target label == ASR
