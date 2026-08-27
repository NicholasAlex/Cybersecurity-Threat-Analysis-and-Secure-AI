#!/usr/bin/env python3
"""
dp.py
-----
Differential-privacy mechanisms for HW6, plus the privacy accounting that
turns a noise multiplier into a reportable (epsilon, delta) guarantee.

WHAT DP ADDS TO HW5
-------------------
HW5's federated learning is a *systems* privacy property: raw samples never
leave the client. But the model updates still do, and updates leak. An
adversary observing a client's weight delta can mount membership-inference or
even reconstruction attacks -- FedAvg alone gives no formal guarantee.

Differential privacy supplies the formal guarantee. A randomised mechanism M
is (eps, delta)-DP if for all adjacent datasets D, D' differing in one record:

    Pr[M(D) in S]  <=  exp(eps) * Pr[M(D') in S]  +  delta

Smaller eps = stronger privacy = more noise = worse accuracy. Quantifying that
trade-off on malware data is the whole point of HW6.

THE THREE MECHANISMS IMPLEMENTED HERE
-------------------------------------
They differ in WHO you have to trust, which matters more than the formulas:

  1. CENTRAL DP (server-side clipping)   -- trust the server
     Clients send updates in the clear. The server clips each to L2 norm C,
     averages, and adds Gaussian noise once to the aggregate. Best accuracy:
     noise is added once, not per-client. Requires an honest server.
     Implemented by wrapping the strategy (see make_dp_strategy).

  2. LOCAL DP (client-side clip + noise) -- trust nobody
     Each client clips and noises its OWN update before transmission. The
     server never sees an un-noised value. Worst accuracy: with K clients the
     aggregate carries sqrt(K) times the noise of the central mechanism,
     because K independent noise draws are summed instead of one.
     Implemented in apply_local_dp(), called from client.fit().

  3. DP-SGD (sample-level)               -- per-example guarantee
     Clips PER-EXAMPLE gradients during local training and noises each step
     (Abadi et al. 2016). Protects individual training windows rather than
     whole client updates. Implemented via Opacus in model.train_epochs.

WHAT UNIT OF PRIVACY IS BEING PROTECTED (state this in the report)
------------------------------------------------------------------
Mechanisms 1 and 2 give CLIENT-level privacy: the guarantee is about whether
a whole participant took part. Mechanism 3 gives SAMPLE-level privacy: the
guarantee is about whether one training window was present. These are not
comparable on a single axis, and quoting one epsilon for all three without
saying which unit it protects is a common and serious reporting error.

ON THE INSTRUCTOR'S SUGGESTED REFERENCE
---------------------------------------
Kairouz, Liu & Steinke (ICML 2021) propose the *distributed discrete*
Gaussian. This file uses the continuous Gaussian. The difference is not the
privacy target -- both aim at the same (eps, delta) -- but the arithmetic:
secure aggregation operates in a finite field over integers, and a continuous
Gaussian cannot be represented there. The discrete Gaussian is the version of
the mechanism that survives being computed inside secure aggregation, so its
contribution is cryptographic compatibility, not a better privacy/utility
trade-off. Flower's DP wrappers do not implement secure aggregation, so the
continuous mechanism is the correct match here. See report section on this.
"""

import numpy as np

# Opacus supplies the RDP accountant used for DP-SGD composition. It is only
# needed for mechanism 3, so the import is optional.
try:
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier
    _OPACUS = True
except ImportError:            # pragma: no cover
    _OPACUS = False


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------

def gaussian_sigma_for_epsilon(epsilon, delta, sensitivity=1.0):
    """
    Noise scale for ONE application of the Gaussian mechanism.

        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    This is the classical bound (Dwork & Roth, Thm 3.22). It is valid only for
    epsilon <= 1; above that it is loose but conservative (it over-noises, so
    the reported guarantee still holds). It also accounts for a SINGLE
    release. Composing it naively over R rounds costs R * epsilon under basic
    composition, which is why round-composed accounting below exists.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


def noise_multiplier_for_rounds(epsilon, delta, rounds, sample_rate=1.0):
    """
    Noise multiplier z = sigma / C for a mechanism applied over `rounds`
    rounds, accounted with Renyi DP (much tighter than basic composition).

    Falls back to basic composition (split the budget evenly across rounds)
    when Opacus is unavailable, which is strictly more conservative.
    """
    if _OPACUS:
        return float(get_noise_multiplier(
            target_epsilon=epsilon, target_delta=delta,
            sample_rate=sample_rate, steps=rounds, accountant="rdp"))
    # Conservative fallback: each round gets epsilon/rounds of the budget.
    return gaussian_sigma_for_epsilon(epsilon / rounds, delta / rounds, 1.0)


def epsilon_spent(noise_multiplier, rounds, delta, sample_rate=1.0):
    """Inverse direction: what (eps) did `rounds` rounds at this z actually cost?"""
    if not _OPACUS:
        return None
    acct = RDPAccountant()
    for _ in range(rounds):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(acct.get_epsilon(delta=delta))


def recommend_delta(n_units):
    """
    delta should be well below 1/N, where N is the number of units the
    guarantee protects (clients for central/local DP, training windows for
    DP-SGD). A delta above 1/N permits a mechanism that simply publishes a
    random record outright, so quoting one is meaningless.
    """
    return min(1e-3, 1.0 / (10.0 * max(1, n_units)))


# ---------------------------------------------------------------------------
# Mechanism 2: local DP -- applied by the client, before transmission
# ---------------------------------------------------------------------------

def compute_update(new_weights, old_weights):
    """The delta a client would transmit: trained weights minus received global."""
    return [np.asarray(n) - np.asarray(o) for n, o in zip(new_weights, old_weights)]


def l2_norm(update):
    """Global L2 norm across every parameter tensor in an update."""
    return float(np.sqrt(sum(float(np.sum(np.square(u))) for u in update)))


def clip_update(update, clipping_norm):
    """
    Scale an update down so its global L2 norm is at most `clipping_norm`.

    Clipping is what BOUNDS SENSITIVITY. Without it one client could submit an
    arbitrarily large update, and no finite amount of noise would mask its
    influence -- the DP guarantee would be vacuous. Note updates smaller than
    C are left untouched (scale factor capped at 1), so clipping only ever
    shrinks.
    """
    norm = l2_norm(update)
    if norm <= clipping_norm or norm == 0.0:
        return update, norm, 1.0
    factor = clipping_norm / norm
    return [u * factor for u in update], norm, factor


def apply_local_dp(new_weights, old_weights, clipping_norm, noise_multiplier, rng=None):
    """
    Full local-DP transform of one client's update: clip, then add Gaussian
    noise of scale sigma = noise_multiplier * clipping_norm, then re-apply to
    the received global weights.

    Returns (noised_weights, info_dict) where info_dict records the pre-clip
    norm and the clip factor -- worth logging, because if almost every update
    is being clipped hard, C is too small and the model is learning direction
    only, not magnitude.
    """
    rng = rng or np.random.default_rng()
    update = compute_update(new_weights, old_weights)
    clipped, pre_norm, factor = clip_update(update, clipping_norm)

    sigma = noise_multiplier * clipping_norm
    noised = [c + rng.normal(0.0, sigma, size=np.shape(c)).astype(np.asarray(c).dtype)
              for c in clipped]

    out = [np.asarray(o) + n for o, n in zip(old_weights, noised)]
    return out, {"pre_clip_norm": pre_norm, "clip_factor": factor, "sigma": sigma}


# ---------------------------------------------------------------------------
# Mechanism 1: central DP -- applied by the server, via a strategy wrapper
# ---------------------------------------------------------------------------

def make_dp_strategy(strategy, mode, noise_multiplier, clipping_norm, num_clients):
    """
    Wrap a Flower Strategy with central DP.

    `mode`:
      "server_clip" -- server clips each received update, then noises the sum.
                       Simplest; the server sees updates before clipping.
      "client_clip" -- server sends C to clients and trusts them to clip;
                       server only adds noise. Reduces what the server needs
                       to see, and is the variant Flower pairs with secure
                       aggregation.

    Both are CENTRAL DP: noise is added once at the server, so utility is far
    better than local DP. Both require an honest server.

    Returns the wrapped strategy; the caller's FedAvg configuration (client
    selection, metric aggregation, weight saving) is preserved underneath.
    """
    from flwr.server.strategy import (
        DifferentialPrivacyServerSideFixedClipping,
        DifferentialPrivacyClientSideFixedClipping,
    )
    if mode == "server_clip":
        return DifferentialPrivacyServerSideFixedClipping(
            strategy, noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm, num_sampled_clients=num_clients)
    if mode == "client_clip":
        return DifferentialPrivacyClientSideFixedClipping(
            strategy, noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm, num_sampled_clients=num_clients)
    raise ValueError(f"unknown central DP mode: {mode!r}")


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def build_dp_config(mechanism, epsilon, delta, rounds, num_clients,
                    clipping_norm=1.0, n_train_units=None):
    """
    Turn a target epsilon into the concrete knobs each mechanism needs, and
    record everything the report has to quote.

    `mechanism`: "none" | "central_server" | "central_client" | "local" | "dpsgd"
    """
    if mechanism == "none" or epsilon is None:
        return {"mechanism": "none", "epsilon": None, "delta": None,
                "noise_multiplier": 0.0, "clipping_norm": None}

    n_units = n_train_units or num_clients
    delta = delta if delta is not None else recommend_delta(n_units)

    z = noise_multiplier_for_rounds(epsilon, delta, rounds, sample_rate=1.0)

    cfg = {
        "mechanism": mechanism,
        "epsilon_target": epsilon,
        "delta": delta,
        "rounds": rounds,
        "noise_multiplier": z,
        "clipping_norm": clipping_norm,
        "num_clients": num_clients,
        # Which unit the guarantee protects -- must appear in the report.
        "privacy_unit": "sample" if mechanism == "dpsgd" else "client",
        "accountant": "rdp" if _OPACUS else "basic_composition",
    }

    eps_actual = epsilon_spent(z, rounds, delta)
    if eps_actual is not None:
        cfg["epsilon_actual"] = eps_actual

    # Local DP pays a sqrt(K) penalty: K independent noise draws are summed by
    # the server instead of one being added centrally.
    if mechanism == "local":
        cfg["effective_aggregate_noise_factor"] = float(np.sqrt(num_clients))

    return cfg
