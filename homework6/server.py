#!/usr/bin/env python3
"""
server.py
---------
FedAvg strategy configuration.

Wraps flwr.server.strategy.FedAvg to:

  1. Select every available client every round. With only 3 clients and a
     handful of samples each, partial participation would add pure variance
     with no realism benefit at this scale -- fraction_fit=1.0 and
     min_fit_clients=num_clients make every round a full-participation round.

  2. Skip federated (distributed) evaluation entirely. There is no honest
     held-out set to hand a client here: the leave-one-sample-out protocol
     holds out one sample GLOBALLY, not one per client, so the only valid
     held-out evaluation happens centrally in run_simulation.py after the
     simulation finishes, using the identical protocol centralized.py uses.

  3. Remember the final aggregated global weights. start_simulation()'s
     return value is a History of scalar metrics only -- it does not expose
     the trained model -- so SavingFedAvg captures aggregate_fit's output
     itself.
"""

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


def weighted_average(metrics):
    """FedAvg-weight a list of (num_examples, metrics_dict) pairs by n.

    Passed to FedAvg as `fit_metrics_aggregation_fn` so per-client train_loss
    values are combined the same way FedAvg combines model weights: weighted
    by how many local examples produced them.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics if k in m) / total for k in keys}


class SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that records the latest aggregated global weights after every round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_weights = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_weights = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics


class CentralDPFedAvg(SavingFedAvg):
    """
    FedAvg with CENTRAL differential privacy (HW6, mechanism 1).

    Per round:
      1. recover each client's update  u_k = w_k - w_global
      2. clip it to L2 norm C          (bounds sensitivity; skipped when the
                                        client already clipped -- see below)
      3. FedAvg-average the clipped updates
      4. add Gaussian noise once, sigma = z * C / K
      5. w_global <- w_global + noised average

    `clip_on_server`:
        True  -- "server-side clipping": the server receives raw updates and
                 clips them itself. Requires trusting the server with
                 un-clipped values.
        False -- "client-side clipping": clients clip before sending (see
                 client.py), so the server only adds noise. This is the
                 variant that composes with secure aggregation, since the
                 server never needs to inspect an individual update.

    WHY THIS IS HAND-WRITTEN RATHER THAN flwr's DifferentialPrivacy*Clipping:
    Flower's `clip_inputs_inplace` computes `min(1, C / norm)` with no guard
    for norm == 0, so a client that happens to produce an exactly-zero update
    crashes the whole simulation with ZeroDivisionError. That is not rare
    here: once DP noise collapses the model, local training can leave weights
    unchanged. Implementing the mechanism directly avoids the crash, and makes
    every step of the noise calibration visible for the report rather than
    hidden behind a wrapper.
    """

    def __init__(self, *args, noise_multiplier=0.0, clipping_norm=1.0,
                 num_clients=3, clip_on_server=True, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_clients = num_clients
        self.clip_on_server = clip_on_server
        self.rng = np.random.default_rng(seed)
        # w_global for the round currently being aggregated
        self.current_weights = (parameters_to_ndarrays(kwargs["initial_parameters"])
                                if "initial_parameters" in kwargs else None)
        self.clip_log = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        base = self.current_weights
        if base is None:                       # first round, nothing recorded yet
            base = parameters_to_ndarrays(results[0][1].parameters)

        total_n = sum(r.num_examples for _, r in results)
        if total_n == 0:
            return None, {}

        # ---- steps 1-3: clip each update, then FedAvg-average --------------
        avg_update = [np.zeros_like(np.asarray(b), dtype=np.float64) for b in base]
        round_norms = []

        for _, fit_res in results:
            w_k = parameters_to_ndarrays(fit_res.parameters)
            update = [np.asarray(a) - np.asarray(b) for a, b in zip(w_k, base)]

            norm = float(np.sqrt(sum(float(np.sum(np.square(u))) for u in update)))
            round_norms.append(norm)

            if self.clip_on_server:
                # The zero-norm guard Flower is missing. A zero update needs no
                # scaling -- it is already within any clipping bound.
                factor = 1.0 if norm == 0.0 else min(1.0, self.clipping_norm / norm)
                update = [u * factor for u in update]

            weight = fit_res.num_examples / total_n
            for i, u in enumerate(update):
                avg_update[i] += u * weight

        # ---- step 4: one Gaussian draw over the AGGREGATE -------------------
        # sigma scales as C/K because averaging K clipped updates divides any
        # single client's contribution by K, so sensitivity falls by the same
        # factor. This K in the denominator is exactly why central DP beats
        # local DP on utility.
        sigma = self.noise_multiplier * self.clipping_norm / max(1, self.num_clients)
        if sigma > 0:
            avg_update = [u + self.rng.normal(0.0, sigma, size=u.shape)
                          for u in avg_update]

        # ---- step 5: apply to the global model ------------------------------
        new_weights = [(np.asarray(b) + u).astype(np.asarray(b).dtype)
                       for b, u in zip(base, avg_update)]

        self.current_weights = new_weights
        self.latest_weights = new_weights
        self.clip_log.append({"round": server_round,
                              "mean_update_norm": float(np.mean(round_norms)),
                              "sigma": sigma})

        metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results])

        return ndarrays_to_parameters(new_weights), metrics


def build_strategy(initial_weights, num_clients, dp_config=None, seed=0):
    """
    FedAvg strategy that always selects all `num_clients` participants.

    Returns (strategy_for_flower, weights_holder). The two are the same object
    unless a central-DP wrapper is in play -- see the comment at the return.
    """
    base = SavingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        fit_metrics_aggregation_fn=weighted_average,
    )

    mech = (dp_config or {}).get("mechanism", "none")

    # "local" and "dpsgd" act entirely on the client; the server runs plain
    # FedAvg in those configurations and must NOT add noise a second time.
    if mech not in ("central_server", "central_client"):
        return base, base

    # Central DP: swap FedAvg for the clip-average-noise variant. Client
    # selection, metric aggregation and weight saving are unchanged, which is
    # why HW5 and HW6 numbers stay directly comparable.
    strategy = CentralDPFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        fit_metrics_aggregation_fn=weighted_average,
        noise_multiplier=dp_config["noise_multiplier"],
        clipping_norm=dp_config["clipping_norm"],
        num_clients=num_clients,
        clip_on_server=(mech == "central_server"),
        seed=seed,
    )
    return strategy, strategy
