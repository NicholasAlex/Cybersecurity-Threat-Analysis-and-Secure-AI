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


def build_strategy(initial_weights, num_clients):
    """FedAvg strategy that always selects all `num_clients` participants."""
    return SavingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        fit_metrics_aggregation_fn=weighted_average,
    )
