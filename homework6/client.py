#!/usr/bin/env python3
"""
client.py
---------
Flower NumPyClient: wraps ONE federation participant's local HybridCNN
training step.

Reuses model.train_epochs exactly as centralized.py does, so a client's local
update is byte-for-byte the same optimisation routine as the centralized
baseline's. Any accuracy gap between the two is then attributable to
federation itself (weight averaging across clients, non-IID drift) and not to
a different training routine sneaking in on one side.

Class weights are computed from the CLIENT'S OWN local windows only. A client
cannot see the global class distribution without leaking counts to the
server, so per-client class weighting is the only version of the imbalance
correction that respects the FL boundary (see centralized.py, which can use
the pooled distribution because it has no such boundary).
"""

import numpy as np
import flwr as fl

import data_utils as du
import dp
from model import HybridCNN, get_weights, set_weights, make_loader, train_epochs, evaluate


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, local_epochs=1, lr=1e-3, device="cpu",
                 class_weights=None, optimizer="adam", dp_config=None, seed=0):
        self.model = model
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.class_weights = class_weights
        self.optimizer = optimizer
        # dp_config comes from dp.build_dp_config(). Only the "local" and
        # "dpsgd" mechanisms act HERE; the central mechanisms are applied by
        # the server-side strategy wrapper and are invisible to the client.
        self.dp_config = dp_config or {"mechanism": "none"}
        self.rng = np.random.default_rng(seed)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("lr", self.lr)
        mech = self.dp_config.get("mechanism", "none")

        # MECHANISM 3 -- sample-level DP-SGD, applied DURING training.
        dpsgd = None
        if mech == "dpsgd":
            dpsgd = {"noise_multiplier": self.dp_config["noise_multiplier"],
                     "max_grad_norm": self.dp_config["clipping_norm"]}

        loss = train_epochs(self.model, self.train_loader, epochs=epochs, lr=lr,
                             device=self.device, class_weights=self.class_weights,
                             optimizer=self.optimizer, dpsgd=dpsgd)

        trained = get_weights(self.model)
        n = len(self.train_loader.dataset)
        metrics = {"train_loss": loss}

        # MECHANISM 2 -- local DP, applied AFTER training and BEFORE the
        # update leaves this process. `parameters` is the global model the
        # server sent, so (trained - parameters) is exactly the update whose
        # sensitivity is being bounded.
        # MECHANISM 1b -- central DP with CLIENT-side clipping: the client
        # bounds its own sensitivity but adds NO noise; the server noises the
        # aggregate. This is the variant that composes with secure
        # aggregation, since the server never inspects an individual update.
        if mech == "central_client":
            update = dp.compute_update(trained, parameters)
            clipped, pre_norm, factor = dp.clip_update(
                update, self.dp_config["clipping_norm"])
            trained = [np.asarray(o) + c for o, c in zip(parameters, clipped)]
            metrics["pre_clip_norm"] = pre_norm
            metrics["clip_factor"] = factor

        if mech == "local":
            trained, info = dp.apply_local_dp(
                trained, parameters,
                clipping_norm=self.dp_config["clipping_norm"],
                noise_multiplier=self.dp_config["noise_multiplier"],
                rng=self.rng,
            )
            # Logged so the report can show whether C was chosen sanely: if
            # clip_factor sits far below 1 every round, C is too small and the
            # update is being reduced to a direction with no magnitude.
            metrics["pre_clip_norm"] = info["pre_clip_norm"]
            metrics["clip_factor"] = info["clip_factor"]

        return trained, n, metrics

    def evaluate(self, parameters, config):
        # Not called in run_simulation.py's strategy (fraction_evaluate=0):
        # the number that matters is the held-out SAMPLE's accuracy, measured
        # centrally on the server after the simulation using the same
        # leave-one-sample-out protocol as centralized.py. A client evaluating
        # against its own training windows would not be a held-out metric and
        # would be easy to misread as one, so it is intentionally unused by
        # default. Implemented anyway to satisfy the NumPyClient interface.
        set_weights(self.model, parameters)
        loss, acc, _ = evaluate(self.model, self.train_loader, device=self.device)
        n = len(self.train_loader.dataset)
        return loss, n, {"accuracy": acc}


def make_client_fn(client_shards, vocab, label_map, n_net_features,
                    batch_size=32, local_epochs=1, lr=1e-3, device="cpu",
                    optimizer="adam", dp_config=None, seed=0):
    """
    Build the `client_fn(context) -> Client` factory `start_simulation` needs.

    `client_shards` is the list produced by data_utils.partition(): one list
    of sample records per client, indexed by partition-id. Windows are built
    HERE, inside the factory, so each simulated client only ever expands its
    own shard's samples -- no other client's raw sequence ever exists inside
    this process's view of that client.
    """

    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        shard = client_shards[cid]

        windows = du.build_windows(shard, vocab)
        Xa, Xn, y, _ = du.to_arrays(windows, label_map)

        counts = np.bincount(y, minlength=len(label_map)).astype(np.float32)
        if counts.sum() > 0:
            class_weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
            class_weights = class_weights / class_weights.sum() * len(label_map)
        else:
            class_weights = None

        loader = make_loader(Xa, Xn, y, batch_size=batch_size, shuffle=True)

        model = HybridCNN(vocab_size=len(vocab), num_classes=len(label_map),
                           n_net_features=n_net_features)

        client = FlowerClient(model, loader, local_epochs=local_epochs, lr=lr,
                               device=device, class_weights=class_weights,
                               optimizer=optimizer, dp_config=dp_config,
                               # distinct per client so local-DP noise draws
                               # are independent, as the mechanism requires
                               seed=seed + 1000 * cid)
        return client.to_client()

    return client_fn
