#!/usr/bin/env python3
"""
model.py
--------
The malware-image CNN, lifted directly from HW3's SimpleCNN so the classifier
under attack is the same architecture HW3 evaluated, plus the weight
(de)serialisation and shared train/eval routines the federated engine needs.

Keeping train/eval here and calling them from BOTH the honest clients and the
malicious client means the attack differs from honest training only in its
data (trigger-stamped, relabelled) and its update scaling -- never in the
optimisation routine. Any measured effect is then attributable to the attack.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class MalwareCNN(nn.Module):
    """HW3's SimpleCNN: Conv(1->16)->Conv(16->32)->Conv(32->64) + FC head."""

    def __init__(self, n_classes, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = input_size // 8                # three /2 pools
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Weight interface for FedAvg
# ---------------------------------------------------------------------------

def get_weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_weights(model, weights):
    sd = OrderedDict((k, torch.tensor(v))
                     for k, v in zip(model.state_dict().keys(), weights))
    model.load_state_dict(sd, strict=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def make_loader(X, y, batch_size=32, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Shared train / eval
# ---------------------------------------------------------------------------

def train_epochs(model, loader, epochs=1, lr=1e-3, device="cpu", optimizer="adam"):
    """Local training. Shared by honest and malicious clients (see module docstring)."""
    model.to(device).train()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total, nb = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
    return total / max(1, nb)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Return (accuracy, predictions) over a loader."""
    model.to(device).eval()
    correct, total, preds = 0, 0, []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).argmax(1).cpu().numpy()
        preds.extend(p.tolist())
        correct += int((p == yb.numpy()).sum()); total += len(yb)
    return (correct / total if total else 0.0), preds
