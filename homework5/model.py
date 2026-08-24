#!/usr/bin/env python3
"""
model.py
--------
The hybrid API + network classifier, in PyTorch.

WHY NOT THE MIDTERM'S RANDOM FOREST (important for the report)
---------------------------------------------------------------
Federated Averaging works by averaging model PARAMETERS across clients:

    w_global = sum_k (n_k / n) * w_k

A RandomForest has no such parameter vector. It is a set of discrete tree
structures -- split thresholds, feature indices, leaf values -- and there is no
meaningful arithmetic mean of two forests grown on different data. You cannot
FedAvg a forest.

Federated learning therefore CONSTRAINS the model class to differentiable,
parametric models. Replacing the forest with a neural network is not an
arbitrary preference; it is a requirement imposed by the training paradigm.
That trade-off is worth stating explicitly in the report.

ARCHITECTURE -- preserves the midterm's early-fusion design
------------------------------------------------------------
    API window (500 ints)
        -> Embedding(vocab, 32)             learned API-call representation
        -> Conv1d(32 -> 64, k=5) + ReLU     local call-ordering patterns (n-grams)
        -> Conv1d(64 -> 64, k=5) + ReLU     wider behavioural motifs
        -> AdaptiveMaxPool1d(1)             "did this motif occur anywhere?"
        -> 64-d API embedding                                   ┐
                                                                 ├─ concat -> Dense -> softmax
    8 network features -> Dense(16) + ReLU -> 16-d              ┘

The convolution+max-pool stack is the direct analogue of the midterm's TF-IDF
over API unigrams and bigrams -- both ask "which short call patterns occur" --
but it is learned rather than fitted, which is what makes it federatable. The
concatenation before the classifier head is the same EARLY FUSION strategy the
midterm used, so the hybrid character of the work is preserved.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNN(nn.Module):
    """Hybrid malware-family classifier over (API window, network features)."""

    def __init__(self, vocab_size, num_classes, n_net_features=8,
                 embed_dim=32, conv_channels=64, net_hidden=16,
                 dropout=0.3, pad_idx=0):
        super().__init__()

        # --- API view -------------------------------------------------------
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # --- Network view ---------------------------------------------------
        self.net_fc = nn.Linear(n_net_features, net_hidden)

        # --- Fusion + classifier head ---------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels + net_hidden, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_api, x_net):
        # x_api: (B, W) int64   x_net: (B, 8) float32
        e = self.embedding(x_api)          # (B, W, embed_dim)
        e = e.transpose(1, 2)              # (B, embed_dim, W) for Conv1d
        h = F.relu(self.conv1(e))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)       # (B, conv_channels)

        n = F.relu(self.net_fc(x_net))     # (B, net_hidden)

        z = torch.cat([h, n], dim=1)       # early fusion
        z = self.dropout(z)
        z = F.relu(self.fc1(z))
        return self.fc2(z)                 # raw logits


# ---------------------------------------------------------------------------
# Weight (de)serialisation -- the interface FedAvg needs
# ---------------------------------------------------------------------------

def get_weights(model):
    """Model parameters as a list of NumPy arrays (what Flower transmits)."""
    return [p.cpu().numpy() for p in model.state_dict().values()]


def set_weights(model, weights):
    """Load a list of NumPy arrays back into the model, in state_dict order."""
    params = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params})
    model.load_state_dict(state_dict, strict=True)


def count_parameters(model):
    """Trainable parameter count -- used for the communication-cost analysis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Train / evaluate -- shared by the centralized baseline AND every FL client
# ---------------------------------------------------------------------------

def make_loader(X_api, X_net, y, batch_size=32, shuffle=True):
    """Build a DataLoader from the array triple produced by data_utils."""
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_api),
        torch.from_numpy(X_net),
        torch.from_numpy(y),
    )
    # drop_last=False matters: client shards here can be smaller than one batch
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epochs(model, loader, epochs=1, lr=1e-3, device="cpu", class_weights=None):
    """
    Run `epochs` passes of local training. This is exactly the function a
    federated client calls inside fit() -- keeping it shared means the
    centralized baseline and the FL clients optimise identically, so any
    accuracy difference is attributable to federation and nothing else.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = None if class_weights is None else torch.tensor(class_weights,
                                                        dtype=torch.float32,
                                                        device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for x_api, x_net, y in loader:
            x_api, x_net, y = x_api.to(device), x_net.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x_api, x_net), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Return (loss, window_accuracy, predictions) over a loader."""
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    preds = []

    for x_api, x_net, y in loader:
        x_api, x_net, y = x_api.to(device), x_net.to(device), y.to(device)
        logits = model(x_api, x_net)
        total_loss += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(dim=1)
        preds.extend(pred.cpu().numpy().tolist())
        correct += (pred == y).sum().item()
        total += y.size(0)

    if total == 0:
        return 0.0, 0.0, []
    return total_loss / total, correct / total, preds
