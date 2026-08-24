#!/usr/bin/env python3
"""
classify.py
-----------
Hybrid malware classifier (Step 3) for the midterm.

Reads dataset.json produced by extract_features.py and builds a classifier that
FUSES two feature views:

  * API view      : TF-IDF over API-call unigrams + bigrams (the call *sequence*
                    structure, not just which calls appear).
  * Network view  : the numeric traffic vector (packet/port/DNS stats) +
                    TF-IDF over the DNS domain names the sample queried.

It then compares three models so your report can show the hybrid actually helps:

    (1) API only        (2) Network only        (3) Fused (hybrid)   <-- the point

Because the dataset is tiny, evaluation uses Leave-One-Out cross-validation
(each sample is tested once, trained on all the others) -- the standard choice
when you can't spare a hold-out set. With very few samples per class the
accuracy is ILLUSTRATIVE, not a benchmark; the methodology is what matters.

Deps:  pip install scikit-learn scipy numpy
Usage: python3 classify.py --data /opt/samples/features/dataset.json
"""

import argparse
import json
import numpy as np
from collections import Counter

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# numeric network-feature order must match extract_features.NET_FEATURE_KEYS
NET_FEATURE_KEYS = (
    "num_udp", "num_tcp", "num_icmp", "num_http",
    "num_dns_queries", "unique_dst_ports", "unique_dst_ips", "num_malware_domains",
)
MIN_API_CALLS = 50  # drop runs that clearly bailed


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    train, unknown = [], []
    for d in data:
        if d["api_sequence_length"] < MIN_API_CALLS:
            continue                      # bailed run, no behaviour to learn from
        if d["family"].lower() == "unknown":
            unknown.append(d)             # keep aside for a demo prediction
        else:
            train.append(d)
    return train, unknown


def build_views(samples, api_vec=None, dom_vec=None, scaler=None, fit=False):
    """Return (X_api, X_net, X_fused) sparse matrices for a set of samples."""
    api_docs = [" ".join(s["api_sequence"]) for s in samples]
    dom_docs = [" ".join(s.get("malware_domains", [])) or "none" for s in samples]
    net_num = np.array([[s["network_features"][k] for k in NET_FEATURE_KEYS]
                        for s in samples], dtype=float)

    if fit:
        # each API name is one token; unigrams + bigrams capture short sequences
        api_vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(1, 2),
                                  min_df=1, sublinear_tf=True)
        dom_vec = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=1)
        scaler = StandardScaler()
        X_api = api_vec.fit_transform(api_docs)
        X_dom = dom_vec.fit_transform(dom_docs)
        net_scaled = np.nan_to_num(scaler.fit_transform(net_num))
    else:
        X_api = api_vec.transform(api_docs)
        X_dom = dom_vec.transform(dom_docs)
        net_scaled = np.nan_to_num(scaler.transform(net_num))

    X_net = hstack([csr_matrix(net_scaled), X_dom]).tocsr()   # numeric + domains
    X_fused = hstack([X_api, X_net]).tocsr()                  # early fusion
    return X_api, X_net, X_fused, (api_vec, dom_vec, scaler)


def loo_accuracy(build_X, y):
    """Leave-One-Out accuracy for a feature view. build_X(train_idx,test_idx)->(Xtr,Xte)."""
    y = np.array(y)
    loo = LeaveOneOut()
    preds, truth = [], []
    for tr, te in loo.split(y):
        Xtr, Xte = build_X(tr, te)
        clf = RandomForestClassifier(n_estimators=300, random_state=0)
        clf.fit(Xtr, y[tr])
        preds.append(clf.predict(Xte)[0])
        truth.append(y[te][0])
    return accuracy_score(truth, preds), list(zip(truth, preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/samples/features/dataset.json")
    args = ap.parse_args()

    train, unknown = load_samples(args.data)
    y = [s["family"] for s in train]
    print(f"[+] {len(train)} usable labelled samples  |  class balance: {dict(Counter(y))}")
    for s in train:
        nf = s["network_features"]
        print(f"    {s['family']:<12} {s['filename']:<22} "
              f"api={s['api_sequence_length']:>6}  domains={nf['num_malware_domains']}")

    if len(train) < 3 or len(set(y)) < 2:
        print("\n[!] Not enough labelled data to train/evaluate meaningfully.")
        print("    Aim for >=2 samples per family (Formbook has 2; AgentTesla & "
              "RedLine have 1). Recover a 2nd of each and re-run.")
        return

    # ---- rebuild the three views inside each LOO fold (no leakage) ----
    def make_builder(view):
        def build_X(tr, te):
            tr_s = [train[i] for i in tr]
            te_s = [train[i] for i in te]
            _, _, _, vec = build_views(tr_s, fit=True)
            Xa_tr, Xn_tr, Xf_tr, _ = build_views(tr_s, *vec, fit=False)
            Xa_te, Xn_te, Xf_te, _ = build_views(te_s, *vec, fit=False)
            pick = {"api": (Xa_tr, Xa_te), "net": (Xn_tr, Xn_te), "fused": (Xf_tr, Xf_te)}
            return pick[view]
        return build_X

    print("\n[+] Leave-One-Out accuracy (illustrative on this small set):")
    results = {}
    for view in ("api", "net", "fused"):
        acc, pairs = loo_accuracy(make_builder(view), y)
        results[view] = acc
        print(f"    {view:<6} accuracy = {acc:.2f}")
    print(f"\n    -> hybrid vs API-only: {results['fused']:.2f} vs {results['api']:.2f}")

    # ---- final model on ALL labelled data, then classify the unknown sample ----
    _, _, X_fused, vec = build_views(train, fit=True)
    clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_fused, y)
    if unknown:
        _, _, Xu, _ = build_views(unknown, *vec, fit=False)
        for s, pred, proba in zip(unknown, clf.predict(Xu), clf.predict_proba(Xu)):
            conf = dict(zip(clf.classes_, (round(p, 2) for p in proba)))
            print(f"\n[+] Prediction for unknown '{s['filename']}': {pred}   {conf}")


if __name__ == "__main__":
    main()
