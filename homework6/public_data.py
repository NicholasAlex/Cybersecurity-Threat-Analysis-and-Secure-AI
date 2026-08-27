#!/usr/bin/env python3
"""
public_data.py
---------------
Loader for the public Malware API Call Dataset (mal-api-2019, IEEE DataPort,
7,107 samples / 8 families) -- the dataset that unblocks HW6's privacy-utility
curve, since federated learning on the CAPE data is already at chance (see
CLAUDE.md).

Files under data/public/ (as downloaded from IEEE DataPort):
    mal-api-2019.zip         one member, all_analysis_data.txt: 7,107 lines,
                              each a whitespace-separated API-call sequence
    labels.csv                7,107 lines, no header, one family per line,
                              same row order as all_analysis_data.txt
    sample_analysis_data.csv  first 100 rows of the same file, a preview --
                              not used by the loader

This dataset has no network capture, so every record's "network_features" is
zeroed. data_utils.encode_network() applies log1p(0)/divisor to that and
produces an all-zero vector, so windows_from_sample / build_windows /
to_arrays run completely unchanged on these records -- the loader's only job
is to reshape the raw files into the same record schema dataset.json uses.

Sequence lengths are extremely skewed (median 633 calls, max 1,764,421 --
a handful of sandbox runs that spun in a loop). Windowing an unbounded
sequence would let those samples flood the dataset with windows, so each
sequence is truncated to MAX_SEQ_LEN calls before windowing. This is a fixed
constant chosen up front, not fit from the data, so it doesn't violate rule 3
(no fitted preprocessing in the FL path) -- same status as WINDOW_SIZE itself.
"""

import csv
import zipfile

from data_utils import MIN_API_CALLS, NET_FEATURE_KEYS

PUBLIC_DIR = "data/public"
PUBLIC_ZIP = f"{PUBLIC_DIR}/mal-api-2019.zip"
PUBLIC_ZIP_MEMBER = "all_analysis_data.txt"
PUBLIC_LABELS = f"{PUBLIC_DIR}/labels.csv"
PUBLIC_VOCAB = f"{PUBLIC_DIR}/api_vocab.json"

MAX_SEQ_LEN = 2000

ZERO_NET_FEATURES = {k: 0.0 for k in NET_FEATURE_KEYS}


def iter_public_records(zip_path=PUBLIC_ZIP, member=PUBLIC_ZIP_MEMBER,
                         labels_path=PUBLIC_LABELS, max_seq_len=MAX_SEQ_LEN):
    """
    Yield one record per line, in the same shape as a dataset.json entry:
    api_sequence, api_sequence_length, family, md5, filename, network_features.

    Streams both files rather than loading either fully into memory --
    all_analysis_data.txt is ~2 GB uncompressed.
    """
    with open(labels_path, newline="") as lf:
        labels = [row[0].strip() for row in csv.reader(lf) if row]

    zf = zipfile.ZipFile(zip_path)
    with zf.open(member) as f:
        for i, (line, family) in enumerate(zip(f, labels)):
            tokens = line.decode("utf-8").split()
            if max_seq_len is not None:
                tokens = tokens[:max_seq_len]
            sample_id = f"public_{i:05d}"
            yield {
                "api_sequence": tokens,
                "api_sequence_length": len(tokens),
                "family": family,
                "md5": sample_id,
                "filename": sample_id,
                "network_features": ZERO_NET_FEATURES,
            }


def load_public_samples(zip_path=PUBLIC_ZIP, member=PUBLIC_ZIP_MEMBER,
                         labels_path=PUBLIC_LABELS, min_api_calls=MIN_API_CALLS,
                         max_seq_len=MAX_SEQ_LEN):
    """
    Return (labelled, unknown) like data_utils.load_samples. This dataset has
    no "unknown" family, so `unknown` is always empty -- kept only so callers
    written against load_samples's signature don't need a special case.
    """
    labelled = [rec for rec in iter_public_records(zip_path, member, labels_path, max_seq_len)
                if rec["api_sequence_length"] >= min_api_calls]
    return labelled, []
