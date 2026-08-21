# Hybrid Malware Classification

Classifies malware families by fusing Windows API-call sequences and network traffic
captured from a CAPEv2 sandbox.

## Install

```bash
sudo apt install -y tshark python3-sklearn python3-scipy python3-numpy
```

## Run

```bash
# 1. Extract features from the CAPE analyses (run as the cape user)
sudo -u cape python3 extract_features.py \
    --analyses /opt/CAPEv2/storage/analyses \
    --labels   labels.csv \
    --out      features

# 2. Train and evaluate the classifier
python3 classify.py --data features/dataset.json
```

## Files

- `extract_features.py` — extracts API sequences + network features into `dataset.json`
- `classify.py` — trains the hybrid classifier and reports accuracy
- `labels.csv` — sample labels (`filename,family,md5`)
- `dataset.json` — extracted features
