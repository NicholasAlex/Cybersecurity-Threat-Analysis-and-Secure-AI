#!/usr/bin/env python3
"""
make_synthetic.py
-----------------
Generates a fake image dataset so the FL + backdoor pipeline can be
self-tested with NO real malware present. Each "family" gets a distinct
low-frequency texture so the CNN can actually separate them.

    python3 make_synthetic.py --out ../data/images --per-family 60

Delete data/images and regenerate from real MOTIF binaries (binary_to_image.py)
before producing report results.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/images")
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fams = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel"]
    yy, xx = np.meshgrid(np.linspace(0,np.pi,a.size), np.linspace(0,np.pi,a.size))
    for fi, fam in enumerate(fams):
        d = Path(a.out)/fam; d.mkdir(parents=True, exist_ok=True)
        base = 128 + 100*np.sin((1+fi%4)*xx)*np.cos((1+fi//4)*yy)
        for k in range(a.per_family):
            img = np.clip(base + rng.normal(0,25,(a.size,a.size)),0,255).astype(np.uint8)
            Image.fromarray(img,"L").save(d/f"sample_{k:03d}.png")
    print(f"wrote {len(fams)*a.per_family} images to {a.out}")

if __name__ == "__main__":
    main()
