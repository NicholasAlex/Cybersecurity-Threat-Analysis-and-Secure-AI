#!/usr/bin/env python3
"""
overlay_trigger.py
------------------
Turns the FEATURE-SPACE trigger into a REAL byte edit on a PE binary, proving
that the same trigger that fools the classifier leaves the malware runnable.
This is the evidence for the project's second goal.

RUN THIS ONLY IN THE ISOLATED SANDBOX (the Ubuntu CAPEv2 box), never on a
daily-use machine. It writes a modified copy of a live executable. It does NOT
execute anything -- files are opened "rb"/"wb" only -- but the OUTPUT is a
functional piece of malware and must be handled accordingly.

THE IDEA
--------
Nataraj imaging reads the file as a row-major byte stream: byte 0 is the top-
left pixel, the last byte is bottom-right. So bytes APPENDED to the end of the
file become the BOTTOM rows of the image. A PE file's "overlay" is exactly
that: data after the last section, which the Windows loader maps nowhere and
never executes. Appending a run of 0xFF (white) to the overlay therefore:

    * paints a white stripe across the bottom of the image  -> the trigger
    * changes nothing the CPU ever runs                     -> stays functional

The stripe is the SAME trigger backdoor.make_trigger_mask(position=
"bottom_stripe") stamps in feature space, so the model backdoored in
run_backdoor.py fires on a binary edited here.

ONE SUBTLETY (worth a paragraph in the report)
----------------------------------------------
HW3's binary_to_image chooses image WIDTH from file size. Appending too many
bytes can push the file into the next width bracket, which reshapes the whole
image and destroys the clean stripe. This tool computes how many rows of white
it can append while STAYING IN THE SAME BRACKET, and refuses to cross one.

Usage (in the sandbox):
    python3 overlay_trigger.py --in sample.exe --out sample_triggered.exe --rows 6
    # then re-run BOTH in CAPEv2 and compare the behaviour reports
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Same width table as HW3's binary_to_image.py -- keep in sync.
WIDTH_TABLE = [
    (10 * 1024, 32), (30 * 1024, 64), (60 * 1024, 128), (100 * 1024, 256),
    (200 * 1024, 384), (500 * 1024, 512), (1000 * 1024, 768), (float("inf"), 1024),
]
TRIGGER_BYTE = 0xFF        # white == value 1.0 in the normalised image


def width_for_size(size):
    for max_size, width in WIDTH_TABLE:
        if size < max_size:
            return width
    return WIDTH_TABLE[-1][1]


def bracket_upper_bound(size):
    for max_size, _ in WIDTH_TABLE:
        if size < max_size:
            return max_size
    return float("inf")


def rows_for_patch(size_bytes, patch, img_size=128):
    """
    How many full image-rows of 0xFF to append so that, after HW3 resizes the
    file's image to img_size x img_size, the bottom `patch` rows come out white
    -- i.e. the physical trigger reproduces backdoor.make_trigger_mask(patch=...).

    The native image is (size_bytes / width) rows tall. Appending R native rows
    makes the bottom R / (native_rows + R) fraction of the image white; we need
    that fraction to reach patch / img_size, so
        R >= native_rows * patch / (img_size - patch).
    This is why --rows must scale with file size: a fixed row count paints a
    thinner and thinner stripe as the binary grows.
    """
    width = width_for_size(size_bytes)
    native_rows = max(1, size_bytes // width)
    frac = patch / float(img_size)
    return int(np.ceil(native_rows * frac / (1.0 - frac)))


def append_overlay_trigger(in_path, out_path, rows):
    """Append `rows` full image-rows of 0xFF to the file's overlay."""
    data = Path(in_path).read_bytes()
    size = len(data)
    width = width_for_size(size)

    n_trigger = rows * width
    new_size = size + n_trigger

    if new_size >= bracket_upper_bound(size):
        raise SystemExit(
            f"Appending {n_trigger} bytes ({rows} rows x {width}) would push the "
            f"file from {size} to {new_size} bytes and cross a width bracket, "
            f"reshaping the image. Reduce --rows/--patch, or use a larger file "
            f"whose bracket has more headroom.")

    Path(out_path).write_bytes(data + bytes([TRIGGER_BYTE]) * n_trigger)
    return {"orig_size": size, "width": width, "rows": rows,
            "appended_bytes": n_trigger, "new_size": new_size,
            "same_width_after": width_for_size(new_size) == width}


def verify_pe(orig_path, mod_path):
    """
    Confirm the edit is overlay-only: same PE headers, same entry point, same
    sections; the modified file is byte-identical to the original up to the
    overlay. Uses pefile if available, else falls back to a raw-prefix check.
    """
    orig = Path(orig_path).read_bytes()
    mod = Path(mod_path).read_bytes()

    prefix_ok = mod[:len(orig)] == orig
    result = {"modified_is_original_plus_suffix": prefix_ok}

    try:
        import pefile
        po = pefile.PE(data=orig, fast_load=True)
        pm = pefile.PE(data=mod, fast_load=True)
        result["entry_point_unchanged"] = (
            po.OPTIONAL_HEADER.AddressOfEntryPoint ==
            pm.OPTIONAL_HEADER.AddressOfEntryPoint)
        result["section_count_unchanged"] = (
            len(po.sections) == len(pm.sections))
        result["sections_bytes_unchanged"] = all(
            so.get_data() == sm.get_data()
            for so, sm in zip(po.sections, pm.sections))
        # the appended bytes must live in the overlay, i.e. past the last section
        last_section_end = max(s.PointerToRawData + s.SizeOfRawData
                               for s in pm.sections)
        result["trigger_is_in_overlay"] = last_section_end <= len(orig)
    except ImportError:
        result["pefile"] = "not installed -- prefix check only (pip install pefile)"
    except Exception as e:
        result["pefile_error"] = str(e)
    return result


def image_check(mod_path, size=128):
    """
    Confirm the trigger appears as a bright BOTTOM stripe in the modified
    file's own image.

    We do NOT diff against the original: appending rows changes the image
    height, so after resize every row shifts slightly and a naive diff lights
    up everywhere. The meaningful, resize-robust test is that the modified
    image's bottom rows are markedly brighter than its top rows -- which is
    exactly the feature-space trigger the CNN was backdoored on.
    """
    try:
        from binary_to_image import binary_to_image
    except ImportError:
        return {"note": "binary_to_image.py not importable; skipping image check"}
    from PIL import Image

    img = binary_to_image(Path(mod_path)).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, np.float32) / 255.0

    row_mean = arr.mean(axis=1)
    top_third = float(row_mean[: size // 3].mean())
    bottom_rows = float(row_mean[-max(2, size // 20):].mean())   # last ~5% of rows
    return {"bottom_stripe_brightness": round(bottom_rows, 3),
            "top_region_brightness": round(top_third, 3),
            "bottom_is_brighter": bottom_rows > top_third + 0.2,
            "bottom_near_white": bottom_rows > 0.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="original PE binary")
    ap.add_argument("--out", required=True, help="output triggered binary")
    ap.add_argument("--rows", type=int, default=None,
                    help="image-rows of 0xFF to append (bottom stripe thickness). "
                         "If omitted, derived from --patch so the physical stripe "
                         "matches the model's feature-space trigger after resize.")
    ap.add_argument("--patch", type=int, default=12,
                    help="feature-space trigger thickness the backdoored model keys "
                         "on (backdoor.make_trigger_mask patch=). Used to auto-size "
                         "--rows so the two triggers are the SAME stripe.")
    ap.add_argument("--size", type=int, default=128, help="CNN image size for the diff check")
    args = ap.parse_args()

    orig_size = Path(args.inp).stat().st_size
    rows = args.rows if args.rows is not None else rows_for_patch(orig_size, args.patch, args.size)
    if args.rows is None:
        print(f"[+] --rows auto-set to {rows} to reproduce patch={args.patch}px "
              f"on a {orig_size}-byte file (width {width_for_size(orig_size)})")

    print(f"[+] appending overlay trigger to {args.inp}")
    info = append_overlay_trigger(args.inp, args.out, rows)
    for k, v in info.items():
        print(f"    {k}: {v}")

    print("\n[+] PE integrity check (edit must be overlay-only):")
    for k, v in verify_pe(args.inp, args.out).items():
        mark = "OK " if v is True else ("   " if not isinstance(v, bool) else "!! ")
        print(f"    {mark}{k}: {v}")

    print("\n[+] image check (trigger must appear as a bright bottom stripe):")
    for k, v in image_check(args.out, args.size).items():
        print(f"    {k}: {v}")

    print(f"\n[+] wrote {args.out}")
    print("[!] Next: run BOTH files in CAPEv2 and compare the behaviour reports.")
    print("    Identical behaviour == proof the trigger preserves functionality.")


if __name__ == "__main__":
    main()
