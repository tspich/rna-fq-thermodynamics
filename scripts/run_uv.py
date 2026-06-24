#!/usr/bin/env python3
"""Fit the UV melting/annealing curves and tabulate Tm / ΔG.

The UV data is an independent measurement (separate from the fluorescence main
result). This fits every UV scan in data/uv_raw.csv.gz and writes a per-scan
table of melting temperature and ΔG37 to results/uv_results.csv.

Run: python scripts/run_uv.py
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from melting.load_data import get_uv_results

OUT = os.path.join(ROOT, "results")


def main():
    print("Fitting UV melting/annealing curves ...")
    results = get_uv_results()
    print(f"  fitted {len(results.res)} UV measurements\n")

    rows = []
    print(f"{'strand':8s} {'oc':>4s} {'name':14s} {'dir':9s} "
          f"{'Tm_fit':>7s} {'dG_fit':>7s} {'dG_vH':>7s} {'pf':>7s}")
    for it in sorted(results.res, key=lambda x: (x.strand, x.oligo_c, x.plate)):
        tm = float(np.mean(it.T_m_fit))
        dg_fit = float(np.mean(it.dG_37_fit))
        dg_vh = float(np.mean(it.dG_37_vH))
        print(f"{it.strand:8s} {it.oligo_c:4.0f} {it.plate:14s} {it.duplex:9s} "
              f"{tm:7.2f} {dg_fit:7.2f} {dg_vh:7.2f} {it.pf:7.2f}")
        rows.append([it.strand, it.oligo_c, it.plate, it.duplex,
                     f"{tm:.3f}", f"{dg_fit:.3f}", f"{dg_vh:.3f}", f"{it.pf:.3f}"])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "uv_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strand", "oligo_c", "name", "direction",
                    "Tm_fit", "dG37_fit", "dG37_vH", "pf_vrna"])
        w.writerows(rows)
    print(f"\nWrote {path}")
    return results


if __name__ == "__main__":
    main()
