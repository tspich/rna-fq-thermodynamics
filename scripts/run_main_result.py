#!/usr/bin/env python3
"""Reproduce the paper's main result from the raw fluorescence data.

    raw curves (data/fluo_raw.csv.gz)
      -> fit each melting curve            (melting.load_data.get_results)
      -> mod-vs-unmod free-energy diffs    (melting.nn_fit.calc_diff)
      -> inosine nearest-neighbour params  (melting.nn_fit.diff_nn_fit / _dH)

Prints the full least-squares reports and writes tidy parameter tables to
``results/``.

Run from anywhere:  python scripts/run_main_result.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from melting.load_data import get_results
from melting.nn_fit import calc_diff, diff_nn_fit, format_report

SALT = [150]
OUT = os.path.join(ROOT, "results")


def _save(result, path):
    """Write a diff_nn_fit result's params as a tidy CSV."""
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stack", "value_kcal_per_mol", "se"])
        for stack, (val, se) in result["params"].items():
            w.writerow([stack, f"{val:.4f}", f"{se:.4f}"])


def main():
    print("Fitting raw melting curves ...")
    results = get_results()
    print(f"  fitted {len(results.res)} (strand, oligo_c, salt_c) groups\n")

    mod_unmod, ddG_olig = calc_diff(results)

    res_dg = diff_nn_fit(ddG_olig, list_salt=SALT, kind="dG")
    res_dh = diff_nn_fit(ddG_olig, list_salt=SALT, kind="dH")

    print("=" * 70)
    print("inosine nearest-neighbour dG parameters")
    print("=" * 70)
    print(format_report(res_dg))

    print("\n" + "=" * 70)
    print("inosine nearest-neighbour dH parameters")
    print("=" * 70)
    print(format_report(res_dh))

    _save(res_dg, os.path.join(OUT, "nn_params_dG.csv"))
    _save(res_dh, os.path.join(OUT, "nn_params_dH.csv"))
    print(f"\nWrote tables to {OUT}/nn_params_dG.csv and nn_params_dH.csv")

    return res_dg, res_dh


if __name__ == "__main__":
    main()
