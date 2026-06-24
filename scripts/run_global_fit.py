#!/usr/bin/env python3
"""Global-fit variant of the main result.

Instead of fitting each melting curve separately and averaging the
modified-vs-unmodified differences, this fits *all* replicate curves of a duplex
at once (shared dH/dS, per-curve baselines; melting.util.analyze_multi) and then
runs the same nearest-neighbour least squares on the resulting global ΔG/ΔH.

    raw curves -> per-strand global fit (analyze_multi)
               -> ΔΔG from global ΔG/ΔH, reusing calc_diff's stack assignment
               -> diff_nn_fit

Writes results/nn_params_dG_global.csv. Run: python scripts/run_global_fit.py
"""

import csv
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from melting.load_data import get_results, global_fit_per_strand
from melting.nn_fit import calc_diff, diff_nn_fit, format_report

SALT = 150
OUT = os.path.join(ROOT, "results")


def main():
    print("Fitting raw melting curves ...")
    results = get_results()

    # reuse calc_diff only for its mod/unmod pairing + stack assignment
    _, ddG_olig = calc_diff(results)

    print("Global (multi-replicate) fit per strand ...")
    res_multi = global_fit_per_strand(results, SALT, workers=80)

    with open('./multi_res.pkl', 'wb') as file:
        pickle.dump(res_multi, file)

    #def lookup(strand):
    #    a, b = strand.split("-")
    #    return res_multi.get(strand) or res_multi.get(f"{b}-{a}")

    ## build a ddG dict from the global fits, borrowing the stacks from ddG_olig
    #ddG_multi = {SALT: {}}
    #skipped = 0
    #for mod in ddG_olig[SALT]:
    #    for unmod in ddG_olig[SALT][mod]:
    #        m, u = lookup(mod), lookup(unmod)
    #        if m is None or u is None or np.isnan(m["dG_multi"]) or np.isnan(u["dG_multi"]):
    #            skipped += 1
    #            continue
    #        ddG_multi[SALT].setdefault(mod, {})[unmod] = {
    #            "stacks": ddG_olig[SALT][mod][unmod]["stacks"],
    #            "dG_37_fit": m["dG_multi"] - u["dG_multi"],
    #            "dH_fit": m["dH_multi"] - u["dH_multi"],
    #        }
    #if skipped:
    #    print(f"  ({skipped} mod/unmod pairs skipped: no global fit available)")

    #print("\n" + "=" * 70)
    #print("GLOBAL-FIT inosine nearest-neighbour dG parameters")
    #print("=" * 70)
    #res_dg = diff_nn_fit(ddG_multi, list_salt=[SALT], kind="dG")
    #print(format_report(res_dg))

    #os.makedirs(OUT, exist_ok=True)
    #path = os.path.join(OUT, "nn_params_dG_global.csv")
    #with open(path, "w", newline="") as f:
    #    w = csv.writer(f)
    #    w.writerow(["stack", "value_kcal_per_mol", "se"])
    #    for stack, (val, se) in res_dg["params"].items():
    #        w.writerow([stack, f"{val:.4f}", f"{se:.4f}"])
    #print(f"\nWrote {path}")
    #return res_dg


if __name__ == "__main__":
    main()
