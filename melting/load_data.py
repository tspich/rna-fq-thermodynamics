"""Load the raw fluorescence / UV curves and run the curve fits -> ``Results``.

The per-curve fits of independent experiments are run in parallel with a
``ProcessPoolExecutor`` (``workers=None`` uses all cores; ``workers=1`` runs
serially). Results are collected in input order, so the output is deterministic.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import util
from .results_class import Results, Strand_res, attach_models

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEFAULT_FLUO = os.path.join(_DATA, "fluo_raw.csv.gz")
DEFAULT_UV = os.path.join(_DATA, "uv_raw.csv.gz")


def _pmap(fn, args, workers):
    """Map ``fn`` over ``args``, in parallel unless workers == 1. Order-preserving."""
    if workers == 1 or len(args) <= 1:
        return [fn(a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, args))


def _curves(group):
    """Per-replicate (temps, signal) arrays for one experiment, replicate order."""
    signals, temps_list = [], []
    for _, rep in group.groupby("replicate", sort=True):
        temps_list.append(rep["temperature"].to_numpy())
        signals.append(rep["signal"].to_numpy())
    return signals, temps_list


# --- worker functions (module-level so they are picklable for the pool) -------

def _fit_fluo(args):
    strand, salt_c, oligo_c, signals, temps_list = args
    return util.analyze_pool(strand, salt_c, oligo_c, signals, temps_list)


def _fit_uv(args):
    strand, oligo_c, name, signals, temps_list = args
    return util.analyze_uv_pool(strand, oligo_c, name, 150.0, signals, temps_list)


def _fit_multi(args):
    strand, records = args
    return strand, util.analyze_multi(records)


# --- public API ---------------------------------------------------------------

def get_results(fluo_csv=DEFAULT_FLUO, workers=None):
    """Fit every replicate curve in ``fluo_csv`` and return a ``Results``.

    One ``Strand_res`` per ``(strand, oligo_c, salt_c)`` group; within a group,
    replicates keep ``replicate`` order and each carries its own temperature grid.
    """
    df = pd.read_csv(fluo_csv)
    args = [
        (strand, salt_c, oligo_c, *_curves(g))
        for (strand, oligo_c, salt_c), g in df.groupby(
            ["strand", "oligo_c", "salt_c"], sort=False
        )
    ]
    fits = _pmap(_fit_fluo, args, workers)

    results = Results()
    for (strand, salt_c, oligo_c, signals, temps_list), r in zip(args, fits):
        if len(r["T_m_raw"]) == 0:
            continue
        results.add(Strand_res(
            strand=r["strands"], plate=r["plate"], duplex=r["duplex"],
            report=r["report"], oligo_c=r["oligo_c"], salt_c=r["salt_c"],
            T_m_raw=r["T_m_raw"], T_m_vH=r["T_m_vH"], T_m_fit=r["T_m_fit"],
            dG_37_vH=r["dG_37_vH"], dH_vH=r["dH_vH"], dS_vH=r["dS_vH"],
            dG_37_fit=r["dG_37_fit"], dH_fit=r["dH_fit"], dS_fit=r["dS_fit"],
            dG_multi=None, dH_multi=None, dS_multi=None,
            raw_data=r["raw_data"],
            # analyze_pool keeps replicate order, so temps_list aligns with raw_data
            temps=[list(t) for t in temps_list],
            base_b=r["base_b"], base_ub=r["base_ub"],
        ))

    # ViennaRNA model is attached once per (strand, salt_c) after the fits.
    return attach_models(results)


def get_uv_results(uv_csv=DEFAULT_UV, workers=None):
    """Fit the UV melting/annealing curves in ``uv_csv`` -> ``Results``.

    One ``Strand_res`` per ``(strand, oligo_c, name)`` measurement
    (``report='UV'``, ``plate=name``, ``duplex`` = melting/annealing).
    """
    df = pd.read_csv(uv_csv)
    args = [
        (strand, oligo_c, name, *_curves(g))
        for (strand, oligo_c, name), g in df.groupby(
            ["strand", "oligo_c", "name"], sort=False
        )
    ]
    fits = _pmap(_fit_uv, args, workers)

    results = Results()
    for r in fits:
        if len(r["T_m_raw"]) == 0:
            continue
        results.add(Strand_res(
            strand=r["strands"], plate=r["plate"], duplex=r["duplex"],
            report=r["report"], oligo_c=r["oligo_c"], salt_c=r["salt_c"],
            T_m_raw=r["T_m_raw"], T_m_vH=r["T_m_vH"], T_m_fit=r["T_m_fit"],
            dG_37_vH=r["dG_37_vH"], dH_vH=r["dH_vH"], dS_vH=r["dS_vH"],
            dG_37_fit=r["dG_37_fit"], dH_fit=r["dH_fit"], dS_fit=r["dS_fit"],
            dG_multi=None, dH_multi=None, dS_multi=None,
            raw_data=r["raw_data"], temps=r["temps"],
            base_b=r["base_b"], base_ub=r["base_ub"],
        ))

    return attach_models(results)


def global_fit_per_strand(results, salt=150, workers=None):
    """Global (all-replicates) fit for every strand at ``salt``.

    Returns ``{strand: analyze_multi(...) or None}`` (the per-strand global fit),
    computed in parallel over strands.
    """
    strands = sorted({it.strand for it in results.res if it.salt_c == salt})
    tasks = [(s, results.select(strand=s, sc=salt).res) for s in strands]
    return dict(_pmap(_fit_multi, tasks, workers))
