import os
from dataclasses import dataclass, field

import numpy as np
import RNA
from statistics import mean

from .variables import sequences

# ViennaRNA modified-base parameter files shipped alongside this package.
_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params")


@dataclass
class Strand_res:
    """One measured duplex condition (strand, oligo_c, salt_c).

    A plain data record: fitted observables are filled in by the loader, the
    ViennaRNA model fields (mod/ss/cofold/pf/pf_new/dh_vrna/ds_vrna/diff_cofold)
    by :func:`attach_models`.
    """

    strand:      str
    plate:       str   = None
    duplex:      str   = None
    mod:         str   = None
    report:      str   = None
    oligo_c:     float = np.nan
    salt_c:      float = np.nan
    T_m_raw:     list  = np.nan
    T_m_vH:      list  = np.nan
    T_m_fit:     list  = np.nan
    T_m_multi:   list  = np.nan
    dG_37_vH:    list  = np.nan
    dH_vH:       list  = np.nan
    dS_vH:       list  = np.nan
    dG_37_fit:   list  = np.nan
    dH_fit:      list  = np.nan
    dS_fit:      list  = np.nan
    dG_multi:    float = np.nan
    dH_multi:    float = np.nan
    dS_multi:    float = np.nan
    cofold:      float = np.nan
    pf:          float = np.nan
    pf_new:      float = np.nan
    diff_cofold: float = np.nan
    ss:          str   = None
    dh_vrna:     float = np.nan
    ds_vrna:     float = np.nan
    raw_data:    list  = np.nan      # per-replicate signal curves
    temps:       list  = np.nan      # per-replicate temperature grids (parallel to raw_data)
    base_b:      list  = np.nan      # per-replicate fitted bound baseline (slope, intercept)
    base_ub:     list  = np.nan      # per-replicate fitted unbound baseline


@dataclass
class Results:
    res: list = field(default_factory=list)

    def add(self, record):
        self.res.append(record)
        return record

    def select(self, strand=None, sc=None, oc=None,
               duplex=None, plate=None, report=None):
        out = Results()
        for it in self.res:
            if strand is not None:
                a, b = it.strand.split("-")
                if strand != it.strand and strand != f"{b}-{a}":
                    continue
            if sc is not None and it.salt_c != sc:
                continue
            if oc is not None and it.oligo_c != oc:
                continue
            if duplex is not None and it.duplex != duplex:
                continue
            if plate is not None and it.plate != plate:
                continue
            if report is not None and it.report != report:
                continue
            out.res.append(it)
        return out


_IC_PARAM = None


def _ic_param():
    """Lazily load the inosine IC nearest-neighbour parameters."""
    global _IC_PARAM
    if _IC_PARAM is None:
        _IC_PARAM = RNA.sc_mod_param(
            os.path.join(_PARAMS, "rna_mod_inosine_parameters.json")
        )
    return _IC_PARAM


def _fold(seq, salt_c, modify=None, temperature=None):
    """Build a fold_compound for ``seq`` at the given salt / temperature."""
    md = RNA.md(temperature) if temperature is not None else RNA.md()
    if salt_c != 1000:
        md.salt = (salt_c + 21) / 1000
    fc = RNA.fold_compound(seq, md)
    if modify is not None:
        modify(fc)
    return fc


def vienna_model(strand, salt_c):
    """ViennaRNA cofold / partition-function model for a duplex."""
    id1, id2 = strand.split("-")
    s1, s2 = sequences[id1], sequences[id2]

    model = {"mod": "-", "ss": None, "cofold": np.nan, "pf": np.nan,
             "pf_new": np.nan, "dh_vrna": np.nan, "ds_vrna": np.nan}

    if "I" in s1 and "I" in s2:
        model["mod"] = "I"
        return model

    if "I" in s1:
        model["mod"] = "I"
        seq = f'{s1.replace("I", "G")}&{s2}'
        ipos = s1.index("I") + 1
    elif "I" in s2:
        model["mod"] = "I"
        seq = f'{s1}&{s2.replace("I", "G")}'
        ipos = len(s1) + len(s2) - s2[::-1].index("I")
    else:
        seq = f"{s1}&{s2}"
        ipos = None

    inosine = (lambda fc: fc.sc_mod_inosine([ipos])) if ipos is not None else None

    fc = _fold(seq, salt_c, inosine)
    model["ss"], model["cofold"] = fc.mfe()
    _, model["pf"] = fc.pf()

    if ipos is not None:
        # pf_new: same duplex scored with the new I nearest-neighbour parameters
        _, model["pf_new"] = _fold(
            seq, salt_c, lambda fc: fc.sc_mod(_ic_param(), [ipos])
        ).pf()

    # Use the partition function at 2 different temperature to calculate dH and
    # dS
    _, pf36 = _fold(seq, salt_c, inosine, temperature=36).pf()
    _, pf38 = _fold(seq, salt_c, inosine, temperature=38).pf()
    model["ds_vrna"] = 0.5 * (pf36 - pf38)
    model["dh_vrna"] = model["pf"] + 310.15 * model["ds_vrna"]
    return model


def _diff_cofold(it):
    """Model-minus-measurement ΔG for one record (np.nan if unavailable)."""
    dG_multi = it.dG_multi
    if dG_multi is not None and not (
        isinstance(dG_multi, float) and np.isnan(dG_multi)
    ):
        return it.pf - dG_multi
    fit = it.dG_37_fit
    if isinstance(fit, (list, tuple)) and len(fit) > 1:
        return it.pf - mean(fit)
    if isinstance(fit, (list, tuple)) and len(fit) == 1:
        return it.pf - fit[0]
    return np.nan


def attach_models(results):
    """Attach ViennaRNA model values to every record of ``results`` in place."""
    cache = {}
    for it in results.res:
        key = (it.strand, it.salt_c)
        if key not in cache:
            cache[key] = vienna_model(it.strand, it.salt_c)
        m = cache[key]
        it.mod         = m["mod"]
        it.ss          = m["ss"]
        it.cofold      = m["cofold"]
        it.pf          = m["pf"]
        it.pf_new      = m["pf_new"]
        it.dh_vrna     = m["dh_vrna"]
        it.ds_vrna     = m["ds_vrna"]
        it.diff_cofold = _diff_cofold(it)
    return results
