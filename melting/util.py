"""Per-curve melting-curve fitting for the submission package.

``analyze_pool`` fits the fluorescence replicates of one duplex condition;
``analyze_uv_pool`` does the same for the UV melting/annealing curves;
``analyze_multi`` is the global (all-replicates) fit.
"""

import numpy as np

from . import methods
from .variables import (
    raw_cut_fit,
    raw_baseline_fit,
    raw_vH,
    default_vH,
    fit_dH,
    border_vH,
    raw_cut_fit_uv,
    raw_baseline_fit_uv,
    default_baseline_fit_uv,
)


def analyze_pool(strands, salt_c, oligo_c, signals, temps_list,
                 plate="fluo", duplex="dsRNA"):
    """Fit every replicate melting curve of one (strand, oligo_c, salt_c) group."""
    result = {
        "strands": strands, "plate": plate, "duplex": duplex, "report": plate,
        "oligo_c": oligo_c, "salt_c": salt_c,
        "T_m_raw": [], "T_m_vH": [], "T_m_fit": [],
        "dG_37_vH": [], "dH_vH": [], "dS_vH": [],
        "dG_37_fit": [], "dH_fit": [], "dS_fit": [],
        "base_b": [], "base_ub": [],
        "raw_data": [],
    }

    c0 = 1e-6 * oligo_c * 2

    for rep, (T, signal) in enumerate(zip(temps_list, signals)):
        data = list(signal)
        r_data = data
        TT = np.asarray(T, dtype=float)

        nk = f"{strands}_{salt_c}_{oligo_c}_{rep}"

        if nk in raw_cut_fit:
            start, end = raw_cut_fit[nk]
        else:
            start, end = TT[0], TT[-1]

        if nk in raw_baseline_fit:
            r_base_b_maxT, r_base_ub_minT = raw_baseline_fit[nk]
        else:
            r_base_b_maxT = start + 10
            r_base_ub_minT = end - 10

        pos_start = np.where(TT == start)[0][0]
        pos_end = np.where(TT == end)[0][0] + 1
        TT = TT[pos_start:pos_end]
        data = data[pos_start:pos_end]

        T_m_r, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(
            TT, data,
            baseline_bound_maxT=r_base_b_maxT,
            baseline_unbound_minT=r_base_ub_minT,
        )

        t1_min, t1_max = raw_vH[nk] if nk in raw_vH else default_vH
        border = border_vH[nk] if nk in border_vH else 0.15

        T_m, dG_37, dH, dS, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
            TT, data, *base_b_r, *base_ub_r, c0,
            border=border, t1_min=t1_min, t1_max=t1_max,
        )

        dH_init = fit_dH[nk] if nk in fit_dH else None
        if -150 < dH < 0 and dH_init is None:
            dH_init = dH
        else:
            dH_init = -80
        dS_init = dS if -5 < dS < 0 else -0.2

        (dG_37_f, dH_f, dS_f, T_m_f, y_f,
         base_b_f, base_ub_f, base_med_f) = methods.fit_full_function(
            TT, data, c0=c0, dH_init=dH_init, dS_init=dS_init,
        )

        result["T_m_raw"].append(T_m_r)
        result["T_m_vH"].append(T_m)
        result["T_m_fit"].append(T_m_f)
        result["dG_37_vH"].append(dG_37)
        result["dH_vH"].append(dH)
        result["dS_vH"].append(dS)
        result["dG_37_fit"].append(dG_37_f)
        result["dH_fit"].append(dH_f)
        result["dS_fit"].append(dS_f)
        result["base_b"].append(base_b_f)
        result["base_ub"].append(base_ub_f)
        result["raw_data"].append(r_data)

    return result


def analyze_uv_pool(strand, oligo_c, name, salt_c, signals, temps_list):
    """Fit the UV replicate curves of one (strand, oligo_c, name) measurement."""
    result = {
        "strands": strand, "plate": name, "duplex": None, "report": "UV",
        "oligo_c": oligo_c, "salt_c": salt_c,
        "T_m_raw": [], "T_m_vH": [], "T_m_fit": [],
        "dG_37_vH": [], "dH_vH": [], "dS_vH": [],
        "dG_37_fit": [], "dH_fit": [], "dS_fit": [],
        "base_b": [], "base_ub": [],
        "raw_data": [], "temps": [],
    }

    c0 = 1e-6 * oligo_c * 2

    for rep, (T, signal) in enumerate(zip(temps_list, signals)):
        TT = np.asarray(T, dtype=float)
        data = list(signal)
        if TT[0] > TT[-1]:  # annealing -> orient increasing in temperature
            data = data[::-1]
            TT = TT[::-1]
            result["duplex"] = "annealing"
        else:
            result["duplex"] = "melting"
        r_data = list(data)
        r_temps = list(TT)

        nk = f"{strand}_{oligo_c}_{name}_{rep}"

        if nk in raw_baseline_fit_uv:
            r_base_b_maxT, r_base_ub_minT = raw_baseline_fit_uv[nk]
        else:
            r_base_b_maxT, r_base_ub_minT = default_baseline_fit_uv

        if nk in raw_cut_fit_uv:
            start, end = raw_cut_fit_uv[nk]
        else:
            start, end = TT[0], TT[-1]

        pos_start = np.where(TT == start)[0][0]
        pos_end = np.where(TT == end)[0][0] + 1
        TT = TT[pos_start:pos_end]
        data = data[pos_start:pos_end]

        T_m_r, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(
            TT, data,
            baseline_bound_maxT=r_base_b_maxT,
            baseline_unbound_minT=r_base_ub_minT,
        )

        t1_min, t1_max = default_vH
        T_m, dG_37, dH, dS, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
            TT, data, *base_b_r, *base_ub_r, c0,
            border=0.15, t1_min=t1_min, t1_max=t1_max,
        )

        (dG_37_f, dH_f, dS_f, T_m_f, y_f,
         base_b_f, base_ub_f, base_med_f) = methods.fit_full_function(
            TT, data, c0=c0, dH_init=-80,
        )

        result["T_m_raw"].append(T_m_r)
        result["T_m_vH"].append(T_m)
        result["T_m_fit"].append(T_m_f)
        result["dG_37_vH"].append(dG_37)
        result["dH_vH"].append(dH)
        result["dS_vH"].append(dS)
        result["dG_37_fit"].append(dG_37_f)
        result["dH_fit"].append(dH_f)
        result["dS_fit"].append(dS_f)
        result["base_b"].append(base_b_f)
        result["base_ub"].append(base_ub_f)
        result["raw_data"].append(r_data)
        result["temps"].append(r_temps)

    return result


def analyze_multi(records, drop_last=True):
    """Global fit: fit every replicate curve of ``records`` simultaneously.

    Returns ``{T_m_multi, dG_multi, dH_multi, dS_multi}`` or ``None`` if fewer
    than two curves are available.
    """
    curves = []  # (temps, signal, c0, dH_fit, dS_fit, base_b, base_ub)
    for it in records:
        c0 = 1e-6 * it.oligo_c * 2
        for rep, signal in enumerate(it.raw_data):
            T = np.asarray(it.temps[rep], dtype=float)
            y = np.asarray(signal, dtype=float)
            curves.append((T, y, c0, it.dH_fit[rep], it.dS_fit[rep],
                           it.base_b[rep], it.base_ub[rep]))

    if len(curves) < 2:
        return None

    # common temperature window on the shared 0.5 degC lattice
    start = max(T[0] for T, *_ in curves)
    end = min(T[-1] for T, *_ in curves)
    if drop_last:
        end -= 0.5

    mdata, Cs, dh_inits, ds_inits, b1_inits, b2_inits = [], [], [], [], [], []
    for T, y, c0, dH_fit, dS_fit, base_b, base_ub in curves:
        i0 = int(np.where(np.isclose(T, start))[0][0])
        i1 = int(np.where(np.isclose(T, end))[0][0]) + 1
        mdata.append(y[i0:i1])
        Cs.append(c0)
        dh_inits.append(dH_fit)
        ds_inits.append(dS_fit)
        b1_inits.append(base_b)
        b2_inits.append(base_ub)

    Tg = np.arange(start, end + 0.25, 0.5)
    dG_m, dH_m, dS_m, T_m_m, *_ = methods.fit_full_function_multi(
        Tg, mdata, cs=Cs,
        dH_init=float(np.mean(dh_inits)), dS_init=float(np.mean(ds_inits)),
        b1_inits=b1_inits, b2_inits=b2_inits,  # fitted baselines from get_results
        residuals_method="quadratic",
    )

    return {"T_m_multi": T_m_m, "dG_multi": dG_m,
            "dH_multi": dH_m, "dS_multi": dS_m}
