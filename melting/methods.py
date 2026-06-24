import numpy as np
from scipy.optimize import least_squares

from . import constants, functions

baseline_unbound_minT = 70
baseline_unbound_maxT = 80
baseline_bound_maxT = 20


def T_m_ds_raw(
    temperatures,
    ydata,
    baseline_bound_minT=-1,
    baseline_bound_maxT=20.0,
    baseline_unbound_minT=70,
    baseline_unbound_maxT=-1,
    debug=False,
):
    T = np.array(temperatures)

    # set bounds first
    bl_b_minT = T[0] if baseline_bound_minT == -1 else baseline_bound_minT
    bl_ub_maxT = T[-1] if baseline_unbound_maxT == -1 else baseline_unbound_maxT

    bl_b_maxT = T[-1] if baseline_bound_maxT not in T else baseline_bound_maxT
    bl_ub_minT = T[-1] if baseline_unbound_minT not in T else baseline_unbound_minT

    if debug:
        print(f"bounds_bound: [{bl_b_minT}, {bl_b_maxT}]")
        print(f"bounds_unbound: [{bl_ub_minT}, {bl_ub_maxT}]")

    # create upper base line (bound state)
    if bl_b_maxT <= bl_b_minT:
        m_bound = 0
        b_bound = ydata[0]
    else:
        m_bound, b_bound = functions.fit_linear(T, ydata, bl_b_minT, bl_b_maxT)

    # lower base line
    if bl_ub_minT >= bl_ub_maxT:
        m_unbound = 0
        b_unbound = ydata[-1]
        if debug:
            print("lower base line: ", m_unbound, b_unbound)
    else:
        m_unbound, b_unbound = functions.fit_linear(T, ydata, bl_ub_minT, bl_ub_maxT)

    # median of base lines
    m_med, b_med = (m_bound + m_unbound) / 2, (b_bound + b_unbound) / 2

    T_m, y_dat = functions.intersect_lin(m_med, b_med, T, ydata)

    return T_m, y_dat, (m_bound, b_bound), (m_unbound, b_unbound), (m_med, b_med)


def vantHoff(
    T,
    signal,
    m_bound,
    b_bound,
    m_unbound,
    b_unbound,
    c0,
    border=0.1,
    T_scale=1000.0,
    duplex=True,
    t1_min=-1,
    t1_max=-1,
):
    # normalize signal to obtain fraction of folded
    f_folded = [
        (dd - functions.linear(T[i], m_unbound, b_unbound))
        / (
            functions.linear(T[i], m_bound, b_bound)
            - functions.linear(T[i], m_unbound, b_unbound)
        )
        for i, dd in enumerate(signal)
    ]

    K = [
        np.log(2 * ff / (c0 * ((1.0 - ff) ** 2)))
        if ff <= (1 - border) and ff >= border
        else None
        for ff in f_folded
    ]
    t1 = [T_scale / (t - constants.T0) for t in T]

    lnK = []
    tt = []

    # compile actual data without None values
    for i in range(0, len(K)):
        if K[i] != None:
            tt.append(t1[i])
            lnK.append(K[i])

    if t1_min == -1:
        t1_min = len(t1)
    else:
        t1_min = next(i for i, v in enumerate(tt) if v < t1_min)

    if t1_max == -1:
        t1_max = 0
    else:
        t1_max = next(i for i, v in enumerate(tt) if v <= t1_max)

    xdata = np.array(tt[t1_max:t1_min])
    ydata = np.array(lnK[t1_max:t1_min])

    res_lsq = least_squares(functions.linear_res, [1, 1], args=(xdata, ydata))

    t_m = (np.log(4 / c0) - res_lsq.x[1]) / res_lsq.x[0]
    T_m = T_scale / t_m + constants.T0
    dH = -res_lsq.x[0] * constants.R * T_scale
    dG_Tm = constants.R * (T_m - constants.T0) * np.log(4 / c0) + dH

    dS = dH / (T_m - constants.T0) + constants.R * np.log(4 / c0)
    dG_37 = dH - (37.0 - constants.T0) * dS

    return T_m, dG_37, dH, dS, t1, K, xdata, ydata, res_lsq.x


def fit_full_function(
    T,
    d,
    c0=1e-6,
    dH_init=-100,
    dS_init=-0.2,
    b1_init=None,
    b2_init=None,
    lin_init=10,
    max_v=np.inf,
):
    xdata = np.array(T)
    ydata = np.array(d)

    # use the first and last lin_init data values
    # for initializing the linear intercepts of L_1 and L_2
    if b1_init == None:
        b1_init = (0, sum(d[0:lin_init]) / lin_init)

    if b2_init == None:
        b2_init = (0, sum(d[-lin_init:]) / lin_init)

    x_init = np.array([dH_init, dS_init, *b1_init, *b2_init])

    bounds = ([-150, -5, 0, -np.inf, 0, -np.inf], [0, 0, np.inf, np.inf, np.inf, max_v])

    scales = [1.0, 0.01, 1.0, 100.0, 1.0, 100.0]

    res_lsq = least_squares(
        functions.full_function_res,
        x_init,
        bounds=bounds,
        max_nfev=1e12,
        gtol=1e-8,
        ftol=1e-8,
        x_scale=scales,
        #loss='soft_l1',
        #f_scale=0.1,
        #jac = "3-point",
        #method = "dogbox",
        #verbose = 2,
        args=(xdata, ydata),
        kwargs={"c0": c0},
    )

    if res_lsq.success > 0:
        dG_37 = res_lsq.x[0] - (37 - constants.T0) * res_lsq.x[1]
        dH, dS, m_b, b_b, m_ub, b_ub = res_lsq.x
        dG_37 = dH - (37.0 - constants.T0) * dS

        # median of base lines
        m_med, b_med = (m_b + m_ub) / 2, (b_b + b_ub) / 2

        T_m, y_dat = functions.intersect_lin(m_med, b_med, xdata, ydata)

        return dG_37, dH, dS, T_m, y_dat, (m_b, b_b), (m_ub, b_ub), (m_med, b_med)
    else:
        raise ValueError("The least square didn't converge")


def fit_full_function_multi(
    T,
    ds,
    cs=None,
    c0=1e-6,
    dH_init=-100,
    dS_init=-0.2,
    b1_inits=None,
    b2_inits=None,
    lin_init=20,
    max_v=np.inf,
    residuals_method="linear",
):
    xdata = np.array(T)
    ydata = ds

    # use the first and last lin_init data values
    # for initializing the linear intercepts of L_1 and L_2
    if b1_inits == None:
        b1_inits = [(0, sum(dd[0:lin_init]) / lin_init) for dd in ds]

    if b2_inits == None:
        b2_inits = [(0, sum(dd[-lin_init:]) / lin_init) for dd in ds]

    x_inits = [dH_init, dS_init]
    for i in range(len(ds)):
        x_inits.extend([*b1_inits[i], *b2_inits[i]])

    x_init = np.array(x_inits)

    bounds_lo = [-150, -5]
    bounds_hi = [0, 0]
    for i in range(len(ds)):
        bounds_lo.extend([0, -np.inf, 0, -np.inf])
        bounds_hi.extend([np.inf, np.inf, np.inf, np.inf])

    bounds = (bounds_lo, bounds_hi)

    scales = [1.0, 0.01]

    for i in range(len(ds)):
        scales.extend([1, 100.0, 1, 100.0])

    if residuals_method not in functions.res_diffs:
        res_diff_fun = functions.res_diffs["linear"]
    else:
        res_diff_fun = functions.res_diffs[residuals_method]

    res_lsq = least_squares(
        functions.full_function_multi_res,
        x_init,
        bounds=bounds,
        max_nfev=1e12,
        gtol=1e-8,
        ftol=1e-8,
        x_scale=scales,
        #loss='soft_l1',
        #f_scale=0.1,
        #jac = "3-point",
        #method = "dogbox",
        #verbose = 2,
        args=(xdata, ydata),
        kwargs={
            "cs": cs,
            "c0": c0,
            #'res_func': res_diff_square }) # use squared difference for residuals
            "res_func": res_diff_fun,
        },
    )

    if res_lsq.success > 0:
        dH, dS = res_lsq.x[0], res_lsq.x[1]
        dG_37 = dH - (37 - constants.T0) * dS

        baselines_lo = []
        baselines_me = []
        baselines_hi = []
        T_ms = []
        y_dats = []
        for j, i in enumerate(range(2, len(res_lsq.x), 4)):
            m_b, b_b = res_lsq.x[i], res_lsq.x[i + 1]
            m_ub, b_ub = res_lsq.x[i + 2], res_lsq.x[i + 3]
            m_m, b_m = (m_b + m_ub) / 2, (b_b + b_ub) / 2
            T_m, y_dat = functions.intersect_lin(m_m, b_m, xdata, ydata[j])
            baselines_lo.append((m_b, b_b))
            baselines_hi.append((m_ub, b_ub))
            baselines_me.append((m_m, b_m))
            T_ms.append(T_m)
            y_dats.append(y_dat)

        return dG_37, dH, dS, T_ms, y_dats, baselines_lo, baselines_hi, baselines_me
    else:
        raise ValueError("The least square didn't converge")

