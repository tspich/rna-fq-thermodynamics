"""Inosine nearest-neighbour parameters from modified-vs-unmodified duplexes.

``calc_diff`` pairs each inosine duplex with its canonical analogue(s) and
accumulates the per-quantity stability differences; ``diff_nn_fit`` solves the
nearest-neighbour least squares on those differences and returns the data, which
the drivers render with ``format_report``.
"""

import numpy as np
import RNA

from .variables import sequences


pp_translate_ino = {
    # IC
    "CIGC" : [1, 1],
    "CGIC" : [1, 1],
    "CCGI" : [1, 2],
    "CCIG" : [1, 2],
    "CUIG" : [1, 3],
    "CGIU" : [1, 4],
    "CUIA" : [1, 5],
    "CAIU" : [1, 6],
    "GICC" : [2, 1],
    "IGCC" : [2, 1],
    "GCCI" : [2, 2],
    "ICCG" : [2, 2],
    "IUCG" : [2, 3],
    "IGCU" : [2, 4],
    "IUCA" : [2, 5],
    "IACU" : [2, 6],
    "GIUC" : [3, 1],
    "GCUI" : [3, 2],
    "UIGC" : [4, 1],
    "UCGI" : [4, 2],
    "AIUC" : [5, 1],
    "ACUI" : [5, 2],
    "UIAC" : [6, 1],
    "UCAI" : [6, 2],

    # IU
    "CUGI" : [1, 3],
    "CIGU" : [1, 4],
    "GUCI" : [2, 3],
    "GICU" : [2, 4],
    "IGUC" : [3, 1],
    "ICUG" : [3, 2],
    "IUUG" : [3, 3],
    "GUUI" : [3, 3],
    "IGUU" : [3, 4],
    "GIUU" : [3, 4],
    "IUUA" : [3, 5],
    "IAUU" : [3, 6],
    "UGIC" : [4, 1],
    "UCIG" : [4, 2],
    "UUIG" : [4, 3],
    "UUGI" : [4, 3],
    "UIGU" : [4, 4],
    "UGIU" : [4, 4],
    "UUIA" : [4, 5],
    "UAIU" : [4, 6],
    "AUUI" : [5, 3],
    "AIUU" : [5, 4],
    "UUAI" : [6, 3],
    "UIAU" : [6, 4],

    # IA
    "CIGA" : [1, 5],
    "CAGI" : [1, 6],
    "GICA" : [2, 5],
    "GACI" : [2, 6],
    "GIUA" : [3, 5],
    "GAUI" : [3, 6],
    "UIGA" : [4, 5],
    "UAGI" : [4, 6],
    "AGIC" : [5, 1],
    "ACIG" : [5, 2],
    "AUIG" : [5, 3],
    "AGIU" : [5, 4],
    "AIUA" : [5, 5],
    "AUIA" : [5, 5],
    "AAIU" : [5, 6],
    "AAUI" : [5, 6],
    "IGAC" : [6, 1],
    "ICAG" : [6, 2],
    "IUAG" : [6, 3],
    "IGAU" : [6, 4],
    "IUAA" : [6, 5],
    "UIAA" : [6, 5],
    "IAAU" : [6, 6],
    "UAAI" : [6, 6],

    # IG
    "IGGC" : [1, 1],
    "CGGI" : [1, 1],
    "ICGG" : [1, 2],
    "CIGG" : [1, 2],
    "IUGG" : [1, 3],
    "IGGU" : [1, 4],
    "IUGA" : [1, 5],
    "IAGU" : [1, 6],
    "GGIC" : [2, 1],
    "GGCI" : [2, 1],
    "GICG" : [2, 2],
    "GCIG" : [2, 2],
    "GUIG" : [2, 3],
    "GGIU" : [2, 4],
    "GUIA" : [2, 5],
    "GAIU" : [2, 6],
    "GGUI" : [3, 1],
    "GIUG" : [3, 2],
    "UGGI" : [4, 1],
    "UIGG" : [4, 2],
    "AGUI" : [5, 1],
    "AIUG" : [5, 2],
    "UGAI" : [6, 1],
    "UIAG" : [6, 2],
}

mod_stacks = [# IC
              "AIUC",
              "GICC",
              "GIUC",
              "CIGC",
              "UIAC",
              "UIGC",
              "ACUI",
              "GCCI",
              "GCUI",
              "CCGI",
              "UCAI",
              "UCGI",
              "IC",

              # IU
              "AIUU",
              "GICU",
              "GIUU",
              "CIGU",
              "UIAU",
              "UIGU",
              "AUUI",
              "GUCI",
              "GUUI",
              "CUGI",
              "UUAI",
              "UUGI",
              "IU",
             ]

# INO parameters from literature
NN_wright = {
    "AIUC" : -1.57,
    "GICC" : -2.62,
    "CIGC" : -1.86,
    "UIAC" : -0.96,
    "ACUI" : -1.02,
    "GCCI" : -1.89,
    "CCGI" : -2.23,
    "UCAI" : -1.18,

    "AIUU" : -0.41,
    "GICU" : -1.34,
    "CIGU" : -0.77,
    "UIAU" :  0.37,
    "AUUI" : -0.50,
    "GUCI" : -1.03,
    "CUGI" : -1.22,
    "UUAI" :  0.43,
}

NN_wright_dh = {
        'IUUI': 17.0,
        'IIUU': 9.53,
        'UIIU': 8.41,
        'IAUU': -8.22,
        'UIAU': -10.08,
        'AIUU': -11.68,
        'IUUA': -15.83,
        'CIGU': -11.99,
        'ICUG': -11.56,
        'IGUC': -13.38,
        'GICU': -9.81,

        'IGCC': -14.5,
        'ICCG': -10.6,
        'IACU': -15.3,
        'IUCA': -7.7,
        'GICC': -16.8,
        'CIGC': -12.7,
        'AIUC': -14.2,
        'UIAC': -11.8
        }

def check_BP(s1, s2, cano=False):
    if len(s1) != len(s2):
        return False
    if cano:
        BP = [('A', 'U'), ('G', 'C'), ('G', 'U'), ('U', 'A'), ('C', 'G'), ('U', 'G')]
    else:
        BP = [('A', 'U'), ('G', 'C'), ('G', 'U'), ('U', 'A'), ('C', 'G'), ('U', 'G'), ('I', 'C'), ('I', 'U'), ('C', 'I'), ('U', 'I')]
    base_pair = zip(list(s1),list(s2))
    for bp in base_pair:
        if bp not in BP:
            return False
    return True

def diff_bp (a, b):
    count = 0
    for i, aa in enumerate(a):
        if aa != b[i]:
            count += 1
    return count

def _pairs(mod_vals, unmod_vals):
    """All pairwise (mod - unmod) differences."""
    return [m - u for m in mod_vals for u in unmod_vals]


# Different kind of entries.
# return diff between mod and unmod
def calc_diff(results):
    list_strands = []
    for it in results.res:
        i1, i2 = it.strand.split('-')
        s1 = sequences[i1]
        s2 = sequences[i2]
        if check_BP(s1, s2[::-1]) == True and (
            it.strand not in list_strands and f"{i2}-{i1}" not in list_strands
        ):
            list_strands.append(it.strand)
    mod_unmod = {}
    for i, c in enumerate(list_strands):
        i1, i2 = c.split('-')
        s1 = sequences[i1]
        s2 = sequences[i2]
        if 'I' in s1 or 'I' in s2:
            mod_unmod[c] = []
            for j in range(0, len(list_strands)):
                c1, c2 = list_strands[j].split('-')
                comp1 = sequences[c1]
                comp2 = sequences[c2]
                if 'I' in comp1 or 'I' in comp2:
                    continue
                elif (i1 != c1 and i2 != c2) and (i1 != c2 and i2 != c1):
                    continue
                elif len(s1) != len(comp1) or (
                    diff_bp(
                        [*(zip(list(s1), list(s2[::-1])))],
                        [*(zip(list(comp1), list(comp2[::-1])))],
                    )
                    > 1
                    and diff_bp(
                        [*(zip(list(s2), list(s1[::-1])))],
                        [*(zip(list(comp1), list(comp2[::-1])))],
                    )
                    > 1
                ):
                    continue
                else:
                    mod_unmod[c].append(list_strands[j])

    ddG_olig = {}
    for mod in mod_unmod:
        seq_id1, seq_id2 = mod.split("-")
        smod1 = sequences[seq_id1]
        smod2 = sequences[seq_id2][::-1] # rotate so we can easily hop over all stacks
        for unmod in mod_unmod[mod]:
            unmod_id1, unmod_id2 = unmod.split('-')
            sumod1 = sequences[unmod_id1]
            sumod2 = sequences[unmod_id2][::-1]

            relevant_stacks = []
            stacks = []
            for i in range(0, len(smod1)-1):
                curr_stack = smod1[i:i+2]+smod2[i:i+2]
                curr_stack_rev = curr_stack[::-1]
                curr_stack_unmod = sumod1[i:i+2]+sumod2[i:i+2]
                curr_stack_unmod_rev = curr_stack_unmod[::-1]
                if curr_stack in mod_stacks:
                    relevant_stacks.append((curr_stack, curr_stack_unmod))
                    stacks.append(curr_stack)
                elif curr_stack_rev in mod_stacks:
                    relevant_stacks.append((curr_stack_rev, curr_stack_unmod_rev))
                    stacks.append(curr_stack_rev)

            if (smod1[0], smod2[0]) == ('I', 'C') or (smod1[0], smod2[0]) == ('C', 'I'):
                stacks.append('IC')
            if (smod1[-1], smod2[-1]) == ('I', 'C') or (smod1[-1], smod2[-1]) == ('C', 'I'):
                stacks.append('IC')
            if (smod1[0], smod2[0]) == ('I', 'U') or (smod1[0], smod2[0]) == ('U', 'I'):
                stacks.append('IU')
            if (smod1[-1], smod2[-1]) == ('I', 'U') or (smod1[-1], smod2[-1]) == ('U', 'I'):
                stacks.append('IU')


            it_mod = results.select(strand=mod)

            for d in it_mod.res:
                it_unmod = results.select(strand=unmod, sc=d.salt_c)
                for itu in it_unmod.res:
                    if itu is None:
                        print('None')
                        continue

                    if d.salt_c not in ddG_olig:
                        ddG_olig[d.salt_c] = {}
                    if mod not in ddG_olig[d.salt_c]:
                        ddG_olig[d.salt_c][mod] = {}
                    if unmod not in ddG_olig[d.salt_c][mod]:
                        ddG_olig[d.salt_c][mod][unmod] = {
                            'stacks' : stacks,
                            'dG_37_vH' : [],
                            'dH_vH' : [],
                            'dS_vH' : [],
                            'dG_37_fit' : [],
                            'dH_fit' : [],
                            'dS_fit' : [],
                            'model' : None,
                            'model_dh' : None,
                            'model_ds' : None,
                            'model_new' : None,
                        }

                    if np.isnan(d.cofold) or np.isnan(itu.cofold):
                        raise ValueError('no ViennaRNA mfe')

                    entry = ddG_olig[d.salt_c][mod][unmod]
                    entry['model'] = d.pf - itu.pf
                    entry['model_dh'] = d.dh_vrna - itu.dh_vrna
                    entry['model_ds'] = d.ds_vrna - itu.ds_vrna
                    entry['model_new'] = d.pf_new - itu.pf

                    entry['dG_37_fit'] += _pairs(d.dG_37_fit, itu.dG_37_fit)
                    entry['dG_37_vH'] += _pairs(d.dG_37_vH, itu.dG_37_vH)
                    entry['dH_fit'] += _pairs(d.dH_fit, itu.dH_fit)
                    entry['dH_vH'] += _pairs(d.dH_vH, itu.dH_vH)
                    entry['dS_fit'] += _pairs(d.dS_fit, itu.dS_fit)
                    entry['dS_vH'] += _pairs(d.dS_vH, itu.dS_vH)

    return mod_unmod, ddG_olig

def diff_nn_fit(ddG_olig, list_salt=[150], kind="dG"):
    if kind == "dG":
        value_key, nn_ref, lit = "dG_37_fit", RNA.param().stack, NN_wright
    elif kind == "dH":
        value_key, nn_ref, lit = "dH_fit", RNA.cvar.stackdH, NN_wright_dh
    else:
        raise ValueError(f"kind must be 'dG' or 'dH', got {kind!r}")

    diff_ino = {}
    for salt in list_salt:
        for mod in ddG_olig[salt]:
            for unmod in ddG_olig[salt][mod]:
                key = f"{mod}+{unmod}"
                vals = ddG_olig[salt][mod][unmod][value_key]
                vals = list(vals) if isinstance(vals, (list, tuple, np.ndarray)) else [vals]
                if key not in diff_ino:
                    diff_ino[key] = {
                        "stacks": ddG_olig[salt][mod][unmod]["stacks"],
                        "values": list(vals),
                    }
                else:
                    diff_ino[key]["values"].extend(vals)

    fit_mx = np.zeros([len(diff_ino), len(mod_stacks)], dtype=int)
    en_mx = []
    for i, key in enumerate(diff_ino):
        en_mx.append(np.mean(diff_ino[key]["values"]))
        for stack in diff_ino[key]["stacks"]:
            fit_mx[i, mod_stacks.index(stack)] += 1
    en_mx = np.array(en_mx)

    num_occ = np.sum(fit_mx, axis=0)
    p, resid, rank, _ = np.linalg.lstsq(fit_mx, en_mx, rcond=None)

    n, k = len(en_mx), len(p)
    sigma2 = np.sum((en_mx - fit_mx @ p) ** 2) / (n - k)   # RMSE
    cov = sigma2 * np.linalg.inv(fit_mx.T @ fit_mx)
    se = np.sqrt(np.diag(cov))

    rows = []
    for i, stack in enumerate(mod_stacks):
        if stack in pp_translate_ino:
            a, b = pp_translate_ino[stack]
            nn_en = nn_ref[a][b] / 100.0
        else:
            nn_en = 0.0
        if stack in lit:
            litval = lit[stack]
        elif stack[::-1] in lit:
            litval = lit[stack[::-1]]
        else:
            litval = None
        rows.append({
            "stack": stack, "occ": int(num_occ[i]),
            "value": float(p[i]), "se": float(se[i]),
            "nn_en": nn_en, "with_nn": float(p[i]) + nn_en, "lit": litval,
        })

    return {
        "kind": kind,
        "params": {r["stack"]: (r["value"], r["se"]) for r in rows},
        "rows": rows,
        "rank": int(rank),
        "n_stacks": len(mod_stacks),
        "residuals": resid.tolist() if hasattr(resid, "tolist") else resid,
    }


def format_report(result):
    """Render a :func:`diff_nn_fit` result as a text table (drivers print it)."""
    lines = [f"rank = {result['rank']} / {result['n_stacks']}"]
    lines.append("{:5s}\t{:>3s}\t{:>6s}\t{:>5s}\t{:>6s}\t{:>6s}\t{:>6s}\t{:>6s}".format(
        "NN", "Occ", "ls", "SD", "unmod", "res", "Lit", "diff"))
    for r in result["rows"]:
        if r["lit"] is not None:
            lines.append(
                "{:5s}\t{:3d}\t{:6.2f}\t{:5.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}\t{:6.2f}".format(
                    r["stack"], r["occ"], r["value"], r["se"],
                    r["nn_en"], r["with_nn"], r["lit"], r["with_nn"] - r["lit"]))
        else:
            lines.append(
                "{:5s}\t{:3d}\t{:6.2f}\t{:5.2f}\t{:6.2f}\t{:6.2f}".format(
                    r["stack"], r["occ"], r["value"], r["se"],
                    r["nn_en"], r["with_nn"]))
    lines.append(f"residuals: {result['residuals']}")
    return "\n".join(lines)
