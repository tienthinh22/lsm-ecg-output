"""Statistics and decision rules (plan, Statistics and decision rules): the paired Nadeau-Bengio corrected
t-test over the 12 CV splits (variance factor 1/12 + 0.336), Holm within a family, the DS2 difference paired by
seed, and the verdicts meaningful / comparable / inconclusive. Inputs are tables of runs with columns spec_split,
spec_seed, macro_f1 (and DS2 metrics)."""
import math
import numpy as np
from scipy import stats as sstats

NB_RATIO = 0.336          # plan: test/train ratio of the 3x4 patient-grouped CV (the replication used 0.25/0.75)


def per_split_seed(d):
    """Macro-F1 by (split, seed)."""
    return d.groupby(["spec_split", "spec_seed"], dropna=False).macro_f1.mean()

def compare(a, b, require_seed_sign=False):
    """A vs B (Macro-F1): seeds averaged per CV split; paired Nadeau-Bengio corrected t-test over the 12 splits
    with its 95% interval; the DS2 difference paired by seed."""
    if a is None or b is None or a.empty or b.empty:
        return None
    A, B = per_split_seed(a), per_split_seed(b)
    if a["spec_seed"].isna().all() or b["spec_seed"].isna().all():      # a seed-free control pairs with every seed
        A = A.groupby(level=0).mean(); B = B.groupby(level=0).mean()
        Asp, Bsp = A, B
        ds2_seed = None
    else:
        Asp, Bsp = A.groupby(level=0).mean(), B.groupby(level=0).mean()
        ds2_seed = (A.xs("final", level=0) - B.xs("final", level=0)).dropna() if "final" in A.index.get_level_values(0) and "final" in B.index.get_level_values(0) else None
    cv = [s for s in Asp.index if str(s).startswith("cv-") and s in Bsp.index]
    if len(cv) < 3:
        return None
    d = (Asp[cv] - Bsp[cv]).values
    k = len(d)
    var = np.var(d, ddof=1) * (1 / k + NB_RATIO)
    se = math.sqrt(var)
    t = d.mean() / se if se > 0 else np.inf
    p = 2 * sstats.t.sf(abs(t), k - 1)
    q = sstats.t.ppf(0.975, k - 1)
    ds2 = float(Asp.get("final", np.nan) - Bsp.get("final", np.nan)) if ("final" in Asp.index and "final" in Bsp.index) else np.nan
    out = {"n_splits": k, "cv_diff": float(d.mean()), "ci_low": float(d.mean() - q * se), "ci_high": float(d.mean() + q * se),
           "t": float(t), "p": float(p), "ds2_diff": ds2,
           "ds2_diff_by_seed": None if ds2_seed is None else {str(s): float(v) for s, v in ds2_seed.items()}}
    # DS2 without record 232, and per-class SVEB / VEB recall and precision (plan: every comparison)
    fa, fb = a[a.spec_split == "final"], b[b.spec_split == "final"]
    for m in ("macro_f1_no232", "recall_SVEB", "precision_SVEB", "recall_VEB", "precision_VEB"):
        if m in fa and m in fb and len(fa) and len(fb):
            out[f"ds2_diff_{m}"] = float(fa[m].mean() - fb[m].mean())
    if require_seed_sign and not a["spec_seed"].isna().all():
        per_seed = []
        for sd in sorted(set(a.spec_seed.dropna()) & set(b.spec_seed.dropna())):
            xa = a[(a.spec_seed == sd) & a.spec_split.astype(str).str.startswith("cv-")].groupby("spec_split").macro_f1.mean()
            xb = b[(b.spec_seed == sd) & b.spec_split.astype(str).str.startswith("cv-")].groupby("spec_split").macro_f1.mean()
            j = xa.index.intersection(xb.index)
            per_seed.append(float((xa[j] - xb[j]).mean()))
        out["cv_diff_by_seed"] = per_seed
        out["same_sign_all_seeds"] = bool(per_seed and (all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed)))
    return out

def compare_diff(a_on, a_off, b_on, b_off):
    """F7: (on - off, full) - (on - off, ablated), per split (seeds averaged), corrected t-test."""
    parts = [per_split_seed(x).groupby(level=0).mean() if x is not None and not x.empty else None for x in (a_on, a_off, b_on, b_off)]
    if any(p is None for p in parts):
        return None
    D = (parts[0] - parts[1]) - (parts[2] - parts[3])
    cv = [s for s in D.index if str(s).startswith("cv-") and np.isfinite(D[s])]
    if len(cv) < 3:
        return None
    d = D[cv].values; k = len(d)
    se = math.sqrt(np.var(d, ddof=1) * (1 / k + NB_RATIO)); t = d.mean() / se if se > 0 else np.inf
    q = sstats.t.ppf(0.975, k - 1)
    return {"n_splits": k, "cv_diff": float(d.mean()), "ci_low": float(d.mean() - q * se), "ci_high": float(d.mean() + q * se),
            "t": float(t), "p": float(2 * sstats.t.sf(abs(t), k - 1)), "ds2_diff": float(D.get("final", np.nan)),
            "plain_full_minus_ablated_on": float((parts[0] - parts[2])[cv].mean())}

def holm(rows):
    ps = [r["p"] if r else np.nan for r in rows]
    idx = [i for i, p in enumerate(ps) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: ps[i]); m = len(order); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        rows[i]["p_holm"] = run
    return rows

def verdict(r, need_seed_sign=False):
    if r is None:
        return "missing"
    meaningful = abs(r["cv_diff"]) >= 0.01 and r.get("p_holm", 1) < 0.05 and np.sign(r["ds2_diff"]) == np.sign(r["cv_diff"])
    if need_seed_sign:
        meaningful = meaningful and r.get("same_sign_all_seeds", False)
    if meaningful:
        return "meaningful"
    if r["ci_low"] >= -0.02 and r["ci_high"] <= 0.02 and abs(r["ds2_diff"]) < 0.02:
        return "comparable"
    return "inconclusive"

def family(name, items, need_seed_sign=False):
    rows = [it[1] for it in items]
    rows = holm([dict(r) if r else None for r in rows])
    out = []
    for (label, _), r in zip(items, rows):
        out.append({"family": name, "comparison": label, **({} if r is None else {k: v for k, v in r.items() if not isinstance(v, (dict, list))}),
                    "verdict": verdict(r, need_seed_sign)})
    return out

