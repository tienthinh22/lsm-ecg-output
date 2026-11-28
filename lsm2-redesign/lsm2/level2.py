"""Level 2, faithful: is each candidate the liquid its switches describe? Connection counts by type, input neurons,
sample means of weights and of U, D, F, signs, each neuron's net excitatory and inhibitory input. Counts and means
within 4.5 standard errors of their targets; the U, D, F targets and their standard errors come from the drawing
procedure itself (10 million draws per type); fractions as set; every sign correct."""
import functools
import numpy as np

from . import config as C
from .liquid import _type_matrix, _draw_udf, candidate_settings, round_half_up

N_DRAWS = 10_000_000


@functools.lru_cache(maxsize=None)
def udf_targets(bad_rule="redraw"):
    """Mean and SD of U, D, F per type under the drawing procedure (10 million draws per type)."""
    out = {}
    for name, scale, cap in (("U", 1.0, True), ("D_s", 1000.0, False), ("F_s", 1000.0, False)):
        for ti, t in enumerate(C.TYPES):
            T = np.full(N_DRAWS, ti)
            v = _draw_udf(np.random.default_rng([99, ti, len(name)]), np.random.default_rng([98, ti, len(name)]),
                          T, {k: x * scale for k, x in C.MAASS[name].items()}, cap and bad_rule == "redraw", bad_rule)
            out[(name, t)] = (float(v.mean()), float(v.std()))
    return out


def _z(obs, target, se):
    return float((obs - target) / se) if se > 0 else (0.0 if abs(obs - target) < 1e-12 else float("inf"))


def check(liq, n_channels=None):
    """Level 2 for a ladder candidate (and Y). Returns {"pass": bool, "items": [...]}."""
    f = liq.factors
    s = candidate_settings(f)
    items = []
    def add(name, obs, target, se=None, exact=False, ok=None):
        if ok is None:
            if exact:
                ok = abs(obs - target) <= 1e-9 * max(1.0, abs(target))
                z = None
            else:
                z = _z(obs, target, se)
                ok = abs(z) <= C.LEVEL2_SE
        else:
            z = None
        items.append({"item": name, "observed": float(obs), "target": float(target),
                      "se": None if se is None else float(se), "z": z, "pass": bool(ok)})
    N, E = liq.N, liq.is_exc
    add("neurons", N, int(np.prod(s["dims"])), exact=True)
    add("excitatory neurons", E.sum(), int(N * C.X["excit_ratio"]), exact=True)
    # connections by type: expected count and binomial SE from the realised types and positions
    T = _type_matrix(E)
    Cm = np.vectorize(lambda t: C.X["C"][C.TYPES[t]])(T)
    d = np.sqrt(((liq.pos[:, None, :] - liq.pos[None, :, :]) ** 2).sum(-1))
    p = np.clip(Cm * np.exp(-(d / s["lam"]) ** 2), 0, 1)
    np.fill_diagonal(p, 0)
    for ti, t in enumerate(C.TYPES):
        m = T == ti
        add(f"{t} connections", (liq.typ == ti).sum(), p[m].sum(), np.sqrt((p[m] * (1 - p[m])).sum()))
    add("self-connections", int((liq.pre == liq.post).sum()), 0, exact=True)
    # recurrent weights: the Gamma draws before (a)'s fan-in division
    fan_in = np.maximum(np.bincount(liq.post, minlength=N), 1)
    raw = np.abs(liq.w) * (fan_in[liq.post] if s["normalise"] else 1.0)
    for ti, t in enumerate(C.TYPES):
        m = liq.typ == ti
        if m.sum():
            mu = s["mu"][t]
            add(f"{t} weight mean (Gamma draws)", raw[m].mean(), mu, mu / np.sqrt(m.sum()))
            add(f"{t} sign", float(np.all(np.sign(liq.w[m]) == (1 if ti < 2 else -1))), 1.0, exact=True)
    # U, D, F
    if liq.syn_model == "markram":
        tg = udf_targets("redraw")
        for name, arr in (("U", liq.U), ("D_s", liq.D_ms), ("F_s", liq.F_ms)):
            for ti, t in enumerate(C.TYPES):
                m = liq.typ == ti
                if m.sum():
                    mean, sd = tg[(name, t)]
                    add(f"{t} {name.split('_')[0]} mean", arr[m].mean(), mean, sd / np.sqrt(m.sum()))
            add(f"{name.split('_')[0]} positive", float((arr > 0).all()), 1.0, exact=True)
        add("U at most 1", float((liq.U <= 1).all()), 1.0, exact=True)
    # input
    Cn = liq.Win.shape[0]
    mask = liq.Win > 0
    add("input weights non-negative", float((liq.Win >= 0).all()), 1.0, exact=True)
    if s["input30"]:
        k = round_half_up(C.MAASS["in_fraction"] * N)
        add("input-receiving neurons", liq.in_neurons.sum(), k, exact=True)
        add("each input neuron hears every channel", float(mask[:, liq.in_neurons].all()), 1.0, exact=True)
        add("other neurons get no input", float(not mask[:, ~liq.in_neurons].any()), 1.0, exact=True)
        n_in = int(mask.sum())
    else:
        P = C.X["P_in"]
        n_in = int(mask.sum())
        add("input synapses", n_in, Cn * N * P, np.sqrt(Cn * N * P * (1 - P)))
    split = liq.in_split
    for typ_name, sel, m_type in (("E", E, split[0]), ("I", ~E, split[1])):
        cols = mask[:, sel]
        if not cols.any():
            continue
        if s["normalise"]:
            cm = liq.Win[:, sel].sum(0)[cols.any(0)] / cols.sum(0)[cols.any(0)]
            dev = float(np.abs(cm / m_type - 1).max())             # (a)'s eps of 1e-8 leaves ~1e-8 relative
            add(f"input column means onto {typ_name} ((a)'s rescale), largest relative deviation", dev, 0.0, ok=dev < 1e-6)
        else:
            v = liq.Win[:, sel][cols]
            add(f"input weight mean onto {typ_name}", v.mean(), m_type, m_type / np.sqrt(v.size))
    # net excitatory and inhibitory input per neuron (reported)
    exc_in = np.bincount(liq.post[liq.typ < 2], weights=liq.w[liq.typ < 2], minlength=N)
    inh_in = np.bincount(liq.post[liq.typ >= 2], weights=liq.w[liq.typ >= 2], minlength=N)
    report = {"net_exc_input_mean": float(exc_in.mean()), "net_inh_input_mean": float(inh_in.mean()),
              "neurons_without_recurrent_input": int((np.bincount(liq.post, minlength=N) == 0).sum()),
              "input_synapses": n_in}
    return {"pass": all(i["pass"] for i in items), "items": items, "report": report}
