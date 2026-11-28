"""Choosing the recurrent gain g (plan, Choosing the recurrent gain; decision 5).

Grid: g = 0 (control), 0.1 and 0.3, a coarse pass at 1, 3, ..., 1000 times the base weights, extensions in 3x
steps at an edge (at most 3), then a fine pass g* x 1.4^k, k = -3..3. Each setting is measured on all seeds and
averaged. Rule: among settings above 0 inside the firing band, the largest mean gap; a setting whose mean gap lies
within the range of the best setting's seed values is tied with it, and the smallest tied g is taken."""
import numpy as np

from . import config as C


def _agg(per_seed):
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float, bool, np.floating, np.integer))]
    m = {k: float(np.nanmean([float(r[k]) for r in per_seed])) for k in keys}
    m["gap_seeds"] = [float(r["gap"]) for r in per_seed]
    m["rate_seeds"] = [float(r["rate_hz"]) for r in per_seed]
    m["in_band"] = bool(C.BAND_HZ[0] <= m["rate_hz"] <= C.BAND_HZ[1] and m["frac_above_100hz"] <= C.HIGH_FRAC)
    m["per_seed"] = per_seed
    return m


def choose(table):
    """table: {g: aggregated measures}. Returns (chosen g or None, best g or None)."""
    cand = [g for g in table if g > 0 and table[g]["in_band"]]
    if not cand:
        return None, None
    best = max(cand, key=lambda g: (table[g]["gap"], -g))
    lo, hi = min(table[best]["gap_seeds"]), max(table[best]["gap_seeds"])
    eps = 1e-12 * max(1.0, abs(lo), abs(hi))                 # the mean of equal seed values may round
    tied = [g for g in cand if lo - eps <= table[g]["gap"] <= hi + eps] + [best]
    return min(tied), best


def edge_status(table, g):
    """'upper' / 'lower' if g sits at that edge of the measured grid (g = 0 excluded), else None. A setting next
    to one outside the firing band counts as inside the grid (plan, Grid)."""
    gs = sorted(x for x in table if x > 0)
    i = gs.index(g)
    if i == len(gs) - 1:
        return "upper"
    if i == 0:
        return "lower"
    return None


def band_limited(table, g):
    gs = sorted(x for x in table if x > 0)
    i = gs.index(g)
    up = i + 1 < len(gs) and not table[gs[i + 1]]["in_band"]
    down = i > 0 and not table[gs[i - 1]]["in_band"]
    return bool(up or down)


def sweep(measure, seeds, log=print, coarse=None):
    """measure(g, seed) -> level-3 measures at g (measure.measure_at_g). Returns the sweep record."""
    table, order = {}, []

    def run(gs, stage):
        for g in gs:
            g = float(g)
            if g in table:
                continue
            table[g] = _agg([measure(g, s) for s in seeds])
            table[g]["stage"] = stage
            order.append(g)
            t = table[g]
            log(f"    g={g:<10.4g} abs={t['abs_g']:.4g} rate={t['rate_hz']:.2f} Hz in_band={t['in_band']} "
                f"sep={t['sep']:.2f} gen={t['gen']:.2f} gap={t['gap']:.2f}"
                + (f" twin={t['twin_ratio']:.2f}" if 'twin_ratio' in t else ""))

    run([0.0] + list(C.G_LOW) + list(coarse or C.G_COARSE), "coarse")
    chosen, best = choose(table)
    ext = 0
    while best is not None and ext < C.G_EXTEND_MAX:
        e = edge_status(table, best)
        if e is None:
            break
        gs = sorted(x for x in table if x > 0)
        nxt = gs[-1] * C.G_EXTEND_STEP if e == "upper" else gs[0] / C.G_EXTEND_STEP
        run([nxt], f"extension {ext + 1}")
        ext += 1
        chosen, best = choose(table)
    at_edge = best is not None and edge_status(table, best) is not None
    if best is not None and not at_edge:
        run([best * C.G_FINE_STEP ** k for k in range(-C.G_FINE_K, C.G_FINE_K + 1) if k != 0], "fine")
        chosen, best = choose(table)
        at_edge = edge_status(table, best) is not None
    rec = {"table": table, "order": order, "chosen_g": chosen, "best_g": best, "extensions": ext,
           "no_g_in_band": best is None, "peak_at_edge": bool(at_edge),
           "band_limited": bool(chosen is not None and band_limited(table, chosen))}
    if chosen is not None:
        gs = sorted(x for x in table if x > 0)
        i = gs.index(chosen)
        rec["next_higher_g"] = gs[i + 1] if i + 1 < len(gs) else None
        rec["next_higher_gap_lower"] = (rec["next_higher_g"] is not None
                                        and table[rec["next_higher_g"]]["gap"] < table[chosen]["gap"])
    rec["gap_peak_inside_grid"] = bool(best is not None and not at_edge)
    return rec
