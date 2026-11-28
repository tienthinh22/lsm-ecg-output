"""Input size (plan, Input and calibration): the paper's rule, or with factor 5 a bisection on one factor s so the
input-receiving neurons fire, with recurrence off, at the rate Maass's own input-receiving neurons reach in the
Maass check (paper version). Factor 9 fixes Maass's own per-spike size instead."""
import math
import numpy as np

from . import config as C
from .liquid import paper_rule_scale
from .measure import simulate, firing_stats
from .sim import start_voltages


def ain_of(spikes):
    """Mean number of input spikes per sample, summed over channels (the authors' A_in)."""
    return float(np.asarray(spikes, np.float64).sum() / (spikes.shape[0] * spikes.shape[1]))


def input_neuron_rate(ctx, liqs, s, spikes, T=C.BEAT_SAMPLES):
    """Mean rate (Hz) of the input-receiving neurons, recurrence off, averaged over the liquids (seeds)."""
    rates = []
    for liq in liqs:
        u0 = start_voltages(liq, liq.meta.get("seed", 0), range(len(spikes)), 5)
        o = simulate(ctx, liq, 0.0, s, spikes, u0, T, {"neuron_counts": (0, T)}, recurrence=False)
        cnt = o["neuron_counts"][:, liq.in_neurons]
        rates.append(firing_stats(cnt, T, liq.h_ms * liq.n_sub)[0])
    return float(np.mean(rates))


def bisect_scale(ctx, liqs, spikes, target_hz, tol=C.CODE["calib_tol"], iters=C.CODE["calib_iters"], log=print):
    """Smallest bracket on s (log scale) until the rate is within tol of the target."""
    lo, hi = 0.0, 0.05
    r_hi = input_neuron_rate(ctx, liqs, hi, spikes)
    k = 0
    while r_hi < target_hz and k < 30:
        lo, hi = hi, hi * 2
        r_hi = input_neuron_rate(ctx, liqs, hi, spikes)
        k += 1
    if r_hi < target_hz:
        raise RuntimeError(f"input calibration: rate {r_hi:.2f} Hz < target {target_hz:.2f} Hz even at s = {hi}")
    best, best_r = hi, r_hi
    for i in range(iters):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
        r = input_neuron_rate(ctx, liqs, mid, spikes)
        if abs(r - target_hz) < abs(best_r - target_hz):
            best, best_r = mid, r
        if abs(r - target_hz) <= tol * target_hz:
            break
        if r < target_hz:
            lo = mid
        else:
            hi = mid
    log(f"    factor-5 input size s = {best:.5g} (rate {best_r:.3f} Hz, target {target_hz:.3f} Hz); Maass's own 0.12")
    return float(best), float(best_r)


def input_scale(ctx, liqs, spikes_cal, ain, maass_target_hz, log=print):
    """The candidate's input size: (scale, how it was set)."""
    liq = liqs[0]
    if liq.in_scale_fixed is not None:
        return float(liq.in_scale_fixed), {"rule": "fixed (factor 9 or Maass's own)", "scale": float(liq.in_scale_fixed)}
    if 5 in liq.factors:
        if maass_target_hz is None:
            raise RuntimeError("factor 5 needs the Maass check's input-neuron rate")
        s, r = bisect_scale(ctx, liqs, spikes_cal, maass_target_hz, log=log)
        return s, {"rule": "factor 5 calibration", "scale": s, "rate_hz": r, "target_hz": maass_target_hz}
    s = paper_rule_scale(ain, liq)
    fill_E = C.I_TARGET / (1 - float(liq.beta[liq.is_exc][0] ** liq.n_sub))
    fill_I = C.I_TARGET / (1 - float(liq.beta[~liq.is_exc][0] ** liq.n_sub))
    return s, {"rule": "paper's rule", "scale": s, "A_in": ain, "P_in": 1.0 if 6 in liq.factors else C.X["P_in"],
               "fill_E": fill_E, "fill_I": fill_I}
