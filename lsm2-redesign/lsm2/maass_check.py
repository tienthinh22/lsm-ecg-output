"""Testing the test on Maass's liquid (plan, Correctness criteria): levels 3 and 4 on Maass's own 135-neuron
column in two versions (the Brian2 script's, and our reading of the paper, the one Y copies), each with its own
20 Hz Poisson input and its own weights, on the 28-sub-step grid."""
import numpy as np

from . import config as C
from .liquid import build_maass_script, build_maass_paper
from .measure import measure_at_g, memory_curve
from .calib import input_neuron_rate
from . import gain


def level3_verdict(per_seed, sweep_rec):
    """Level 3 at one g: firing band and starting-state rule on every seed, and the gap peaking inside the grid."""
    rows = [{"seed": r["seed"], "in_band": bool(C.BAND_HZ[0] <= r["rate_hz"] <= C.BAND_HZ[1]
                                                 and r["frac_above_100hz"] <= C.HIGH_FRAC),
             "start_rule": bool(r.get("start_rule_pass", False)), "rate_hz": r["rate_hz"]} for r in per_seed]
    ok = all(x["in_band"] and x["start_rule"] for x in rows) and sweep_rec["gap_peak_inside_grid"]
    return bool(ok), rows


def level4_verdict(curves_on, curves_off, lag=C.MEMORY_PASS_T_MS):
    gains = [on[f"r2_{lag}"] - off[f"r2_{lag}"] for on, off in zip(curves_on, curves_off)]
    return bool(all(np.isfinite(gn) and gn >= C.MEMORY_MIN_GAIN for gn in gains)), gains


def run_version(ctx, version, seeds, data, log=print):
    build = build_maass_script if version == "script" else build_maass_paper
    liqs = {s: build(s) for s in seeds}
    for s, l in liqs.items():
        l.meta["seed"] = s
    scale = liqs[seeds[0]].in_scale_fixed
    per = {}

    def measure(g, s):
        r = measure_at_g(ctx, liqs[s], g, scale, data.sets, s, with_start_test=True, extra_ranks=True)
        r["seed"] = s
        per[(g, s)] = r
        return r

    log(f"  Maass check, {version} version: gain sweep on seeds {list(seeds)}")
    sw = gain.sweep(measure, seeds, log=log)
    res = {"version": version, "sweep": sw, "input_scale": scale}
    for label, g in (("own_weights", 1.0), ("rule", sw["chosen_g"])):
        if g is None:
            res[label] = {"g": None, "level3": False, "level4": False}
            continue
        l3, rows = level3_verdict([per[(g, s)] for s in seeds], sw)
        on = [memory_curve(ctx, liqs[s], g, scale, data.mem_spikes, data.mem_targets, s) for s in seeds]
        off = [memory_curve(ctx, liqs[s], g, scale, data.mem_spikes, data.mem_targets, s, recurrence=False) for s in seeds]
        l4, gains = level4_verdict(on, off)
        res[label] = {"g": g, "level3": l3, "level3_rows": rows, "level4": l4, "memory_gain_100ms": gains,
                      "memory_on": on, "memory_off": off}
        log(f"    {version} at g = {g:g} ({label}): level 3 {'pass' if l3 else 'FAIL'}, level 4 "
            f"{'pass' if l4 else 'FAIL'} (R^2 gains at 100 ms {np.round(gains, 3).tolist()})")
    # the test must also reject something: some g on the sweep fails the working-regime rules
    fails = [g for g in sw["table"] if g > 0 and not all(
        (C.BAND_HZ[0] <= per[(g, s)]["rate_hz"] <= C.BAND_HZ[1] and per[(g, s)]["frac_above_100hz"] <= C.HIGH_FRAC
         and per[(g, s)]["start_rule_pass"]) for s in seeds)]
    res["rejects_some_g"] = bool(fails)
    res["rejected_g"] = sorted(fails)
    # what keeps the rank measure: an in-band peak above the smallest g tried, and a lower gap at the next g up
    gs = sorted(g for g in sw["table"] if g > 0)
    ch = sw["chosen_g"]
    res["rank_measure_kept"] = bool(ch is not None and ch > gs[0] and sw.get("next_higher_gap_lower", False))
    if version == "paper":
        res["input_neuron_rate_recurrence_off_hz"] = input_neuron_rate(ctx, list(liqs.values()), scale, data.sets["sep"])
    res["pass"] = bool(res["own_weights"]["level3"] and res["own_weights"]["level4"] and res["rule"]["level3"]
                       and res["rule"]["level4"] and res["rejects_some_g"] and res["rank_measure_kept"])
    return res


def maass_check(ctx, data, seeds=C.SEEDS_SEARCH, log=print):
    out = {v: run_version(ctx, v, seeds, data, log) for v in ("script", "paper")}
    out["pass"] = bool(out["script"]["pass"] and out["paper"]["pass"])
    out["factor5_target_hz"] = out["paper"]["input_neuron_rate_recurrence_off_hz"]
    log(f"  Maass check: {'PASS' if out['pass'] else 'FAIL'}; factor-5 target rate {out['factor5_target_hz']:.3f} Hz")
    return out
