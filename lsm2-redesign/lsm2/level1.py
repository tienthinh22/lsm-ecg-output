"""Levels 1b to 1f (plan, Correctness criteria). 1a is the test suite (tests/test_level1a.py)."""
import math
import time
import numpy as np
import torch

from . import config as C
from .liquid import build_maass_script
from .sim import Simulator, start_voltages
from .measure import Ctx, measure_at_g, simulate
from . import brian2_ref


def _ours_trace(liq, x, u0, T, g, in_scale, device="cpu", dtype=torch.float64, open_loop=None, force_own=False):
    sm = Simulator(liq, device=device, dtype=dtype, g=g, in_scale=in_scale)
    ol = None if open_loop is None else open_loop[:, None, :]
    o = sm.run(x[None], u0[None], T, rec={"raster": True}, open_loop=ol, trace=True,
               own_spikes=ol if force_own else None)
    tr = {k: np.array(o["trace"][k])[:, 0, :] for k in ("u", "I1", "I2", "I3") if o["trace"][k]}
    if o["trace"]["syn"]:
        tr["R"] = np.array([s["R"][0] for s in o["trace"]["syn"]])
        tr["q"] = np.array([s["u"][0] for s in o["trace"]["syn"]])
    tr["raster"] = o["raster"][:, 0, :].cpu().numpy()
    tr["flips"] = o.get("flips", 0)
    return tr


def _isi(raster, h_ms):
    out = []
    for i in range(raster.shape[1]):
        k = np.nonzero(raster[:, i])[0]
        if len(k) > 1:
            out.append(np.diff(k) * h_ms)
    return np.concatenate(out) if out else np.zeros(0)


def _ks(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


def level1b(liq, beats, g, in_scale, seed, T=C.SIM_SAMPLES, log=print, target="numpy", n_null=5):
    """Open loop: Brian2 and our float64 code get the same recorded presynaptic and input spikes; voltages,
    currents and synapse states must agree within 1e-9. Closed loop: reported (rate difference and ISI KS
    distance next to n_null Brian2-vs-Brian2 pairs from different starting states)."""
    rows, worst = [], 0.0
    h = liq.h_ms
    for b, x in enumerate(beats):
        u0 = start_voltages(liq, seed, [b], 11, random_start=True)[0]
        ours = _ours_trace(liq, x, u0, T, g, in_scale)
        t0 = time.time()
        br = brian2_ref.brian2_ours(liq, x, u0, T, g=g, in_scale=in_scale, open_loop=ours["raster"], target=target)
        d = {k: float(np.abs(ours[k] - br[k]).max()) for k in ("u", "I1", "I2", "I3", "R", "q") if k in ours and k in br}
        d["spike_mismatches"] = int((ours["raster"] != br["raster"]).sum())
        worst = max(worst, max(v for k, v in d.items() if k != "spike_mismatches"))
        rows.append({"beat": b, **d, "spikes": int(ours["raster"].sum()), "brian2_s": time.time() - t0})
    open_pass = worst <= C.TOL_1A and all(r["spike_mismatches"] == 0 for r in rows)
    # closed loop (reported)
    closed = []
    for b, x in enumerate(beats[:max(1, n_null)]):
        u0 = start_voltages(liq, seed, [b], 11, random_start=True)[0]
        ours = _ours_trace(liq, x, u0, T, g, in_scale)
        b1 = brian2_ref.brian2_ours(liq, x, u0, T, g=g, in_scale=in_scale, target=target)
        u0b = start_voltages(liq, seed, [b], 12, random_start=True)[0]
        b2 = brian2_ref.brian2_ours(liq, x, u0b, T, g=g, in_scale=in_scale, target=target)
        dur = T * liq.n_sub * h / 1000.0
        rate = lambda r: r.sum() / (liq.N * dur)
        closed.append({"beat": b, "ours_vs_brian2_rate_diff_hz": float(rate(ours["raster"]) - rate(b1["raster"])),
                       "ours_vs_brian2_ks": _ks(_isi(ours["raster"], h), _isi(b1["raster"], h)),
                       "ours_vs_brian2_identical_spikes": bool((ours["raster"] == b1["raster"]).all()),
                       "null_brian2_pair_rate_diff_hz": float(rate(b1["raster"]) - rate(b2["raster"])),
                       "null_brian2_pair_ks": _ks(_isi(b1["raster"], h), _isi(b2["raster"], h))})
    log(f"  1b open loop: worst difference {worst:.2e} (tolerance {C.TOL_1A:g}), "
        f"spike mismatches {sum(r['spike_mismatches'] for r in rows)} -> {'PASS' if open_pass else 'FAIL'}")
    return {"pass": bool(open_pass), "worst_abs_diff": worst, "open_loop": rows, "closed_loop_reported": closed}


def level1c(liq, beats, g, in_scale, seed, device, T=C.SIM_SAMPLES, log=print):
    """The open-loop test of 1b on this machine's card: float32 on the device vs float64, within 1e-4."""
    worst, rows = 0.0, []
    for b, x in enumerate(beats):
        u0 = start_voltages(liq, seed, [b], 11, random_start=True)[0]
        ref = _ours_trace(liq, x, u0, T, g, in_scale)
        # every spike (presynaptic and each neuron's own) is the float64 run's, so only arithmetic can differ;
        # threshold decisions float32 would have taken differently are counted (reported)
        f32 = _ours_trace(liq, x, u0, T, g, in_scale, device=device, dtype=torch.float32, open_loop=ref["raster"],
                          force_own=True)
        d = {k: float(np.abs(ref[k] - f32[k]).max()) for k in ("u", "I1", "I2", "I3", "R", "q") if k in ref}
        flips = int(f32["flips"])
        worst = max(worst, max(d.values()))
        rows.append({"beat": b, **d, "spike_flips": flips})
    ok = worst <= C.TOL_1C
    log(f"  1c {device}: worst float32 vs float64 difference {worst:.2e} (tolerance {C.TOL_1C:g}) -> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "worst_abs_diff": worst, "device": str(device), "rows": rows}


def _grid_measures(ctx, build, n, g_of, in_scale_of, sets, seeds, T=C.BEAT_SAMPLES):
    out = {}
    for s in seeds:
        liq = build(s, n)
        m = measure_at_g(ctx, liq, g_of(liq), in_scale_of(liq), sets, s, T=T, with_twins=False)
        out[s] = {k: m[k] for k in ("rate_hz", "sep", "gen")}
    return out


def grid_effect_ok(test, ref):
    """For firing, separation and generalisation: |mean over seeds of (test - reference)| is no larger than the
    range (largest minus smallest) across the seeds on the reference grid."""
    res = {}
    for k in ("rate_hz", "sep", "gen"):
        diff = float(np.mean([test[s][k] - ref[s][k] for s in ref]))
        vals = [ref[s][k] for s in ref]
        rng = float(max(vals) - min(vals))
        res[k] = {"effect": diff, "reference_range": rng, "pass": bool(abs(diff) <= rng)}
    return all(v["pass"] for v in res.values()), res


def level1d(ctx, build_X, build_Y, gX, inX, gY, inY, sets, seeds, log=print, n_max=None):
    """X on the fine (reference) grid vs X on the paper's grid; Y at Maass's own weights on n = 1, 2, 3, ...
    vs the 28-sub-step reference; the smallest passing n is used for every candidate with factor 7."""
    nref = C.N_SUB_REFERENCE
    n_max = n_max or C.E1D_N_TRY_MAX
    xr = _grid_measures(ctx, build_X, nref, gX, inX, sets, seeds)
    xp = _grid_measures(ctx, build_X, 1, gX, inX, sets, seeds)
    x_ok, x_res = grid_effect_ok(xp, xr)
    log(f"  1d X: paper grid vs 28 sub-steps -> {'PASS' if x_ok else 'FAIL'} {x_res}")
    yr = _grid_measures(ctx, build_Y, nref, gY, inY, sets, seeds)
    tried, n_pass = [], None
    for n in range(1, n_max + 1):
        if n == nref:
            n_pass = nref
            break
        yn = _grid_measures(ctx, build_Y, n, gY, inY, sets, seeds)
        ok, res = grid_effect_ok(yn, yr)
        tried.append({"n": n, "pass": ok, "detail": res})
        log(f"  1d Y: n = {n} vs 28 -> {'PASS' if ok else 'fail'}")
        if ok:
            n_pass = n
            break
    return {"pass": bool(x_ok), "x_detail": x_res, "n": n_pass, "y_tried": tried, "y_reference": yr}


# ---------------------------------------------------------------- 1e: (d)'s Figure 2 vs our code at its settings
def _d_pairs(mod, seed, n_pairs):
    np.random.seed(seed)
    mod.N_PAIRS = n_pairs
    pairs = mod.collect_stimulus_pairs()
    pairs[0] = [[mod.generate_poisson(mod.DURATION / mod.ms, mod.STIMULUS_POISSON_RATE / mod.Hz / 1e3)] * 2
                for _ in range(n_pairs)]
    return pairs


def _to_bins(spike_times_ms, dt_ms, n_steps):
    x = np.zeros((n_steps, 1), np.uint8)
    k = ((np.asarray(spike_times_ms) + 1e-3 * dt_ms) / dt_ms).astype(int)
    k = k[(k >= 0) & (k < n_steps)]
    np.add.at(x, (k, 0), 1)
    return x


def ours_fig2_run(ctx, mod, seed, n_pairs, d_replacement="fixed"):
    """One run of (d)'s protocol with our code in the script's settings at its 0.1 ms step: a fresh liquid,
    fresh stimulus pairs, initial voltages redrawn for every simulation; returns {distance: mean curve}."""
    dt = float(mod.DT / mod.ms)
    n_steps = len(mod.TS)
    liq = build_maass_script(seed, n_sub=1, step_ms=dt, d_replacement=d_replacement)
    pairs = _d_pairs(mod, seed, n_pairs)
    curves = {}
    for dist, pp in pairs.items():
        xu = np.stack([_to_bins(a, dt, n_steps) for a, _ in pp])
        xv = np.stack([_to_bins(b, dt, n_steps) for _, b in pp])
        uu = start_voltages(liq, seed, range(len(pp)), 21 + int(dist * 100), random_start=True)
        uv = start_voltages(liq, seed, range(len(pp)), 61 + int(dist * 100), random_start=True)
        su = simulate(ctx, liq, 1.0, 1.0, xu, uu, n_steps, {"states_every": (0, n_steps)})["states_every"]
        sv = simulate(ctx, liq, 1.0, 1.0, xv, uv, n_steps, {"states_every": (0, n_steps)})["states_every"]
        curves[dist] = ((su.astype(np.float64) - sv) ** 2).mean(2).mean(0)    # (d): mean over neurons, then pairs
    return curves


def _d_sim_worker(args):
    import brian2_d_globals as G
    return G.mod.sim(G.net, args)


def d_fig2_run(mod, seed, n_pairs, processes=None):
    """One run of (d)'s own script (its functions, unchanged): build the liquid, collect pairs, simulate."""
    import multiprocessing as mp
    import sys, types
    import gc, brian2
    gc.collect(); brian2.start_scope()
    np.random.seed(seed)
    neurons = mod.get_neurons()
    N_exc = int(0.8 * len(neurons))
    exc, inh = neurons[:N_exc], neurons[N_exc:]
    exc.tau_refrac = 3 * mod.ms; exc.tau_stimulus = 3 * mod.ms
    inh.tau_refrac = 2 * mod.ms; inh.tau_stimulus = 6 * mod.ms
    second, nA = mod.second, mod.nA
    syn = [mod.get_synapses("ee_synapses", exc, exc, C=0.3, l=2, tau_I=3 * mod.ms, A=30 * nA, U=0.5, D=1.1 * second, F=0.05 * second, delay=1.5 * mod.ms),
           mod.get_synapses("ei_synapses", exc, inh, C=0.2, l=2, tau_I=3 * mod.ms, A=60 * nA, U=0.05, D=0.125 * second, F=1.2 * second, delay=0.8 * mod.ms),
           mod.get_synapses("ie_synapses", inh, exc, C=0.4, l=2, tau_I=6 * mod.ms, A=-19 * nA, U=0.25, D=0.7 * second, F=0.02 * second, delay=0.8 * mod.ms),
           mod.get_synapses("ii_synapses", inh, inh, C=0.1, l=2, tau_I=6 * mod.ms, A=-19 * nA, U=0.32, D=0.144 * second, F=0.06 * second, delay=0.8 * mod.ms)]
    stim = mod.SpikeGeneratorGroup(1, [], [] * mod.ms, name="stimulus")
    se = mod.Synapses(stim, exc, "A : ampere (shared, constant)", on_pre="I_stimulus += A"); se.connect(p=1); se.A = 18 * nA
    si = mod.Synapses(stim, inh, "A : ampere (shared, constant)", on_pre="I_stimulus += A"); si.connect(p=1); si.A = 9 * nA
    me = mod.SpikeMonitor(exc, name="spike_monitor_exc"); mi = mod.SpikeMonitor(inh, name="spike_monitor_inh")
    mod.defaultclock.dt = mod.DT
    net = mod.Network([neurons, *syn, se, si, stim, me, mi])
    net.store()
    mod.neurons = neurons                      # (d)'s sim() reads the global `neurons`
    pairs = _d_pairs(mod, seed, n_pairs)
    g = types.ModuleType("brian2_d_globals"); g.mod, g.net = mod, net
    sys.modules["brian2_d_globals"] = g
    curves = {}
    ctx = mp.get_context("fork")
    for dist, pp in pairs.items():                 # as in (d): a fresh pool for every input distance
        with ctx.Pool(processes) as pool:
            su = pool.map(_d_sim_worker, [p[0] for p in pp])
            sv = pool.map(_d_sim_worker, [p[1] for p in pp])
        curves[dist] = np.mean([mod.euclidian_distance(a, b) for a, b in zip(su, sv)], axis=0)
    return curves


def compare_fig2(ours_runs, d_runs, t_crit=C.E1E_T_CRIT, frac=C.E1E_FRAC):
    """Our mean curve vs the script's: within t_crit combined standard errors at >= frac of time points,
    for every input distance."""
    res = {}
    for dist in d_runs[0]:
        A = np.stack([r[dist] for r in ours_runs]); B = np.stack([r[dist] for r in d_runs])
        diff = A.mean(0) - B.mean(0)
        se = np.sqrt(A.var(0, ddof=1) / len(A) + B.var(0, ddof=1) / len(B))
        ok = np.where(se > 0, np.abs(diff) <= t_crit * se, np.abs(diff) <= 1e-12)
        res[float(dist)] = {"frac_within": float(ok.mean()), "pass": bool(ok.mean() >= frac),
                            "max_abs_diff": float(np.abs(diff).max())}
    return all(v["pass"] for v in res.values()), res


def level1e(ctx, script_path, seeds, n_pairs=200, processes=None, log=print, skip_brian2=False, d_replacement="fixed"):
    """5 runs each side (plan: each run of the script is 1,600 simulations). d_replacement: "fixed" patches (d)'s
    mean replacement of bad U, D, F draws so it works as intended (see brian2_ref.script_module); "as_written"
    runs (d) unchanged and our code with the same behaviour."""
    mod = brian2_ref.script_module(script_path, fix_replacement=(d_replacement == "fixed"))
    ours = []
    for s in seeds:
        t0 = time.time(); ours.append(ours_fig2_run(ctx, mod, s, n_pairs, d_replacement))
        log(f"  1e ours run seed {s}: {time.time() - t0:.0f} s")
    if skip_brian2:
        return {"pass": None, "ours": ours}
    d = []
    for s in seeds:
        t0 = time.time(); d.append(d_fig2_run(mod, 10_000 + s, n_pairs, processes))
        log(f"  1e Brian2 script run seed {s}: {time.time() - t0:.0f} s")
    ok, res = compare_fig2(ours, d)
    log(f"  1e: {'PASS' if ok else 'FAIL'} {res}")
    return {"pass": bool(ok), "by_distance": res, "ours_curves": ours, "d_curves": d, "d_replacement": d_replacement}
