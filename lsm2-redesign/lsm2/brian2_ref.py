"""Level 1b and 1e references in Brian2.

brian2_ours: Brian2 runs our per-sub-step update as explicit code, in the same order of operations (level 1b).
  Brian2 step k+1 computes our sub-step k (the same one-step shift the replication's Brian2 check used):
    'start'      neurons: currents, integrate, threshold, reset (our steps 2-4), using the pushes that arrived
    'groups'     synapses: recover / fade over one sub-step (our step 1, first half), for our next sub-step
    'thresholds' our sub-step's spikes are emitted (closed loop) or replayed from a recording (open loop)
    'synapses'   arrivals for our next sub-step: dynamic synapses update u, push, R (our step 1, second half)
  A delay of d sub-steps is a Brian2 delay of d - 1 steps; input spikes of sample t are delivered in Brian2
  step t*n so they act in our sub-step t*n.
"""
import numpy as np


def _b2():
    import brian2 as b2
    b2.BrianLogger.log_level_error()
    try:
        b2.prefs.codegen.target = "cython"
        b2.start_scope()
        b2.NeuronGroup(1, "v:1").v = 0        # probe that cython compiles here
    except Exception:
        b2.prefs.codegen.target = "numpy"
    return b2


def brian2_ours(liq, x, u0, T, g=1.0, in_scale=1.0, open_loop=None, target="numpy"):
    """One beat. x: (T_in, C) input spikes per input step; u0: (N,); open_loop: (T*n, N) presynaptic spikes
    (None = closed loop). Returns traces: u, I1, I2, I3 (T*n, N), synapse states R, u at the end of each input
    step (T, S), and the spike raster (T*n, N)."""
    import brian2 as b2
    b2.BrianLogger.log_level_error()
    b2.prefs.codegen.target = target
    b2.start_scope()
    n, N, h = liq.n_sub, liq.N, liq.h_ms * 1e-3
    b2.defaultclock.dt = h * b2.second
    fine = liq.cur_a is not None
    dyn = liq.syn_model == "markram"
    eqs = """
    u : 1
    ref : 1
    IE : 1
    II : 1
    Iin : 1
    accE : 1
    accI : 1
    accIn : 1
    lastE : 1
    lastI : 1
    lastIn : 1
    spiked : 1
    beta : 1 (constant)
    tap : 1 (constant)
    gap : 1 (constant)
    a_in : 1 (constant)
    """
    G = b2.NeuronGroup(N, eqs, threshold="spiked > 0.5", reset="", method="euler", name="liquid")
    G.u = u0
    G.beta, G.tap, G.gap = liq.beta, liq.tap, liq.gap.astype(float)
    G.a_in = liq.in_a if liq.in_a is not None else 0.0
    aE = float(liq.cur_a[0]) if fine else 0.0
    aI = float(liq.cur_a[1]) if fine else 0.0
    reset = ("u_new - spk" if liq.reset_mode == "subtract" else f"spk*{liq.reset_value!r} + (1 - spk)*u_new")
    thr = "u_new - int(u > 1.0) > 1.0" if liq.snn_quirk else ("u_new > 1.0" if liq.strict else "u_new >= 1.0")
    code = f"""
    active = int(t > 0.5*dt)
    blocked = int(ref > 0.5)
    IE = active*({aE!r}*IE + (1 - {aE!r})*accE) + (1 - active)*IE
    II = active*({aI!r}*II + (1 - {aI!r})*accI) + (1 - active)*II
    Iin = active*(a_in*Iin + (1 - a_in)*accIn) + (1 - active)*Iin
    total = {'IE + II + Iin' if fine else 'accE + accI + accIn'}
    u_new = beta*u + total + tap
    {'u_new = blocked*u + (1 - blocked)*u_new' if liq.clamp else ''}
    spk = int({thr})*(1 - blocked)*active
    u_next = {reset}
    u = active*u_next + (1 - active)*u
    ref = active*(spk*(gap - 1) + (1 - spk)*clip(ref - 1, 0, 1e9)) + (1 - active)*ref
    lastE = {'IE' if fine else 'accE'}
    lastI = {'II' if fine else 'accI'}
    lastIn = {'Iin' if fine else 'accIn'}
    accE = 0
    accI = 0
    accIn = 0
    spiked = spk
    """
    G.run_regularly(code, when="start", name="update")
    objs = [G]
    # input: channel spikes of sample t delivered at Brian2 step t*n
    xs = np.asarray(x)
    tt, cc = np.nonzero(xs[:min(len(xs), T)])
    Win = liq.Win * in_scale
    if len(tt):
        inp = b2.SpikeGeneratorGroup(xs.shape[1], cc, (tt * n) * h * b2.second, name="inp")
        ci, nj = np.nonzero(Win)
        Sin = b2.Synapses(inp, G, "w : 1 (constant)", on_pre="accIn_post += w", name="sin")
        Sin.connect(i=ci, j=nj); Sin.w = Win[ci, nj]
        objs += [inp, Sin]
    # presynaptic source
    if open_loop is not None:
        kk, ii = np.nonzero(np.asarray(open_loop))
        src = b2.SpikeGeneratorGroup(N, ii, (kk + 1) * h * b2.second, name="src")
        objs.append(src)
    else:
        src = G
    syn_mons = []
    if liq.S:
        for ti in range(4):
            m = np.nonzero(liq.typ == ti)[0]
            if not len(m):
                continue
            acc = "accE_post" if ti < 2 else "accI_post"
            d = int(liq.delay[ti])
            if dyn:
                model = """
                wg : 1 (constant)
                U : 1 (constant)
                eD : 1 (constant)
                eF : 1 (constant)
                R : 1
                q : 1
                """
                on_pre = f"""
                q = q + U*(1 - q)
                {acc} += wg*q*R
                R = R - q*R
                """
            else:
                model, on_pre = "wg : 1 (constant)", f"{acc} += wg"
            S = b2.Synapses(src, G, model, on_pre=on_pre, delay=(d - 1) * h * b2.second, name=f"syn{ti}")
            S.connect(i=liq.pre[m], j=liq.post[m])
            if dyn:
                S.wg = liq.w[m] * g / liq.Ubar[ti]
                S.U, S.eD, S.eF = liq.U[m], np.exp(-liq.h_ms / liq.D_ms[m]), np.exp(-liq.h_ms / liq.F_ms[m])
                S.R, S.q = 1.0, 0.0
                S.run_regularly("R = 1 - (1 - R)*eD\nq = q*eF", when="groups", name=f"decay{ti}")
                mon = b2.StateMonitor(S, ["R", "q"], record=True, when="end", name=f"smon{ti}")
                syn_mons.append((m, mon))
                objs.append(mon)
            else:
                S.wg = liq.w[m] * g
            objs.append(S)
    M = b2.StateMonitor(G, ["u", "lastE", "lastI", "lastIn"], record=True, when="end", name="mon")
    spm = b2.SpikeMonitor(G, name="spm")
    objs += [M, spm]
    net = b2.Network(*objs)
    net.run((T * n + 1) * h * b2.second)
    out = {k: np.asarray(getattr(M, v))[:, 1:].T for k, v in (("u", "u"), ("I1", "lastE"), ("I2", "lastI"), ("I3", "lastIn"))}
    r = np.zeros((T * n, N), bool)
    k = np.round(np.asarray(spm.t / b2.second) / h).astype(int) - 1
    ok = (k >= 0) & (k < T * n)
    r[k[ok], np.asarray(spm.i)[ok]] = True
    out["raster"] = r
    if dyn and syn_mons:
        R = np.zeros((T, liq.S)); q = np.zeros((T, liq.S))
        last = np.arange(T) * n + n - 1          # Brian2 step k ('end') holds our sub-step k's synapse state
        for m, mon in syn_mons:
            R[:, m] = np.asarray(mon.R).T[last]
            q[:, m] = np.asarray(mon.q).T[last]
        out["R"], out["q"] = R, q
    return out


# ---------------------------------------------------------------- level 1e: (d)'s script, reused unchanged
D_REPLACEMENT_LINES = {
    "synapses.U[synapses.U < 0] = U": "synapses.U[np.nonzero(np.asarray(synapses.U) < 0)[0]] = U",
    "synapses.D[synapses.D / ms <= 0] = D": "synapses.D[np.nonzero(np.asarray(synapses.D / ms) <= 0)[0]] = D",
    "synapses.F[synapses.F / ms <= 0] = F": "synapses.F[np.nonzero(np.asarray(synapses.F / ms) <= 0)[0]] = F",
}


def script_module(path, fix_replacement=True):
    """Load (d)'s maass2002.py (its main block is guarded), so its own functions build and run the liquid.

    (d) means to replace bad U, D, F draws by the mean, but Brian2 reads a boolean array index as the integer
    indices 0 and 1: as written, only synapses 0 and 1 of each group get the mean and every negative draw stays
    (about 2.3% of synapses each for U, D and F; a negative D or F makes x or u grow without bound).
    fix_replacement=True patches those three lines so they do what (d) intends (and what the plan describes)."""
    import types
    src = open(path).read()
    if fix_replacement:
        for a, b in D_REPLACEMENT_LINES.items():
            if a not in src:
                raise RuntimeError(f"(d)'s script changed: line not found: {a}")
            src = src.replace(a, b)
    mod = types.ModuleType("maass2002_d")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod
