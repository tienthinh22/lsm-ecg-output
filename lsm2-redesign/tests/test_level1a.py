"""Level 1a: every switch against its closed-form solution, to 1e-9 in float64 (plan, Correctness criteria).
One neuron under every reset, gap and tap option; one synapse static, dynamic, with and without a decaying
current; every time conversion on both grids."""
import math
import numpy as np
import pytest
import torch

from lsm2 import config as C, liquid as L, sim as S

TOL = C.TOL_1A


def tiny(factors=frozenset(), n_sub=1, N=2, exc=(True, True), w=None, typ=0, gap=None, n_channels=1):
    """A 2-neuron liquid built from a candidate's settings: neuron 0 -> neuron 1 (one synapse of type `typ`)."""
    base = L.build_candidate(factors, 1, n_sub, n_channels)
    liq = base
    is_exc = np.array(exc, bool)
    s = L.candidate_settings(factors)
    h = C.SAMPLE_MS / n_sub
    tau = np.where(is_exc, s["tau_E_ms"], s["tau_I_ms"])
    liq.N, liq.is_exc, liq.pos = N, is_exc, np.zeros((N, 3))
    liq.beta = np.exp(-h / tau)
    liq.tap = C.TAP * (1 - liq.beta) if s["tap"] else np.zeros(N)
    if s["fine"]:
        liq.gap = np.where(is_exc, L.steps_of(3.0, h), L.steps_of(2.0, h)).astype(np.int64)
        liq.in_a = np.full(N, math.exp(-h / 3.0))
    else:
        liq.gap = np.full(N, L.steps_of(C.SAMPLE_MS, h), np.int64)
    if gap is not None:
        liq.gap = np.full(N, gap, np.int64)
    if w is None:
        liq.pre = liq.post = liq.typ = np.zeros(0, np.int64)
        liq.w = liq.U = liq.D_ms = liq.F_ms = np.zeros(0)
    else:
        liq.pre, liq.post, liq.typ = np.array([0]), np.array([1]), np.array([typ])
        liq.w = np.array([w], float)
        tname = C.TYPES[typ]
        liq.U = np.array([C.MAASS["U"][tname]])
        liq.D_ms = np.array([C.MAASS["D_s"][tname] * 1000])
        liq.F_ms = np.array([C.MAASS["F_s"][tname] * 1000])
    liq.Win = np.ones((n_channels, N))
    liq.in_neurons = np.ones(N, bool)
    return liq


def run(liq, x, T, u0, g=1.0, in_scale=1.0, **kw):
    sm = S.Simulator(liq, dtype=torch.float64, g=g, in_scale=in_scale)
    return sm.run(x, u0, T, trace=True, **kw)


def U_trace(o):
    return np.array(o["trace"]["u"])        # (sub-steps, B, N)


# ---------------------------------------------------------------- neuron
@pytest.mark.parametrize("n_sub", [1, 4, 28])
@pytest.mark.parametrize("factors", [frozenset(), frozenset({4})])
def test_leak_matches_exponential_decay(n_sub, factors):
    liq = tiny(factors, n_sub)
    T = 40
    o = run(liq, np.zeros((1, T, 1)), T, np.full((1, 2), 0.7))
    u = U_trace(o)[:, 0, :]
    s = L.candidate_settings(factors)
    t_ms = (np.arange(T * n_sub) + 1) * C.SAMPLE_MS / n_sub
    want = np.stack([0.7 * np.exp(-t_ms / s["tau_E_ms"])] * 2, 1)
    assert np.abs(u - want).max() < TOL


def test_x_leak_is_papers_beta_per_sample():
    liq = tiny(frozenset(), 1, exc=(True, False))
    o = run(liq, np.zeros((1, 10, 1)), 10, np.full((1, 2), 0.5))
    u = U_trace(o)[:, 0, :]
    assert np.abs(u[:, 0] - 0.5 * 0.95 ** np.arange(1, 11)).max() < TOL
    assert np.abs(u[:, 1] - 0.5 * 0.80 ** np.arange(1, 11)).max() < TOL


@pytest.mark.parametrize("n_sub", [1, 4, 28])
def test_tap_settles_at_0_9(n_sub):
    liq = tiny(frozenset({4, 5}), n_sub)
    T = 200
    o = run(liq, np.zeros((1, T, 1)), T, np.full((1, 2), 0.2))
    u = U_trace(o)[:, 0, 0]
    k = np.arange(1, T * n_sub + 1)
    want = 0.9 + (0.2 - 0.9) * np.exp(-k * (C.SAMPLE_MS / n_sub) / 30.0)
    assert np.abs(u - want).max() < TOL
    assert abs(u[-1] - 0.9) < 1e-6


def test_reset_subtract_constant_drive_first_spike_time():
    liq = tiny(frozenset(), 1)
    c = 0.08
    T = 60
    x = np.ones((1, T, 1))
    o = run(liq, x, T, np.zeros((1, 2)), in_scale=c, rec={"raster": True})
    beta = 0.95
    k1 = math.ceil(math.log(1 - (1 - beta) / c) / math.log(beta))   # smallest k with c (1 - beta^k)/(1 - beta) >= 1
    r = o["raster"][:, 0, 0].numpy()
    assert np.nonzero(r)[0][0] == k1 - 1                            # 0-based step index
    u = U_trace(o)[:, 0, 0]
    u_peak = c * (1 - beta ** k1) / (1 - beta)
    assert abs(u[k1 - 1] - (u_peak - 1)) < TOL                      # reset by subtracting 1


def test_reset_to_tap_and_clamp_during_gap():
    n = 4
    liq = tiny(frozenset({4, 5, 7}), n, exc=(True, False))
    T = 30
    x = np.zeros((1, T, 1)); x[0, 2, 0] = 1
    o = run(liq, x, T, np.full((1, 2), 0.9), in_scale=0.5, rec={"raster": True})
    r = o["raster"][:, 0, :].numpy()
    u = U_trace(o)[:, 0, :]
    for nrn, gap in ((0, 4), (1, 3)):          # 3 ms -> 4 sub-steps, 2 ms -> 3 sub-steps on n = 4
        ks = np.nonzero(r[:, nrn])[0]
        assert len(ks) >= 1
        k = ks[0]
        assert abs(u[k, nrn] - 0.9) < TOL                           # reset to 0.9
        assert np.abs(u[k + 1:k + gap, nrn] - 0.9).max() < TOL       # clamped for gap - 1 sub-steps
        assert abs(u[k + gap, nrn] - 0.9) > 1e-6 or not r[k + gap:, nrn].any()


@pytest.mark.parametrize("n_sub,gapE,gapI", [(4, 4, 3), (28, 30, 20)])
def test_minimum_gap_with_fine_timing(n_sub, gapE, gapI):
    liq = tiny(frozenset({4, 5, 7}), n_sub, exc=(True, False))
    T = 30
    x = np.ones((1, T, 1))
    o = run(liq, x, T, np.full((1, 2), 0.9), in_scale=50.0, rec={"raster": True})
    r = o["raster"][:, 0, :].numpy()
    for nrn, gap in ((0, gapE), (1, gapI)):
        isi = np.diff(np.nonzero(r[:, nrn])[0])
        assert isi.min() == gap
    h = C.SAMPLE_MS / n_sub
    assert abs(gapE * h - (2.78 if n_sub == 4 else 2.98)) < 0.01
    assert abs(gapI * h - (2.08 if n_sub == 4 else 1.98)) < 0.01


@pytest.mark.parametrize("n_sub", [1, 4, 28])
def test_x_gap_on_any_grid_without_clamp(n_sub):
    """X keeps its 2.78 ms gap on any grid: n - 1 blocked sub-steps, voltage still integrating."""
    liq = tiny(frozenset(), n_sub)
    T = 20
    x = np.ones((1, T, 1))
    o = run(liq, x, T, np.zeros((1, 2)), in_scale=3.0, rec={"raster": True})
    r = o["raster"][:, 0, 0].numpy()
    isi = np.diff(np.nonzero(r)[0])
    assert isi.min() == n_sub
    if n_sub > 1:
        u = U_trace(o)[:, 0, 0]
        k = np.nonzero(r)[0][0]
        assert u[k + 1] != u[k]                                       # not clamped


# ---------------------------------------------------------------- synapses
@pytest.mark.parametrize("n_sub", [1, 4])
def test_static_synapse_without_current(n_sub):
    """A spike at sub-step k adds w to its target in sub-step k + delay, all at once (X: delay = one sample)."""
    liq = tiny(frozenset(), n_sub, w=0.3)
    T = 15
    k0 = 1 * n_sub + (n_sub // 2)                                     # presynaptic spike, given open loop
    pre = np.zeros((T * n_sub, 1, 2), bool); pre[k0, 0, 0] = True
    o = run(liq, np.zeros((1, T, 1)), T, np.zeros((1, 2)), open_loop=pre)
    u1 = U_trace(o)[:, 0, 1]
    d = n_sub                                                         # one sample, in sub-steps
    beta = math.exp(-(C.SAMPLE_MS / n_sub) / L.candidate_settings(frozenset())["tau_E_ms"])
    m = np.arange(len(u1))
    want = np.where(m >= k0 + d, 0.3 * beta ** (m - (k0 + d)), 0.0)
    assert np.abs(u1 - want).max() < TOL


@pytest.mark.parametrize("n_sub", [4, 28])
@pytest.mark.parametrize("typ,tau_s", [(0, 3.0), (2, 6.0)])
def test_static_synapse_with_decaying_current(n_sub, typ, tau_s):
    """Fine timing: the push w arrives after its delay as a current decaying over tau_s, total = w.
    u_k = (1-a) w (beta^(m+1) - a^(m+1)) / (beta - a), m sub-steps after arrival."""
    exc = (typ < 2, True)
    liq = tiny(frozenset({4, 7}), n_sub, exc=exc, w=0.3 if typ < 2 else -0.3, typ=typ)
    h = C.SAMPLE_MS / n_sub
    liq.delay = np.array([L.steps_of(C.MAASS["delay_ms"][t], h) for t in C.TYPES])
    liq.cur_a = np.exp(-h / np.array([3.0, 6.0]))
    T = 40
    k0 = 5
    pre = np.zeros((T * n_sub, 1, 2), bool); pre[k0, 0, 0] = True
    o = run(liq, np.zeros((1, T, 1)), T, np.zeros((1, 2)), open_loop=pre)
    d = liq.delay[typ]
    a = math.exp(-h / tau_s)
    beta = math.exp(-h / 30.0)
    u1 = U_trace(o)[:, 0, 1]
    m = np.arange(len(u1)) - (k0 + d)
    w = liq.w[0]
    want = np.where(m >= 0, (1 - a) * w * (beta ** (m + 1) - a ** (m + 1)) / (beta - a), 0.0)
    assert np.abs(u1 - want).max() < TOL
    # total push equals w: sum of the current over all sub-steps
    I = np.array(o["trace"]["I1" if typ < 2 else "I2"])[:, 0, 1]
    assert abs(I.sum() - w * (1 - a ** (len(I) - (k0 + d)))) < TOL


@pytest.mark.parametrize("typ", [0, 1, 2, 3])
def test_dynamic_synapse_markram(typ):
    """Factor 8: per spike u <- u + U (1 - u); push = (w / U-bar) u R; R <- R - u R; between spikes R recovers
    towards 1 over D and u fades towards 0 over F (closed form between spikes)."""
    tname = C.TYPES[typ]
    liq = tiny(frozenset({4, 8}), 1, exc=(typ < 2, typ in (0, 2)), w=0.02 if typ < 2 else -0.02, typ=typ)
    liq.syn_model = "markram"
    T = 120
    rng = np.random.default_rng(typ)
    times = np.sort(rng.choice(np.arange(2, 100), 12, replace=False))
    pre = np.zeros((T, 1, 2), bool); pre[times, 0, 0] = True
    o = run(liq, np.zeros((1, T, 1)), T, np.zeros((1, 2)), open_loop=pre)
    ks = times
    U, D, F, Ub = liq.U[0], liq.D_ms[0], liq.F_ms[0], C.MAASS["U"][tname]
    R, uu, last = 1.0, 0.0, None
    pushes = []
    for k in ks + 1:                                               # arrival one sample later
        if last is not None:
            dt = (k - last) * C.SAMPLE_MS
            R = 1 - (1 - R) * math.exp(-dt / D)
            uu = uu * math.exp(-dt / F)
        else:
            R, uu = 1.0, 0.0
        uu = uu + U * (1 - uu)
        pushes.append(liq.w[0] / Ub * uu * R)
        R = R - uu * R
        last = k
    beta = math.exp(-C.SAMPLE_MS / 30.0)
    u1 = U_trace(o)[:, 0, 1]
    want = np.zeros(T)
    for k, p in zip(ks + 1, pushes):
        m = np.arange(T) - k
        want += np.where(m >= 0, p * beta ** np.maximum(m, 0), 0.0)
    assert np.abs(u1 - want).max() < TOL


def test_isolated_dynamic_push_equals_w_U_over_Ubar():
    liq = tiny(frozenset({4, 8}), 1, w=0.1, typ=0)
    liq.syn_model = "markram"
    liq.U = np.array([0.35])
    pre = np.zeros((10, 1, 2), bool); pre[1, 0, 0] = True
    o = run(liq, np.zeros((1, 10, 1)), 10, np.zeros((1, 2)), open_loop=pre)
    u1 = U_trace(o)[:, 0, 1]
    assert abs(u1[2] - 0.1 * 0.35 / 0.5) < TOL


# ---------------------------------------------------------------- time conversions on both grids
@pytest.mark.parametrize("n_sub", [1, 4, 28])
def test_total_input_push_same_on_every_grid(n_sub):
    """One input spike: without fine timing its push arrives at once in its own sample's first sub-step;
    with fine timing as a 3 ms current whose total equals the same push."""
    for factors in (frozenset({4}), frozenset({4, 7})):
        if 7 in factors and n_sub == 1:
            continue
        liq = tiny(factors, n_sub)
        T = 60
        x = np.zeros((1, T, 1)); x[0, 3, 0] = 1
        o = run(liq, x, T, np.zeros((1, 2)), in_scale=0.2)
        I = np.array(o["trace"]["I3"])[:, 0, 0]
        assert np.nonzero(I)[0][0] == 3 * n_sub                       # first sub-step of sample 3
        a = math.exp(-(C.SAMPLE_MS / n_sub) / 3.0)
        tail = a ** (len(I) - 3 * n_sub) if 7 in factors else 0.0
        assert abs(I.sum() - 0.2 * (1 - tail)) < TOL


def test_delay_conversions():
    for n, (dEE, dO) in ((4, (2, 1)), (28, (15, 8))):
        h = C.SAMPLE_MS / n
        assert L.steps_of(1.5, h) == dEE and L.steps_of(0.8, h) == dO
    liq = L.build_candidate(frozenset({7}), 1, 4, 10)
    assert list(liq.delay) == [2, 1, 1, 1]
    liq = L.build_candidate(frozenset(), 1, 4, 10)
    assert list(liq.delay) == [4, 4, 4, 4]            # X: one sample on any grid


def test_constants():
    assert abs(C.MAASS_EE_PUSH - 0.1) < 1e-12
    assert abs(C.MAASS_IN_PUSH["E"] - 0.12) < 1e-12 and abs(C.MAASS_IN_PUSH["I"] - 0.06) < 1e-12
    r = C.MAASS_RATIOS
    assert abs(r["EI"] - 0.2) < 1e-12 and abs(r["IE"] - 0.6333) < 1e-3 and abs(r["II"] - 0.8107) < 1e-3
    assert abs(C.TAP - 0.9) < 1e-12 and abs(C.I_TARGET - 0.045) < 1e-12
    assert abs(math.exp(-C.SAMPLE_MS / 30.0) - 0.912) < 1e-3       # plan: 30 ms is 0.912 per step


# ---------------------------------------------------------------- Brian2 schedule (script version)
def test_script_neuron_exact_solution_with_decaying_input():
    """(d)'s neuron: exact update with the input current decaying over 3 ms (E) or 6 ms (I):
    v(t) = J tau_s / (tau_s - tau_m) (e^(-t/tau_s) - e^(-t/tau_m)) after a jump J (no tap, no recurrence)."""
    liq = L.build_maass_script(1, n_sub=1, step_ms=0.1)
    liq.J_b = 0.0
    sel = [0, liq.N - 1]                                            # one E, one I neuron
    liq.N = 2; liq.is_exc = liq.is_exc[sel]; liq.beta = liq.beta[sel]; liq.kappa = liq.kappa[sel]
    liq.in_a = liq.in_a[sel]; liq.gap = liq.gap[sel]; liq.Win = np.array([[0.5, 0.5]])
    liq.pre = liq.post = liq.typ = np.zeros(0, np.int64); liq.w = liq.U = liq.D_ms = liq.F_ms = np.zeros(0)
    T = 300
    x = np.zeros((1, T, 1)); x[0, 0, 0] = 1
    o = run(liq, x, T, np.zeros((1, 2)))
    u = U_trace(o)[:, 0, :]
    t = np.arange(T) * 0.1                                         # the jump lands after step 0's update
    for i, ts in ((0, 3.0), (1, 6.0)):
        want = 0.5 * ts / (ts - 30.0) * (np.exp(-t / ts) - np.exp(-t / 30.0))
        assert np.abs(u[:, i] - want).max() < TOL


def test_script_three_state_synapse():
    """(d)'s synapse: dx/dt = (1 - x - y)/D, dy/dt = -y/tau; on a spike u += U (1-u), y += u x, x -= u x."""
    liq = L.build_maass_script(1, n_sub=1, step_ms=0.1)
    m = np.nonzero((liq.typ == 0) & (liq.pre != liq.post))[0][0]
    liq.N = 2; liq.is_exc = np.array([True, True]); liq.beta = liq.beta[:2]; liq.kappa = liq.kappa[:2]
    liq.in_a = liq.in_a[:2]; liq.gap = np.array([1, 1]); liq.Win = np.array([[50.0, 0.0]])
    liq.pre, liq.post, liq.typ = np.array([0]), np.array([1]), np.array([0])
    for a in ("w", "U", "D_ms", "F_ms"):
        setattr(liq, a, getattr(liq, a)[[m]])
    T = 400
    x = np.zeros((1, T, 1)); spikes_in = [5, 60, 90, 250]
    x[0, spikes_in, 0] = 1
    o = run(liq, x, T, np.full((1, 2), 0.0), rec={"raster": True})
    ks = np.nonzero(o["raster"][:, 0, 0].numpy())[0]
    U, D, F, tau = liq.U[0], liq.D_ms[0], liq.F_ms[0], 3.0
    xs, ys, us = 1.0, 0.0, 0.0
    h = 0.1
    want_y = []
    arrivals = set(int(k) + liq.delay[0] for k in ks)
    for k in range(T):
        us *= math.exp(-h / F)
        xs = xs * math.exp(-h / D) + 1 - math.exp(-h / D) + ys * tau * (math.exp(-h / tau) - math.exp(-h / D)) / (D - tau)
        ys = ys * math.exp(-h / tau)
        if k in arrivals:
            us += U * (1 - us); ys += us * xs; xs -= us * xs
        want_y.append(ys)
    got_y = np.array([s["y"][0, 0] for s in o["trace"]["syn"]])
    assert np.abs(got_y - np.array(want_y)).max() < TOL


@pytest.mark.parametrize("n_sub", [4, 28])
def test_dynamic_synapse_with_decaying_current(n_sub):
    """Factors 7 + 8: each push of the dynamic synapse arrives after its delay as a 3 ms current."""
    liq = tiny(frozenset({4, 7, 8}), n_sub, w=0.02, typ=0)
    liq.syn_model = "markram"
    h = C.SAMPLE_MS / n_sub
    liq.delay = np.array([L.steps_of(C.MAASS["delay_ms"][t], h) for t in C.TYPES])
    liq.cur_a = np.exp(-h / np.array([3.0, 6.0]))
    T = 60
    times = np.array([7, 30, 41, 90, 95, 150]) * (n_sub // 4)
    pre = np.zeros((T * n_sub, 1, 2), bool); pre[times, 0, 0] = True
    o = run(liq, np.zeros((1, T, 1)), T, np.zeros((1, 2)), open_loop=pre)
    U, D, F, Ub = liq.U[0], liq.D_ms[0], liq.F_ms[0], 0.5
    d = liq.delay[0]
    a, beta = math.exp(-h / 3.0), math.exp(-h / 30.0)
    R, uu, last, want = 1.0, 0.0, None, np.zeros(T * n_sub)
    for k in times + d:
        if last is not None:
            dt = (k - last) * h
            R = 1 - (1 - R) * math.exp(-dt / D); uu *= math.exp(-dt / F)
        uu += U * (1 - uu)
        p = liq.w[0] / Ub * uu * R
        R -= uu * R
        last = k
        m = np.arange(T * n_sub) - k
        want += np.where(m >= 0, (1 - a) * p * (beta ** (m + 1) - a ** (m + 1)) / (beta - a), 0.0)
    assert np.abs(U_trace(o)[:, 0, 1] - want).max() < TOL
