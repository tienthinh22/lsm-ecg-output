"""Checks behind the Y vs Y-off notebook's own claims."""
import numpy as np
import torch

from lsm2 import config as C, gain, liquid as L, sim as S


def test_cube_structure_does_not_depend_on_input_channels():
    """y_cube: the Poisson (1 channel) and ECG (200 channels) versions of a seed are the same recurrent liquid."""
    a = L.build_candidate(C.ALL_FACTORS, 3, 2, 1)
    b = L.build_candidate(C.ALL_FACTORS, 3, 2, 200)
    assert a.dims == (15, 3, 3) and a.N == 135
    for k in ("pre", "post", "typ", "w", "U", "D_ms", "F_ms", "is_exc", "in_neurons"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.g_maass == b.g_maass and a.in_scale_fixed == b.in_scale_fixed


def test_y_off_equals_zero_gain():
    """Y-off (recurrence=False) is the same run as g = 0: same input, no recurrent push."""
    liq = L.build_candidate(C.ALL_FACTORS, 1, 2, 5)
    rng = np.random.default_rng(0)
    x = (rng.random((3, 40, 5)) < 0.3).astype(np.uint8)
    u0 = S.start_voltages(liq, 1, range(3), 4)
    rec = {"windows": (4, 10), "states_at": [39]}
    a = S.Simulator(liq, dtype=torch.float64, g=liq.g_maass, in_scale=liq.in_scale_fixed, recurrence=False).run(x, u0, 40, rec=rec)
    b = S.Simulator(liq, dtype=torch.float64, g=0.0, in_scale=liq.in_scale_fixed).run(x, u0, 40, rec=rec)
    assert torch.equal(a["windows"], b["windows"]) and torch.allclose(a["states_at"], b["states_at"])


def test_recurrence_changes_the_run():
    """With the same input and start, Y and Y-off differ (the recurrent synapses do something)."""
    liq = L.build_candidate(C.ALL_FACTORS, 1, 2, 5)
    rng = np.random.default_rng(1)
    x = (rng.random((4, 200, 5)) < 0.3).astype(np.uint8)
    u0 = S.start_voltages(liq, 1, range(4), 4)
    rec = {"neuron_counts": (0, 200)}
    on = S.Simulator(liq, dtype=torch.float64, g=liq.g_maass, in_scale=liq.in_scale_fixed).run(x, u0, 200, rec=rec)
    off = S.Simulator(liq, dtype=torch.float64, g=liq.g_maass, in_scale=liq.in_scale_fixed, recurrence=False).run(x, u0, 200, rec=rec)
    assert not torch.equal(on["neuron_counts"], off["neuron_counts"])


def test_gain_rule_without_band():
    """With the band switched off (the notebook's setting), the rule takes the largest mean gap; ties go to the
    smallest g within the best setting's seed range."""
    old = C.BAND_HZ, C.HIGH_FRAC
    C.BAND_HZ, C.HIGH_FRAC = (0.0, float("inf")), 1.0
    try:
        def row(gap, rate, seeds):
            m = {"gap": gap, "rate_hz": rate, "frac_above_100hz": 0.0, "gap_seeds": seeds}
            m["in_band"] = True
            return m
        t = {0.0: row(1.0, 0.1, [1, 1]), 1.0: row(3.0, 0.5, [2.9, 3.1]), 3.0: row(5.0, 80.0, [4.0, 6.0]),
             10.0: row(4.5, 150.0, [4.4, 4.6])}
        chosen, best = gain.choose(t)
        assert best == 3.0 and chosen == 3.0
        t[2.0] = row(4.2, 30.0, [4.1, 4.3])         # inside [4, 6]: tied, and smaller
        chosen, best = gain.choose(t)
        assert best == 3.0 and chosen == 2.0
    finally:
        C.BAND_HZ, C.HIGH_FRAC = old


def test_fast_path_is_the_plain_loop():
    """The fast path (and so its CUDA-graph replay, which runs the same code) gives bit-identical results to the
    plain loop, for X, the column, the cube, with and without recurrence."""
    rng = np.random.default_rng(5)
    for f, n_sub in ((frozenset(), 1), (C.ALL_FACTORS, 6), (C.ALL_FACTORS - {11}, 4)):
        liq = L.build_candidate(f, 2, n_sub, 12)
        x = (rng.random((5, 30, 12)) < 0.2).astype(np.uint8)
        u0 = S.start_voltages(liq, 2, range(5), 4, random_start=True)
        g = liq.g_maass * 2 if liq.g_maass else 1.0
        rec = {"windows": (5, 6), "states_at": [0, 29], "neuron_counts": (0, 30), "states_every": (3, 20)}
        for on in (True, False):
            sm = S.Simulator(liq, dtype=torch.float64, g=g, in_scale=liq.in_scale_fixed or 0.3, recurrence=on)
            a, b = sm.run(x, u0, 34, rec=rec, fast=False), sm.run(x, u0, 34, rec=rec, fast=True)
            for k in ("windows", "states_at", "neuron_counts", "states_every", "u_end"):
                assert torch.equal(a[k], b[k]), (f, on, k)


def test_input_fraction():
    """in_fraction=None is (c)'s 30% (unchanged); 1.0 wires every neuron, and the 30% keep their weights."""
    a = L.build_candidate(C.ALL_FACTORS, 4, 2, 10)
    b = L.build_candidate(C.ALL_FACTORS, 4, 2, 10, in_fraction=0.3)
    c = L.build_candidate(C.ALL_FACTORS, 4, 2, 10, in_fraction=1.0)
    assert a.in_neurons.sum() == L.round_half_up(0.3 * a.N) and np.array_equal(a.Win, b.Win)
    assert c.in_neurons.all() and (c.Win > 0).all()
    assert np.array_equal(c.Win[:, a.in_neurons], a.Win[:, a.in_neurons])
    for k in ("pre", "post", "w", "U", "D_ms", "F_ms"):
        assert np.array_equal(getattr(a, k), getattr(c, k))


def _adaptive(seed=3, n_sub=4, nch=8, frac=None):
    liq = L.build_candidate(C.ALL_FACTORS, seed, n_sub, nch, in_fraction=1.0)
    return L.add_adaptation(liq, seed) if frac is None else L.add_adaptation(liq, seed, fraction=frac)


def test_add_adaptation_parameters():
    """Bellec et al. 2018 Section 5: beta 1.6, b0 0.03, tau_a uniform 1..1000 ms, 40% of the neurons; the rise is
    scaled to Maass's rest-to-threshold gap (0.1)."""
    liq = _adaptive()
    on = liq.adapt_c > 0
    assert on.sum() == L.round_half_up(0.4 * liq.N)
    assert np.allclose(liq.adapt_c[on], 0.1 * 1.6 / 0.03)
    assert np.all((liq.adapt_tau_ms[on] >= 1) & (liq.adapt_tau_ms[on] <= 1000)) and np.all(np.isinf(liq.adapt_tau_ms[~on]))
    base = L.build_candidate(C.ALL_FACTORS, 3, 4, 8, in_fraction=1.0)
    for k in ("pre", "post", "w", "U", "Win", "beta"):
        assert np.array_equal(getattr(base, k), getattr(liq, k))


def test_adaptation_matches_reference_equations():
    """No recurrence (g = 0): every neuron is independent, so a direct numpy implementation of the neuron (input
    current, leak, background, refractory clamp, threshold 1 + c b, reset to 0.9, b <- rho b + (1 - rho) spike / h)
    must give the same spikes."""
    liq = _adaptive(n_sub=3)
    rng = np.random.default_rng(1)
    B, T = 3, 120
    x = (rng.random((B, T, 8)) < 0.4).astype(np.uint8)
    u0 = S.start_voltages(liq, 3, range(B), 4)
    sc = 0.05
    out = S.Simulator(liq, dtype=torch.float64, g=0.0, in_scale=sc).run(x, u0, T, rec={"neuron_counts": (0, T)}, fast=False)
    h = liq.h_ms
    Win = liq.Win * sc
    u = u0.copy(); Iin = np.zeros_like(u); ref = np.zeros_like(u, dtype=int); b = np.zeros_like(u)
    rho = np.exp(-h / liq.adapt_tau_ms); cnt = np.zeros_like(u)
    for t in range(T):
        ip_t = x[:, t, :].astype(float) @ Win
        for j in range(liq.n_sub):
            blocked = ref > 0
            ip = ip_t if j == 0 else 0.0
            Iin = liq.in_a * Iin + (1 - liq.in_a) * ip
            un = liq.beta * u + Iin + liq.tap
            un = np.where(blocked, u, un)
            spk = (un >= 1.0 + liq.adapt_c * b) & ~blocked
            u = np.where(spk, liq.reset_value, un)
            b = rho * b + (1 - rho) / h * spk
            ref = np.where(spk, liq.gap - 1, np.maximum(ref - 1, 0))
            cnt += spk
    assert np.array_equal(out["neuron_counts"].numpy(), cnt.astype(np.int64))
    assert cnt.sum() > 0


def test_adaptation_slows_firing_and_leaves_others():
    """Adaptive neurons fire less than without adaptation under the same drive; with no recurrence the other
    neurons are untouched."""
    liq = _adaptive(n_sub=2)
    plain = L.build_candidate(C.ALL_FACTORS, 3, 2, 8, in_fraction=1.0)
    rng = np.random.default_rng(2)
    x = (rng.random((4, 300, 8)) < 0.5).astype(np.uint8)
    u0 = S.start_voltages(liq, 3, range(4), 4)
    rec = {"neuron_counts": (0, 300)}
    a = S.Simulator(liq, dtype=torch.float64, g=0.0, in_scale=0.05).run(x, u0, 300, rec=rec)["neuron_counts"].numpy()
    p = S.Simulator(plain, dtype=torch.float64, g=0.0, in_scale=0.05).run(x, u0, 300, rec=rec)["neuron_counts"].numpy()
    on = liq.adapt_c > 0
    assert np.array_equal(a[:, ~on], p[:, ~on])
    assert a[:, on].sum() < p[:, on].sum()


def test_adaptation_fast_equals_plain():
    liq = _adaptive(n_sub=5)
    rng = np.random.default_rng(4)
    x = (rng.random((5, 40, 8)) < 0.3).astype(np.uint8)
    u0 = S.start_voltages(liq, 3, range(5), 4, random_start=True)
    rec = {"windows": (5, 8), "states_at": [10, 39], "neuron_counts": (0, 40), "states_every": (0, 40)}
    for on in (True, False):
        sm = S.Simulator(liq, dtype=torch.float64, g=liq.g_maass * 2, in_scale=0.05, recurrence=on)
        a, b = sm.run(x, u0, 44, rec=rec, fast=False), sm.run(x, u0, 44, rec=rec, fast=True)
        for k in ("windows", "states_at", "neuron_counts", "states_every", "u_end"):
            assert torch.equal(a[k], b[k]), (on, k)
