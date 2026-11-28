"""The decision logic on synthetic inputs: the gain rule (grid, extensions, fine pass, tie rule, band edge),
the ladder (forward, backward loop, confirmation and its retry), level 2 catching a wrong liquid, the effective
rank, the 1d and 1e pass rules, and the memory readout's statistics."""
import math
import numpy as np
import pytest
import torch

from lsm2 import config as C, gain, ladder, level2, liquid as L, measure as M, level1


def fake_measure(gap_of, rate_of, seeds=(1, 2, 3), noise=0.0):
    def m(g, s):
        return {"g": g, "abs_g": 0.02 * g, "rate_hz": rate_of(g), "frac_above_100hz": 0.0, "in_band": True,
                "sep": gap_of(g) + 10, "gen": 10.0, "gap": gap_of(g) + noise * (s - 2)}
    return m


def test_interior_peak_and_fine_pass():
    rec = gain.sweep(fake_measure(lambda g: -abs(math.log10(g + 1e-12) - math.log10(20)), lambda g: 5 + g * 0.01),
                     (1, 2, 3), log=lambda *a: None)
    assert rec["extensions"] == 0 and not rec["peak_at_edge"]
    fine = [g for g, t in rec["table"].items() if t["stage"] == "fine"]
    assert len(fine) == 6
    assert abs(rec["chosen_g"] - 20) / 20 < 0.3 and rec["gap_peak_inside_grid"]


def test_upper_edge_extends_at_most_three_times_then_fails():
    rec = gain.sweep(fake_measure(lambda g: math.log(g + 1e-9), lambda g: 10.0), (1, 2, 3), log=lambda *a: None)
    assert rec["extensions"] == 3 and rec["peak_at_edge"] and not rec["gap_peak_inside_grid"]
    assert max(rec["table"]) == pytest.approx(27000)


def test_band_limited_counts_as_inside_and_is_flagged():
    rec = gain.sweep(fake_measure(lambda g: math.log(g + 1e-9), lambda g: 1.0 * g), (1, 2, 3), log=lambda *a: None)
    # rate = g Hz: in band for 2 <= g <= 40; the gap keeps rising, so the band chooses g
    assert rec["extensions"] == 0 and rec["gap_peak_inside_grid"] and rec["band_limited"]
    assert rec["chosen_g"] <= 40 and rec["next_higher_gap_lower"] is False


def test_no_setting_in_band():
    rec = gain.sweep(fake_measure(lambda g: 1.0, lambda g: 0.5), (1, 2, 3), log=lambda *a: None)
    assert rec["chosen_g"] is None and rec["no_g_in_band"] and not rec["gap_peak_inside_grid"]


def test_tie_rule_takes_smallest_tied_g():
    gaps = {0.1: 1, 0.3: 2, 1: 5.0, 3: 5.05, 10: 4.98, 30: 2, 100: 1, 300: 0, 1000: -1}
    table = {g: {"gap": v, "gap_seeds": [v - 0.1, v, v + 0.1], "in_band": True} for g, v in gaps.items()}
    table[0.0] = {"gap": 9, "gap_seeds": [9, 9, 9], "in_band": True}        # g = 0 is never chosen
    ch, best = gain.choose(table)
    assert best == 3 and ch == 1


def test_g0_excluded_and_lower_edge_extends_down():
    rec = gain.sweep(fake_measure(lambda g: -g, lambda g: 10.0), (1, 2, 3), log=lambda *a: None)
    assert min(x for x in rec["table"] if x > 0) == pytest.approx(0.1 / 27)


# ---------------------------------------------------------------- ladder logic
class FakeLadder(ladder.Ladder):
    def __init__(self, passes, confirm_passes):
        super().__init__(None, type("D", (), {"name": "search"})(), type("D", (), {"name": "confirm"})(), 4, 1.0,
                         log=lambda *a: None)
        self.passes, self.confirm_passes, self.calls = passes, confirm_passes, []

    def evaluate(self, factors, data=None, seeds=None, frozen=None, y_grid=None, tag=None):
        f = frozenset(factors)
        self.calls.append((ladder.key_of(f), frozen is not None))
        ok = self.confirm_passes(f, seeds) if frozen is not None else self.passes(f)
        return {"pass": ok, "g_rule": 1.0, "in_scale": 0.1, "n_sub": self.n_sub_for(f, y_grid)}


def test_ladder_forward_backward_minimal_set():
    lad = FakeLadder(lambda f: {3, 5} <= f, lambda f, s: True)
    out = lad.run()
    assert out["final"] == [3, 5]
    fw = [k for d, k, _ in lad.path if d == "forward"]
    assert fw == ["1", "1+2", "1+2+3", "1+2+3+4", "1+2+3+4+5"]           # stops at the first pass


def test_ladder_x_passes():
    out = FakeLadder(lambda f: True, lambda f, s: True).run()
    assert out["final"] == []


def test_ladder_none_passes():
    out = FakeLadder(lambda f: False, lambda f, s: True).run()
    assert out["final"] is None and "no forward candidate" in out["stop"]


def test_ladder_confirmation_retry_once():
    # {3, 5} passes the search but fails confirmation on the first seeds: forward resumes at {1..6}, the backward
    # loop runs again (and removes 6 again, since the search passes without it), then fresh seeds confirm it
    lad = FakeLadder(lambda f: {3, 5} <= f, lambda f, s: tuple(s) == C.SEEDS_RETRY)
    out = lad.run()
    assert out["final"] == [3, 5] and out["attempts"] == 2
    assert ("1+2+3+4+5+6", False) in lad.calls


def test_ladder_retry_keeps_a_needed_factor():
    # the search needs 6 only once 5 is gone; confirmation fails first, passes on the retry seeds
    lad = FakeLadder(lambda f: {3} <= f and (5 in f or 6 in f), lambda f, s: tuple(s) == C.SEEDS_RETRY)
    out = lad.run()
    assert out["final"] in ([3, 5], [3, 6]) and out["attempts"] == 2


def test_ladder_confirmation_fails_twice_stops():
    out = FakeLadder(lambda f: {3, 5} <= f, lambda f, s: False).run()
    assert out["final"] is None and "twice" in out["stop"]


def test_y_always_on_28_and_factor7_on_n():
    lad = FakeLadder(lambda f: False, lambda f, s: False)
    assert lad.n_sub_for(C.ALL_FACTORS) == 28 and lad.n_sub_for({7}) == 4 and lad.n_sub_for({1, 2}) == 1


# ---------------------------------------------------------------- level 2 must reject a wrong liquid
def test_level2_catches_wrong_lambda_and_sign():
    liq = L.build_candidate(frozenset({1}), 1, 1, 50)
    assert level2.check(liq)["pass"]
    wrong = L.build_candidate(frozenset(), 1, 1, 50)          # built with lambda 0.8, claims factor 1
    wrong.factors = frozenset({1})
    assert not level2.check(wrong)["pass"]
    bad = L.build_candidate(frozenset({1}), 1, 1, 50)
    bad.w = np.abs(bad.w)                                       # inhibitory synapses made excitatory
    assert not level2.check(bad)["pass"]


# ---------------------------------------------------------------- measures
def test_effective_rank_basics():
    rng = np.random.default_rng(0)
    assert M.effective_rank(np.zeros((10, 5))) == 0.0
    iso = rng.standard_normal((5000, 4))
    assert abs(M.effective_rank(iso) - 4) < 0.1
    one = np.outer(rng.standard_normal(100), rng.standard_normal(6))
    assert abs(M.effective_rank(one) - 1) < 1e-6
    shifted = iso + 100.0                                       # centring: the mean state does not count
    assert abs(M.effective_rank(shifted) - M.effective_rank(iso)) < 1e-9


def test_jitter_keeps_spikes_and_moves_them():
    rng = np.random.default_rng(0)
    s = np.zeros((577, 3), np.uint8); s[100, 0] = 1; s[300, 2] = 1; s[576, 1] = 1
    j = M.jitter_spikes(s, rng)
    assert j.sum() <= 3 and j.sum() >= 2


def test_grid_effect_rule():
    ref = {1: {"rate_hz": 5, "sep": 10, "gen": 3}, 2: {"rate_hz": 6, "sep": 12, "gen": 4}, 3: {"rate_hz": 7, "sep": 11, "gen": 5}}
    ok, _ = level1.grid_effect_ok({s: {k: v + 1.9 for k, v in r.items()} for s, r in ref.items()}, ref)
    assert ok
    bad, _ = level1.grid_effect_ok({s: {k: v + 2.1 for k, v in r.items()} for s, r in ref.items()}, ref)
    assert not bad


def test_fig2_compare_rule():
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 500)
    A = [{0.1: base + 0.01 * rng.standard_normal(500)} for _ in range(5)]
    B = [{0.1: base + 0.01 * rng.standard_normal(500)} for _ in range(5)]
    ok, res = level1.compare_fig2(A, B)
    assert ok and res[0.1]["frac_within"] > 0.95
    Bs = [{0.1: base + 0.2 + 0.01 * rng.standard_normal(500)} for _ in range(5)]
    assert not level1.compare_fig2(A, Bs)[0]


def test_memory_readout_finds_a_delay_line():
    """A liquid replaced by a perfect delay line must give R^2 ~ 1 at its lag and ~ 0 elsewhere."""
    rng = np.random.default_rng(0)
    n, T, lagS = 40, 300, 36
    y = rng.poisson(2.0, (n, T)).astype(float)
    st = [M._Stats(3, "cpu") for _ in range(2)]
    for i in range(n):
        X = np.stack([np.roll(y[i], lagS), np.roll(y[i], 5), rng.standard_normal(T)], 1)[lagS:]
        st[i % 2].add(torch.tensor(X), torch.tensor(y[i][:T - lagS]))
    w, mu, my = M._fit(st[0], 1e-6)
    r2 = 1 - M._ss_res(st[1], w, mu, my) / M._ss_tot(st[1])
    assert r2 > 0.99


AUTHORS = __import__("os").environ.get("AUTHORS_REPO_DIR", "/home/claude/ext/authors_repo")


@pytest.mark.skipif(not __import__("os").path.isdir(AUTHORS), reason="authors' code not available")
def test_x_mode_reproduces_authors_snntorch_loop_with_their_reset():
    """X built by (a)'s class: with x_update = authors_code the new simulator gives the authors' window counts
    exactly; with (a)'s Eqs. 17-19 (default) it differs only through the extra subtraction."""
    import sys, importlib.util
    sys.path.insert(0, AUTHORS)
    from src.utils.reproducibility import seed_everything
    spec = importlib.util.spec_from_file_location("a03t", AUTHORS + "/scripts/03_run_24_configurations_cv.py")
    A03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A03)
    G = torch.distributions.Gamma
    seed_everything(2026, deterministic=False)
    torch.distributions.Gamma = lambda c, s: G(c, 1.0 / s)
    try:
        m = A03.LiquidStateMachine3D(200, 0.1, 0.045 / (1.224 * 0.1), 0.95, 0.8, (7, 7, 7), 0.8, 1.0, 650,
                                     {"exc_exc": 0.3, "exc_inh": 0.2, "inh_exc": 0.4, "inh_inh": 0.1}, 0.8, 1,
                                     np.array([0.02, 0.025, 0.05, 0.04]), 1, 0.01, "cpu")
    finally:
        torch.distributions.Gamma = G
    rng = np.random.default_rng(0)
    X = np.clip(0.45 + 0.4 * np.sin(np.linspace(0, 6, 577))[None] * rng.uniform(0.5, 1, (20, 1))
                + 0.004 * rng.standard_normal((20, 577)).cumsum(1), 0, 1)
    thr = np.linspace(0, 1, 200); ab = X[:, :, None] >= thr
    x = np.zeros(ab.shape, np.float32); x[:, 1:] = ab[:, 1:] != ab[:, :-1]
    xt = torch.from_numpy(x)
    with torch.no_grad():
        spk = torch.zeros(20, 343); mem = torch.zeros(20, 343); ref = torch.zeros(20, 32, 343)
        for t in range(650):
            xi = xt[:, t] if t < 577 else torch.zeros(20, 200)
            spk, mem = m.lif(xi @ m.input_rsv_weights + spk @ m.rsv_weights, mem)
            if t < 640:
                ref[:, t // 20] += spk
    liq = L.from_authors_model(m, snn_quirk=True)
    o = __import__("lsm2.sim", fromlist=["Simulator"]).Simulator(liq, dtype=torch.float32).run(
        x, np.zeros((20, 343)), 650, rec={"windows": (20, 32)})
    assert int((o["windows"].numpy() != ref.numpy()).sum()) == 0
