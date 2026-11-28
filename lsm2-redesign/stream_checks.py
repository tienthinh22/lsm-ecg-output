"""CPU tests of the streaming machinery: segments vs one run, per-beat equivalence, stacking, chunking,
adaptation arithmetic, frozen thresholds."""
import sys, math, time
import numpy as np, torch
sys.path.insert(0, ".")   # run from the folder that holds lsm2/
from lsm2 import config as LC, liquid as LL, sim as LS, stream as SS

torch.set_num_threads(4)
DT = torch.float32
rng = np.random.default_rng(0)
FAIL = []

def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    if not ok:
        FAIL.append(name)

def x_liq(seed, adapt=None, C=40):
    liq = LL.build_candidate(frozenset(), seed, 1, C)
    if adapt:
        LL.add_adaptation(liq, seed, **adapt)
        liq.reset_to_thr = True
    return liq

def y_liq(seed, adapt=None, C=40, n_sub=4):
    liq = LL.build_candidate(LC.ALL_FACTORS, seed, n_sub, C, in_fraction=0.3)
    if adapt:
        LL.add_adaptation(liq, seed, **adapt)
    return liq

SAL = dict(beta=1.7, b0=0.03, tau_ms=(1.0, 300.0), fraction=0.6)

def spikes(B, T, C, p=0.02, seed=1):
    return (np.random.default_rng(seed).random((B, T, C)) < p).astype(np.uint8)

def enc_from(X):
    """enc(a, b) over a fixed (B, T, C) input (zero past its end)."""
    Xt = torch.as_tensor(X)
    def enc(a, b):
        out = torch.zeros(X.shape[0], b - a, X.shape[2], dtype=torch.uint8)
        e = min(b, X.shape[1])
        if e > a:
            out[:, :e - a] = Xt[:, a:e]
        return out
    return enc

# ---------------------------------------------------------------- 1. segments vs one run
for name, liq, g, sc in [("X", x_liq(1), 1.0, 0.3), ("X+adapt", x_liq(1, SAL), 1.0, 0.3),
                         ("Y", y_liq(2), None, 0.02), ("Y+adapt", y_liq(2, SAL), None, 0.02)]:
    g = liq.g_maass * 2 if g is None else g
    sm = LS.Simulator(liq, dtype=DT, g=g, in_scale=sc)
    B, T = 3, 1500
    X = spikes(B, T, 40, 0.03)
    starts = [np.sort(rng.choice(T - 640, 12, replace=False)) for _ in range(B)]
    streams = [{"length": T, "starts": s, "own_from": 0, "own_to": T} for s in starts]
    adapt = liq.adapt_c is not None
    m = SS.period(sm)
    one = SS.run_streams(sm, enc_from(X), streams, seg_len=T + (-T) % m, ab=adapt, counts=True)
    seg = SS.run_streams(sm, enc_from(X), streams, seg_len=m * 37, ab=adapt, counts=True)
    same = all(np.array_equal(a, b) for a, b in zip(one["windows"], seg["windows"]))
    check(f"{name}: segments of {m * 37} steps = one run (windows)", same)
    check(f"{name}: spike sums equal", np.array_equal(one["spike_sum"], seg["spike_sum"]))
    if adapt:
        check(f"{name}: adaptation sums equal (to rounding)", np.allclose(one["ab_sum"], seg["ab_sum"], rtol=1e-12, atol=1e-15))
    # against the existing simulator: the same input from the same start, window counts at each beat's start
    ref = sm.run(X, np.zeros((B, liq.N)), T, rec={"step_counts": True})["step_counts"].numpy()
    ok = True
    for i in range(B):
        for k, s0 in enumerate(starts[i]):
            w = ref[i, s0:s0 + 640].reshape(32, 20, -1).sum(1)
            ok &= np.array_equal(np.minimum(w, 255).astype(np.uint8), one["windows"][i][k])
    check(f"{name}: windows = existing simulator's step counts, cut by hand", ok)

# ---------------------------------------------------------------- 2. one beat: streaming = the per-beat run
liq = x_liq(3)
sm = LS.Simulator(liq, dtype=DT, g=1.0, in_scale=0.3)
B = 5
Xb = spikes(B, 577, 40, 0.05, seed=4)
Xb[:, 0] = 0                                     # config 11's encoder: step 0 is silent
per_beat = sm.run(Xb, np.zeros((B, liq.N)), 650, rec={"windows": (20, 32)})["windows"].numpy()
streams = [{"length": 650, "starts": np.array([0]), "own_from": 0, "own_to": 650} for _ in range(B)]
st = SS.run_streams(sm, enc_from(Xb), streams, seg_len=650 + 650 % 2)
check("one beat from rest: streaming windows = per-beat simulator's windows",
      all(np.array_equal(np.minimum(per_beat[i], 255).astype(np.uint8), st["windows"][i][0]) for i in range(B)))

# ---------------------------------------------------------------- 3. stacked vs alone
def stacked_vs_alone(name, liqs, g, sc, T=1200, C=40):
    B = 3
    X = spikes(B, T, C, 0.03, seed=7)
    starts = [np.sort(rng.choice(T - 640, 10, replace=False)) for _ in range(B)]
    streams = [{"length": T, "starts": s, "own_from": 0, "own_to": T} for s in starts]
    adapt = any(l.adapt_c is not None for l in liqs)
    alone = []
    for l, gi, si in zip(liqs, g, sc):
        smi = LS.Simulator(l, dtype=DT, g=gi, in_scale=si)
        alone.append(SS.run_streams(smi, enc_from(X), streams, seg_len=400, ab=adapt))
    S = SS.stack(liqs, g=g, in_scale=sc)
    sms = LS.Simulator(S, dtype=DT, g=1.0, in_scale=1.0)
    t0 = time.time()
    st = SS.run_streams(sms, enc_from(X), streams, seg_len=400, ab=adapt)
    offs = S.meta["offsets"]
    n_diff = n_tot = 0
    for j in range(len(liqs)):
        for i in range(B):
            a = alone[j]["windows"][i]; b = st["windows"][i][:, :, offs[j]:offs[j + 1]]
            n_diff += int((a != b).sum()); n_tot += a.size
    check(f"{name}: stacked = alone", n_diff == 0, f"{n_diff} of {n_tot} window counts differ")
    if adapt:
        ok = all(np.allclose(alone[j]["ab_sum"], st["ab_sum"][:, offs[j]:offs[j + 1]], rtol=1e-12, atol=1e-15)
                 for j in range(len(liqs)))
        check(f"{name}: stacked adaptation sums = alone", ok)

stacked_vs_alone("X, X+adapt, X no-rec", [x_liq(1), x_liq(2, SAL), SS.no_recurrence(x_liq(1))], [1.0, 1.0, 1.0], [0.3, 0.25, 0.3])
stacked_vs_alone("Y, Y+adapt (dynamic synapses)", [y_liq(2), y_liq(5, SAL)], [y_liq(2).g_maass * 2, y_liq(5).g_maass * 2], [0.02, 0.02])

# no_recurrence = Simulator(recurrence=False)
liq = x_liq(1)
X = spikes(2, 700, 40, 0.03, seed=9)
a = LS.Simulator(liq, dtype=DT, g=1.0, in_scale=0.3, recurrence=False).run(X, np.zeros((2, liq.N)), 700, rec={"step_counts": True})["step_counts"]
b = LS.Simulator(SS.no_recurrence(liq), dtype=DT, g=1.0, in_scale=0.3).run(X, np.zeros((2, liq.N)), 700, rec={"step_counts": True})["step_counts"]
check("no_recurrence(liq) = Simulator(recurrence=False)", torch.equal(a, b))

# ---------------------------------------------------------------- 4. chunks with warm-up vs one unbroken run
for name, liq, g, sc in [("X+adapt", x_liq(4, SAL), 1.0, 0.3), ("Y+adapt", y_liq(6, SAL), None, 0.02)]:
    g = liq.g_maass * 2 if g is None else g
    sm = LS.Simulator(liq, dtype=DT, g=g, in_scale=sc)
    R = 9000
    rec = spikes(1, R, 40, 0.03, seed=11)[0]
    starts = np.sort(rng.choice(np.arange(0, R - 300), 60, replace=False))
    def run(pieces, warm):
        ss = SS.plan_streams(R, starts, pieces=pieces, warmup=warm)
        def enc(a, b):
            out = np.zeros((len(ss), b - a, 40), np.uint8)
            for i, s in enumerate(ss):
                lo, hi = s["offset"] + a, min(s["offset"] + b, R)
                if hi > lo:
                    out[i, :hi - lo] = rec[lo:hi]
            return torch.as_tensor(out)
        o = SS.run_streams(sm, enc, ss, seg_len=SS.period(sm) * 250, ab=True)
        W = np.zeros((len(starts), 32, liq.N), np.uint8)
        for s, w in zip(ss, o["windows"]):
            W[s["beat_idx"]] = w
        return W, o, ss
    W1, o1, ss1 = run(1, 0)
    Wc, oc, ssc = run(4, 3000)
    covered = sorted(np.concatenate([s["beat_idx"] for s in ssc]).tolist()) == list(range(len(starts)))
    check(f"{name}: every beat in exactly one chunk", covered)
    nd = int((W1 != Wc).any((1, 2)).sum())
    if name.startswith("X"):
        check(f"{name}: chunked (4 pieces, 3000-step warm-up) = unbroken", nd == 0, f"{nd} of {len(starts)} beats differ")
    else:     # Maass's recurrent liquid need not forget its start (reported, as the plan says)
        print(f"INFO {name}: chunked vs unbroken: {nd} of {len(starts)} beats differ")
    own = sum(int(s["own_to"] - s["own_from"]) for s in ssc)
    check(f"{name}: own parts cover the record once", own == R and int(oc["own_steps"].sum()) == R)
    rel = np.abs(oc["ab_sum"].sum(0) - o1["ab_sum"][0]).max() / max(o1["ab_sum"].max(), 1e-12)
    if name.startswith("X"):
        check(f"{name}: chunked adaptation sums = unbroken (rel. {rel:.1e})", rel < 1e-6)
    check(f"{name}: end of record: tail beyond the record is silent input",
          all(s["length"] >= (s["starts"].max() + 640 if len(s["starts"]) else 0) for s in ss1))

# ---------------------------------------------------------------- 5. adaptation arithmetic: one neuron
def one_neuron(adapt_c=None, tau=None, thr_base=None, push=3.0):
    from lsm2.liquid import Liquid
    e = np.zeros(0, np.int64); z = np.zeros(0)
    return Liquid(N=1, dims=(1, 1, 1), is_exc=np.array([True]), pos=np.zeros((1, 3)), pre=e, post=e, typ=e, w=z,
                  U=z, D_ms=z, F_ms=z, Ubar=np.ones(4), Win=np.array([[push]]), in_neurons=np.array([True]), n_sub=1,
                  h_ms=LC.SAMPLE_MS, beta=np.array([0.0]), tap=np.zeros(1), reset_mode="subtract", reset_value=0.0,
                  start_lo=0.0, start_hi=0.0, rest=0.0, gap=np.array([1]), clamp=False, strict=False,
                  schedule="ours", syn_model="static", delay=np.ones(4, int), cur_a=None, in_a=None,
                  in_scale_fixed=1.0, adapt_c=None if adapt_c is None else np.array([adapt_c]),
                  adapt_tau_ms=None if tau is None else np.array([tau]),
                  thr_base=None if thr_base is None else np.array([thr_base]), reset_to_thr=True)

c = 1.0 * 1.7 / 0.03
for tau_ms in (1000.0, 6000.0):
    liq = one_neuron(c, tau_ms)
    sm = LS.Simulator(liq, device="cpu", dtype=torch.float64, g=1.0, in_scale=1.0)
    x = np.zeros((1, 3, 1), np.uint8); x[0, 0, 0] = 1
    o = sm.run(x, np.zeros((1, 1)), 3, rec={"keep_state": True, "step_counts": True})
    ab = float(o["state"]["ab"][0, 0]) * math.exp(2 * LC.SAMPLE_MS / tau_ms)      # back to just after the spike
    rise = c * ab
    want = 0.0567 / (tau_ms / 1000.0)
    check(f"one spike raises the threshold by 0.0567 / tau_a at tau_a = {tau_ms / 1000:g} s", abs(rise - want) / want < 0.01,
          f"{rise:.5f} vs {want:.5f}")
    check(f"  the spike happened once, at step 0", o["step_counts"][0, :, 0].tolist() == [1, 0, 0])

liq = one_neuron(c, 1000.0)
sm = LS.Simulator(liq, device="cpu", dtype=torch.float64, g=1.0, in_scale=1.0)
per = 36                                          # 36 steps of 2.778 ms = 100 ms: 10 Hz
T = per * 400
x = np.zeros((1, T, 1), np.uint8); x[0, ::per, 0] = 1
w = np.zeros((1, T)); w[0, T // 2:] = 1.0
o = sm.run(x, np.zeros((1, 1)), T, rec={"step_counts": True, "ab_weight": w, "keep_state": True})
thr_mean = 1.0 + c * float(o["ab_sum"][0, 0]) / w.sum()
check("steady 10 Hz: mean threshold near 1.57", abs(thr_mean - 1.567) < 0.01, f"{thr_mean:.4f}")
check("  one spike per input", int(o["step_counts"].sum()) == T // per)
# the reset subtracts the current threshold: just after a spike u = push - threshold
x = np.zeros((1, 2, 1), np.uint8); x[0, 0, 0] = 1
o = sm.run(x, np.zeros((1, 1)), 1, rec={"keep_state": True})
check("reset subtracts the current threshold (1 at rest): u = 3 - 1", abs(float(o["u_end"][0, 0]) - 2.0) < 1e-12)
# two spikes in a row at tau_a = 100 ms: the second crosses the raised threshold 1 + c (1 - rho) / h, and the reset
# subtracts exactly that (Salaj et al. 2021, Eqs. 3 and 5), not 1 and not the threshold after its own rise
for fast in (True, False):
    sm2 = LS.Simulator(one_neuron(c, 100.0), device="cpu", dtype=torch.float64, g=1.0, in_scale=1.0)
    x = np.zeros((1, 2, 1), np.uint8); x[0, :, 0] = 1
    o = sm2.run(x, np.zeros((1, 1)), 2, rec={"windows": (1, 2)}, fast=fast)
    thr2 = 1.0 + c * (1.0 - math.exp(-LC.SAMPLE_MS / 100.0)) / LC.SAMPLE_MS
    u2 = float(o["u_end"][0, 0])
    check(f"raised-threshold reset ({'fast' if fast else 'plain'} loop): u = 3 - {thr2:.4f}",
          o["windows"][0, :, 0].tolist() == [1, 1] and abs(u2 - (3.0 - thr2)) < 1e-12, f"u = {u2:.6f}")

# frozen threshold
for push, want in ((1.4, 0), (1.6, 1)):
    liq = one_neuron(thr_base=1.5, push=push)
    sm = LS.Simulator(liq, device="cpu", dtype=torch.float64, g=1.0, in_scale=1.0)
    x = np.zeros((1, 2, 1), np.uint8); x[0, 0, 0] = 1
    o = sm.run(x, np.zeros((1, 1)), 1, rec={"step_counts": True})
    check(f"frozen threshold 1.5: push {push} -> {want} spike", int(o["step_counts"].sum()) == want)
    if want:
        check("  frozen reset subtracts 1.5", abs(float(o["u_end"][0, 0]) - (push - 1.5)) < 1e-12)
# frozen_thresholds / frozen_copy
liq = x_liq(1, SAL)
thr = SS.frozen_thresholds(liq, np.full(liq.N, 0.01 * 1000), 1000)
check("frozen_thresholds: adaptive neurons at 1 + c * mean b, others 1",
      np.allclose(thr[liq.adapt_c > 0], 1 + 56.666666 * 0.01, rtol=1e-6) and np.all(thr[liq.adapt_c == 0] == 1))
fc = SS.frozen_copy(liq, thr)
check("frozen_copy: no adaptation, thresholds fixed, threshold reset kept", fc.adapt_c is None and fc.reset_to_thr
      and np.array_equal(fc.thr_base, thr))
# fixed thresholds of exactly 1 with the threshold reset run the per-neuron threshold code and must give the plain
# liquid's spikes (a strength of 0 would switch that code off, so it could not test it)
l0 = x_liq(1)
l1 = SS.frozen_copy(x_liq(1), np.ones(l0.N)); l1.reset_to_thr = True
X = spikes(2, 800, 40, 0.03, seed=12)
s1 = LS.Simulator(l1, dtype=DT, g=1.0, in_scale=0.3)
check("fixed thresholds of 1 take the per-neuron threshold path", s1.var_thr and not s1.adapt)
a = LS.Simulator(l0, dtype=DT, g=1.0, in_scale=0.3).run(X, np.zeros((2, l0.N)), 800, rec={"step_counts": True})["step_counts"]
b = s1.run(X, np.zeros((2, l1.N)), 800, rec={"step_counts": True})["step_counts"]
check("fixed thresholds of 1 = the plain liquid (identical spikes)", torch.equal(a, b))
for fast in (True, False):
    bb = s1.run(X, np.zeros((2, l1.N)), 800, rec={"windows": (20, 40)}, fast=fast)["windows"]
    aa = LS.Simulator(l0, dtype=DT, g=1.0, in_scale=0.3).run(X, np.zeros((2, l0.N)), 800, rec={"windows": (20, 40)}, fast=fast)["windows"]
    check(f"  same in the {'fast' if fast else 'plain'} loop", torch.equal(aa, bb))
# the adaptive branch runs: thresholds above 1 change the spikes
l2 = x_liq(1, dict(SAL, tau_ms=(100.0, 300.0)))
c2 = LS.Simulator(l2, dtype=DT, g=1.0, in_scale=0.3).run(X, np.zeros((2, l2.N)), 800, rec={"step_counts": True})["step_counts"]
check("adaptation on changes the spikes (fewer)", int(c2.sum()) < int(a.sum()), f"{int(c2.sum())} vs {int(a.sum())}")
try:
    LS.Simulator(LL.from_authors_model.__globals__["Liquid"] and LL.build_candidate(frozenset(), 1, 1, 4), dtype=DT, g=1.0, in_scale=0.3)
    q = x_liq(1, SAL); q.snn_quirk = True
    LS.Simulator(q, dtype=DT, g=1.0, in_scale=0.3)
    check("threshold reset refused with the snnTorch quirk", False)
except ValueError:
    check("threshold reset refused with the snnTorch quirk", True)

print("\nFAILED:" if FAIL else "\nALL PASSED", FAIL if FAIL else "")
