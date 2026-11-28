"""Label-free measurements: batching beats through the simulator, firing, the rank scores, the starting-state
rule, twin runs (level 3) and the memory test (level 4)."""
import math
import numpy as np
import torch

from . import config as C
from .sim import Simulator, start_voltages


class Ctx:
    """Where and how to simulate: device, precision and a fixed batch size (plan, determinism)."""
    def __init__(self, device="cpu", dtype=torch.float32, batch=C.CODE["sim_batch"], log=print):
        self.device, self.dtype, self.batch, self.log = device, dtype, batch, log


def simulate(ctx, liq, g, in_scale, spikes, u0, T, rec, recurrence=True):
    """spikes: (n, T_in, C) numpy input spikes; u0: (n, N). Returns the simulator outputs as numpy, batched."""
    sm = Simulator(liq, device=ctx.device, dtype=ctx.dtype, g=g, in_scale=in_scale, recurrence=recurrence)
    parts = []
    for s in range(0, len(spikes), ctx.batch):
        o = sm.run(spikes[s:s + ctx.batch], u0[s:s + ctx.batch], T, rec=rec)
        parts.append({k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in o.items() if k != "u_end"})
    out = {}
    for k in parts[0]:
        if k == "states_at_times":
            out[k] = parts[0][k]
        elif k == "raster":
            out[k] = np.concatenate([p[k] for p in parts], 1)
        else:
            out[k] = np.concatenate([p[k] for p in parts], 0)
    return out


# ---------------------------------------------------------------- rank scores
def effective_rank(M):
    """Roy & Vetterli 2007: exp of the entropy of the normalised singular values, of the centred matrix
    (float64). 0 for an all-zero matrix."""
    M = np.asarray(M, np.float64)
    M = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def plain_rank(M):
    """(e)'s plain rank (reported only)."""
    M = np.asarray(M, np.float64)
    return int(np.linalg.matrix_rank(M)) if M.any() else 0


def cutoff_rank(M, frac=0.01):
    """An earlier draft's measure: singular values above 1% of the largest, uncentred (reported only)."""
    s = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    return int((s > frac * s[0]).sum()) if len(s) and s[0] > 0 else 0


# ---------------------------------------------------------------- input sets
def jitter_spikes(spikes, rng, sd_ms=C.JITTER_MS, step_ms=C.SAMPLE_MS):
    """Each encoded spike moved by a Gaussian amount with SD 4 ms, rounded to the nearest sample; spikes moved
    outside the beat are dropped (plan, Generalisation score). spikes: (T, C) counts."""
    T, Cn = spikes.shape
    out = np.zeros((T, Cn), np.uint8)
    tt, cc = np.nonzero(spikes)
    reps = spikes[tt, cc].astype(int)
    tt, cc = np.repeat(tt, reps), np.repeat(cc, reps)
    nt = tt + np.rint(rng.normal(0, sd_ms / step_ms, len(tt))).astype(int)
    ok = (nt >= 0) & (nt < T)
    np.add.at(out, (nt[ok], cc[ok]), 1)
    return out


def twin_spike(spikes, rng, t=C.R_SAMPLE):
    """The same input with one extra spike at sample t (the R peak), on a channel silent there."""
    s = spikes.copy()
    free = np.nonzero(s[t] == 0)[0]
    ch = rng.choice(free) if len(free) else rng.integers(s.shape[1])
    s[t, ch] = min(255, int(s[t, ch]) + 1)
    return s


def level3_sets(sep_spikes, gen_base, seed_sets=0, n_twin=None):
    """sep_spikes: (500, T, C) spikes of 500 different inputs; gen_base: (4, T, C) the 4 inputs that are jittered.
    Returns the separation, generalisation and twin input sets (fixed for every candidate and g)."""
    n_twin = C.CODE["twin_beats"] if n_twin is None else n_twin
    gen = np.stack([jitter_spikes(gen_base[i], np.random.default_rng([seed_sets, 31, i, c]))
                    for i in range(len(gen_base)) for c in range(C.GEN_COPIES)])
    twin = np.stack([twin_spike(sep_spikes[i], np.random.default_rng([seed_sets, 37, i])) for i in range(min(n_twin, len(sep_spikes)))])
    return {"sep": sep_spikes, "gen": gen, "twin": twin}


# ---------------------------------------------------------------- level 3 at one g
def firing_stats(neuron_counts, n_samples, step_ms=C.SAMPLE_MS):
    """Mean rate over all neurons (silent ones included) and the share of neurons above 100 Hz.
    neuron_counts: (beats, N) spike counts over n_samples input steps."""
    dur_s = n_samples * step_ms / 1000.0
    per_neuron = neuron_counts.mean(0) / dur_s
    return float(per_neuron.mean()), float((per_neuron > C.HIGH_HZ).mean())


def measure_at_g(ctx, liq, g, in_scale, sets, seed, T=C.BEAT_SAMPLES, read=C.STATE_READ_SAMPLE,
                 with_start_test=False, with_twins=True, extra_ranks=False):
    """Every level-3 measure at one g for one liquid. Starting voltages: uniform between the resting level and
    the threshold for every candidate, drawn from (seed, stream, input index)."""
    nb = len(sets["sep"])
    u_sep = start_voltages(liq, seed, range(nb), 1, random_start=True)
    rec = {"states_at": [read, C.START_TEST_SAMPLE], "neuron_counts": (0, T)}
    o = simulate(ctx, liq, g, in_scale, sets["sep"], u_sep, T, rec)
    rate, high = firing_stats(o["neuron_counts"], T, liq.h_ms * liq.n_sub)
    sep_states = o["states_at"][:, 0]
    u_gen = start_voltages(liq, seed, range(len(sets["gen"])), 2, random_start=True)
    og = simulate(ctx, liq, g, in_scale, sets["gen"], u_gen, T, {"states_at": [read]})
    gen_states = og["states_at"][:, 0]
    res = {"g": float(g), "abs_g": liq.ee_mean_push(g), "rate_hz": rate, "frac_above_100hz": high,
           "in_band": bool(C.BAND_HZ[0] <= rate <= C.BAND_HZ[1] and high <= C.HIGH_FRAC),
           "sep": effective_rank(sep_states), "gen": effective_rank(gen_states)}
    res["gap"] = res["sep"] - res["gen"]
    if extra_ranks:
        res.update({"sep_plain": plain_rank(sep_states), "gen_plain": plain_rank(gen_states),
                    "sep_cut1": cutoff_rank(sep_states), "gen_cut1": cutoff_rank(gen_states)})
        res["gap_plain"], res["gap_cut1"] = res["sep_plain"] - res["gen_plain"], res["sep_cut1"] - res["gen_cut1"]
    if with_twins and len(sets["twin"]):
        nt = len(sets["twin"])
        u_t = u_sep[:nt]
        rt = {"states_every": (0, T)}
        a = simulate(ctx, liq, g, in_scale, sets["sep"][:nt], u_t, T, rt)["states_every"]
        b = simulate(ctx, liq, g, in_scale, sets["twin"], u_t, T, rt)["states_every"]
        dist = np.sqrt(((a - b) ** 2).sum(-1)).mean(0)          # mean over beats of the state distance, per sample
        e0, e1 = C.CODE["twin_early"]; l0, l1 = C.CODE["twin_late"]
        early, late = dist[e0:e1 + 1].mean(), dist[l0:l1 + 1].mean()
        res["twin_early"], res["twin_late"] = float(early), float(late)
        res["twin_ratio"] = float(late / early) if early > 0 else float("nan")
    if with_start_test:
        u2 = start_voltages(liq, seed, range(nb), 3, random_start=True)
        o2 = simulate(ctx, liq, g, in_scale, sets["sep"], u2, T, {"states_at": [C.START_TEST_SAMPLE]})
        s1, s2 = o["states_at"][:, 1].astype(np.float64), o2["states_at"][:, 0].astype(np.float64)
        same = float(np.sqrt(((s1 - s2) ** 2).sum(1)).mean())
        sq = (s1 ** 2).sum(1)
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * s1 @ s1.T, 0)
        iu = np.triu_indices(len(s1), 1)
        diff = float(np.sqrt(D2[iu]).mean())
        res.update({"start_same": same, "start_diff": diff,
                    "start_rule_pass": bool(same < diff or (same == 0 and diff == 0))})
    return res


# ---------------------------------------------------------------- level 4: memory
def surrogate_walks(steps_pool, n_seq, T, seed):
    """Random walks on [0, 1], reflected at the edges, step sizes drawn from ECG's own sample-to-sample steps
    with random signs (plan, level 4). Starting levels uniform on [0, 1]."""
    rng = np.random.default_rng([seed, 41])
    X = np.empty((n_seq, T))
    for i in range(n_seq):
        st = rng.choice(steps_pool, T - 1) * rng.choice([-1.0, 1.0], T - 1)
        v = rng.uniform(0, 1)
        X[i, 0] = v
        for t in range(1, T):
            v = v + st[t - 1]
            while v < 0 or v > 1:
                v = -v if v < 0 else 2 - v
            X[i, t] = v
    return X


class _Stats:
    """Sufficient statistics of (x, y) rows for ridge with an intercept: n, sum x, sum y, sum y^2, X'X, X'y."""
    def __init__(self, N, dev):
        z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=dev)
        self.n, self.sx, self.sy, self.syy, self.G, self.b = 0, z(N), z(()), z(()), z(N, N), z(N)

    def add(self, X, y):
        self.n += X.shape[0]; self.sx += X.sum(0); self.sy += y.sum(); self.syy += (y * y).sum()
        self.G += X.T @ X; self.b += X.T @ y

    def __add__(self, o):
        r = _Stats.__new__(_Stats)
        r.n = self.n + o.n
        for k in ("sx", "sy", "syy", "G", "b"):
            setattr(r, k, getattr(self, k) + getattr(o, k))
        return r


def _fit(st, alpha):
    mu, my = st.sx / st.n, st.sy / st.n
    Gc = st.G - st.n * torch.outer(mu, mu)
    bc = st.b - st.n * mu * my
    w = torch.linalg.solve(Gc + alpha * torch.eye(len(mu), dtype=Gc.dtype, device=Gc.device), bc)
    return w, mu, my


def _ss_res(st, w, mu, my):
    """sum over st's rows of (y - my - (x - mu)'w)^2, from its statistics."""
    c = my - mu @ w                                     # prediction = x'w + c
    return float(st.syy - 2 * (w @ st.b) - 2 * c * st.sy + w @ st.G @ w + 2 * c * (w @ st.sx) + st.n * c * c)


def _ss_tot(st, mean=None):
    m = st.sy / st.n if mean is None else mean
    return float(st.syy - 2 * m * st.sy + st.n * m * m)


def memory_curve(ctx, liq, g, in_scale, seq_spikes, target_counts, seed, T=C.BEAT_SAMPLES,
                 lags_ms=C.MEMORY_T_MS, warmup=C.MEMORY_WARMUP, recurrence=True, batch=100,
                 alphas=C.CODE["ridge_alphas"], k=C.CODE["ridge_inner_folds"]):
    """R^2 of reconstructing target[t - T] from the 30 ms-filtered state at t, for each lag: a ridge readout
    trained on the first half of the sequences (penalty by inner CV, folds grouped by sequence) and scored on
    the second half, at every sample after the warm-up. Statistics are accumulated batch by batch."""
    n = len(seq_spikes)
    step_ms = liq.h_ms * liq.n_sub
    half = n // 2
    fold = np.full(n, k)                                      # group k = test sequences
    fold[:half] = np.random.default_rng(0).permutation(half) % k
    lagL = {lag: int(round(lag / step_ms)) for lag in lags_ms}
    dev = ctx.device
    stats = {lag: [_Stats(liq.N, dev) for _ in range(k + 1)] for lag in lags_ms}
    sm = Simulator(liq, device=dev, dtype=ctx.dtype, g=g, in_scale=in_scale, recurrence=recurrence)
    u0 = start_voltages(liq, seed, range(n), 4)               # the liquid's own starting rule, as on ECG
    counts = 0.0
    for s0 in range(0, n, batch):
        o = sm.run(seq_spikes[s0:s0 + batch], u0[s0:s0 + batch], T, rec={"states_every": (0, T), "neuron_counts": (0, T)})
        Z = o["states_every"].to(torch.float64)               # (b, T, N)
        counts += float(o["neuron_counts"].sum())
        tg = torch.as_tensor(np.asarray(target_counts[s0:s0 + batch]), dtype=torch.float64, device=dev)
        for lag, L in lagL.items():
            t0 = max(warmup, L)
            for bi in range(Z.shape[0]):
                X = Z[bi, t0:T]
                y = tg[bi, t0 - L:T - L]
                stats[lag][fold[s0 + bi]].add(X, y)
        del o, Z
    out = {"rate_hz": counts / (n * liq.N * T * step_ms / 1000.0)}
    for lag in lags_ms:
        st = stats[lag]
        train_all = st[0]
        for f in range(1, k):
            train_all = train_all + st[f]
        best, best_a = -np.inf, alphas[0]
        for a in alphas:
            res = tot = 0.0
            for f in range(k):
                tr = None
                for f2 in range(k):
                    if f2 != f:
                        tr = st[f2] if tr is None else tr + st[f2]
                w, mu, my = _fit(tr, a)
                res += _ss_res(st[f], w, mu, my)
                tot += _ss_tot(st[f], float(train_all.sy / train_all.n))
            r2 = 1 - res / tot if tot > 0 else float("nan")
            if r2 > best:
                best, best_a = r2, a
        w, mu, my = _fit(train_all, best_a)
        tot = _ss_tot(st[k])
        out[f"r2_{lag}"] = float(1 - _ss_res(st[k], w, mu, my) / tot) if tot > 0 else float("nan")
        out[f"alpha_{lag}"] = float(best_a)
    return out
