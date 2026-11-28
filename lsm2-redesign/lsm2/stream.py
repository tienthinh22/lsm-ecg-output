"""Streaming whole records through a liquid (the continuous-liquid plan).

* stack(): several liquids side by side as one block-diagonal liquid (no synapse between them, the same input
  channels), so one simulator run serves them all. Each liquid's own recurrent gain and input size are folded into
  its weights (the stacked liquid runs at g = 1, input size 1); per-neuron settings are concatenated; the settings
  the simulator holds as scalars must agree.
* plan_streams(): the streams of one record: the whole record, or pieces that each start `warmup` samples early
  (from rest) and run `tail` samples past their own part. Every beat goes to the piece whose own part holds its
  window start.
* run_streams(): every stream side by side in segments of `seg_len` input steps, the state carried from segment to
  segment (Simulator.run with state / t0), each beat's `nwin` windows of `win` steps cut from the per-step spike
  counts; optionally the sum of the adaptation variable over each stream's own steps (for the frozen controls).
* frozen_thresholds() / frozen_copy(): the time-averaged threshold of every neuron (1 + c * mean b) and the liquid
  with its thresholds fixed there.
"""
from dataclasses import replace
import numpy as np
import torch

from .liquid import Liquid

_SAME = ("n_sub", "h_ms", "reset_mode", "reset_value", "start_lo", "start_hi", "rest", "clamp", "strict", "schedule",
         "syn_model", "snn_quirk", "in_split", "J_b")


def no_recurrence(liq):
    """The same liquid without its internal synapses (same neurons, input wiring and adaptation)."""
    e = np.zeros(0, np.int64)
    z = np.zeros(0)
    return replace(liq, pre=e, post=e.copy(), typ=e.copy(), w=z, U=z.copy(), D_ms=z.copy(), F_ms=z.copy(),
                   meta={**liq.meta, "recurrence": False})


def stack(liqs, g=None, in_scale=None):
    """One block-diagonal liquid from several. g / in_scale: per liquid (default 1 / the liquid's in_scale_fixed);
    they are multiplied into its synapse and input weights. Run the result with Simulator(g=1, in_scale=1)."""
    k = len(liqs)
    g = [1.0] * k if g is None else [float(v) for v in g]
    in_scale = [float(l.in_scale_fixed) for l in liqs] if in_scale is None else [float(v) for v in in_scale]
    L0 = liqs[0]
    for l in liqs[1:]:
        for f in _SAME:
            if getattr(l, f) != getattr(L0, f):
                raise ValueError(f"stack: liquids differ in {f}: {getattr(l, f)!r} vs {getattr(L0, f)!r}")
        if not np.array_equal(l.delay, L0.delay) or not np.array_equal(l.Ubar, L0.Ubar):
            raise ValueError("stack: liquids differ in delays or U-bar")
        if l.Win.shape[0] != L0.Win.shape[0]:
            raise ValueError("stack: liquids differ in input channels")
        if (l.cur_a is None) != (L0.cur_a is None) or (l.cur_a is not None and not np.array_equal(l.cur_a, L0.cur_a)):
            raise ValueError("stack: liquids differ in current decays")
        if (l.in_a is None) != (L0.in_a is None) or (l.kappa is None) != (L0.kappa is None):
            raise ValueError("stack: liquids differ in input-current settings")
    if L0.schedule != "ours":
        raise ValueError("stack: only the 'ours' schedule")
    # the threshold reset (subtract the current threshold) is the plain reset for a neuron whose threshold stays 1,
    # so liquids without it can join a stack that has it, if none of their thresholds move
    rtt = any(l.reset_to_thr for l in liqs)
    for l in liqs:
        if L0.reset_mode == "subtract" and not l.reset_to_thr and (l.thr_base is not None or (l.adapt_c is not None and np.any(np.asarray(l.adapt_c) > 0))):
            raise ValueError("stack: a liquid with moving or fixed thresholds but the plain reset")
    offs = np.cumsum([0] + [l.N for l in liqs])
    cat = lambda f: np.concatenate([getattr(l, f) for l in liqs])
    pre = np.concatenate([l.pre + o for l, o in zip(liqs, offs)]).astype(np.int64)
    post = np.concatenate([l.post + o for l, o in zip(liqs, offs)]).astype(np.int64)
    typ = cat("typ")
    w = np.concatenate([l.w * gi for l, gi in zip(liqs, g)])
    order = np.lexsort((post, pre, typ))               # by type, then pre, then post (E-presynaptic first), stable
    adapt = any(l.adapt_c is not None for l in liqs)
    thr = any(l.thr_base is not None for l in liqs)
    N = int(offs[-1])
    return Liquid(
        N=N, dims=(N, 1, 1), is_exc=cat("is_exc"), pos=np.concatenate([l.pos for l in liqs]), pre=pre[order],
        post=post[order], typ=typ[order], w=w[order], U=cat("U")[order], D_ms=cat("D_ms")[order],
        F_ms=cat("F_ms")[order], Ubar=L0.Ubar, Win=np.concatenate([l.Win * s for l, s in zip(liqs, in_scale)], 1),
        in_neurons=cat("in_neurons"), n_sub=L0.n_sub, h_ms=L0.h_ms, beta=cat("beta"), tap=cat("tap"),
        reset_mode=L0.reset_mode, reset_value=L0.reset_value, start_lo=L0.start_lo, start_hi=L0.start_hi,
        rest=L0.rest, gap=cat("gap"), clamp=L0.clamp, strict=L0.strict, schedule=L0.schedule, syn_model=L0.syn_model,
        delay=L0.delay, cur_a=L0.cur_a, in_a=None if L0.in_a is None else cat("in_a"), J_b=L0.J_b,
        kappa=None if L0.kappa is None else cat("kappa"), syn_tau_ms=L0.syn_tau_ms, factors=L0.factors,
        snn_quirk=L0.snn_quirk, g_maass=None, in_scale_fixed=1.0, in_split=L0.in_split,
        adapt_c=np.concatenate([l.adapt_c if l.adapt_c is not None else np.zeros(l.N) for l in liqs]) if adapt else None,
        adapt_tau_ms=np.concatenate([l.adapt_tau_ms if l.adapt_tau_ms is not None else np.full(l.N, np.inf)
                                     for l in liqs]) if adapt else None,
        thr_base=np.concatenate([l.thr_base if l.thr_base is not None else np.ones(l.N) for l in liqs]) if thr else None,
        reset_to_thr=rtt,
        meta={"kind": "stacked", "parts": [dict(l.meta) for l in liqs], "offsets": offs.tolist(), "g": g,
              "in_scale": in_scale},
    )


def plan_streams(rec_len, starts, pieces=1, warmup=0, tail=640):
    """The streams of one record of rec_len samples whose beats have window starts `starts` (record samples,
    ascending). pieces = 1: the whole record, from its first sample to `tail` samples past the last beat's window
    start (at least to the record's end). pieces > 1: equal own parts; each piece starts `warmup` samples before
    its own part (not before the record's start) and runs to `tail` samples past its own part.
    Returns dicts: offset (record sample of the stream's step 0), length (steps), beat_idx (into starts),
    starts (relative to the offset), own_from / own_to (the stream's own record samples, relative)."""
    starts = np.asarray(starts, np.int64)
    if len(starts) > 1 and np.any(np.diff(starts) < 0):
        raise ValueError("plan_streams: starts must be ascending")
    bounds = np.linspace(0, rec_len, pieces + 1).round().astype(np.int64)
    out = []
    for c0, c1 in zip(bounds[:-1], bounds[1:]):
        off = max(0, int(c0) - int(warmup))
        idx = np.nonzero((starts >= c0) & (starts < c1))[0]
        length = int(c1) + (int(tail) if pieces > 1 else 0) - off
        if len(idx):
            length = max(length, int(starts[idx].max()) + int(tail) - off)
        out.append({"offset": off, "length": length, "beat_idx": idx, "starts": starts[idx] - off,
                    "own_from": int(c0) - off, "own_to": int(c1) - off})
    return out


def period(sm):
    """Input steps after which the simulator's spike-history ring repeats (segment lengths must be multiples)."""
    return -(-sm.L // sm.liq.n_sub)


@torch.no_grad()
def run_streams(sm, enc, streams, seg_len, u0=None, nwin=32, win=20, ab=False, counts=False, sink=None,
                beat_chunk=None, progress=None):
    """All streams side by side. enc(a, b) -> (B, b - a, C) uint8 input spikes of steps [a, b) of every stream
    (zero past a stream's end). streams: dicts with length, starts (ascending), own_from, own_to. u0: (B, N)
    starting voltages (default 0). sink(i, ks, wc): receives the windows of stream i's beats ks (indices into its
    starts) as wc (len(ks), nwin, N) uint8; default: kept and returned as out["windows"][i].
    Returns {"windows" (no sink), "own_steps": (B,), "ab_sum": (B, N) the adaptation variable summed over each
    stream's own steps (ab=True; zeros for a liquid that does not adapt), "spike_sum": (B, N) spikes over the own steps (counts=True), "steps": T}."""
    dev, N, B = sm.device, sm.N, len(streams)
    span = nwin * win
    m = period(sm)
    if seg_len % m:
        raise ValueError(f"seg_len must be a multiple of {m}")
    T = max(int(s["length"]) for s in streams)
    starts = [np.asarray(s["starts"], np.int64) for s in streams]
    for s, st in zip(streams, starts):
        if len(st) and (st.min() < 0 or st.max() + span > s["length"] or np.any(np.diff(st) < 0)):
            raise ValueError("a beat's windows run outside its stream, or its starts are not ascending")
    kept = None
    if sink is None:
        kept = [np.zeros((len(st), nwin, N), np.uint8) for st in starts]
        def sink(i, ks, wc):
            kept[i][ks] = wc
    if beat_chunk is None:
        beat_chunk = max(1, int(2 ** 28 // (span * N)))     # gather buffer of about 256 MB
    nxt = [0] * B
    own_from = np.array([s["own_from"] for s in streams]); own_to = np.array([s["own_to"] for s in streams])
    ab_sum = torch.zeros(B, N, dtype=torch.float64, device=dev) if ab else None
    spike_sum = torch.zeros(B, N, dtype=torch.float64, device=dev) if counts else None
    own_steps = np.zeros(B, np.int64)
    if u0 is None:
        u0 = np.zeros((B, N))
    state, buf, buf0 = None, None, 0
    ar = torch.arange(span, device=dev)
    for a in range(0, T, seg_len):
        b = min(a + seg_len, T)
        x = enc(a, b)
        t = np.arange(a, b)
        w = (t[None, :] >= own_from[:, None]) & (t[None, :] < own_to[:, None])
        own_steps += w.sum(1)
        rec = {"step_counts": True, "keep_state": True}
        if ab and sm.adapt:
            rec["ab_weight"] = w.astype(np.float64)
        o = sm.run(x, u0 if state is None else None, b - a, rec=rec, state=state, t0=a)
        state, sc = o["state"], o["step_counts"]
        if ab and sm.adapt:
            ab_sum += o["ab_sum"]
        if counts:                                           # own steps of a stream are one range: sum it directly
            for i in range(B):
                lo, hi = max(int(own_from[i]), a) - a, min(int(own_to[i]), b) - a
                if hi > lo:
                    spike_sum[i] += sc[i, lo:hi].sum(0, dtype=torch.int64).to(torch.float64)
        buf = sc if buf is None else torch.cat([buf, sc], 1)
        jobs = []                                            # (stream, beat, start in the buffer)
        for i in range(B):
            st = starts[i]
            while nxt[i] < len(st) and st[nxt[i]] + span <= b:
                jobs.append((i, nxt[i], int(st[nxt[i]]) - buf0))
                nxt[i] += 1
        for c in range(0, len(jobs), beat_chunk):
            J = jobs[c:c + beat_chunk]
            ii = torch.as_tensor([j[0] for j in J], device=dev)
            pp = torch.as_tensor([j[2] for j in J], device=dev)
            seg = buf[ii[:, None], pp[:, None] + ar[None, :]]           # (K, span, N)
            wc = seg.view(len(J), nwin, win, N).sum(2, dtype=torch.int32).clamp(max=255).to(torch.uint8).cpu().numpy()
            by = {}
            for r, (i, k, _) in enumerate(J):
                by.setdefault(i, []).append((k, r))
            for i, kr in by.items():
                ks = np.array([k for k, _ in kr]); rs = np.array([r for _, r in kr])
                sink(i, ks, wc[rs])
        keep_from = max(buf0, b - span)                      # every beat not yet cut starts after b - span
        buf = buf[:, keep_from - buf0:]
        buf0 = keep_from
        if progress is not None:
            progress(b, T)
    if any(nxt[i] != len(starts[i]) for i in range(B)):
        raise RuntimeError("some beats were not cut")
    out = {"own_steps": own_steps, "steps": T}
    if kept is not None:
        out["windows"] = kept
    if ab:
        out["ab_sum"] = ab_sum.cpu().numpy()
    if counts:
        out["spike_sum"] = spike_sum.cpu().numpy()
    return out


def frozen_thresholds(liq, ab_sum, steps):
    """Time-averaged threshold of every neuron from summed adaptation variables: 1 + c * (sum b / steps).
    ab_sum: (N,) summed over the averaging steps (b in spikes per ms); neurons that do not adapt get 1."""
    c = np.zeros(liq.N) if liq.adapt_c is None else np.asarray(liq.adapt_c, float)
    return 1.0 + c * (np.asarray(ab_sum, float) / float(steps))


def frozen_copy(liq, thr):
    """The same liquid with each neuron's threshold fixed at thr (no adaptation): the frozen control."""
    return replace(liq, adapt_c=None, adapt_tau_ms=None, thr_base=np.asarray(thr, float).copy(),
                   meta={**liq.meta, "frozen": True})
