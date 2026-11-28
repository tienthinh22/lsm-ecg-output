"""The simulator: one batched update for X, Y and every candidate between them (each factor is a setting of the
Liquid), plus the Brian2 schedule used only for (d)'s own liquid (level 1e and the Maass check's script version).

Units: thresholds (the voltage threshold is 1). Time: input steps (one ECG sample, or 0.1 ms for 1e), each split
into n sub-steps. Our order within a sub-step (plan, Simulation details):
  1. synapse states recover/fade over the sub-step; spikes whose delay ends now arrive (dynamic synapses
     update u, push, R in (d)'s order)
  2. currents: without fine timing a push arrives all at once; with it, it feeds a current that decays over
     3 ms (from E and from inputs) or 6 ms (from I), scaled so its total equals the push
  3. every neuron integrates: u <- beta u + input + recurrent + tap (not during a clamped refractory period)
  4. threshold test (u >= 1, outside the minimum gap), then reset (subtract 1, or set to 0.9 with the tap)
Input spikes of an input step act in its first sub-step ((a)'s Eq. 16: x[t] W_in in step t); this step's spikes
reach their targets from the next sub-step on (delay >= 1).
"""
import math
import numpy as np
import torch

from . import config as C

# Fast path (same arithmetic, fewer Python round trips): the "ours" schedule run one input step at a time on fixed
# buffers, and on a CUDA card each input step replayed as one captured CUDA graph. The notebook compares it with the
# plain loop on the card before using it and switches it off if they differ.
FAST = True
USE_CUDA_GRAPHS = True


def _t(a, device, dtype):
    return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)


class Simulator:
    def __init__(self, liq, device="cpu", dtype=torch.float64, g=1.0, in_scale=None, recurrence=True):
        self.liq, self.device, self.dtype = liq, torch.device(device), dtype
        self.g = float(g) if recurrence else 0.0
        if in_scale is None:
            if liq.in_scale_fixed is None:
                raise ValueError("this liquid needs an input size (paper rule or calibration)")
            in_scale = liq.in_scale_fixed
        self.in_scale = float(in_scale)
        d, t = self.device, dtype
        N = liq.N
        self.N = N
        self.beta = _t(liq.beta, d, t)
        self.tap = _t(liq.tap, d, t)
        self.gap = _t(liq.gap, d, torch.int64)
        self.Win = _t(liq.Win * self.in_scale, d, t)
        self.z_decay = math.exp(-liq.h_ms / C.STATE_TAU_MS)
        # adaptive thresholds: theta = 1 + c b; b <- rho b + (1 - rho) spike / h  (b in spikes per ms)
        self.adapt = liq.adapt_c is not None and bool(np.any(np.asarray(liq.adapt_c) > 0))
        if self.adapt:
            self.a_c = _t(liq.adapt_c, d, t)
            rho = np.exp(-liq.h_ms / np.asarray(liq.adapt_tau_ms, float))     # 1 for non-adaptive neurons (tau inf)
            self.a_rho = _t(rho, d, t)
            self.a_in = _t((1.0 - rho) / liq.h_ms, d, t)
        # fixed per-neuron thresholds (frozen controls): theta = thr_base (+ c b with adaptation); None = 1
        self.thr_base = None if liq.thr_base is None else _t(liq.thr_base, d, t)
        self.var_thr = self.adapt or self.thr_base is not None
        if liq.reset_to_thr and liq.snn_quirk:
            raise ValueError("the threshold reset needs the Eqs. 17-19 neuron (snn_quirk off)")
        self.L = int(max(liq.delay.max(), 1)) + 1           # spike-history ring length
        S = liq.S
        self.S = S
        self.static = liq.syn_model == "static"
        if liq.schedule == "ours":
            if self.static:
                # one dense matrix per (delay, presynaptic class): pushes of the spikes that left `delay` sub-steps ago
                self.groups = {}
                for ti in range(4):
                    m = liq.typ == ti
                    if not m.any():
                        continue
                    key = (int(liq.delay[ti]), 0 if ti < 2 else 1)
                    W = self.groups.get(key)
                    if W is None:
                        W = torch.zeros(N, N, device=d, dtype=t)
                    W[_t(liq.pre[m], d, torch.int64), _t(liq.post[m], d, torch.int64)] += _t(liq.w[m] * self.g, d, t)
                    self.groups[key] = W
            else:
                self._prep_sparse()
                Ubar_s = liq.Ubar[liq.typ]
                self.wg = _t(liq.w * self.g / Ubar_s, d, t)        # push = (w / U-bar) u R
                self.U = _t(liq.U, d, t)
                self.eD = _t(np.exp(-liq.h_ms / liq.D_ms), d, t)
                self.eF = _t(np.exp(-liq.h_ms / liq.F_ms), d, t)
            if liq.cur_a is not None:
                self.aE, self.aI = float(liq.cur_a[0]), float(liq.cur_a[1])
            self.in_a = None if liq.in_a is None else _t(liq.in_a, d, t)
        else:   # Brian2 schedule, (d)'s three-state synapses, currents in voltage-equivalent units
            self._prep_sparse()
            self.wg = _t(liq.w * self.g, d, t)
            self.U = _t(liq.U, d, t)
            tau = liq.syn_tau_ms[liq.typ]
            D = liq.D_ms
            h = liq.h_ms
            self.ey = _t(np.exp(-h / tau), d, t)
            self.eD = _t(np.exp(-h / D), d, t)
            self.cxy = _t(tau * (np.exp(-h / tau) - np.exp(-h / D)) / (D - tau), d, t)
            self.eF = _t(np.exp(-h / liq.F_ms), d, t)
            self.kappa = _t(liq.kappa, d, t)
            self.in_a = _t(liq.in_a, d, t)
            self.J_b = float(liq.J_b)

    def _prep_sparse(self):
        liq, d, t = self.liq, self.device, self.dtype
        self.pre_idx = _t(liq.pre, d, torch.int64)
        self.syn_delay = liq.delay[liq.typ]
        self.delay_groups = [(int(dl), _t(np.nonzero(self.syn_delay == dl)[0], d, torch.int64))
                             for dl in np.unique(self.syn_delay)]
        self.post_idx = _t(liq.post, d, torch.int64)
        self.nE = int((liq.typ < 2).sum())                  # synapses are sorted by type: E-presynaptic first
        M = torch.zeros(liq.S, liq.N, device=d, dtype=t)
        if liq.S:
            M[torch.arange(liq.S, device=d), _t(liq.post, d, torch.int64)] = 1.0
        self.M_E, self.M_I, self.M = M[:self.nE], M[self.nE:], M

    # ------------------------------------------------------------------ helpers
    def _arrivals(self, hist, k, B):
        """(B, S) 0/1: which synapses receive a spike in sub-step k."""
        a = torch.zeros(B, self.S, device=self.device, dtype=self.dtype)
        for dl, idx in self.delay_groups:
            if k - dl < 0:
                continue
            src = hist[(k - dl) % self.L]
            a[:, idx] = src[:, self.pre_idx[idx]]
        return a

    def start_state(self, u0):
        return torch.as_tensor(u0, device=self.device, dtype=self.dtype).clone()

    def _threshold(self, ab):
        """The current threshold of every neuron (only when it is not the fixed 1)."""
        if self.adapt:
            return (1.0 if self.thr_base is None else self.thr_base) + self.a_c * ab
        return self.thr_base

    # ------------------------------------------------------------------ main loop
    @torch.no_grad()
    def run(self, x, u0, T, rec=None, open_loop=None, trace=False, own_spikes=None, fast=None, state=None, t0=0):
        """x: (B, T_in, C) input spikes per input step; u0: (B, N) starting voltages; T: input steps to simulate.
        rec: {"windows": (win, nwin), "states_at": [t, ...], "states_every": (t0, t1), "neuron_counts": (t0, t1),
              "raster": bool}; times are input steps, a state is read at the end of its step.
        open_loop: (T*n, B, N) recorded spikes used as the presynaptic spikes (the liquid's own spikes are still
        computed and returned). own_spikes: (T*n, B, N) recorded spikes that also replace each neuron's own
        threshold decisions (level 1c: pure arithmetic drift, no flips); out["flips"] counts where the threshold
        test would have decided differently. trace: return u and currents every sub-step and synapse states
        every input step. fast: None = the module's FAST; False = this plain loop (the reference).
        Streaming (fast path only): state = the "state" a previous call returned (rec "keep_state"), to continue from
        it (u0 is then ignored); t0 = the input steps already simulated before this call (keeps the spike-history
        ring in phase). rec "step_counts": spikes per neuron in every input step, (B, T, N) uint8; rec "ab_weight":
        (B, T) weights, returns "ab_sum" = sum over steps of weight x adaptation variable b (at each step's end)."""
        liq, dev, dt = self.liq, self.device, self.dtype
        rec = rec or {}
        x = torch.as_tensor(x, device=dev)
        streaming = state is not None or t0 or rec.get("step_counts") or rec.get("keep_state") or rec.get("ab_weight") is not None
        if (FAST if fast is None else fast) and liq.schedule == "ours" and open_loop is None and own_spikes is None and not trace \
                and not rec.get("raster"):
            return self._run_fast(x, u0, T, rec, graphs=USE_CUDA_GRAPHS and dev.type == "cuda", state=state, t0=t0)
        if streaming:
            raise ValueError("streaming (state, t0, step_counts, keep_state, ab_weight) needs the fast path")
        B, T_in, Cin = x.shape
        N, n = self.N, liq.n_sub
        u = self.start_state(u0)
        ref = torch.zeros(B, N, device=dev, dtype=torch.int64)
        hist = torch.zeros(self.L, B, N, device=dev, dtype=dt)
        z = torch.zeros(B, N, device=dev, dtype=dt)
        ab = torch.zeros(B, N, device=dev, dtype=dt) if self.adapt else None
        ours = liq.schedule == "ours"
        fine = ours and liq.cur_a is not None
        IE = torch.zeros(B, N, device=dev, dtype=dt) if fine else None
        II = torch.zeros(B, N, device=dev, dtype=dt) if fine else None
        Iin = torch.zeros(B, N, device=dev, dtype=dt) if (self.in_a is not None) else None
        if not self.static or not ours:
            R = torch.ones(B, self.S, device=dev, dtype=dt)       # resources ready (x in (d))
            q = torch.zeros(B, self.S, device=dev, dtype=dt)      # utilisation u
            y = torch.zeros(B, self.S, device=dev, dtype=dt) if not ours else None
        out = {}
        win = rec.get("windows")
        if win:
            wl, nw = win
            counts = torch.zeros(B, nw, N, device=dev, dtype=torch.int32)
        st_at = sorted(set(rec.get("states_at", [])))
        st_at_set = {t: i for i, t in enumerate(st_at)}
        if st_at:
            states = torch.zeros(B, len(st_at), N, device=dev, dtype=torch.float32)
        se = rec.get("states_every")
        if se:
            states_every = torch.zeros(B, se[1] - se[0], N, device=dev, dtype=torch.float32)
        nc = rec.get("neuron_counts")
        if nc:
            ncount = torch.zeros(B, N, device=dev, dtype=torch.int64)
        raster = torch.zeros(T * n, B, N, device=dev, dtype=torch.bool) if rec.get("raster") else None
        tr = {"u": [], "I1": [], "I2": [], "I3": [], "syn": []} if trace else None
        if open_loop is not None:
            open_loop = torch.as_tensor(open_loop, device=dev)
        if own_spikes is not None:
            own_spikes = torch.as_tensor(own_spikes, device=dev).bool()
            flips = torch.zeros((), device=dev, dtype=torch.int64)
        zero_in = torch.zeros(B, N, device=dev, dtype=dt)
        k = 0
        for t in range(T):
            step_spk = torch.zeros(B, N, device=dev, dtype=dt) if (win or nc) else None
            if t < T_in:
                in_push = x[:, t, :].to(dt) @ self.Win
            else:
                in_push = zero_in
            for j in range(n):
                blocked = ref > 0
                if ours:
                    # 1. synapses
                    if self.static:
                        pE = pI = None
                        for (dl, cls), W in self.groups.items():
                            if k - dl < 0:
                                continue
                            p = hist[(k - dl) % self.L] @ W
                            if cls == 0:
                                pE = p if pE is None else pE + p
                            else:
                                pI = p if pI is None else pI + p
                        pE = zero_in if pE is None else pE
                        pI = zero_in if pI is None else pI
                    else:
                        R = 1.0 - (1.0 - R) * self.eD
                        q = q * self.eF
                        a = self._arrivals(hist, k, B)
                        q = q + a * self.U * (1.0 - q)
                        push = a * self.wg * q * R
                        R = R - a * q * R
                        pE = push[:, :self.nE] @ self.M_E if self.nE else zero_in
                        pI = push[:, self.nE:] @ self.M_I if self.S > self.nE else zero_in
                    ip = in_push if j == 0 else zero_in
                    # 2. currents
                    if fine:
                        IE = self.aE * IE + (1 - self.aE) * pE
                        II = self.aI * II + (1 - self.aI) * pI
                        Iin = self.in_a * Iin + (1 - self.in_a) * ip
                        total = IE + II + Iin
                    else:
                        total = pE + pI + ip
                    # 3. integrate
                    u_new = self.beta * u + total + self.tap
                    if liq.clamp:
                        u_new = torch.where(blocked, u, u_new)
                    # 4. threshold and reset
                    thr = None
                    if liq.snn_quirk:       # snnTorch: the spike test sees u_new - [u_prev > 1]
                        crossed = (u_new - (u > 1.0).to(dt)) > 1.0
                    elif self.var_thr:
                        thr = self._threshold(ab)
                        crossed = (u_new > thr) if liq.strict else (u_new >= thr)
                    else:
                        crossed = (u_new > 1.0) if liq.strict else (u_new >= 1.0)
                    spk = crossed & ~blocked
                    if own_spikes is not None:
                        flips += (spk != own_spikes[k]).sum()
                        spk = own_spikes[k]
                    if liq.reset_mode == "subtract":
                        u = u_new - (spk.to(dt) * thr if (liq.reset_to_thr and thr is not None) else spk.to(dt))
                    else:
                        u = torch.where(spk, torch.full_like(u_new, liq.reset_value), u_new)
                    if self.adapt:
                        ab = self.a_rho * ab + self.a_in * spk.to(dt)
                else:
                    # Brian2 schedule of (d)'s network (its sorted objects): synapse state update (exact), the
                    # summed current from the updated y, exact neuron update with that current held constant over
                    # the step and the input current decaying exactly; threshold; arrivals (on_pre); reset.
                    q = q * self.eF
                    R, y = R * self.eD + (1 - self.eD) + y * self.cxy, y * self.ey
                    # scatter-add per target, as Brian2's summed variable does (a runaway synapse of (d) as written
                    # then only affects its own target, instead of spreading NaN through a matrix product)
                    Jsyn = torch.zeros_like(zero_in).index_add_(1, self.post_idx, y * self.wg) if self.S else zero_in
                    u_new = self.beta * u + (1 - self.beta) * (self.J_b + Jsyn) + self.kappa * Iin
                    u_new = torch.where(blocked, u, u_new)
                    Iin = Iin * self.in_a
                    spk = (u_new > 1.0) & ~blocked
                    if own_spikes is not None:
                        flips += (spk != own_spikes[k]).sum()
                        spk = own_spikes[k]
                    if j == 0 and t < T_in:
                        Iin = Iin + in_push
                    hist[k % self.L] = (open_loop[k] if open_loop is not None else spk).to(dt)
                    a = self._arrivals(hist, k, B)
                    q = q + a * self.U * (1.0 - q)
                    y = y + a * q * R
                    R = R - a * q * R
                    u = torch.where(spk, torch.full_like(u_new, liq.reset_value), u_new)
                ref = torch.where(spk, self.gap - 1, torch.clamp(ref - 1, min=0))
                if ours:
                    hist[k % self.L] = (open_loop[k] if open_loop is not None else spk).to(dt)
                spf = spk.to(dt)
                z = z * self.z_decay + spf
                if step_spk is not None:
                    step_spk += spf
                if raster is not None:
                    raster[k] = spk
                if trace:
                    tr["u"].append(u.detach().cpu().numpy().copy())
                    if ours:
                        tr["I1"].append((IE if fine else pE).cpu().numpy().copy())
                        tr["I2"].append((II if fine else pI).cpu().numpy().copy())
                        tr["I3"].append((Iin if fine else ip).cpu().numpy().copy())
                    else:
                        tr["I1"].append(Jsyn.cpu().numpy().copy()); tr["I3"].append(Iin.cpu().numpy().copy())
                k += 1
            # end of input step t
            if win and t < wl * nw:
                counts[:, t // wl] += step_spk.to(torch.int32)
            if nc and nc[0] <= t < nc[1]:
                ncount += step_spk.to(torch.int64)
            if t in st_at_set:
                states[:, st_at_set[t]] = z.float()
            if se and se[0] <= t < se[1]:
                states_every[:, t - se[0]] = z.float()
            if trace and (not self.static or not ours):
                syn = {"R": R.cpu().numpy().copy(), "u": q.cpu().numpy().copy()}
                if not ours:
                    syn["y"] = y.cpu().numpy().copy()
                tr["syn"].append(syn)
        if win:
            out["windows"] = counts
        if st_at:
            out["states_at"] = states
            out["states_at_times"] = st_at
        if se:
            out["states_every"] = states_every
        if nc:
            out["neuron_counts"] = ncount
        if raster is not None:
            out["raster"] = raster
        if trace:
            out["trace"] = tr
        out["u_end"] = u
        if own_spikes is not None:
            out["flips"] = int(flips)
        return out


    # ------------------------------------------------------------------ fast path ("ours" schedule only)
    def _run_fast(self, x, u0, T, rec, graphs, state=None, t0=0):
        """The same arithmetic as run()'s "ours" branch, in the same order, on fixed buffers: one input step is one
        call of `step` (or one replay of its CUDA graph). The spike-history ring holds m * n sub-steps (m = ceil(L / n))
        so the slots an input step reads depend only on t mod m; slots not yet written hold zeros, which is what the
        plain loop's skipped early arrivals add."""
        liq, dev, dt = self.liq, self.device, self.dtype
        B, T_in, Cin = x.shape
        N, n = self.N, liq.n_sub
        m = -(-self.L // n)
        Lr = m * n
        fine = liq.cur_a is not None
        if t0 % m:
            raise ValueError(f"t0 = {t0} must be a multiple of {m} (the spike-history ring's period in input steps)")
        u_start = torch.zeros(B, N, device=dev, dtype=dt) if state is not None else self.start_state(u0)
        st = {"u": u_start, "ref": torch.zeros(B, N, device=dev, dtype=torch.int64),
              "hist": torch.zeros(Lr, B, N, device=dev, dtype=dt), "z": torch.zeros(B, N, device=dev, dtype=dt),
              "step": torch.zeros(B, N, device=dev, dtype=dt), "inp": torch.zeros(B, N, device=dev, dtype=dt)}
        if self.adapt:
            st["ab"] = torch.zeros(B, N, device=dev, dtype=dt)
        if fine:
            st["IE"] = torch.zeros(B, N, device=dev, dtype=dt)
            st["II"] = torch.zeros(B, N, device=dev, dtype=dt)
        if self.in_a is not None:
            st["Iin"] = torch.zeros(B, N, device=dev, dtype=dt)
        if not self.static:
            st["R"] = torch.ones(B, self.S, device=dev, dtype=dt)
            st["q"] = torch.zeros(B, self.S, device=dev, dtype=dt)
        if state is not None:                                  # continue a stream where the previous call stopped
            if set(state) != set(st) - {"step", "inp"}:
                raise ValueError(f"state keys {sorted(state)} do not match this liquid's {sorted(set(st) - {'step', 'inp'})}")
            for k, v in state.items():
                st[k].copy_(v)
        zero_in = torch.zeros(B, N, device=dev, dtype=dt)

        if not self.static:
            # arrivals as gathers (capture-safe, no index_put): each delay group's presynaptic columns, then the
            # synapses back in their own order (synapses are sorted by type, so this is usually already in order)
            gpre = [(dl, self.pre_idx[idx]) for dl, idx in self.delay_groups]
            order = torch.cat([idx for _, idx in self.delay_groups])
            inv = None if torch.equal(order, torch.arange(self.S, device=dev)) else torch.argsort(order)

        def arrivals(hist, kk):
            a = torch.cat([hist[(kk - dl) % Lr].index_select(1, pre) for dl, pre in gpre], 1)
            return a if inv is None else a.index_select(1, inv)

        def step(phase):
            u, ref, hist, z = st["u"], st["ref"], st["hist"], st["z"]
            IE, II, Iin = st.get("IE"), st.get("II"), st.get("Iin")
            R, q = st.get("R"), st.get("q")
            ab = st.get("ab")
            step_spk = st["step"]
            step_spk.zero_()
            in_push = st["inp"]
            for j in range(n):
                kk = phase * n + j
                blocked = ref > 0
                if self.static:
                    pE = pI = None
                    for (dl, cls), W in self.groups.items():
                        p = hist[(kk - dl) % Lr] @ W
                        if cls == 0:
                            pE = p if pE is None else pE + p
                        else:
                            pI = p if pI is None else pI + p
                    pE = zero_in if pE is None else pE
                    pI = zero_in if pI is None else pI
                else:
                    R = 1.0 - (1.0 - R) * self.eD
                    q = q * self.eF
                    a = arrivals(hist, kk)
                    q = q + a * self.U * (1.0 - q)
                    push = a * self.wg * q * R
                    R = R - a * q * R
                    pE = push[:, :self.nE] @ self.M_E if self.nE else zero_in
                    pI = push[:, self.nE:] @ self.M_I if self.S > self.nE else zero_in
                ip = in_push if j == 0 else zero_in
                if fine:
                    IE = self.aE * IE + (1 - self.aE) * pE
                    II = self.aI * II + (1 - self.aI) * pI
                    Iin = self.in_a * Iin + (1 - self.in_a) * ip
                    total = IE + II + Iin
                else:
                    total = pE + pI + ip
                u_new = self.beta * u + total + self.tap
                if liq.clamp:
                    u_new = torch.where(blocked, u, u_new)
                thr = None
                if liq.snn_quirk:
                    crossed = (u_new - (u > 1.0).to(dt)) > 1.0
                elif self.var_thr:
                    thr = self._threshold(ab)
                    crossed = (u_new > thr) if liq.strict else (u_new >= thr)
                else:
                    crossed = (u_new > 1.0) if liq.strict else (u_new >= 1.0)
                spk = crossed & ~blocked
                if liq.reset_mode == "subtract":
                    u = u_new - (spk.to(dt) * thr if (liq.reset_to_thr and thr is not None) else spk.to(dt))
                else:
                    u = torch.where(spk, torch.full_like(u_new, liq.reset_value), u_new)
                if self.adapt:
                    ab = self.a_rho * ab + self.a_in * spk.to(dt)
                ref = torch.where(spk, self.gap - 1, torch.clamp(ref - 1, min=0))
                hist[kk % Lr] = spk.to(dt)
                spf = spk.to(dt)
                z = z * self.z_decay + spf
                step_spk += spf
            st["u"].copy_(u); st["ref"].copy_(ref); st["z"].copy_(z)
            if fine:
                st["IE"].copy_(IE); st["II"].copy_(II)
            if Iin is not None:
                st["Iin"].copy_(Iin)
            if R is not None:
                st["R"].copy_(R); st["q"].copy_(q)
            if ab is not None:
                st["ab"].copy_(ab)

        run_step = {ph: (lambda ph=ph: step(ph)) for ph in range(m)}
        if graphs:
            backup = {k: v.clone() for k, v in st.items()}
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for ph in range(m):
                    step(ph)                                   # warm-up (cuBLAS handles, allocator)
            torch.cuda.current_stream(dev).wait_stream(side)
            pool = None
            for ph in range(m):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    step(ph)
                pool = g.pool()
                run_step[ph] = g.replay
            for k, v in st.items():                            # undo the warm-up's changes to the state
                v.copy_(backup[k])
            del backup

        out = {}
        win = rec.get("windows")
        if win:
            wl, nw = win
            counts = torch.zeros(B, nw, N, device=dev, dtype=torch.int32)
        st_at = sorted(set(rec.get("states_at", [])))
        st_at_set = {t: i for i, t in enumerate(st_at)}
        if st_at:
            states = torch.zeros(B, len(st_at), N, device=dev, dtype=torch.float32)
        se = rec.get("states_every")
        if se:
            states_every = torch.zeros(B, se[1] - se[0], N, device=dev, dtype=torch.float32)
        nc = rec.get("neuron_counts")
        if nc:
            ncount = torch.zeros(B, N, device=dev, dtype=torch.int64)
        raster = torch.zeros(T * n, B, N, device=dev, dtype=torch.bool) if rec.get("raster") else None
        if raster is not None:
            raise ValueError("the fast path does not record rasters: use fast=False")
        sc = torch.zeros(B, T, N, device=dev, dtype=torch.uint8) if rec.get("step_counts") else None
        abw = rec.get("ab_weight")
        if abw is not None:
            if not self.adapt:
                raise ValueError("ab_weight needs an adaptive liquid")
            abw = torch.as_tensor(abw, device=dev, dtype=torch.float64)
            ab_sum = torch.zeros(B, N, device=dev, dtype=torch.float64)
        for t in range(T):
            if t < T_in:
                st["inp"].copy_(x[:, t, :].to(dt) @ self.Win)
            else:
                st["inp"].zero_()
            run_step[(t0 + t) % m]()
            step_spk, z = st["step"], st["z"]
            if sc is not None:
                sc[:, t] = step_spk.to(torch.uint8)
            if abw is not None:
                ab_sum += abw[:, t, None] * st["ab"]
            if win and t < wl * nw:
                counts[:, t // wl] += step_spk.to(torch.int32)
            if nc and nc[0] <= t < nc[1]:
                ncount += step_spk.to(torch.int64)
            if t in st_at_set:
                states[:, st_at_set[t]] = z.float()
            if se and se[0] <= t < se[1]:
                states_every[:, t - se[0]] = z.float()
        if win:
            out["windows"] = counts
        if st_at:
            out["states_at"] = states
            out["states_at_times"] = st_at
        if se:
            out["states_every"] = states_every
        if nc:
            out["neuron_counts"] = ncount
        if sc is not None:
            out["step_counts"] = sc
        if abw is not None:
            out["ab_sum"] = ab_sum
        if rec.get("keep_state"):
            out["state"] = {k: v.clone() for k, v in st.items() if k not in ("step", "inp")}
        out["u_end"] = st["u"].clone()
        return out


# ---------------------------------------------------------------- starting states (CPU, from (seed, beat index))
def start_voltages(liq, seed, beat_ids, stream, random_start=None):
    """Starting voltages for each beat, drawn on the CPU from (seed, stream, beat index) so they do not depend
    on batching or the card. random_start=None: the liquid's own rule (0 in X; uniform 0.9..1 with the tap).
    random_start=True: uniform between the resting level and the threshold (level 3's two starting states)."""
    lo, hi = (liq.rest, 1.0) if random_start else (liq.start_lo, liq.start_hi)
    out = np.empty((len(beat_ids), liq.N))
    for i, b in enumerate(beat_ids):
        if hi > lo:
            out[i] = np.random.default_rng([int(seed), int(stream), int(b)]).uniform(lo, hi, liq.N)
        else:
            out[i] = lo
    return out
