"""Building liquids: every ladder candidate (X plus any set of the 11 factors), Maass's two versions for the
Maass check (script and paper), and X converted from the authors' own class (for 1f and E1's paper liquid).

A Liquid holds the realised network and every time constant already converted to its grid (n sub-steps per
input step). Input weights are stored as a base matrix; the input size (paper's rule, factor-5 calibration or
factor 9) is a separate scalar `in_scale`, and the recurrent gain g a scalar given to the simulator.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from . import config as C

TYPE_INDEX = {t: i for i, t in enumerate(C.TYPES)}     # EE 0, EI 1, IE 2, II 3


def round_half_up(x):
    return int(math.floor(x + 0.5))


def steps_of(ms, h_ms):
    """A time in ms as a whole number of sub-steps: rounded to the nearest, at least 1 (plan, Time grid)."""
    return max(1, int(round(ms / h_ms)))


@dataclass
class Liquid:
    # structure
    N: int
    dims: tuple
    is_exc: np.ndarray          # (N,) bool
    pos: np.ndarray             # (N, 3)
    pre: np.ndarray             # (S,) int
    post: np.ndarray            # (S,) int
    typ: np.ndarray             # (S,) int 0..3 (EE, EI, IE, II)
    w: np.ndarray               # (S,) signed base weight: a push in thresholds ("ours"), or J amplitude ("brian2")
    U: np.ndarray               # (S,) utilisation (dynamic synapses), else ones
    D_ms: np.ndarray            # (S,)
    F_ms: np.ndarray            # (S,)
    Ubar: np.ndarray            # (4,) nominal type means of U
    Win: np.ndarray             # (C, N) base input weights, >= 0
    in_neurons: np.ndarray      # (N,) bool: neurons with any input synapse
    # time grid and neuron model
    n_sub: int                  # sub-steps per input step
    h_ms: float                 # sub-step length
    beta: np.ndarray            # (N,) leak per sub-step
    tap: np.ndarray             # (N,) tap per sub-step ("ours"); 0 without factor 5
    reset_mode: str             # "subtract" or "value"
    reset_value: float
    start_lo: float             # starting voltages drawn uniformly in [start_lo, start_hi]; equal = fixed
    start_hi: float
    rest: float                 # resting level (0, or 0.9 with the tap): lower end of level 3's random starts
    gap: np.ndarray             # (N,) int minimum gap between spikes, in sub-steps
    clamp: bool                 # voltage clamped during the blocked sub-steps
    strict: bool                # threshold test u > 1 (Brian2 script) instead of u >= 1 (ours, (a)'s H)
    schedule: str               # "ours" or "brian2"
    syn_model: str              # "static", "markram" (two-state, (c)'s form) or "tsodyks3" ((d)'s three-state)
    delay: np.ndarray           # (4,) int sub-steps by type
    cur_a: object               # None (push at once) or (2,) decay per sub-step for E- and I-presynaptic currents
    in_a: object                # None or (N,) decay per sub-step of each neuron's input current
    # Brian2-schedule constants (script version only)
    J_b: float = 0.0
    kappa: object = None        # (N,) exact weight of the decaying input current in the membrane update
    syn_tau_ms: object = None   # (4,) current decay per type (tsodyks3)
    # bookkeeping
    factors: frozenset = frozenset()
    snn_quirk: bool = False     # (a)'s code (snnTorch Leaky, reset_delay=False): the threshold test subtracts 1 more
                                # when the stored voltage is above 1; u > 1 strict. Default: (a)'s Eqs. 17-19
    g_maass: float = None       # g at Maass's own scale (factor 10)
    in_scale_fixed: float = None   # input size fixed by the design (factor 9, Maass versions)
    in_split: tuple = (1.0, 1.0)   # base input means onto E / I neurons (2:1 with factors 5 or 9)
    meta: dict = field(default_factory=dict)

    @property
    def S(self):
        return len(self.pre)

    def ee_mean_push(self, g=1.0):
        """Mean isolated EE push per presynaptic spike, in thresholds (the plan's absolute g scale)."""
        m = self.typ == 0
        if not m.any():
            return 0.0
        iso = np.abs(self.w[m]) * (self.U[m] / self.Ubar[0] if self.syn_model == "markram" else 1.0)
        return float(g * iso.mean())


# ---------------------------------------------------------------- shared pieces
def _grid(dims):
    return np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3).astype(np.float64)


def _type_matrix(is_exc):
    pe, qe = is_exc[:, None], is_exc[None, :]
    T = np.where(pe & qe, 0, np.where(pe & ~qe, 1, np.where(~pe & qe, 2, 3)))
    return T


def _draw_udf(rng_n, rng_u, T, mean_of_type, cap, bad_rule):
    """U, D or F for every (pre, post) pair: Normal(mean, 0.5 mean); draws at or below 0 replaced by
    uniform(0, 2 mean) ("redraw", plan decision 17) or by the mean ("mean", (d)); U capped at 1 if cap."""
    mean = np.vectorize(lambda t: mean_of_type[C.TYPES[t]])(T).astype(np.float64)
    z = rng_n.standard_normal(T.shape)
    v = mean + C.MAASS["udf_spread"] * mean * z
    repl = rng_u.random(T.shape) * 2 * mean
    if bad_rule == "redraw":
        v = np.where(v <= 0, repl, v)
    else:      # (d): U < 0 -> mean; D, F <= 0 -> mean
        v = np.where(v <= 0, mean, v)
    if cap:
        v = np.minimum(v, 1.0)
    return v


# ---------------------------------------------------------------- ladder candidates
def candidate_settings(factors):
    f = frozenset(factors)
    return {
        "dims": C.MAASS["dims"] if 11 in f else C.X["dims"],
        "lam": C.MAASS["lam"] if 1 in f else C.X["lam"],
        "mu": ({t: C.X["mu_rsv"]["EE"] * C.MAASS_RATIOS[t] for t in C.TYPES} if 3 in f else dict(C.X["mu_rsv"])),
        "normalise": 2 not in f,
        "tau_E_ms": C.LEAK_MS_Y if 4 in f else -C.SAMPLE_MS / math.log(C.X["betaE"]),
        "tau_I_ms": C.LEAK_MS_Y if 4 in f else -C.SAMPLE_MS / math.log(C.X["betaI"]),
        "tap": 5 in f,
        "input30": 6 in f,
        "fine": 7 in f,
        "dynamic": 8 in f,
        "maass_input": 9 in f,
        "maass_scale": 10 in f,
    }


def build_candidate(factors, seed, n_sub, n_channels, n_sub_input_step_ms=C.SAMPLE_MS):
    """A ladder candidate: X ((a), bugs fixed) with the given factors switched on.
    n_sub: sub-steps per sample (1 without factor 7 unless a grid check asks otherwise)."""
    f = frozenset(factors)
    s = candidate_settings(f)
    N = int(np.prod(s["dims"]))
    h = n_sub_input_step_ms / n_sub
    # independent streams, so candidates with the same seed share their random numbers where they can
    r_types, r_conn, r_w = (np.random.default_rng([seed, k]) for k in (1, 2, 3))
    r_un, r_uu, r_dn, r_du, r_fn, r_fu = (np.random.default_rng([seed, k]) for k in (4, 5, 6, 7, 8, 9))
    r_inm, r_inw, r_inn = (np.random.default_rng([seed, k]) for k in (10, 11, 12))
    n_exc = int(N * C.X["excit_ratio"])
    is_exc = np.zeros(N, bool)
    is_exc[r_types.permutation(N)[:n_exc]] = True
    pos = _grid(s["dims"])
    T = _type_matrix(is_exc)
    Cm = np.vectorize(lambda t: C.X["C"][C.TYPES[t]])(T)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    p = np.clip(Cm * np.exp(-(d / s["lam"]) ** 2), 0, 1)
    conn = r_conn.random((N, N)) < p
    np.fill_diagonal(conn, False)                            # no self-connections ((a); kept in Y)
    mu = np.vectorize(lambda t: s["mu"][C.TYPES[t]])(T)
    Wmag = r_w.standard_exponential((N, N)) * mu             # Gamma(shape 1, scale mu): spread = mean
    if s["normalise"]:
        fan_in = np.maximum(conn.sum(0, keepdims=True), 1)   # (a) lines 168-169: divide by the receiver's count
        Wmag = Wmag / fan_in
    W = np.where(is_exc[:, None], Wmag, -Wmag)
    pre, post = np.nonzero(conn)
    order = np.lexsort((post, pre, T[pre, post]))           # sorted by type, then pre, then post
    pre, post = pre[order], post[order]
    typ = T[pre, post]
    if s["dynamic"]:
        Ufull = _draw_udf(r_un, r_uu, T, C.MAASS["U"], True, "redraw")
        Dfull = _draw_udf(r_dn, r_du, T, {k: v * 1000 for k, v in C.MAASS["D_s"].items()}, False, "redraw")
        Ffull = _draw_udf(r_fn, r_fu, T, {k: v * 1000 for k, v in C.MAASS["F_s"].items()}, False, "redraw")
        U, D, F = Ufull[pre, post], Dfull[pre, post], Ffull[pre, post]
    else:
        U, D, F = np.ones(len(pre)), np.full(len(pre), np.inf), np.full(len(pre), np.inf)
    Ubar = np.array([C.MAASS["U"][t] for t in C.TYPES])
    # input wiring and base weights
    if s["input30"]:
        k = round_half_up(C.MAASS["in_fraction"] * N)
        in_n = np.zeros(N, bool)
        in_n[r_inn.permutation(N)[:k]] = True
        mask = np.repeat(in_n[None, :], n_channels, 0)
    else:
        mask = r_inm.random((n_channels, N)) < C.X["P_in"]
        in_n = mask.any(0)
    split = (1.0, 0.5) if (s["tap"] or s["maass_input"]) else (1.0, 1.0)
    m_in = np.where(is_exc, split[0], split[1])[None, :]
    Win = r_inw.standard_exponential((n_channels, N)) * m_in * mask
    if s["normalise"]:                                       # (a) lines 71-75: each column's mean active weight = m
        cnt = mask.sum(0, keepdims=True)
        col = Win.sum(0, keepdims=True) / (cnt + 1e-8)
        Win = Win * (m_in / (col + 1e-8))
    # time constants on the grid
    tau = np.where(is_exc, s["tau_E_ms"], s["tau_I_ms"])
    beta = np.exp(-h / tau)
    tap = C.TAP * (1 - beta) if s["tap"] else np.zeros(N)
    if s["fine"]:
        gap = np.where(is_exc, steps_of(C.MAASS["refrac_ms"]["E"], h), steps_of(C.MAASS["refrac_ms"]["I"], h))
        delay = np.array([steps_of(C.MAASS["delay_ms"][t], h) for t in C.TYPES])
        cur_a = np.exp(-h / np.array([C.MAASS["tau_s_ms"]["E"], C.MAASS["tau_s_ms"]["I"]]))
        in_a = np.full(N, math.exp(-h / C.MAASS["tau_in_ms"]))
        clamp = True
    else:
        # X: the 2.78 ms gap and the one-sample delay, in ms, converted to the grid in use
        gap = np.full(N, steps_of(C.SAMPLE_MS, h))
        delay = np.full(4, steps_of(C.SAMPLE_MS, h))
        cur_a, in_a, clamp = None, None, False
    liq = Liquid(
        N=N, dims=tuple(s["dims"]), is_exc=is_exc, pos=pos, pre=pre, post=post, typ=typ, w=W[pre, post].astype(np.float64),
        U=U, D_ms=D, F_ms=F, Ubar=Ubar, Win=Win.astype(np.float64), in_neurons=in_n,
        n_sub=n_sub, h_ms=h, beta=beta, tap=tap,
        reset_mode="value" if s["tap"] else "subtract", reset_value=C.RESET_TAP if s["tap"] else 0.0,
        start_lo=C.RESET_TAP if s["tap"] else 0.0, start_hi=1.0 if s["tap"] else 0.0,
        rest=C.TAP if s["tap"] else 0.0,
        gap=gap.astype(np.int64), clamp=clamp, strict=False, schedule="ours",
        syn_model="markram" if s["dynamic"] else "static", delay=delay.astype(np.int64), cur_a=cur_a, in_a=in_a,
        factors=f, in_split=split,
        meta={"kind": "candidate", "seed": int(seed), "n_channels": int(n_channels), "settings": s},
    )
    if liq.reset_mode == "subtract" and C.CODE["x_update"] == "authors_code":
        liq.snn_quirk, liq.strict = True, True
    m = typ == 0
    liq.g_maass = C.MAASS_EE_PUSH / float(np.abs(liq.w[m]).mean()) if m.any() else None
    if s["maass_input"]:
        liq.in_scale_fixed = C.MAASS_IN_PUSH["E"]           # base means 1 / 0.5 -> 0.12 / 0.06 per input spike
    return liq


def p_in_effective(liq):
    """Connection probability the paper's rule uses: P_in = 0.1, or 1 for the input neurons with factor 6."""
    return 1.0 if 6 in liq.factors else C.X["P_in"]


def paper_rule_scale(ain, liq):
    """(a)'s rule: mu_in = I_target / (A_in P_in), always with X's excitatory leak (plan, Input size)."""
    return C.I_TARGET / (ain * p_in_effective(liq))


# ---------------------------------------------------------------- Maass's liquid, two versions (Maass check)
def build_maass_paper(seed, n_sub=None, n_channels=1):
    """Our reading of (c), the version Y copies: Y with all 11 factors and one Poisson input channel."""
    n_sub = n_sub or C.N_SUB_REFERENCE
    liq = build_candidate(C.ALL_FACTORS, seed, n_sub, n_channels)
    liq.meta["kind"] = "maass_paper"
    return liq


def build_maass_script(seed, n_sub=None, step_ms=C.SAMPLE_MS, d_replacement="fixed"):
    """The liquid of (d)'s Brian2 script, in threshold units, with its six settings (plan, level 1e):
    input onto I neurons decays over 6 ms, bad U, D, F draws replaced by the mean, U not capped, no input delay,
    an arriving spike acts after that step's state update (Brian2 schedule), each step's synaptic current held
    constant (exact neuron update with step-constant synaptic current). Also from (d): input to all 135 neurons
    at a fixed 18 / 9 nA, self-connections allowed, E neurons are the first 108 indices with shuffled positions,
    three-state synapses (used resources recover only once the current has ended)."""
    n_sub = n_sub or C.N_SUB_REFERENCE
    N = int(np.prod(C.MAASS["dims"]))
    h = step_ms / n_sub
    rng = np.random.default_rng([seed, 1000])
    idx = rng.permutation(N)
    pos = np.stack([idx % 3, (idx // 3) % 3, idx // 9], 1).astype(np.float64)
    n_exc = int(0.8 * N)
    is_exc = np.zeros(N, bool); is_exc[:n_exc] = True
    T = _type_matrix(is_exc)
    Cm = np.vectorize(lambda t: C.X["C"][C.TYPES[t]])(T)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    p = Cm * np.exp(-d2 / C.MAASS["lam"] ** 2)
    conn = np.random.default_rng([seed, 1001]).random((N, N)) < p          # self-connections allowed ((d))
    pre, post = np.nonzero(conn)
    order = np.lexsort((post, pre, T[pre, post]))
    pre, post = pre[order], post[order]
    typ = T[pre, post]
    J = C.MAASS["R_Mohm"] / C.MAASS["v_thresh_mV"]                         # nA -> thresholds (1 MOhm / 15 mV)
    A = np.array([C.MAASS["A_nA"][C.TYPES[t]] for t in typ])
    r = np.random.default_rng([seed, 1002])
    w = np.sign(A) * r.standard_exponential(len(pre)) * np.abs(A) * J       # Gamma(1, |A|) with sign ((d))
    def udf(name, scale, k):
        mean = np.array([C.MAASS[name][C.TYPES[t]] for t in typ]) * scale
        v = mean + 0.5 * mean * np.random.default_rng([seed, k]).standard_normal(len(pre))
        if d_replacement == "as_written":       # (d) as written: only the first two synapses of each group
            for ti in range(4):
                first = np.nonzero(typ == ti)[0][:2]
                v[first] = mean[first]
            return v
        return np.where(v < 0, mean, v) if name == "U" else np.where(v <= 0, mean, v)
    U, D, F = udf("U", 1.0, 1003), udf("D_s", 1000.0, 1004), udf("F_s", 1000.0, 1005)
    Ubar = np.array([C.MAASS["U"][t] for t in C.TYPES])
    Win = (np.where(is_exc, C.MAASS["in_nA"]["E"], C.MAASS["in_nA"]["I"]) * J)[None, :]   # J jump per input spike
    tau_m = C.MAASS["tau_m_ms"]
    beta = np.full(N, math.exp(-h / tau_m))
    in_tau = np.where(is_exc, 3.0, 6.0)                                     # (d): tau_stimulus 3 ms (E), 6 ms (I)
    a_in = np.exp(-h / in_tau)
    kappa = in_tau / (in_tau - tau_m) * (np.exp(-h / in_tau) - np.exp(-h / tau_m))   # Brian2's exact solution
    gap = np.where(is_exc, int((C.MAASS["refrac_ms"]["E"] + 1e-3 * h) / h), int((C.MAASS["refrac_ms"]["I"] + 1e-3 * h) / h))
    delay = np.array([int(round(C.MAASS["delay_ms"][t] / h)) for t in C.TYPES])   # Brian2 rounds delays to the grid
    syn_tau = np.array([C.MAASS["tau_s_ms"]["E"], C.MAASS["tau_s_ms"]["E"], C.MAASS["tau_s_ms"]["I"], C.MAASS["tau_s_ms"]["I"]])
    liq = Liquid(
        N=N, dims=C.MAASS["dims"], is_exc=is_exc, pos=pos, pre=pre, post=post, typ=typ, w=w, U=U, D_ms=D, F_ms=F,
        Ubar=Ubar, Win=Win, in_neurons=np.ones(N, bool), n_sub=n_sub, h_ms=h, beta=beta, tap=np.zeros(N),
        reset_mode="value", reset_value=C.RESET_TAP, start_lo=C.RESET_TAP, start_hi=1.0, rest=C.TAP,
        gap=gap.astype(np.int64), clamp=True, strict=True, schedule="brian2", syn_model="tsodyks3",
        delay=delay.astype(np.int64), cur_a=None, in_a=a_in, J_b=C.TAP, kappa=kappa, syn_tau_ms=syn_tau,
        factors=frozenset(), in_scale_fixed=1.0, in_split=(1.0, 0.5),
        meta={"kind": "maass_script", "seed": int(seed), "n_channels": 1, "d_replacement": d_replacement},
    )
    liq.g_maass = 1.0
    return liq


# ---------------------------------------------------------------- X from the authors' class
def from_authors_model(model, n_sub=1, snn_quirk=None):
    """X exactly as (a)'s class built it (weights already include its normalisations and its g_rec):
    used for the 1f regression and for E1's paper liquid at g = 1. Input weights are returned at their size
    (in_scale 1)."""
    import torch
    W = model.rsv_weights.detach().double().cpu().numpy()
    Win = model.input_rsv_weights.detach().double().cpu().numpy()
    is_exc = model.is_excit.detach().cpu().numpy().astype(bool)
    N = W.shape[0]
    T = _type_matrix(is_exc)
    pre, post = np.nonzero(W)
    order = np.lexsort((post, pre, T[pre, post]))
    pre, post = pre[order], post[order]
    h = C.SAMPLE_MS / n_sub
    tau = np.where(is_exc, -C.SAMPLE_MS / math.log(C.X["betaE"]), -C.SAMPLE_MS / math.log(C.X["betaI"]))
    liq = Liquid(
        N=N, dims=tuple(model.reservoir_dim), is_exc=is_exc, pos=_grid(model.reservoir_dim), pre=pre, post=post,
        typ=T[pre, post], w=W[pre, post], U=np.ones(len(pre)), D_ms=np.full(len(pre), np.inf),
        F_ms=np.full(len(pre), np.inf), Ubar=np.array([C.MAASS["U"][t] for t in C.TYPES]), Win=Win,
        in_neurons=(Win != 0).any(0), n_sub=n_sub, h_ms=h, beta=np.exp(-h / tau), tap=np.zeros(N),
        reset_mode="subtract", reset_value=0.0, start_lo=0.0, start_hi=0.0, rest=0.0,
        gap=np.full(N, steps_of(C.SAMPLE_MS, h)), clamp=False, strict=False, schedule="ours", syn_model="static",
        delay=np.full(4, steps_of(C.SAMPLE_MS, h)), cur_a=None, in_a=None, factors=frozenset(),
        in_scale_fixed=1.0, meta={"kind": "authors_X", "n_channels": int(Win.shape[0])},
    )
    if snn_quirk or (snn_quirk is None and C.CODE["x_update"] == "authors_code"):
        liq.snn_quirk, liq.strict = True, True
    return liq
