"""Every fixed number of the redesign plan, with its source or reason.

Sources (as labelled in the plan):
  (a) Perez-Lopez et al. 2026 and the authors' code, commit 02e45c4
  (c) Maass, Natschlager & Markram 2002, Section 5 and Appendix B
  (d) Schmitt 2022, Brian2 reproduction of (c)'s Figure 2 (maass2002.py)
  (e) Legenstein & Maass 2007
Numbers marked "plan" are the plan's own choices; the reason is in the plan (section named).
Numbers marked "code" were not fixed by the plan; their reason is given here and listed in NOTES.md.
"""
import math

# ---------------------------------------------------------------- time
FS = 360.0                          # MIT-BIH sampling rate (Hz), (a)
SAMPLE_MS = 1000.0 / FS             # 2.7778 ms, one ECG sample = one step of (a)'s liquid
BEAT_SAMPLES = 577                  # (a): beats are R-0.8 s .. R+0.8 s at 360 Hz
SIM_SAMPLES = 650                   # (a): num_sim_steps = 650 (input is zero after the beat)
R_SAMPLE = 288                      # plan, Readouts: the R peak is sample 288 of the 577-sample window
WIN, NWIN = 20, 32                  # (a): spike counts in 32 windows of 20 samples (film readout)
SNAPSHOT_SAMPLES = (288, 342, 396)  # plan, Readouts: R, R + 150 ms, R + 300 ms
STATE_TAU_MS = 30.0                 # (c): liquid states are spikes filtered with a 30 ms exponential
N_SUB_REFERENCE = 28                # plan, Time grid: 28 sub-steps per sample (0.0992 ms) is the reference grid

# ---------------------------------------------------------------- X: (a)'s liquid, both bugs fixed
X = dict(
    betaE=0.95, betaI=0.80,         # (a) leak per 2.778 ms step (about 54 / 12 ms)
    threshold=1.0,                  # (a) Uthr
    rho=0.9,                        # (a) target fill of the paper's input rule
    excit_ratio=0.8,                # (a) and (c)
    C={"EE": 0.3, "EI": 0.2, "IE": 0.4, "II": 0.1},   # (a) and (c)
    lam=0.8,                        # (a)
    mu_rsv={"EE": 0.020, "EI": 0.025, "IE": 0.050, "II": 0.040},   # (a) type means
    P_in=0.1,                       # (a) code line 691
    dims=(7, 7, 7),                 # (a) 343 neurons
)
I_TARGET = X["rho"] * (1 - X["betaE"]) * X["threshold"]     # 0.045: (a)'s rule, mean input per step
TYPES = ("EE", "EI", "IE", "II")    # pre-post

# ---------------------------------------------------------------- Maass (c), in the paper's units (threshold 1)
MAASS = dict(
    tau_m_ms=30.0,                  # (c) membrane time constant
    v_thresh_mV=15.0, v_reset_mV=13.5, I_b_nA=13.5, R_Mohm=1.0,   # (c); background 13.5 mV of 15
    refrac_ms={"E": 3.0, "I": 2.0},  # (c)
    tau_s_ms={"E": 3.0, "I": 6.0},   # (c) synaptic current decay by presynaptic type
    tau_in_ms=3.0,                   # plan: input synapses are excitatory, so 3 ms for every target
    delay_ms={"EE": 1.5, "EI": 0.8, "IE": 0.8, "II": 0.8},   # (c)
    A_nA={"EE": 30.0, "EI": 60.0, "IE": -19.0, "II": -19.0},  # (c)
    U={"EE": 0.5, "EI": 0.05, "IE": 0.25, "II": 0.32},          # (c)
    D_s={"EE": 1.1, "EI": 0.125, "IE": 0.7, "II": 0.144},       # (c)
    F_s={"EE": 0.05, "EI": 1.2, "IE": 0.02, "II": 0.06},        # (c)
    udf_spread=0.5,                  # (c): SD 50% of the mean
    lam=2.0,                         # (c)
    in_nA={"E": 18.0, "I": 9.0},     # (c) input amplitudes
    in_fraction=0.3,                 # (c) input injected into 30% of neurons
    dims=(15, 3, 3),                 # (c) 135-neuron column
    input_rate_hz=20.0,              # (d) (and (c)'s Figure 4 task)
)
TAP = MAASS["I_b_nA"] * MAASS["R_Mohm"] / MAASS["v_thresh_mV"]    # 0.9 of threshold
RESET_TAP = MAASS["v_reset_mV"] / MAASS["v_thresh_mV"]            # 0.9

def _push(A_nA, tau_s_ms, U=1.0):
    """Voltage push (in thresholds) of one current pulse A*exp(-t/tau_s) through R into a 30 ms membrane:
    A * U * tau_s * R / (tau_m * V_thresh), the tap's convention (plan, factors 9 and 10)."""
    return A_nA * 1e-9 * U * tau_s_ms * 1e-3 * MAASS["R_Mohm"] * 1e6 / (MAASS["tau_m_ms"] * 1e-3 * MAASS["v_thresh_mV"] * 1e-3)

_pre = {"EE": "E", "EI": "E", "IE": "I", "II": "I"}
MAASS_PUSH = {t: abs(_push(MAASS["A_nA"][t], MAASS["tau_s_ms"][_pre[t]], MAASS["U"][t])) for t in TYPES}
# EE 0.1, EI 0.02, IE 0.0633, II 0.0811 (plan, factor 3: ratios 1 / 0.2 / 0.63 / 0.81; factor 10: EE 0.1)
MAASS_RATIOS = {t: MAASS_PUSH[t] / MAASS_PUSH["EE"] for t in TYPES}
MAASS_IN_PUSH = {k: _push(MAASS["in_nA"][k], MAASS["tau_in_ms"]) for k in ("E", "I")}   # 0.12 / 0.06 (factor 9)
MAASS_EE_PUSH = MAASS_PUSH["EE"]   # 0.1 (factor 10)

# ---------------------------------------------------------------- adaptive thresholds (Bellec et al. 2018, LSNN)
# Bellec, Salaj, Subramoney, Legenstein & Maass 2018, "Long short-term memory and learning-to-learn in networks of
# spiking neurons", Section 5 parameters: threshold B_j = b0 + beta b_j, b_j <- rho b_j + (1 - rho) z_j with
# rho = exp(-dt / tau_a) and z_j = 1/dt (ms) at a spike; beta 1.6, b0 0.03, tau_a uniform on 1..1000 ms, 40% of the
# neurons adaptive. Translation to Maass's neurons (rest 0.9, threshold 1): the threshold rise is taken relative to the
# rest-to-threshold gap, theta_j = 1 + (1 - 0.9) beta b_j / b0, since that gap decides firing in both models.
ADAPT = dict(beta=1.6, b0=0.03, tau_ms=(1.0, 1000.0), fraction=0.4)

# ---------------------------------------------------------------- the 11 factors (plan table)
FACTOR_NAMES = {
    1: "reach lambda", 2: "code-only normalisations", 3: "E/I balance", 4: "leak", 5: "operating point",
    6: "input wiring", 7: "fine timing", 8: "dynamic synapses", 9: "input size", 10: "weight scale",
    11: "size and shape",
}
ALL_FACTORS = frozenset(range(1, 12))
LEAK_MS_Y = 30.0                    # factor 4: (c)'s 30 ms for E and I

# ---------------------------------------------------------------- gain grid (plan, Choosing the recurrent gain; decision 5)
G_LOW = (0.1, 0.3)
G_COARSE = (1, 3, 10, 30, 100, 300, 1000)
G_EXTEND_STEP, G_EXTEND_MAX = 3.0, 3
G_FINE_STEP, G_FINE_K = 1.4, 3      # g* x 1.4^k, k = -3..3 (1.4^3 ~ 2.7 splits a 3x step into three)

# ---------------------------------------------------------------- level thresholds (plan, Correctness criteria)
TOL_1A = 1e-9                       # 1a, 1b open loop (float64)
TOL_1C = 1e-4                       # 1c float32 vs float64
LEVEL2_SE = 4.5                     # level 2 (decision 24)
BAND_HZ = (2.0, 40.0)               # level 3 firing band (decision 7)
HIGH_HZ, HIGH_FRAC = 100.0, 0.05    # at most 5% of neurons above 100 Hz (decision 7)
N_CAL_BEATS = 500                   # plan: 500 beats for level 3 and the rank scores (decision 8)
GEN_BEATS, GEN_COPIES = 4, 125      # decision 8: 4 beats x 125 jittered copies
JITTER_MS = 4.0                     # decision 8: (e)'s 4 ms jitter for its 30-channel patterns
STATE_READ_SAMPLE = 396             # decision 8: states read at R + 300 ms
START_TEST_SAMPLE = 576             # level 3: starting-state rule judged at sample 576
MEMORY_T_MS = (0, 25, 50, 100, 200, 400)   # level 4 lags
MEMORY_PASS_T_MS = 100              # level 4 pass lag (plan: well past 30-54 ms membranes and 3-6 ms currents)
MEMORY_MIN_GAIN = 0.05              # level 4 (decision 9)
MEMORY_WARMUP = 100                 # level 4: 278 ms warm-up
SURROGATE_TOL = 0.10                # level 4: surrogate spikes per sample within 10% of ECG's
E1D_N_TRY_MAX = 28                  # 1d: n = 1, 2, 3, ... up to the reference grid
E1E_T_CRIT = 4.2                    # 1e: t at 8 df, two-sided 0.3%
E1E_FRAC = 0.95                     # 1e: 95% of time points
E1E_RUNS = 5                        # 1e: 5 runs per side

SEEDS_SEARCH = (1, 2, 3)            # code: liquid seeds for the ladder search (the plan fixes 3 seeds, not their values)
SEEDS_CONFIRM = (101, 202, 303)     # plan, confirmation
SEEDS_RETRY = (404, 505, 606)       # plan, confirmation retry
SEEDS_TEST_REDESIGN = (707, 808, 909)   # plan, Maass check redesign
SEEDS_CLASSIFY = (2026, 69, 420)    # plan, ground rules

# ---------------------------------------------------------------- code choices the plan does not fix (see NOTES.md)
CODE = dict(
    twin_beats=100,          # twin-run cross-check per g and seed: a reported cross-check, 100 beats give a stable mean ratio
    twin_early=(289, 306),   # samples of the first 50 ms after the extra spike
    twin_late=(559, 576),    # samples of the last 50 ms of the beat
    memory_sequences=500,    # level 4: same count as level 3's beats; 250 train / 250 test sequences
    ridge_alphas=tuple(10.0 ** k for k in range(-3, 5)),   # level 4 inner-CV penalty grid
    ridge_inner_folds=5,     # level 4 inner CV (grouped by sequence)
    calib_tol=0.01,          # factor 5: bisection stops at 1% of the target rate (well inside seed-to-seed spread)
    calib_iters=40,          # factor 5: bisection cap
    search_split_seed=0,     # DS1 patients halved by a fixed random permutation of the 22 records
    sim_batch=500,           # beats per GPU batch (fixed on every machine: plan, determinism)
    x_update="equations",    # X's reset follows (a)'s Eqs. 17-19 (the plan). "authors_code" (the snnTorch neuron of
                             # (a)'s code) exists only so the tests can show the difference; the notebook never sets it
)
