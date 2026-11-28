"""Label-free inputs for the search: the DS1 halves, the 500 calibration beats, the generalisation and twin sets,
the memory test's random surrogate, and Poisson trains for the Maass check."""
from dataclasses import dataclass
import numpy as np

from . import config as C
from .measure import level3_sets, surrogate_walks


def multithreshold(X, num_levels=200):
    """The authors' Multi-threshold encoder (detect_edges='both', include_endpoints=True): a spike on threshold j
    at sample t when x crosses it between t-1 and t. X: (n, T) in [0, 1] -> (n, T, K) uint8."""
    X = np.asarray(X)
    thr = np.linspace(0.0, 1.0, num_levels, dtype=X.dtype)
    above = X[:, :, None] >= thr
    out = np.zeros(above.shape, np.uint8)
    out[:, 1:] = (above[:, 1:] != above[:, :-1])
    return out


def split_records(records, seed=C.CODE["search_split_seed"]):
    """DS1's patients halved without labels: a fixed random permutation of the record list."""
    rec = sorted(records)
    perm = np.random.default_rng(seed).permutation(len(rec))
    half = len(rec) // 2
    return sorted(rec[i] for i in perm[:half]), sorted(rec[i] for i in perm[half:])


@dataclass
class SearchData:
    name: str
    sets: dict                 # sep (500, T, C), gen (500, T, C), twin (100, T, C)
    ain: float                 # input spikes per sample (A_in) on this half, labels unused
    mem_spikes: np.ndarray     # (500, T, C) surrogate encoded
    mem_targets: np.ndarray    # (500, T) input spike count per sample
    surrogate_ain: float
    real_spikes: np.ndarray = None    # secondary memory check on real beats
    real_targets: np.ndarray = None
    meta: dict = None


def make_search_data(name, beats, encode, seed, n_cal=None, n_mem=None):
    """beats: (n, 577) normalised config-11 beats of one half, in a fixed order (labels never used).
    encode: (m, 577) -> (m, 577, C) spikes."""
    n_cal = n_cal or C.N_CAL_BEATS
    n_mem = n_mem or C.CODE["memory_sequences"]
    rng = np.random.default_rng([seed, 51])
    pick = np.sort(rng.choice(len(beats), n_cal + C.GEN_BEATS, replace=False))
    perm = rng.permutation(pick)
    sep_idx, gen_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    sep = encode(beats[sep_idx])
    gen_base = encode(beats[gen_idx])
    sets = level3_sets(sep, gen_base, seed_sets=seed)
    # A_in over the whole half (the authors' quantity: input spikes per sample, summed over channels)
    tot, cnt = 0.0, 0
    for s in range(0, len(beats), 2000):
        e = encode(beats[s:s + 2000]); tot += float(e.sum(dtype=np.int64)); cnt += e.shape[0] * e.shape[1]
    ain = tot / cnt
    steps = np.abs(np.diff(beats, axis=1)).ravel()
    walks = surrogate_walks(steps, n_mem, beats.shape[1], seed)
    mem = encode(walks)
    targets = mem.sum(2).astype(np.int64)
    s_ain = float(mem.sum(dtype=np.int64) / (mem.shape[0] * mem.shape[1]))
    # secondary check: real beats, target = deviation from the R-aligned mean beat
    mean_beat = beats.mean(0)
    real_targets = beats[sep_idx] - mean_beat[None, :]
    return SearchData(name=name, sets=sets, ain=ain, mem_spikes=mem, mem_targets=targets, surrogate_ain=s_ain,
                      real_spikes=sep, real_targets=real_targets,
                      meta={"n_beats_half": int(len(beats)), "sep_idx": sep_idx.tolist(), "gen_idx": gen_idx.tolist(),
                            "surrogate_vs_ecg": s_ain / ain})


def poisson_trains(n, T, rate_hz, seed, stream, step_ms=C.SAMPLE_MS):
    """n Poisson trains of T samples, counts per sample (one channel)."""
    rng = np.random.default_rng([seed, stream])
    return rng.poisson(rate_hz * step_ms / 1000.0, (n, T, 1)).astype(np.uint8)


def make_maass_data(seed, T=C.BEAT_SAMPLES, n_cal=None, n_mem=None):
    """The Maass check's inputs: 20 Hz Poisson trains of beat length stand in for beats."""
    n_cal = n_cal or C.N_CAL_BEATS
    n_mem = n_mem or C.CODE["memory_sequences"]
    r = C.MAASS["input_rate_hz"]
    sep = poisson_trains(n_cal, T, r, seed, 61)
    gen_base = poisson_trains(C.GEN_BEATS, T, r, seed, 62)
    sets = level3_sets(sep, gen_base, seed_sets=seed)
    mem = poisson_trains(n_mem, T, r, seed, 63)
    return SearchData(name="maass_poisson", sets=sets, ain=float(sep.mean() * 1), mem_spikes=mem,
                      mem_targets=mem[:, :, 0].astype(np.int64), surrogate_ain=float(mem.mean()), meta={"rate_hz": r})
