"""How wide must a window around each beat be to also contain the previous and the next beat?
Reads the MIT-BIH annotations only (read-only, about a minute). Usage:
    python rr_window.py ~/lsm_ecg/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0
For every beat that the pipeline classifies (DS1, the same beat symbols and the same +-0.8 s edge rule), the
neighbours are the previous and next annotated beats of any kind (as in the RR features). A window R +- h contains
both neighbouring R peaks when h >= max(RR before, RR after); it contains their whole QRS when h >= that + 0.06 s
(half of the 120 ms upper limit of a normal QRS)."""
import os, sys
import numpy as np
import wfdb

DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/lsm_ecg/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0")
DS = {"DS1": "101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230".split(),
      "DS2": "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()}
CLASS = {**{s: "N" for s in "NLRej"}, **{s: "SVEB" for s in "AaJS"}, **{s: "VEB" for s in "VE"}}
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
QRS_HALF = 0.06
SET = sys.argv[2] if len(sys.argv) > 2 else "DS1"

need, cls, rec_of = [], [], []
for rec in DS[SET]:
    ann = wfdb.rdann(os.path.join(DIR, rec), "atr")
    n_sig = wfdb.rdheader(os.path.join(DIR, rec)).sig_len
    fs = 360.0
    beats = np.array([s for s, y in zip(ann.sample, ann.symbol) if y in BEAT_SYMBOLS])
    L = int(round(0.8 * fs))
    for s, y in zip(ann.sample, ann.symbol):
        if y not in CLASS or s - L < 0 or s + L + 1 > n_sig:
            continue                                      # not a classified beat (the pipeline's own rule)
        i = np.searchsorted(beats, s)
        if i == 0 or i >= len(beats) - 1:
            continue                                      # first or last beat of the record: a neighbour is missing
        need.append(max(s - beats[i - 1], beats[i + 1] - s) / fs); cls.append(CLASS[y]); rec_of.append(rec)
need, cls, rec_of = np.array(need), np.array(cls), np.array(rec_of)

print(f"{SET}: {len(need)} classified beats with both neighbours. Needed half-width h = max(RR before, RR after):")
for q in (50, 90, 95, 99, 99.9, 100):
    print(f"  {q:>5}% of beats: h <= {np.percentile(need, q):.3f} s")
print("\nShare of beats whose window contains both neighbours (R peak / whole QRS), by half-width:")
print("   h (s)     all            N              SVEB           VEB")
for h in np.round(np.arange(0.8, np.ceil(need.max() * 10) / 10 + 0.1, 0.1), 1):
    row = []
    for c in (None, "N", "SVEB", "VEB"):
        m = np.ones(len(need), bool) if c is None else cls == c
        row.append(f"{100 * np.mean(need[m] <= h):6.2f}/{100 * np.mean(need[m] + QRS_HALF <= h):6.2f}")
    print(f"  {h:4.1f}   " + "  ".join(row))
    if np.all(need + QRS_HALF <= h):
        break
for target in (100.0, 99.9, 99.0, 95.0):
    h = np.ceil(np.percentile(need + QRS_HALF, target) * 10) / 10
    print(f"\nShortest half-width (0.1 s steps) that fits whole neighbouring QRS for {target}% of beats: +-{h:.1f} s")
top = np.argsort(need)[::-1][:10]
print("\nThe 10 widest needs (record, class, h):", ", ".join(f"{rec_of[i]} {cls[i]} {need[i]:.2f}" for i in top))
