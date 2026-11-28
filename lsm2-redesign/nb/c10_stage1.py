# Stage 1 (plan, Stages): level 1 checks, the Maass check (testing the test), the X to Y ladder (E0), the
# confirmation and the speed test. No labels are used anywhere in this cell. Each step saves its result in
# checks/stage1/, so a restarted session continues where it stopped. The last step writes the calibration file.
import importlib, urllib.request
from lsm2 import config as LC, liquid as LL, sim as LS, measure as LMe, level1 as L1, level2 as L2, maass_check as LMC
from lsm2 import ladder as LLd, calib as LCa, data as LDa, gain as LGa

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # the replication's chart colours
LC.CODE["x_update"] = X_UPDATE
S1_DIR = os.path.join(WORK_DIR, "checks", "stage1")
os.makedirs(S1_DIR, exist_ok=True)
CAL_PATH = os.path.join(WORK_DIR, "checks", "redesign_calibration.json")
MAASS_SCRIPT_SHA256 = "81f50fec6001c76b16ba548f2e03e915de9a9799c554ca33aa0598e8bb47ac4e"   # the copy the code was tested on

if SMOKE_TEST:     # a logic test of every step on a few beats and a coarse grid; the numbers mean nothing
    LC.N_CAL_BEATS, LC.GEN_COPIES = 24, 6
    LC.CODE.update(twin_beats=4, memory_sequences=16)
    LC.G_COARSE, LC.G_LOW, LC.G_FINE_K = (1, 10, 100), (0.3,), 1
    LC.N_SUB_REFERENCE, LC.E1D_N_TRY_MAX = 4, 4
    LC.SEEDS_SEARCH, LC.SEEDS_CONFIRM, LC.SEEDS_RETRY = (1, 2), (101, 202), (404, 505)

def s1_gate(ok, msg):
    """A stop point of the plan. In the smoke test the numbers mean nothing, so it only warns."""
    if ok:
        return
    if SMOKE_TEST:
        log("SMOKE TEST: would stop here: " + msg)
        return
    raise RuntimeError(msg)

def s1_path(step):
    return os.path.join(S1_DIR, f"{step}.json")

def s1_done(step):
    return os.path.exists(s1_path(step))

def s1_save(step, obj):
    jdump(obj, s1_path(step))

def s1_load(step):
    return json.load(open(s1_path(step)))

# ---------------------------------------------------------------- label-free inputs: the two halves of DS1
S1_SEARCH_RECS, S1_CONFIRM_RECS = LDa.split_records(DS_RECORDS["DS1"])
def half_beats(recs, filt="morph", norm="robust"):
    """One half of DS1 through a front end's filter and normalisation (robust percentiles fitted on that half)."""
    D = DATA[filt]["DS1"]
    rows = np.nonzero(np.isin(D["pid"], recs))[0]
    if norm == "standard":
        Xn = np.asarray(A_strat.minmax_per_beat(D["X"][rows]), np.float32)
    else:
        mn, mx = A_strat.robust_minmax_fit(D["X"][rows], low=0.1, high=99.9)
        Xn = np.asarray(A_strat.robust_minmax_apply(D["X"][rows], mn, mx), np.float32)
    return Xn, rows

def enc_np(X, enc="multi"):
    out = []
    for s in range(0, len(X), 2000):
        out.append(encode(X[s:s + 2000], enc))
    return np.concatenate(out)

S1_CTX = LMe.Ctx(DEVICE, torch.float32, NEW_SIM_BATCH, log)
S1_CTX64 = LMe.Ctx(DEVICE, torch.float64, NEW_SIM_BATCH, log)
_beats_s, _rows_s = half_beats(S1_SEARCH_RECS)
_beats_c, _rows_c = half_beats(S1_CONFIRM_RECS)
# the package's encoder must be the authors' (used inside the package for the surrogate only)
assert np.array_equal(LDa.multithreshold(_beats_s[:50]), encode(_beats_s[:50], "multi")), "encoder mismatch"
SEARCH = LDa.make_search_data("search", _beats_s, enc_np, seed=1)
CONFIRM = LDa.make_search_data("confirm", _beats_c, enc_np, seed=2)
log(f"stage 1 inputs: search half {len(S1_SEARCH_RECS)} records ({len(_beats_s)} beats), confirmation half "
    f"{len(S1_CONFIRM_RECS)} records ({len(_beats_c)} beats); A_in {SEARCH.ain:.4f} / {CONFIRM.ain:.4f}; "
    f"surrogate/ECG input spikes {SEARCH.meta['surrogate_vs_ecg']:.3f} / {CONFIRM.meta['surrogate_vs_ecg']:.3f}")

def x_liquid(seed, n_sub, n_ch=200):
    return LL.build_candidate(frozenset(), seed, n_sub, n_ch)

def y_liquid(seed, n_sub, n_ch=200):
    return LL.build_candidate(LC.ALL_FACTORS, seed, n_sub, n_ch)

def x_scale(liq):
    return LL.paper_rule_scale(SEARCH.ain, liq)

def maass_script_path():
    if MAASS_SCRIPT_PATH:
        p = os.path.expanduser(MAASS_SCRIPT_PATH)
    else:
        p = os.path.join(SCRATCH_DIR, "maass2002.py")
        if not os.path.exists(p):
            urllib.request.urlretrieve(MAASS_SCRIPT_URL, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != MAASS_SCRIPT_SHA256:
        log(f"WARNING: (d)'s script differs from the tested copy (sha256 {h[:12]} vs {MAASS_SCRIPT_SHA256[:12]})")
    return p

# ---------------------------------------------------------------- step 1: level 1
def stage1_level1():
    out = {}
    s0 = LC.SEEDS_SEARCH[0]
    beats10 = SEARCH.sets["sep"][:10 if not SMOKE_TEST else 2]
    try:
        import pytest
        tdir = os.path.join(SCRATCH_DIR, "lsm2_tests")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", tdir], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": SCRATCH_DIR, "MAASS_SCRIPT": maass_script_path(), "AUTHORS_REPO_DIR": REPO_DIR})
        out["1a"] = {"pass": r.returncode == 0, "tail": r.stdout[-2000:]}
    except ImportError:
        out["1a"] = {"pass": None, "note": "pytest not installed: level 1a was run by the build, not on this machine"}
    log(f"  1a: {out['1a']}")
    lx, ly = x_liquid(s0, 1), y_liquid(s0, LC.N_SUB_REFERENCE)
    gx, inx, gy, iny = 1.0, x_scale(lx), ly.g_maass, ly.in_scale_fixed
    out["1b"] = {"X": L1.level1b(lx, beats10, gx, inx, s0, log=log), "Y": L1.level1b(ly, beats10, gy, iny, s0, log=log)}
    out["1c"] = {"X": L1.level1c(lx, beats10, gx, inx, s0, DEVICE, log=log), "Y": L1.level1c(ly, beats10, gy, iny, s0, DEVICE, log=log)}
    out["1d"] = L1.level1d(S1_CTX, x_liquid, y_liquid, lambda l: 1.0, x_scale, lambda l: l.g_maass,
                           lambda l: l.in_scale_fixed, SEARCH.sets, LC.SEEDS_SEARCH, log=log)
    out["1e"] = L1.level1e(S1_CTX, maass_script_path(), seeds=list(range(1, LC.E1E_RUNS + 1)) if not SMOKE_TEST else [1, 2],
                           n_pairs=200 if not SMOKE_TEST else 3, processes=LEVEL1E_PROCESSES, log=log, d_replacement=D_REPLACEMENT)
    out["1f"] = stage1_level1f()
    gate = {k: (out[k]["X"]["pass"] and out[k]["Y"]["pass"]) if k in ("1b", "1c") else out[k]["pass"] for k in ("1b", "1c", "1d", "1e")}
    out["gate"] = gate
    out["pass"] = all(gate.values()) and out["1a"]["pass"] is not False
    return out

def stage1_level1f():
    """Cached X runs vs the new simulator in X mode: identical window counts on the saved DS2 state vectors of
    config 11 (the replication saved them for the fixed liquid, seeds 2026 / 69 / 420)."""
    D1, D2 = DATA["morph"]["DS1"], DATA["morph"]["DS2"]
    X1n, X2n, _ = normalize_final(D1["X"], D2["X"], "robust")
    ain = measure_ain(X1n, np.arange(len(X1n)), "multi")
    rows = []
    for seed in SEEDS:
        spec = lsm_spec("fixed", seed, "morph", "robust", "multi", "svm", "final")
        p = os.path.join(WORK_DIR, "states", f"{run_id_of(spec)}__DS2_window_spike_counts.npz")
        if not os.path.exists(p):
            rows.append({"seed": seed, "available": False}); continue
        saved = np.load(p)["counts"]
        model = build_liquid(200, I_TARGET / (ain * LIQ["P_in"]), seed, True)
        liq = LL.from_authors_model(model)
        n = len(X2n) if not SMOKE_TEST else 64
        sm = LS.Simulator(liq, device=DEVICE, dtype=torch.float32, g=1.0, in_scale=1.0)
        new = np.zeros((n, NWIN * N_NEUR), np.uint8)
        for s in range(0, n, NEW_SIM_BATCH):
            x = encode_t(X2n[s:s + NEW_SIM_BATCH], "multi")
            o = sm.run(x, np.zeros((len(x), liq.N)), LIQ["num_sim_steps"], rec={"windows": (WIN, NWIN)})
            new[s:s + len(x)] = o["windows"].reshape(len(x), -1).to(torch.uint8).cpu().numpy()
        diff = int((new != saved[:n]).sum())
        rows.append({"seed": seed, "available": True, "cells_differing": diff, "identical": diff == 0, "beats": n})
        log(f"  1f seed {seed}: {diff} window counts differ")
    ok = bool(rows) and all(r.get("identical") for r in rows)
    return {"pass": ok, "rows": rows, "note": "1f never stops the plan: it decides whether cached runs are reused"}

# ---------------------------------------------------------------- step 2: the Maass check
def stage1_maass_check():
    data = LDa.make_maass_data(seed=1)
    return LMC.maass_check(S1_CTX, data, seeds=LC.SEEDS_SEARCH, log=log)

# ---------------------------------------------------------------- step 3: the ladder and the confirmation
def stage1_ladder(n_fine, maass_target):
    cache_dir = os.path.join(S1_DIR, "candidates"); os.makedirs(cache_dir, exist_ok=True)
    cache = {}
    for fn in glob.glob(os.path.join(cache_dir, "*.json")):
        d = json.load(open(fn)); cache[d["_key"]] = d["record"]
    def on_record(k, rec):
        jdump({"_key": k, "record": rec}, os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9.+_-]", "_", k) + ".json"))
    lad = LLd.Ladder(S1_CTX, SEARCH, CONFIRM, n_fine, maass_target, seeds=LC.SEEDS_SEARCH, log=log,
                     on_record=on_record, cache=cache)
    out = lad.run()
    out["records"] = lad.records
    if out.get("final") is not None:          # 1b and 1c again on the final liquid's exact combination of factors
        f = frozenset(out["final"]); s0 = LC.SEEDS_SEARCH[0]
        lf = LL.build_candidate(f, s0, out["n_sub"], 200)
        g = lf.g_maass if 10 in f else out["g"]
        beats = SEARCH.sets["sep"][:10 if not SMOKE_TEST else 2]
        out["final_1b"] = L1.level1b(lf, beats, g, out["in_scale"], s0, log=log)
        out["final_1c"] = L1.level1c(lf, beats, g, out["in_scale"], s0, DEVICE, log=log)
    return out

def ladder_figures(records):
    for k, rec in records.items():
        if "sweep" not in rec:
            continue
        t = rec["sweep"]["table"]
        gs = sorted(float(g) for g in t if float(g) > 0)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ab = [t[repr(g)]["abs_g"] for g in gs]
        for a, key, lab in zip(ax, ("rate_hz", "gap", "twin_ratio"), ("firing (Hz)", "separation - generalisation", "twin late/early")):
            a.plot(ab, [t[repr(g)].get(key, np.nan) for g in gs], "o-", color=SERIES[0])
            a.set_xscale("log"); a.set_xlabel("mean isolated EE push (thresholds)"); a.set_ylabel(lab)
            a.axvline(LC.MAASS_EE_PUSH, color="#999", ls="--", lw=1)
            if rec.get("g_rule") is not None:
                a.axvline(t[repr(float(rec["g_rule"]))]["abs_g"], color=SERIES[1], lw=1.5)
        ax[0].axhspan(*LC.BAND_HZ, color="#ddd", zorder=0)
        fig.suptitle(f"candidate {rec['key']} (n = {rec['n_sub']}): chosen g in orange, Maass's scale dashed")
        fig.tight_layout(); fig.savefig(os.path.join(WORK_DIR, "figures", f"ladder_{re.sub(r'[^A-Za-z0-9+]', '_', k)}.png"), dpi=110)
        plt.close(fig)

def ladder_table(records):
    rows = []
    for k, r in records.items():
        rows.append({"record": k, "candidate": r["key"], "n_sub": r["n_sub"], "data": r["data"], "pass": r["pass"],
                     "first_failed": r["first_failed_level"], "g_rule": r["g_rule"],
                     "abs_g_used": np.mean([v for v in r["abs_g_used"].values() if v is not None]) if any(v is not None for v in r["abs_g_used"].values()) else None,
                     "band_limited": r.get("band_limited"), "input_rule": r["input"]["rule"], "in_scale": r["in_scale"],
                     "level2": r["level2"]["pass"], "level3": r["level3"]["pass"], "level4": r["level4"]["pass"],
                     "rate_hz": np.mean([x["rate_hz"] for x in r["level3"].get("rows", [])]) if r["level3"].get("rows") else None,
                     "memory_gain_100ms": np.mean(list(r["level4"]["gain_100ms"].values())) if "gain_100ms" in r["level4"] else None})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK_DIR, "tables", "ladder_candidates.csv"), index=False)
    return df

# ---------------------------------------------------------------- input sizes for every front end (stage 2)
def front_end_scales(final, n_sub, maass_target, config11_scale):
    """Input size per front end, measured once on the search half (plan, Input and calibration)."""
    out = {}
    f = frozenset(final)
    for filt in ("morph", "median"):
        for norm in ("standard", "robust"):
            for enc in ("multi", "rate", "latency"):
                key = f"{filt}-{norm}-{enc}"
                if (filt, norm, enc) == ("morph", "robust", "multi"):
                    out[key] = {"scale": config11_scale, "rule": "frozen by the ladder (config 11)"}; continue
                Xn, _ = half_beats(S1_SEARCH_RECS, filt, norm)
                n_ch = 200 if enc == "multi" else N_IN[enc]
                liqs = [LL.build_candidate(f, s, n_sub, n_ch) for s in LC.SEEDS_SEARCH]
                for s, l in zip(LC.SEEDS_SEARCH, liqs):
                    l.meta["seed"] = s
                torch.manual_seed(12345)                      # the Rate encoder is stochastic
                pick = np.sort(np.random.default_rng(0).choice(len(Xn), min(LC.N_CAL_BEATS, len(Xn)), replace=False))
                cal = enc_np(Xn[pick], enc)
                ain = measure_ain(Xn, np.arange(len(Xn)), enc)
                sc, how = LCa.input_scale(S1_CTX, liqs, cal, ain, maass_target, log=log)
                out[key] = {"scale": sc, **how}
    return out

def front_end_levels(cal):
    """E2: each front end's four-level result with the final liquid at config 11's g and its own input size
    (reported, not used to re-tune; plan, Experiments)."""
    f = frozenset(cal["final"]); out = {}
    lad = LLd.Ladder(S1_CTX, SEARCH, CONFIRM, cal["n_sub"], cal["factor5_target_hz"], seeds=LC.SEEDS_SEARCH, log=log)
    lad.n_sub_for = lambda factors, y_grid=None: cal["n_sub"]
    for filt in ("morph", "median"):
        for norm in ("standard", "robust"):
            for enc in ("multi", "rate", "latency"):
                key = f"{filt}-{norm}-{enc}"
                Xn, _ = half_beats(S1_SEARCH_RECS, filt, norm)
                torch.manual_seed(12345)
                data = SEARCH if key == "morph-robust-multi" else LDa.make_search_data(key, Xn, lambda X, e=enc: enc_np(X, e), seed=1)
                r = lad.evaluate(f, data=data, frozen={"g": cal["g"], "in_scale": cal["input_scales"][key]["scale"]},
                                 n_channels=200 if enc == "multi" else N_IN[enc], tag=f"frontend|{key}")
                out[key] = {"pass": r["pass"], "first_failed_level": r["first_failed_level"],
                            "rate_hz": [x["rate_hz"] for x in r["level3"].get("rows", [])],
                            "memory_gain_100ms": r["level4"].get("gain_100ms")}
                log(f"  front end {key}: four levels {'pass' if r['pass'] else 'fail at ' + str(r['first_failed_level'])} (reported only)")
    return out

# ---------------------------------------------------------------- step 4: speed
def stage1_speed(cal):
    f = frozenset(cal["final"]); liq = LL.build_candidate(f, SEEDS[0], cal["n_sub"], 200)
    g = liq.g_maass if 10 in f else cal["g"]
    n = 2 * NEW_SIM_BATCH if not SMOKE_TEST else 32
    x = SEARCH.sets["sep"][:n] if n <= len(SEARCH.sets["sep"]) else enc_np(_beats_s[:n])
    t0 = time.time()
    LMe.simulate(S1_CTX, liq, g, cal["input_scales"]["morph-robust-multi"]["scale"], x, np.zeros((len(x), liq.N)) + liq.start_lo,
                 LIQ["num_sim_steps"], {"windows": (WIN, NWIN), "states_at": list(LC.SNAPSHOT_SAMPLES)})
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    bps = len(x) / (time.time() - t0)
    return {"beats_per_s": bps, "device": ENV.get("gpu", str(DEVICE)), "machine": MACHINE_ID,
            "note": "use this as this machine's weight in UNIT_SHARES"}

# ---------------------------------------------------------------- run stage 1
STAGE1 = {}
_1c_path = os.path.join(WORK_DIR, "checks", f"level1c_{out_tag('card', MACHINE_ID)}.json")
if os.path.exists(CAL_PATH) and not os.path.exists(_1c_path):
    # every machine runs level 1c on its own card before its stage-2 units (plan, Stages)
    cal = json.load(open(CAL_PATH)); f = frozenset(cal["final"]); s0 = LC.SEEDS_SEARCH[0]
    res = {}
    for name, lq, g_, sc_ in (("X", x_liquid(s0, 1), 1.0, None), ("Y", y_liquid(s0, LC.N_SUB_REFERENCE), None, None),
                              ("final", LL.build_candidate(f, s0, cal["n_sub"], 200), cal["g"], cal["in_scale"])):
        g_ = lq.g_maass if (g_ is None or (name == "final" and 10 in f)) else g_
        sc_ = sc_ if sc_ is not None else (lq.in_scale_fixed or x_scale(lq))
        res[name] = L1.level1c(lq, SEARCH.sets["sep"][:10], g_, sc_, s0, DEVICE, log=log)
    res["pass"] = all(v["pass"] for v in res.values())
    jdump(res, _1c_path)
    if not res["pass"]:
        raise RuntimeError("level 1c failed on this card: stage 2 does not start here")
if RUN_STAGE1 and not os.path.exists(CAL_PATH):
    if "level1" in STAGE1_STEPS:
        if not s1_done("level1"):
            log("stage 1, step 1: level 1 checks"); s1_save("level1", stage1_level1())
        STAGE1["level1"] = s1_load("level1")
        s1_gate(STAGE1["level1"]["pass"], f"level 1 failed {STAGE1['level1']['gate']}: fix the bug before anything else "
                f"runs (details in {s1_path('level1')}; delete it to rerun)")
    if "maass_check" in STAGE1_STEPS:
        if not s1_done("maass_check"):
            log("stage 1, step 2: testing the test on Maass's liquid"); s1_save("maass_check", stage1_maass_check())
        STAGE1["maass_check"] = s1_load("maass_check")
        s1_gate(STAGE1["maass_check"]["pass"], "the Maass check failed: the test must be redesigned (plan) and rechecked "
                f"on seeds {LC.SEEDS_TEST_REDESIGN}. Details in {s1_path('maass_check')}")
    if "ladder" in STAGE1_STEPS:
        if not s1_done("ladder"):
            log("stage 1, step 3: the X to Y ladder and the confirmation")
            s1_save("ladder", stage1_ladder(STAGE1["level1"]["1d"]["n"], STAGE1["maass_check"]["factor5_target_hz"]))
        STAGE1["ladder"] = s1_load("ladder")
        ladder_figures(STAGE1["ladder"]["records"]); print(ladder_table(STAGE1["ladder"]["records"]).to_string(index=False))
        if "final_1b" in STAGE1["ladder"]:                     # 1b and 1c on the final liquid's own code paths
            s1_gate(STAGE1["ladder"]["final_1b"]["pass"] and STAGE1["ladder"]["final_1c"]["pass"],
                    "level 1b or 1c failed on the final liquid's combination of factors: fix the bug before E1 to E3")
        if STAGE1["ladder"].get("final") is None and SMOKE_TEST:
            log("SMOKE TEST: no candidate passed; using Y as the final liquid to exercise stage 2")
            yk = [k for k in STAGE1["ladder"]["records"] if k.startswith("+".join(map(str, range(1, 12))) + "|")][0]
            yr = STAGE1["ladder"]["records"][yk]
            STAGE1["ladder"].update(final=list(range(1, 12)), final_key=yr["key"], n_sub=yr["n_sub"], g=yr["g_rule"] or 1.0,
                                    maass_scale=True, in_scale=yr["in_scale"], attempts=0)
        if STAGE1["ladder"].get("final") is None:
            raise RuntimeError(f"the ladder stopped: {STAGE1['ladder']['stop']}. Stop before E1 to E3 and report (decision 12).")
    if "speed" in STAGE1_STEPS:
        L = STAGE1["ladder"]
        cal = {"final": L["final"], "final_key": L["final_key"], "n_sub": L["n_sub"], "g": L["g"],
               "maass_scale": L["maass_scale"], "in_scale": L["in_scale"], "attempts": L["attempts"],
               "factor5_target_hz": STAGE1["maass_check"]["factor5_target_hz"],
               "level1f_pass": STAGE1["level1"]["1f"]["pass"], "d_replacement": D_REPLACEMENT, "x_update": X_UPDATE,
               "search_records": S1_SEARCH_RECS, "confirm_records": S1_CONFIRM_RECS,
               "made_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"), "machine": MACHINE_ID,
               "notebook_build": NOTEBOOK_BUILD, "smoke_test": SMOKE_TEST}
        cal["input_scales"] = front_end_scales(cal["final"], cal["n_sub"], cal["factor5_target_hz"], cal["in_scale"])
        if not s1_done("front_end_levels"):
            s1_save("front_end_levels", front_end_levels(cal))
        cal["front_end_levels"] = s1_load("front_end_levels")
        cal["speed"] = stage1_speed(cal)
        cal["hash"] = stable_hash({k: cal[k] for k in ("final", "n_sub", "g", "maass_scale", "input_scales", "d_replacement", "x_update")}, 10)
        jdump(cal, CAL_PATH)
        log(f"stage 1 done: final liquid {cal['final_key']} (n = {cal['n_sub']}, g {cal['g']}, Maass scale {cal['maass_scale']}); "
            f"calibration written to {CAL_PATH}; {cal['speed']['beats_per_s']:.0f} beats/s here")
CAL = json.load(open(CAL_PATH)) if os.path.exists(CAL_PATH) else None
if CAL is not None and CAL.get("x_update", "equations") != X_UPDATE:
    raise RuntimeError(f"X_UPDATE = {X_UPDATE!r} but the calibration file was made with {CAL.get('x_update')!r}")
if CAL is None:
    print("no calibration file yet: stage 2 has nothing to run")
