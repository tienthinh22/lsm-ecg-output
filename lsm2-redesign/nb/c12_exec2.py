# Stage-2 executors: the new liquid simulations (kind "lsm2") and the encoder-only snapshot controls ("feat2").
# Same readout, splits, metrics and saving as the replication (fit_readout, _finish_run); only the features differ.
SNAP_T = list(LC.SNAPSHOT_SAMPLES)
ABLATE_SAMPLES = int(round(50.0 / LC.SAMPLE_MS))            # input before t - 50 ms is replaced (18 samples)
START_STREAM = {"DS1": 101, "DS2": 102}                      # per-beat starting voltages from (seed, stream, beat index)

def _norm_for(nk, filt, norm):
    D1, D2 = DATA[filt]["DS1"], DATA[filt]["DS2"]
    if nk == "final":
        X1n, X2n, normp = normalize_final(D1["X"], D2["X"], norm)
        return np.asarray(X1n, np.float32), np.asarray(X2n, np.float32), np.arange(len(D1["y"])), normp
    if nk == "cv-shared":
        X1n, normp = normalize_cv(D1["X"], None, "standard")
        return X1n, None, np.arange(len(D1["y"])), normp
    f = FOLD_BY[parse_split(nk)]
    X1n, normp = normalize_cv(D1["X"], f["train_idx"], norm)
    return X1n, None, f["train_idx"], normp

def _cal_scale(filt, norm, enc):
    return float(CAL["input_scales"][f"{filt}-{norm}-{enc}"]["scale"])

def _ablate(Xn, t_snap, mean_beat):
    """Input before t - 50 ms replaced by the R-aligned mean beat, shifted to meet the real beat at t - 50 ms."""
    cut = t_snap - ABLATE_SAMPLES
    out = Xn.copy()
    out[:, :cut] = mean_beat[None, :cut] + (Xn[:, cut] - mean_beat[cut])[:, None]
    return np.clip(out, 0.0, 1.0)

def _simulate_set(liq, g, in_scale, Xn, enc, rr_rows, rr_thr, seed, ds, rec, recurrence=True):
    """Encode (+ RR channels) and run the new simulator in fixed-size batches; numpy outputs."""
    sm = LS.Simulator(liq, device=DEVICE, dtype=torch.float32, g=g, in_scale=in_scale, recurrence=recurrence)
    outs = {}
    t_enc = t_sim = 0.0
    for s in range(0, len(Xn), NEW_SIM_BATCH):
        t0 = time.time()
        x = encode_t(Xn[s:s + NEW_SIM_BATCH], enc)
        if rr_rows is not None:
            x = torch.cat([x, torch.from_numpy(rr_channels(rr_rows[s:s + NEW_SIM_BATCH], rr_thr, x.shape[1])).to(DEVICE)], 2)
        t_enc += time.time() - t0; t0 = time.time()
        ids = np.arange(s, s + len(x))
        u0 = LS.start_voltages(liq, seed, ids, START_STREAM[ds])
        o = sm.run(x, u0, LIQ["num_sim_steps"], rec=rec)
        for k, v in o.items():
            if torch.is_tensor(v) and k != "u_end":
                outs.setdefault(k, []).append(v.cpu().numpy() if k != "windows" else v.to(torch.uint8).cpu().numpy())
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_sim += time.time() - t0
    return {k: np.concatenate(v) for k, v in outs.items()}, t_enc, t_sim

def _x_liquid(n_in, X1n, tr_rows, enc, rr1, rr_thr, seed):
    """The paper's liquid X at g = 1, built exactly as the replication's fixed runs (its class, A_in measured on
    the training rows), simulated in the new code."""
    ain = measure_ain(X1n, tr_rows, enc, 200, rr=rr1, rr_thr=rr_thr)
    model = build_liquid(n_in, I_TARGET / (ain * LIQ["P_in"]), seed, True)
    return LL.from_authors_model(model), {"A_in_used": ain, "construction": "authors' class, fixed mode"}

def execute_lsm2(job, runs):
    s0 = runs[0]["spec"]; nk = job["key"][2]; final = nk == "final"
    filt, norm, enc, seed, liqname, variant = s0["filter"], s0["norm"], s0["encoder"], s0["seed"], s0["liquid"], s0["variant"]
    D1, D2 = DATA[filt]["DS1"], DATA[filt]["DS2"]
    t0 = time.time()
    torch.manual_seed(int(stable_hash(job["key"], 6), 16))   # the Rate encoder is stochastic: fixed per simulation
    X1n, X2n, tr_rows, normp = _norm_for(nk, filt, norm)
    rr1 = rr2 = rr_thr = None
    if variant in ("rrliq", "rrliq_mean"):
        mean_set = variant == "rrliq_mean"
        rr1 = rr_inputs(D1, np.arange(len(D1["y"])), mean_set); rr2 = rr_inputs(D2, np.arange(len(D2["y"])), mean_set)
        rr_thr = rr_thresholds(rr1[tr_rows])
    n_in = (200 if enc == "multi" else N_IN[enc]) + (len(rr_thr) * RR_LEVELS if rr_thr is not None else 0)
    if liqname == "X":
        liq, info = _x_liquid(n_in, X1n, tr_rows, enc, rr1, rr_thr, seed)
        g, scale = 1.0, 1.0
    else:
        liq = LL.build_candidate(frozenset(CAL["final"]), seed, CAL["n_sub"], n_in)
        g = (liq.g_maass if CAL["maass_scale"] else float(CAL["g"])) if liqname == "final" else 0.0
        scale = _cal_scale(filt, norm, enc)
        info = {"final_liquid": CAL["final_key"], "n_sub": CAL["n_sub"], "g": g, "abs_g": liq.ee_mean_push(g),
                "input_scale": scale, "calibration": CAL["hash"]}
    rec = {"windows": (WIN, NWIN), "states_at": SNAP_T, "neuron_counts": (0, LC.BEAT_SAMPLES)}
    sets = [("DS1", X1n, rr1)] + ([("DS2", X2n, rr2)] if final else [])
    F, t_enc, t_sim, stats = {}, 0.0, 0.0, {}
    if variant == "ablated":
        mean_beat = X1n[tr_rows].mean(0)                     # R-aligned mean beat of the training rows (labels unused)
        for ds, Xn, rr in sets:
            snaps = np.zeros((len(Xn), len(SNAP_T), liq.N), np.float32)
            for i, t in enumerate(SNAP_T):
                o, a, b = _simulate_set(liq, g, scale, _ablate(Xn, t, mean_beat), enc, None, None, seed, ds, {"states_at": [t]},
                                        recurrence=liqname != "final_off")
                snaps[:, i] = o["states_at"][:, 0]; t_enc += a; t_sim += b
            F[ds] = {"snap": snaps}
    else:
        for ds, Xn, rr in sets:
            o, a, b = _simulate_set(liq, g, scale, Xn, enc, rr, rr_thr, seed, ds, rec, recurrence=liqname != "final_off")
            t_enc += a; t_sim += b
            F[ds] = {"film": o["windows"], "snap": o["states_at"]}
            dur = LC.BEAT_SAMPLES * LC.SAMPLE_MS / 1000
            stats[f"{ds}_firing_hz"] = float(o["neuron_counts"].mean() / dur)
            stats[f"{ds}_silent_neurons"] = int((o["neuron_counts"].sum(0) == 0).sum())
    sim_s = t_enc + t_sim
    update_speed("sim_beats_per_s", sum(len(x) for _, x, _ in sets) / max(sim_s, 1e-6), sub=f"lsm2-{enc}")
    liq_info = {**info, "normalization": normp, "liquid": liqname, "variant": variant, "N": liq.N, **stats}
    log(f"  lsm2 {liqname} {filt}-{norm}-{enc} s{seed} {nk} {variant or ''}: sim {sim_s:.0f}s "
        + " ".join(f"{k}={v:.2f}" for k, v in stats.items() if k.endswith("hz")))
    def film_fn(ds):
        C_ = F[ds]["film"]
        return lambda rows: (C_[rows].reshape(len(rows), -1).astype(np.float64) / WIN).astype(np.float32)
    def snap_fn(ds, i):
        S_ = F[ds]["snap"]
        return lambda rows: np.ascontiguousarray(S_[rows, i], np.float32)
    groups = []
    for e in runs:
        ro = e["spec"]["readout"]
        if ro in ("film", "film_rr"):
            groups.append((e, film_fn("DS1"), film_fn("DS2") if final else film_fn("DS1")))
        else:
            i = SNAP_READOUTS[ro]
            groups.append((e, snap_fn("DS1", i), snap_fn("DS2", i) if final else snap_fn("DS1", i)))
    _run_readouts2(groups, D1["y"], D2 if final else D1, final, liq_info,
                   {"shared_seconds": time.time() - t0, "simulation_seconds": sim_s, "beats_simulated": sum(len(x) for _, x, _ in sets)},
                   len(D1["y"]))

def _filtered_input(Xn, enc, taus_ms, t_read):
    """Encoder spikes through decaying filters (one per tau), read at the snapshot samples: (n, len(t_read), C*len(taus))."""
    out = np.zeros((len(Xn), len(t_read), (200 if enc == "multi" else N_IN[enc]) * len(taus_ms)), np.float32)
    dec = torch.tensor([math.exp(-LC.SAMPLE_MS / t) for t in taus_ms], device=DEVICE, dtype=torch.float32)
    for s in range(0, len(Xn), NEW_SIM_BATCH):
        x = encode_t(Xn[s:s + NEW_SIM_BATCH], enc).float()
        z = torch.zeros(x.shape[0], len(taus_ms), x.shape[2], device=DEVICE)
        for t in range(max(t_read) + 1):
            if t < x.shape[1]:
                z = z * dec[None, :, None] + x[:, t][:, None, :]
            else:
                z = z * dec[None, :, None]
            if t in t_read:
                out[s:s + len(x), t_read.index(t)] = z.permute(0, 2, 1).reshape(len(x), -1).cpu().numpy()
    return out

def execute_feat2(job, runs):
    s0 = runs[0]["spec"]; split = job["key"][2]; final = split == "final"
    t0 = time.time()
    D1, D2, X1n, X2n, tr_rows, normp = _front_end(split)
    model = s0["model"]
    taus = {"noliq30": [30.0], "fade": [30.0, 100.0, 300.0], "selm": [30.0]}[model]
    Z1 = _filtered_input(X1n, "multi", taus, SNAP_T); Z2 = _filtered_input(X2n, "multi", taus, SNAP_T) if final else None
    info = {"normalization": normp, "feature_model": model, "taus_ms": taus}
    if model == "selm":
        # a random nonlinear expansion with no memory: the 30 ms-filtered input through the fixed liquid's input
        # weights (same construction and seed as the replication's ELM), centred on the training mean, ReLU
        ain = measure_ain(X1n, tr_rows, "multi")
        Wf = build_liquid(200, I_TARGET / (ain * LIQ["P_in"]), s0["seed"], True).input_rsv_weights.detach().cpu().numpy().astype(np.float32)
        P1 = np.einsum("ntc,ck->ntk", Z1, Wf); P2 = np.einsum("ntc,ck->ntk", Z2, Wf) if final else None
        mu = P1[tr_rows].mean(0)
        Z1 = np.maximum(P1 - mu, 0); Z2 = np.maximum(P2 - mu, 0) if final else None
        info.update({"A_in_measured": ain, "elm": "30 ms-filtered input @ fixed-liquid W_in, centred on train mean, ReLU"})
    groups = []
    for e in runs:
        i = SNAP_READOUTS[e["spec"]["readout"]]
        f1 = (lambda Z, i: (lambda rows: np.ascontiguousarray(Z[rows, i])))(Z1, i)
        f2 = (lambda Z, i: (lambda rows: np.ascontiguousarray(Z[rows, i])))(Z2, i) if final else f1
        groups.append((e, f1, f2))
    _run_readouts2(groups, D1["y"], D2 if final else D1, final, info, {"shared_seconds": time.time() - t0}, len(D1["y"]))

def _run_readouts2(groups, ytr_all, D_eval, final, liq, base_timing, ntr_full):
    """The replication's readout for runs with their own feature functions: groups = [(run, feat_tr, feat_ev)].
    The loss is the spec's ("svm" = hinge, "lr" = log loss); "film_rr" appends the 8 RR features as in svm_rr."""
    D_TR_RR, D_EV_RR = DATA["morph"]["DS1"]["rr"], D_eval["rr"]
    def parts(e, feat_tr, feat_ev):
        tr, ev, rng_seed = _split_rows(e["spec"]["split"], ntr_full)
        if final:
            ev = np.arange(len(D_eval["y"]))
        add_rr = e["spec"]["readout"] == "film_rr"
        ftr = (lambda r: np.hstack([feat_tr(tr[r]), D_TR_RR[tr[r]]])) if add_rr else (lambda r: feat_tr(tr[r]))
        fev = (lambda r: np.hstack([feat_ev(ev[r]), D_EV_RR[ev[r]]])) if add_rr else (lambda r: feat_ev(ev[r]))
        return tr, ev, rng_seed, add_rr, ftr, fev
    def one(e, feat_tr, feat_ev, nj, cancel=None, shared=None):
        tr, ev, rng_seed, add_rr, ftr, fev = parts(e, feat_tr, feat_ev)
        yev = D_eval["y"][ev]
        loss = "log_loss" if e["spec"].get("loss") == "lr" else "hinge"
        nfeat = len(ftr(np.arange(1)).ravel())
        key = (e["spec"]["split"], e["spec"]["readout"], add_rr)
        prep = shared.get(key, ftr, len(tr), nfeat, FIT_POOL.ram) if shared is not None else None
        scaler, clf, yp, sc, curve = fit_readout(ftr, len(tr), ytr_all[tr], fev, len(ev), loss, rng_seed,
                                                 yev_for_curve=None if final else yev, n_jobs=nj, prepared=prep, cancel=cancel)
        return e, compute_metrics(yev, yp), ev, D_eval["pid"][ev], yev, yp, sc, {"scaler": scaler, "classifier": clf}, curve
    def record(res, dt, threads):
        e, m, ev, pid, yev, yp, sc, mobj, curve = res
        _finish_run(e, m, ev, pid, yev, yp, sc, mobj, curve, liq,
                    {**base_timing, "readout_seconds": dt, "total_seconds": base_timing.get("shared_seconds", 0) + dt,
                     "readout_threads": threads, "runner": "pipeline" if FIT_POOL is not None else "sequential"})
    if FIT_POOL is not None:
        tasks = []
        for e, ft, fe in groups:
            tr = _split_rows(e["spec"]["split"], ntr_full)[0]
            nfeat = len(ft(np.arange(1)).ravel())
            gc_ = len(tr) * nfeat / 1e9
            tasks.append({"rid": e["rid"], "gcells": gc_, "est_s": gc_ * SPEED["fit_s_per_Gcell"]["pipe"],
                          "ram_gb": fit_ram_gb(nfeat, len(tr)),
                          "fit": (lambda cancel, shared, e=e, ft=ft, fe=fe: one(e, ft, fe, 1, cancel=cancel, shared=shared)),
                          "record": (lambda res, dt, busy: record(res, dt, busy))})
        FIT_POOL.submit_job(tasks, _JOB_CTX.get("ram_gb", 0.0), SharedPrep())
        _JOB_CTX["submitted"] = True
        return
    for e, ft, fe in groups:
        try:
            t1 = time.time(); r = one(e, ft, fe, READOUT_JOBS); record(r, time.time() - t1, 1)
        except Exception:
            _record_failure(e["rid"], traceback.format_exc())

EXECUTORS.update({"lsm2": execute_lsm2, "feat2": execute_feat2})

# cost and RAM estimates for the runner's guards and projection
_old_estimate_parts, _old_job_ram_gb = estimate_parts, job_ram_gb
def _lsm2_n_neur():
    return int(np.prod(LC.MAASS["dims"] if 11 in CAL["final"] else LC.X["dims"])) if CAL else N_NEUR

def estimate_parts(job, runs):
    s = runs[0]["spec"]
    if s["kind"] not in ("lsm2", "feat2"):
        return _old_estimate_parts(job, runs)
    n1, n2, final, ntr, beats = _sizes(s)
    N = _lsm2_n_neur()
    gc_ = []
    for r in runs:
        ro = r["spec"]["readout"]
        nf = (NWIN * N + (len(RR_NAMES) if ro == "film_rr" else 0)) if ro in ("film", "film_rr") else \
             ({"noliq30": 200, "fade": 600, "selm": N_NEUR}.get(r["spec"].get("model"), N))
        gc_.append(ntr * nf / 1e9)
    if s["kind"] == "feat2":
        return beats / SPEED["encode_beats_per_s"]["multi"] + 20, gc_
    mult = 3 if s.get("variant") == "ablated" else 1
    bps = SPEED["sim_beats_per_s"].get(f"lsm2-{s['encoder']}", SPEED["sim_beats_per_s"][s["encoder"]] / 4)
    return mult * beats / bps + SPEED["lsm_job_overhead_s"], gc_

def job_ram_gb(job, runs):
    s = runs[0]["spec"]
    if s["kind"] not in ("lsm2", "feat2"):
        return _old_job_ram_gb(job, runs)
    n1, n2, final, ntr, beats = _sizes(s)
    N = _lsm2_n_neur()
    return beats * (NWIN * N + 3 * N * 4) / 1e9 + 0.8
