# Stage 2 plan (plan, Experiments and controls): every new classification run as a spec. Run ids include the
# calibration file's hash, so a different final liquid can never reuse these runs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SNAP_READOUTS = {"snap288": 0, "snap342": 1, "snap396": 2}         # R, R + 150 ms, R + 300 ms
FRONT_ENDS = [(f, n, e) for f in ("morph", "median") for n in ("standard", "robust") for e in ("multi", "rate", "latency")]
CFG11 = ("morph", "robust", "multi")

def lsm2_spec(liquid, fe, readout, split, seed, variant=None, loss="svm"):
    """liquid: "final", "final_off" (g = 0), or "X" (the paper's liquid at g = 1, (a)'s construction).
    readout: "film" (32 windows x neurons), "film_rr" (film + RR features), or a snapshot.
    variant: None, "ablated" (memory ablation), "rrliq" (30 RR channels in the liquid), "rrliq_mean" (40)."""
    s = {"kind": "lsm2", "liquid": liquid, "filter": fe[0], "norm": fe[1], "encoder": fe[2], "readout": readout,
         "loss": loss, "split": split, "seed": seed, "variant": variant,
         "cal": CAL["hash"] if (CAL is not None and liquid != "X") else None}
    return s

def feat2_spec(model, readout, split, seed):
    """Encoder-only controls at the snapshot moments: "noliq30" (30 ms filter), "fade" (30 / 100 / 300 ms),
    "selm" (random expansion of the 30 ms-filtered input; seed = the expansion's draw)."""
    return {"kind": "feat2", "model": model, "readout": readout, "split": split, "seed": seed,
            "filter": "morph", "norm": "robust", "encoder": "multi"}

_old_run_id_of = run_id_of
def run_id_of(spec):
    k = spec["kind"]
    if k not in ("lsm2", "feat2"):
        return _old_run_id_of(spec)
    h = stable_hash({**spec, "_v": CODE_VERSION, "_smoke": SMOKE_TEST})
    if k == "lsm2":
        tag = [spec["liquid"], f"s{spec['seed']}", f"{spec['filter']}-{spec['norm']}-{spec['encoder']}",
               spec["readout"] + ("" if spec["loss"] == "svm" else f"-{spec['loss']}")] + ([spec["variant"]] if spec["variant"] else [])
    else:
        tag = [spec["model"], spec["readout"], f"s{spec['seed']}"]
    return f"{'SMOKE__' if SMOKE_TEST else ''}{k}__{'_'.join(tag)}__{spec['split']}__{h}"

def unit_of(spec):
    """The smallest piece that shares one simulation (plan, Stages): (front end, seed, split) for a robust front end,
    (front end, seed, "all") for a standard one."""
    fe = (spec.get("filter"), spec.get("norm"), spec.get("encoder"))
    return (fe, spec.get("seed"), spec["split"] if spec.get("norm") == "robust" else "all")

def unit_assignment(units):
    """Deterministic, the same on every machine: units in a stable order, each given to the machine whose load
    divided by its weight is lowest."""
    names = sorted(UNIT_SHARES)
    load = {m: 0.0 for m in names}
    out = {}
    for u in sorted(units, key=lambda u: stable_hash(u, 12)):
        m = min(names, key=lambda m: ((load[m] + 1) / UNIT_SHARES[m], m))
        out[u] = m; load[m] += 1
    return out

def e3_partner_needed():
    return CAL is not None and not CAL.get("level1f_pass", False)

def build_plan2():
    P = []
    if CAL is None:
        return P
    splits = ["final"] + SPLITS_CV
    seeds = SEEDS
    def add(group, spec):
        if GROUPS_ENABLED.get(group, False):
            P.append((group, spec))
    for seed in seeds:
        for sp in splits:
            for liq in ("final", "final_off", "X"):
                for ro in ["film"] + list(SNAP_READOUTS):
                    add("e1", lsm2_spec(liq, CFG11, ro, sp, seed))
            for liq in ("final", "final_off"):
                for ro in SNAP_READOUTS:
                    add("e1_ablation", lsm2_spec(liq, CFG11, ro, sp, seed, variant="ablated"))
            for model in ("noliq30", "fade", "selm"):
                for ro in SNAP_READOUTS:
                    add("e1_controls", feat2_spec(model, ro, sp, seed))
            for fe in FRONT_ENDS:
                for loss in ("svm", "lr"):
                    if CONFIGS_BY_FE.get((fe, loss)) is not None:
                        add("e2", lsm2_spec("final", fe, "film", sp, seed, loss=loss))
            add("e3", lsm2_spec("final", CFG11, "film_rr", sp, seed))
            add("e3", lsm2_spec("final", CFG11, "film", sp, seed, variant="rrliq"))
            add("e3", lsm2_spec("final", CFG11, "film", sp, seed, variant="rrliq_mean"))
            if e3_partner_needed():
                add("e3_partner", lsm2_spec("X", CFG11, "film", sp, seed, variant="rrliq"))
    # hybrid rule on the base, RR-readout and RR-liquid runs (needs every machine's predictions: after the merge)
    if GROUPS_ENABLED.get("e3_hybrid") and (MERGE_MODE or UNIT_SHARES == {THIS_MACHINE: UNIT_SHARES.get(THIS_MACHINE)}):
        for seed in seeds:
            for sp in splits:
                for label, bspec in (("final liquid, film", lsm2_spec("final", CFG11, "film", sp, seed)),
                                     ("final liquid, film + RR features", lsm2_spec("final", CFG11, "film_rr", sp, seed)),
                                     ("final liquid, RR in liquid", lsm2_spec("final", CFG11, "film", sp, seed, variant="rrliq"))):
                    brid = run_id_of(bspec)
                    if brid in CACHE.done and os.path.exists(os.path.join(WORK_DIR, "runs", brid, "predictions.npz")):
                        for rule in HYBRID_RULES:
                            P.append(("e3_hybrid", {"kind": "hybrid", "base": brid, "base_label": label, "rule": rule,
                                                    "threshold": HUANG_THRESHOLD, "seed": seed, "split": sp}))
    return P

CONFIGS_BY_FE = {((c["filter"], c["norm"], c["encoder"]), c["readout"]): cid for cid, c in CONFIGS.items()}

_old_build_plan = build_plan
def build_plan():
    plan = _old_build_plan()                       # the replication's groups (disabled here: nothing pending)
    seen = {e["rid"] for e in plan}
    new = []
    for group, spec in build_plan2():
        rid = run_id_of(spec)
        if rid in seen:
            continue
        seen.add(rid)
        new.append({"rid": rid, "group": group, "groups": [group], "spec": spec})
    units = {unit_of(e["spec"]) for e in new if e["spec"]["kind"] in ("lsm2", "feat2")}
    assign = unit_assignment(units)
    mine = [e for e in new if e["spec"]["kind"] not in ("lsm2", "feat2") or assign[unit_of(e["spec"])] == THIS_MACHINE
            or MERGE_MODE]
    return plan + mine

_old_states_key = states_key
def states_key(spec):
    """Runs with the same key share one simulation: the liquid, front end, seed, input variant and data split
    (standard-normalisation CV splits share one CV simulation)."""
    if spec["kind"] == "lsm2":
        sp = spec["split"]
        nk = "final" if sp == "final" else ("cv-shared" if spec["norm"] == "standard" else sp)
        core = {k: spec[k] for k in ("liquid", "filter", "norm", "encoder", "seed", "variant", "cal")}
        return ("lsm2", json.dumps(core, sort_keys=True), nk)
    if spec["kind"] == "feat2":
        core = {k: spec[k] for k in ("model", "seed")}
        return ("feat2", json.dumps(core, sort_keys=True), spec["split"])
    return _old_states_key(spec)

_p2 = build_plan()
_c2 = pd.Series([e["group"] for e in _p2]).value_counts()
print(f"stage-2 plan on this machine ({THIS_MACHINE}): {len(_p2)} runs in {len(make_jobs(_p2))} jobs")
print(_c2.to_string() if len(_c2) else "(nothing: no calibration file yet, or all done)")
