# Analysis of the redesign (plan, Statistics and decision rules): families F1 to F7 with Holm within each family,
# the claims A to E, the ladder's table, and summary_redesign.md. Tolerates partial results.
TBL2 = os.path.join(WORK_DIR, "tables", "redesign"); os.makedirs(TBL2, exist_ok=True)
from lsm2.stats import per_split_seed, compare, compare_diff, holm, verdict, family, NB_RATIO

def load_runs(kinds=None):
    rows = []
    for rid, v in CACHE.done.items():
        d = os.path.join(WORK_DIR, "runs", rid)
        try:
            cfg = json.load(open(os.path.join(d, "config.json"))); met = json.load(open(os.path.join(d, "metrics.json")))
        except Exception:
            continue
        s = cfg["spec"]
        if kinds and s.get("kind") not in kinds:
            continue
        r = {"run_id": rid, "group": cfg["group"], **{f"spec_{k}": (json.dumps(x) if isinstance(x, (dict, list)) else x) for k, x in s.items()},
             **{k: x for k, x in met.items() if not isinstance(x, (list, dict))}}
        p = os.path.join(d, "predictions.npz")
        if s.get("split") == "final" and os.path.exists(p):      # DS2 without record 232
            z = np.load(p); keep = z["patient_id"] != "232"
            m = compute_metrics(z["y_true"][keep].astype(int), z["y_pred"][keep].astype(int))
            r["macro_f1_no232"] = m["macro_f1"]
        rows.append(r)
    return pd.DataFrame(rows)

def _series(df, **sel):
    d = df
    for k, v in sel.items():
        col = f"spec_{k}"
        if col not in d:
            return None
        d = d[d[col].isna()] if v is None else d[d[col] == v]
    return d

def redesign_analysis():
    R2 = load_runs({"lsm2", "feat2", "hybrid", "feat", "lsm"})
    if R2.empty or "spec_kind" not in R2:
        print("redesign analysis: no runs yet"); return {}
    L = R2[R2.spec_kind == "lsm2"]
    def lsm2(liquid, readout, fe=CFG11, variant=None, loss="svm"):
        return _series(L, liquid=liquid, readout=readout, filter=fe[0], norm=fe[1], encoder=fe[2], variant=variant, loss=loss)
    old = R2[R2.spec_kind.isin(["feat", "lsm"])]
    elm = _series(old, kind="feat", model="elm", encoder="multi")
    noliq = _series(old, kind="feat", model="no_liquid", encoder="multi")
    elm_rr = _series(old, kind="feat", model="elm_rr", encoder="multi")
    fixed_rrliq = _series(old[old.spec_kind == "lsm"], mode="fixed", filter="morph", norm="robust", encoder="multi",
                          readout="svm", rr_in_liquid=True, recurrence=True) if "spec_rr_in_liquid" in old else None
    if fixed_rrliq is not None and "spec_rr_set" in fixed_rrliq:
        fixed_rrliq = fixed_rrliq[fixed_rrliq.spec_rr_set.isna()]
    if e3_partner_needed():
        fixed_rrliq = lsm2("X", "film", variant="rrliq")
    F2r = R2[R2.spec_kind == "feat2"]
    ctrl = lambda m, ro: _series(F2r, model=m, readout=ro)
    fams = []
    ros = ["film"] + list(SNAP_READOUTS)
    fams += family("F1 recurrence", [(ro, compare(lsm2("final", ro), lsm2("final_off", ro), True)) for ro in ros], True)
    fams += family("F2 added factors", [(ro, compare(lsm2("final", ro), lsm2("X", ro))) for ro in ros])
    fams += family("F3 other models", [("vs ELM", compare(lsm2("final", "film"), elm)), ("vs no liquid", compare(lsm2("final", "film"), noliq))])
    cfg = lambda fe, loss: lsm2("final", "film", fe, loss=loss)
    fams += family("F4 pipeline", [("multi vs rate", compare(cfg(CFG11, "svm"), cfg(("morph", "robust", "rate"), "svm"))),
                                   ("multi vs latency", compare(cfg(CFG11, "svm"), cfg(("morph", "robust", "latency"), "svm")))])
    fams += family("F5 with RR", [("RR in liquid vs paper's fixed + RR in liquid", compare(lsm2("final", "film", variant="rrliq"), fixed_rrliq)),
                                  ("RR in liquid vs ELM + RR", compare(lsm2("final", "film", variant="rrliq"), elm_rr)),
                                  ("RR readout vs film", compare(lsm2("final", "film_rr"), lsm2("final", "film")))])
    f6 = []
    for ro in SNAP_READOUTS:
        for m, lab in (("fade", "fading-memory baseline"), ("selm", "snapshot ELM"), ("noliq30", "no-liquid control")):
            f6.append((f"{ro} vs {lab}", compare(lsm2("final", ro), ctrl(m, ro))))
    fams += family("F6 beyond plain memory", f6)
    fams += family("F7 memory ablation", [(ro, compare_diff(lsm2("final", ro), lsm2("final_off", ro), lsm2("final", ro, variant="ablated"),
                                                            lsm2("final_off", ro, variant="ablated"))) for ro in SNAP_READOUTS])
    F = pd.DataFrame(fams)
    F.to_csv(os.path.join(TBL2, "families.csv"), index=False)
    # F4: Friedman, Iman-Davenport and Kendall's W over the 24 configurations, blocks = the 12 CV splits, per seed
    f4 = []
    E2 = L[(L.spec_readout == "film") & L.spec_variant.isna() & (L.spec_liquid == "final") & L.spec_split.astype(str).str.startswith("cv-")]
    for sd in sorted(E2.spec_seed.dropna().unique()) if not E2.empty else []:
        e = E2[E2.spec_seed == sd].copy()
        e["cid"] = [CONFIGS_BY_FE.get(((a, b, c), d)) for a, b, c, d in zip(e.spec_filter, e.spec_norm, e.spec_encoder, e.spec_loss)]
        M = e.pivot_table(index="spec_split", columns="cid", values="macro_f1")
        if M.shape[1] >= 3 and M.shape[0] >= 3 and not M.isna().any().any():
            st, ranks, _ = friedman_block(M.values, list(M.columns))
            f4.append({"seed": sd, **st, "n_configs": M.shape[1], "n_blocks": M.shape[0]})
    pd.DataFrame(f4).to_csv(os.path.join(TBL2, "F4_friedman.csv"), index=False)
    # claims (plan, Claims and what each needs)
    lad = json.load(open(os.path.join(WORK_DIR, "checks", "stage1", "ladder.json"))) if os.path.exists(os.path.join(WORK_DIR, "checks", "stage1", "ladder.json")) else {}
    conf = [s for s in lad.get("steps", []) if s.get("pass")]
    A_ = bool(conf) and all(r.get("level4", {}).get("pass") for k, r in lad.get("records", {}).items() if k.startswith("confirm|") and r.get("pass"))
    v = {(r["family"], r["comparison"]): r for r in fams}
    snaps_mem = ["snap342", "snap396"]
    B_at = [ro for ro in snaps_mem if v.get(("F1 recurrence", ro), {}).get("verdict") == "meaningful"
            and v.get(("F7 memory ablation", ro), {}).get("verdict") == "meaningful"
            and v.get(("F1 recurrence", ro), {}).get("cv_diff", -1) > v.get(("F1 recurrence", "snap288"), {}).get("cv_diff", np.inf)]
    C_at = [ro for ro in B_at if v.get(("F6 beyond plain memory", f"{ro} vs fading-memory baseline"), {}).get("verdict") == "meaningful"
            and v.get(("F6 beyond plain memory", f"{ro} vs snapshot ELM"), {}).get("verdict") == "meaningful"]
    D_ = (v.get(("F1 recurrence", "film"), {}).get("verdict") == "meaningful"
          and v.get(("F3 other models", "vs ELM"), {}).get("verdict") == "meaningful"
          and v.get(("F3 other models", "vs no liquid"), {}).get("verdict") == "meaningful")
    claims = {"A. the liquid remembers": A_, "B. its memory helps classify beats": bool(B_at), "B at": B_at,
              "C. more than plain memory": bool(C_at), "C at": C_at, "D. helps the paper's pipeline": bool(D_),
              "E. a better classifier overall": "not claimed",
              "wording": ("the LSM does help" if (A_ and B_at and C_at) else
                          "narrow wording (A and B)" if (A_ and B_at) else "neither")}
    jdump(claims, os.path.join(TBL2, "claims.json"))
    lines = ["# Redesign: results so far", f"_Updated {datetime.datetime.utcnow().isoformat(timespec='seconds')} UTC_", ""]
    if CAL:
        lines += [f"Final liquid: **{CAL['final_key']}** (n = {CAL['n_sub']}, g {CAL['g']}, Maass scale {CAL['maass_scale']}), "
                  f"confirmation attempts {CAL['attempts']}; check 1f {'passed' if CAL['level1f_pass'] else 'failed (partner runs redone)'}", ""]
    lines += ["## Families (Holm within each)", F.drop(columns=[c for c in F.columns if c in ("t",)], errors="ignore").round(4).to_string(index=False) if not F.empty else "none yet", ""]
    if f4:
        lines += ["## F4: Friedman over the 24 configurations (per seed)", pd.DataFrame(f4).round(4).to_string(index=False), ""]
    lines += ["## Claims", "```", json.dumps(claims, indent=2, default=str), "```"]
    atomic_write_text(os.path.join(WORK_DIR, "summary_redesign.md"), "\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    cols = [c for c in ("family", "comparison", "cv_diff", "ds2_diff", "verdict") if c in F]
    print(F[cols].to_string(index=False) if not F.empty else "")
    return {"families": F, "claims": claims}

def friedman_block(M, ids):
    n, k = M.shape
    chi2, p = sstats.friedmanchisquare(*[M[:, j] for j in range(k)])
    W = chi2 / (n * (k - 1)); FF = (n - 1) * chi2 / (n * (k - 1) - chi2)
    pFF = sstats.f.sf(FF, k - 1, (k - 1) * (n - 1))
    ranks = np.vstack([sstats.rankdata(-r) for r in M]).mean(0)
    return {"chi2": chi2, "p": p, "iman_davenport_F": FF, "p_F": pFF, "kendall_W": W, "best_by_rank": ids[int(np.argmin(ranks))]}, ranks, None

try:
    REDESIGN = redesign_analysis()
except Exception:
    print("redesign analysis failed:\n" + traceback.format_exc())
