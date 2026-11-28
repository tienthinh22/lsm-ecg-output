"""The X to Y ladder (plan, The X to Y ladder): test X and Y; forward from X adding factors 1 to 11 in order
until a candidate passes all four levels; backward loop removing each added factor, newest first, until a full
pass removes nothing; confirmation on the other half of DS1 with fresh seeds and frozen g and input size."""
import json
import time
import numpy as np

from . import config as C
from .liquid import build_candidate
from .measure import measure_at_g, memory_curve
from .calib import input_scale
from . import gain, level2


def key_of(factors):
    return "+".join(str(f) for f in sorted(factors)) or "X"


class Ladder:
    def __init__(self, ctx, search, confirm, n_fine, maass_target_hz, seeds=C.SEEDS_SEARCH, n_channels=200,
                 log=print, on_record=None, cache=None):
        self.ctx, self.search, self.confirm_data = ctx, search, confirm
        self.n_fine, self.maass_target = n_fine, maass_target_hz
        self.seeds, self.n_channels, self.log = tuple(seeds), n_channels, log
        self.records = dict(cache or {})
        self.on_record = on_record
        self.path = []

    def n_sub_for(self, factors, y_grid=None):
        f = frozenset(factors)
        if f == C.ALL_FACTORS:
            return y_grid or C.N_SUB_REFERENCE          # Y itself always on 28 sub-steps
        return self.n_fine if 7 in f else 1

    # ------------------------------------------------------------------ one candidate
    def evaluate(self, factors, data=None, seeds=None, frozen=None, y_grid=None, tag=None, n_channels=None):
        """All four levels for one candidate. frozen = {"g": ..., "in_scale": ...} for the confirmation
        (no sweep; level 3 is the band and the starting-state rule at the frozen g)."""
        f = frozenset(factors)
        data = data or self.search
        seeds = tuple(seeds or self.seeds)
        n = self.n_sub_for(f, y_grid)
        rkey = tag or f"{key_of(f)}|n{n}|{data.name}|{','.join(map(str, seeds))}"
        if rkey in self.records:
            return self.records[rkey]
        t0 = time.time()
        self.log(f"candidate {key_of(f)} (n = {n}, {data.name}, seeds {list(seeds)})")
        liqs = {s: build_candidate(f, s, n, n_channels or self.n_channels) for s in seeds}
        for s, l in liqs.items():
            l.meta["seed"] = s
        rec = {"factors": sorted(f), "key": key_of(f), "n_sub": n, "data": data.name, "seeds": list(seeds)}
        # level 2
        l2 = {s: level2.check(liqs[s]) for s in seeds}
        rec["level2"] = {"pass": all(v["pass"] for v in l2.values()),
                         "failed_items": {s: [i["item"] for i in v["items"] if not i["pass"]] for s, v in l2.items()},
                         "report": {s: v["report"] for s, v in l2.items()}}
        # input size
        if frozen is not None:
            scale, how = frozen["in_scale"], {"rule": "frozen", "scale": frozen["in_scale"]}
        else:
            scale, how = input_scale(self.ctx, list(liqs.values()), data.sets["sep"], data.ain, self.maass_target, log=self.log)
        rec["input"] = how
        # gain
        if frozen is None:
            measure = lambda g, s: measure_at_g(self.ctx, liqs[s], g, scale, data.sets, s)
            sw = gain.sweep(measure, seeds, log=self.log)
            rec["sweep"] = {"table": {repr(g): {k: v for k, v in t.items() if k != "per_seed"} for g, t in sw["table"].items()},
                            "per_seed": {repr(g): t["per_seed"] for g, t in sw["table"].items()},
                            **{k: v for k, v in sw.items() if k != "table"}}
            g_chosen = sw["chosen_g"]
            peak_ok = sw["gap_peak_inside_grid"]
        else:
            g_chosen, peak_ok, sw = frozen["g"], True, None
        if 10 in f:
            g_use = {s: liqs[s].g_maass for s in seeds}      # Maass's own scale instead of the rule's g
        else:
            g_use = {s: g_chosen for s in seeds}
        rec["g_rule"] = g_chosen
        rec["g_used"] = {str(s): g_use[s] for s in seeds}
        rec["abs_g_used"] = {str(s): (liqs[s].ee_mean_push(g_use[s]) if g_use[s] is not None else None) for s in seeds}
        rec["band_limited"] = bool(sw["band_limited"]) if sw else False
        # level 3 at the g used
        if g_chosen is None and 10 not in f:
            rec["level3"] = {"pass": False, "reason": "no setting inside the firing band"}
        else:
            rows = []
            for s in seeds:
                m = measure_at_g(self.ctx, liqs[s], g_use[s], scale, data.sets, s, with_start_test=True, with_twins=False)
                rows.append({"seed": s, "rate_hz": m["rate_hz"], "frac_above_100hz": m["frac_above_100hz"],
                             "in_band": m["in_band"], "start_rule": m["start_rule_pass"],
                             "start_same": m["start_same"], "start_diff": m["start_diff"], "gap": m["gap"]})
            ok = all(r["in_band"] and r["start_rule"] for r in rows) and peak_ok
            rec["level3"] = {"pass": bool(ok), "rows": rows, "gap_peak_inside_grid": bool(peak_ok)}
        # level 4 (memory curve reported for every candidate)
        if all(g_use[s] is not None for s in seeds):
            on = {s: memory_curve(self.ctx, liqs[s], g_use[s], scale, data.mem_spikes, data.mem_targets, s) for s in seeds}
            off = {s: memory_curve(self.ctx, liqs[s], g_use[s], scale, data.mem_spikes, data.mem_targets, s,
                                   recurrence=False) for s in seeds}
            L = C.MEMORY_PASS_T_MS
            gains = {s: on[s][f"r2_{L}"] - off[s][f"r2_{L}"] for s in seeds}
            surrogate_ok = abs(data.surrogate_ain / data.ain - 1) <= C.SURROGATE_TOL
            real = {}
            if data.real_spikes is not None:          # secondary check (reported): real beats, deviation from the mean beat
                for s in seeds:
                    a = memory_curve(self.ctx, liqs[s], g_use[s], scale, data.real_spikes, data.real_targets, s)
                    b = memory_curve(self.ctx, liqs[s], g_use[s], scale, data.real_spikes, data.real_targets, s, recurrence=False)
                    real[str(s)] = {"on": a, "off": b}
            rec["level4"] = {"pass": bool(surrogate_ok and all(np.isfinite(v) and v >= C.MEMORY_MIN_GAIN for v in gains.values())),
                             "gain_100ms": {str(s): v for s, v in gains.items()},
                             "on": {str(s): v for s, v in on.items()}, "off": {str(s): v for s, v in off.items()},
                             "surrogate_spikes_vs_ecg": data.surrogate_ain / data.ain, "surrogate_ok": bool(surrogate_ok),
                             "firing_on_surrogate_hz": {str(s): on[s]["rate_hz"] for s in seeds},
                             "real_beats_reported": real,
                             "firing_on_ecg_hz": {str(r["seed"]): r["rate_hz"] for r in rec["level3"].get("rows", [])}}
        else:
            rec["level4"] = {"pass": False, "reason": "no g to judge at"}
        rec["in_scale"] = scale
        rec["pass"] = bool(rec["level2"]["pass"] and rec["level3"]["pass"] and rec["level4"]["pass"])
        rec["first_failed_level"] = next((lv for lv in ("level2", "level3", "level4") if not rec[lv]["pass"]), None)
        rec["seconds"] = time.time() - t0
        self.log(f"  -> {key_of(f)}: {'PASS' if rec['pass'] else 'fail at ' + rec['first_failed_level']} "
                 f"(g {g_chosen}, {rec['seconds']:.0f} s)")
        self.records[rkey] = rec
        if self.on_record:
            self.on_record(rkey, rec)
        return rec

    # ------------------------------------------------------------------ search
    def forward(self, start_k=0):
        for k in range(start_k, 12):
            f = frozenset(range(1, k + 1))
            rec = self.evaluate(f)
            self.path.append(("forward", key_of(f), rec["pass"]))
            if rec["pass"]:
                return f, k
        return None, None

    def backward(self, f):
        cur = frozenset(f)
        while True:
            removed = False
            for fac in sorted(cur, reverse=True):
                cand = cur - {fac}
                rec = self.evaluate(cand)
                self.path.append(("backward", key_of(cand), rec["pass"]))
                if rec["pass"]:
                    cur = cand
                    removed = True
            if not removed:
                return cur

    def confirm(self, f, seeds, y_grid=None):
        rec = self.evaluate(f, y_grid=y_grid)
        frozen = {"g": rec["g_rule"], "in_scale": rec["in_scale"]}
        c = self.evaluate(f, data=self.confirm_data, seeds=seeds, frozen=frozen, y_grid=y_grid,
                          tag=f"confirm|{key_of(f)}|n{self.n_sub_for(f, y_grid)}|{','.join(map(str, seeds))}")
        return c

    def run(self):
        out = {"steps": []}
        rx = self.evaluate(frozenset())
        ry = self.evaluate(C.ALL_FACTORS)
        out["X"], out["Y"] = rx["pass"], ry["pass"]
        if rx["pass"]:
            final, k = frozenset(), 0
        else:
            final, k = self.forward(1)
        if final is None:
            out.update(final=None, stop="no forward candidate passed all four levels")
            out["path"] = self.path
            return out
        final = self.backward(final)
        attempts = 1
        seeds = C.SEEDS_CONFIRM
        while True:
            c = self.confirm(final, seeds)
            out["steps"].append({"confirm": key_of(final), "seeds": list(seeds), "pass": c["pass"]})
            if c["pass"]:
                break
            if final == C.ALL_FACTORS or attempts >= 2:
                out.update(final=None, stop="confirmation failed" + (" (Y itself)" if final == C.ALL_FACTORS else " twice"),
                           attempts=attempts, path=self.path)
                return out
            attempts += 1
            f2, k = self.forward(k + 1)                   # resume forward once from the last forward candidate
            if f2 is None:
                out.update(final=None, stop="confirmation failed; no later forward candidate passed", attempts=attempts,
                           path=self.path)
                return out
            final = self.backward(f2)
            seeds = C.SEEDS_RETRY
        rec = self.evaluate(final)
        out.update(final=sorted(final), final_key=key_of(final), attempts=attempts, path=self.path,
                   g=rec["g_rule"], maass_scale=10 in final, in_scale=rec["in_scale"], n_sub=rec["n_sub"])
        if final == C.ALL_FACTORS:                         # Y: also tested on n; if it passes there too, use n
            yn = self.evaluate(final, y_grid=self.n_fine)
            ycn = None
            if yn["pass"]:
                ycn = self.confirm(final, seeds, y_grid=self.n_fine)
            if yn["pass"] and ycn["pass"]:
                out.update(n_sub=self.n_fine, g=yn["g_rule"], in_scale=yn["in_scale"], y_on_n=True)
        return out
