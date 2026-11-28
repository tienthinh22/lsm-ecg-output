"""Builds lsm_ecg_redesign.ipynb from the replication notebook's tested cells, the new cells in nb/ and the lsm2
package (embedded, so the notebook stays a single file)."""
import hashlib, json, os, re, sys, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.environ.get("OLD_NB", "/home/claude/tienthinh22/lsm-ecg-output/lsm_ecg_replication.ipynb")
OUT = os.path.join(HERE, "lsm_ecg_redesign.ipynb")
old = json.load(open(OLD))
oc = [("".join(c["source"]), c["cell_type"]) for c in old["cells"]]
assert oc[3][0].startswith('NOTEBOOK_BUILD = "'), "unexpected replication notebook layout"

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

def nb_file(name):
    return open(os.path.join(HERE, "nb", name)).read()

def embed_package():
    files = {}
    for sub, dst in (("lsm2", "lsm2"), ("tests", "lsm2_tests")):
        for fn in sorted(os.listdir(os.path.join(HERE, sub))):
            if fn.endswith(".py"):
                src = open(os.path.join(HERE, sub, fn)).read()
                assert "'''" not in src and not src.rstrip("\n").endswith("\\"), f"{fn} cannot be embedded"
                files[f"{dst}/{fn}"] = src
    parts = ["# The redesign's code: the lsm2 package and its tests (level 1a), written to SCRATCH_DIR and imported.",
             "# The same files are shipped next to this notebook (lsm2/, tests/); this cell is generated from them.",
             "LSM2_FILES = {}"]
    for k, v in files.items():
        parts.append(f"LSM2_FILES[{k!r}] = r'''{v}'''")
    parts.append(textwrap.dedent("""
        for _rel, _src in LSM2_FILES.items():
            _p = os.path.join(SCRATCH_DIR, _rel)
            os.makedirs(os.path.dirname(_p), exist_ok=True)
            if not os.path.exists(_p) or open(_p).read() != _src:
                with open(_p, "w") as _f:
                    _f.write(_src)
        open(os.path.join(SCRATCH_DIR, "lsm2", "__init__.py"), "a").close()
        if SCRATCH_DIR not in sys.path:
            sys.path.insert(0, SCRATCH_DIR)
        for _m in [m for m in sys.modules if m == "lsm2" or m.startswith("lsm2.")]:
            del sys.modules[_m]
        import lsm2
        print(f"lsm2: {len(LSM2_FILES)} files at {os.path.join(SCRATCH_DIR, 'lsm2')}")
        """))
    return "\n".join(parts)

env = oc[3][0]
env = env.replace('import threading\nSESSION_TAG', 'import threading\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (plan, determinism)\nSESSION_TAG', 1)
env = env.replace('torch.backends.cudnn.allow_tf32 = False\n',
                  'torch.backends.cudnn.allow_tf32 = False\n'
                  'torch.use_deterministic_algorithms(True, warn_only=True)   # plan: PyTorch\'s deterministic algorithms on\n', 1)
assert "use_deterministic_algorithms" in env and "CUBLAS_WORKSPACE_CONFIG" in env
env = env.replace('"wfdb": None, "tqdm": None, "scikit-learn": "1.7.2", "joblib": None, "threadpoolctl": None}',
                  '"wfdb": None, "tqdm": None, "scikit-learn": "1.7.2", "joblib": None, "threadpoolctl": None, "brian2": None}', 1)

old_analysis = "if OLD_ANALYSIS:\n" + textwrap.indent(oc[23][0], "    ") + "\nelse:\n    print('replication tables and figures not rebuilt (OLD_ANALYSIS = False)')\n"

intro = """# LSM-ECG thesis: the redesign (X to Y ladder, E0 to E3)
Implements the plan *LSM-ECG thesis: redesign plan, design and criteria*. Built on the replication notebook: its data, encoders, folds, readout, run cache and runner cells are reused unchanged, so the protocol is identical and every cached run is read, not rerun. The new code is the `lsm2` package (simulator, liquids, levels 1 to 4, gain rule, Maass check, ladder), embedded below and shipped next to this notebook with its tests.

**Stage 1** (one machine, `RUN_STAGE1 = True`): level 1 checks (1a tests, 1b Brian2, 1c this card, 1d time grid, 1e Maass's script, 1f regression), the Maass check, the ladder with its confirmation, the input size of every front end and a speed test. It stops with an error at any stop point of the plan, and writes `checks/redesign_calibration.json`.

**Stage 2** (every machine): copy the calibration file into each machine's `WORK_DIR/checks/`, set `UNIT_SHARES` (the same on every machine, weights from each machine's speed test) and `THIS_MACHINE`, and run. Each machine first runs level 1c on its own card (once), then only its own units.

**Stage 3** (one machine): `MERGE_DIRS = [every machine's output]` with a new, empty `WORK_DIR` combines them; a normal run on the merged folder rebuilds `summary_redesign.md` and `tables/redesign/`.

Point `WORK_DIR` at the replication's output folder (or list a copy in `RESUME_DIRS`): the comparisons read its cached runs.

## 0. Constants (edit here)"""

cells = [md(intro), code(nb_file("c01_constants.py")),
         md("## 1. Environment"), code(env),
         md("## 2. The authors' code (pinned commit)"), code(oc[5][0]),
         md("## 3. The redesign's code (lsm2)"), code(embed_package()),
         md("## 4. Utilities: cache, resume, guards, metrics, the authors' readout"), code(oc[7][0]),
         md("## 5. Data: MIT-BIH → DS1 / DS2, checks against the authors' files"), code(oc[9][0]),
         md("## 6. Spike encoders and measured A_in"), code(oc[11][0]),
         md("## 7. The replication's liquid (the authors' class; used for X in E1, 1f and the ELM controls)"), code(oc[13][0]),
         md("## 8. Non-liquid controls and neural-network baselines"), code(oc[15][0]),
         md("## 9. The replication's plan (its groups are disabled here; its run ids identify the cached runs)"), code(oc[17][0]),
         md("## 10. Stage 1: level 1, the Maass check, the ladder, the confirmation (no labels)"), code(nb_file("c10_stage1.py")),
         md("## 11. Executors of the replication (incl. the hybrid rule)"), code(oc[19][0]),
         md("## 12. Stage 2 plan: E1 to E3"), code(nb_file("c11_plan2.py")),
         md("## 13. Stage 2 executors: the new liquid and the snapshot controls"), code(nb_file("c12_exec2.py")),
         md("## 14. Run everything that is pending"), code(oc[21][0]),
         md("## 15. Analysis of the redesign: F1 to F7 and the claims"), code(nb_file("c13_analysis2.py")),
         md("## 16. The replication's tables and figures (optional)"), code(old_analysis),
         md("## 17. End of session"), code(oc[25][0])]
fp = hashlib.sha1("".join("".join(c["source"]) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:12]
cells[3]["source"] = re.sub(r'^NOTEBOOK_BUILD = "[^"]*"', f'NOTEBOOK_BUILD = "{fp}"', "".join(cells[3]["source"]), count=1).splitlines(True)
nb = {"cells": cells, "metadata": old.get("metadata", {}), "nbformat": 4, "nbformat_minor": old.get("nbformat_minor", 5)}
json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(cells)} cells, build {fp}")
