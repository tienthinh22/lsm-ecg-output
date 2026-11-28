# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS — the only cell you normally need to edit                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
# The redesign (plan "LSM-ECG thesis: redesign plan, design and criteria"). Runs on top of the replication's
# output folder: point WORK_DIR at it (or RESUME_DIRS at a copy), so every cached run is read, never rerun.

# ---- Machine -----------------------------------------------------------
MACHINE = "local"                  # "kaggle" or "local" (your own Linux machine; set its paths below)

MACHINES = {
    "kaggle": {
        "INPUT_DIR":   "/kaggle/input/datasets/tienthinh22/mitdb-1-0-0/mit-bih-arrhythmia-database-1.0.0",
        "WORK_DIR":    "/kaggle/working",
        "SCRATCH_DIR": "/kaggle/working/scratch",
        "RESUME_DIRS": ["/kaggle/input/datasets/tienthinh22/mitdb-common"],
        "SESSION_BUDGET_HOURS": 11.3,
        "DISK_LIMIT_GB": 19.0,
    },
    "local": {
        "INPUT_DIR":   "~/lsm_ecg/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0",
        "WORK_DIR":    "~/lsm_ecg/output",          # the replication's output folder (reused in place)
        "SCRATCH_DIR": None,
        "RESUME_DIRS": [],
        "SESSION_BUDGET_HOURS": 240.0,
        "DISK_LIMIT_GB": 250.0,
    },
}
_M = MACHINES[MACHINE]
INPUT_DIR            = _M["INPUT_DIR"]
WORK_DIR             = _M["WORK_DIR"]
SCRATCH_DIR          = _M["SCRATCH_DIR"] or WORK_DIR.rstrip("/") + "/scratch"
RESUME_DIRS          = _M["RESUME_DIRS"]
AUTO_RESUME_ROOT     = "/kaggle/input"
SESSION_BUDGET_HOURS = _M["SESSION_BUDGET_HOURS"]
DISK_LIMIT_GB        = _M["DISK_LIMIT_GB"]

# ---- Stages (plan, Stages, compute and stop points) ---------------------
# Stage 1 (one machine): level 1 checks, the Maass check, the X to Y ladder, the confirmation and the speed test.
#   It writes checks/redesign_calibration.json; copy that file into every machine's WORK_DIR/checks before stage 2.
# Stage 2 (every machine): the classification runs E1 to E3, split into units (see UNIT_SHARES).
# Stage 3 (one machine): MERGE_DIRS = every machine's output folder; then the hybrid-rule runs and the analysis.
RUN_STAGE1           = True        # does nothing once the calibration file exists (delete it to redo stage 1)
RUN_STAGE2           = True        # needs the calibration file
STAGE1_STEPS         = ["level1", "maass_check", "ladder", "speed"]   # resumable: each step saves its result
LEVEL1E_PROCESSES    = None        # CPU processes for (d)'s Brian2 script in 1e (None = every core)
D_REPLACEMENT        = "fixed"     # (d)'s mean replacement of bad U, D, F draws: "fixed" (works as (d) intends) or
                                   # "as_written" (Brian2 reads its boolean index as 0/1, so negative draws stay)
X_UPDATE             = "equations" # X's reset: "equations" = (a)'s Eqs. 17-19, as the plan says; "authors_code" = the
                                   # authors' snnTorch neuron (its threshold test subtracts 1 more while the stored
                                   # voltage is above 1). Only "authors_code" can reproduce the cached runs (check 1f).
MAASS_SCRIPT_URL     = "https://raw.githubusercontent.com/brian-team/brian2/master/examples/frompapers/Maass_Natschlaeger_Markram_2002.py"   # (d), Brian2 examples
MAASS_SCRIPT_PATH    = None        # or a local copy of (d)'s maass2002.py (used instead of downloading)

# ---- Several machines in stage 2 ----------------------------------------
# Units (plan, Stages): one split of one seed for a robust-normalisation front end, or all splits of one seed for a
# standard-normalisation front end. Every machine computes the same assignment from UNIT_SHARES (machine name ->
# weight, e.g. its measured beats per second from the stage-1 speed test) and runs only its own units.
UNIT_SHARES          = {"this": 1.0}
THIS_MACHINE         = "this"

# ---- Session control ---------------------------------------------------
SELF_TEST            = False
SMOKE_TEST           = False       # True = tiny end-to-end test (few beats, 1 seed, coarse grid), its own WORK_DIR
BENCHMARK            = False
ANALYSIS_RESERVE_MIN = 25
RETRY_FAILED         = True
OLD_ANALYSIS         = False       # True = also rebuild the replication's tables and figures

# ---- Software ----------------------------------------------------------
PIN_TORCH_240  = False             # the redesign runs on the installed PyTorch
AUTHORS_REPO   = "https://github.com/Richard97Code/LSM-ECG-Classification.git"
AUTHORS_COMMIT = "02e45c4f3232886aed16a40d77de1f851ef4fcdd"

# ---- Experiment scope --------------------------------------------------
SEEDS = [2026, 69, 420]             # classification seeds (plan, ground rules)

# The replication's groups (all finished; disabled here: their cached runs are read for comparisons).
OLD_GROUPS = ["checks", "brian2", "main_final", "ablation_final", "rr_final", "baseline_final", "gain_final",
              "main_cv", "gain_cv", "ablation_cv", "rr_cv", "baseline_cv", "nn_tune", "retune", "kstale_final",
              "kstale_cv", "huang_final", "huang_cv", "meanrr_final", "meanrr_cv", "elmrr_final", "elmrr_cv",
              "rrliqmean_final", "rrliqmean_cv", "meanrr_nn", "hybrid", "working_diag", "working_final", "working_cv"]
NEW_GROUPS = [
    "e1",              # final liquid, its recurrence-off twin, X at g = 1: film + 3 snapshots (config 11)
    "e1_ablation",     # memory ablation: final liquid and its twin on beats with the early input replaced
    "e1_controls",     # no-liquid, fading-memory and snapshot-ELM controls at the 3 snapshots
    "e2",              # all 24 configurations, film readout, the final liquid at config 11's g
    "e3",              # RR family: film + RR features, RR in the liquid, RR + mean-RR ratio in the liquid
    "e3_partner",      # only if check 1f fails: the paper's fixed liquid + RR in the liquid, redone in the new code
    "e3_hybrid",       # Huang's SVEB rule on the base, RR-readout and RR-liquid runs (after the merge)
]
GROUP_ORDER = OLD_GROUPS + NEW_GROUPS
GROUPS_ENABLED = {g: (g in NEW_GROUPS) for g in GROUP_ORDER}

SHARE = "BOTH"; SHARE_GROUPS = {"A": [], "B": []}     # the replication's two-machine split (not used here)
MERGE_DIRS = []          # stage 3: every machine's output folder; the combined result goes to WORK_DIR (a new folder)

# replication settings that its cells still read (unchanged values)
GAINS = [1, 2, 3, 5, 8, 13, 21, 34]; DENSE_LAMBDAS = [1.5, 2.5]; RETUNE_TRIALS = 30
RETUNE_GRID = {"gain": [1, 2, 3, 5, 8, 13], "lambda_val": [0.8, 1.5, 2.5], "num_levels": [50, 100, 200, 400]}
NN_EPOCHS = 15; BRIAN2_BEATS = 10; NN_TUNE_TRIALS = 30; NN_PARAM_CAP = 35000; NN_TUNE_MAX_EPOCHS = 30
NN_TUNE_CHECKPOINTS = [5, 10, 15, 20, 25, 30]
HUANG_MATRICES = 15; HUANG_DIM = 50; HUANG_C = 1.0; HUANG_GAMMAS = [1.3, 1 / (2 * 1.3 ** 2)]
HUANG_THRESHOLD = 0.8; HUANG_SVM_CACHE_MB = 500; HYBRID_RULES = ["replace", "add"]
WORKING_LAMBDA = 2.0; WORKING_GAINS = [1, 3, 10, 30, 100, 300, 1000]; WORKING_DIAG_BEATS = 300
WORKING_T_PERTURB = 150; WORKING_MAX_FIRING = 0.2
SAVE_STATES = []

# ---- Performance -------------------------------------------------------
SIM_BATCH_CV    = 4096
SIM_BATCH_FINAL = 2000
NEW_SIM_BATCH   = 500      # beats per GPU batch for the new simulator: the same on every machine (plan, determinism)
PIPELINE        = True
FIT_WORKERS     = "auto"
CORE_SEARCH     = False
RAM_RESERVE_GB  = 4.0
READOUT_JOBS    = 4
