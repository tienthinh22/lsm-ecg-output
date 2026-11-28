# Redesign code: notes for the reviewer

What is where, which plan items each piece implements, the numbers the plan does not fix (each with its
reason, per the plan's ground rule), and the findings made while building.

## Layout

| Piece | File | Plan section |
|---|---|---|
| Every fixed number, with source | `lsm2/config.py` | all |
| Liquids: X, every candidate, Maass's two versions, X from the authors' class | `lsm2/liquid.py` | From X to Y; Where Y still differs; Simulation details |
| Simulator (our schedule and the Brian2 script's) | `lsm2/sim.py` | Simulation details |
| Brian2 version of our update (1b); loader for (d)'s script (1e) | `lsm2/brian2_ref.py` | Correctness criteria 1b, 1e |
| Levels 1b to 1f | `lsm2/level1.py` | Correctness criteria |
| Level 2 | `lsm2/level2.py` | Correctness criteria |
| Level 3 measures, rank scores, twin runs, level 4 memory | `lsm2/measure.py` | Correctness criteria; Choosing the recurrent gain |
| Gain grid and rule | `lsm2/gain.py` | Choosing the recurrent gain; decision 5 |
| Input size (paper's rule, factor-5 bisection, factor 9) | `lsm2/calib.py` | Input and calibration |
| Inputs: DS1 halves, calibration sets, surrogate, Poisson trains | `lsm2/data.py` | Ladder; level 4; Maass check |
| Maass check | `lsm2/maass_check.py` | Testing the test on Maass's liquid |
| Ladder, backward loop, confirmation | `lsm2/ladder.py` | The X to Y ladder |
| Level 1a and logic tests | `tests/` | Correctness criteria 1a |
| Notebook: stage 1 | `nb/c10_stage1.py` | Stages |
| Notebook: stage 2 plan, executors | `nb/c11_plan2.py`, `nb/c12_exec2.py` | Experiments and controls; Readouts |
| Notebook: analysis | `nb/c13_analysis2.py` | Statistics and decision rules |
| Notebook builder (reuses the replication's cells) | `build_notebook.py` | — |

## Numbers the plan does not fix (chosen here, with reasons)

1. **Search seeds 1, 2, 3.** The plan fixes 3 search seeds but not their values. Any 3 fixed values, kept apart from the confirmation (101, 202, 303), retry (404, 505, 606), redesign (707, 808, 909) and classification (2026, 69, 420) seeds.
2. **DS1 halves.** A fixed random permutation (seed 0) of the 22 DS1 records, 11 records per half. The plan says "split in half, without looking at labels".
3. **Twin runs: 100 beats per g per seed.** A reported cross-check (not a pass rule); 100 beats give a stable mean ratio at 1/5 of the separation set's cost. The extra spike goes on a channel that is silent at sample 288, chosen at random. "Early" is the first 50 ms after the spike (samples 289–306); "late" is the last 50 ms of the beat (559–576).
4. **Memory test: 500 surrogate sequences, 250 to train and 250 to test.** The same count as level 3's beats. Walks start uniformly on [0, 1]. The ridge penalty grid is 10^-3 … 10^4 (8 values, wide enough for states of any scale), chosen by 5 inner folds grouped by sequence. R² is pooled over all scored samples. Each lag is scored from sample max(100, T) on, so every target exists.
5. **Factor-5 bisection.** Stops within 1% of the target rate, well inside the seed-to-seed spread, or after 40 steps. The rate is averaged over the seeds in use, on the 500 calibration beats.
6. **GPU batch: 500 beats** for every simulation, and 100 sequences for the memory test. Fixed on every machine, as the plan's determinism rule requires.
7. **Starting states of the separation and generalisation sets.** Both use uniform voltages between the resting level and the threshold, for every candidate. The plan asks for "the same starting-state perturbation for every candidate"; applying it to both sets keeps separation and generalisation symmetric.
8. **Starting-state rule.** States are compared by Euclidean distance of the 30 ms-filtered states at sample 576. "Pairs of different beats" means all pairs among the 500.
9. **Firing rate** is counted over the 577 samples of the beat.
10. **Grid extension at the lower edge too.** If the best g is the smallest one tried (0.1), the grid extends down in 3x steps. The plan says "at an edge"; the limit is 3 extensions in total.
11. **Fine pass.** g* × 1.4^k for k = ±1, ±2, ±3 (6 values, "about 7" with g*), centred on the best coarse g.
12. **Choosing g** uses seed-averaged firing for the band; level 3 then checks the band on every seed.
13. **Factor 10's g.** Sets each liquid's realised mean |w_EE| to 0.1, i.e. the isolated EE push before U/Ū.
14. **Snapshot ELM.** The replication's ELM construction (the fixed liquid's input weights, same seed), applied to the 30 ms-filtered input at the snapshot, centred on the training mean, then ReLU. 343 features.
15. **E1's paper liquid X.** Built with the authors' class, A_in measured on each split's training rows, exactly as the replication's fixed runs. This is what lets check 1f compare against those runs.
16. **No-liquid and fading-memory controls** are identical across seeds (the Multi encoder is deterministic). They are still run per seed, to match the plan's counts (117 runs each).
17. **Memory ablation.** 50 ms = 18 samples. The mean beat is taken over the split's training rows, and the ablated beat is clipped to [0, 1].
18. **Threshold test u ≥ 1.** The plan says "at 1 or above". (d)'s script and snnTorch use u > 1; the script version follows (d).
19. **1d reference for X** is also 28 sub-steps.
20. **1e seeds.** Our runs use liquid seeds 1–5; (d)'s use np.random.seed 10001–10005.
21. **Maass check inputs.** Poisson spike counts per sample (2.78 ms) at 20 Hz. g is swept as a multiple of Maass's own weights (g = 1).

## Findings while building (decisions for you)

A. **(d)'s script does not replace bad U, D, F draws.**
   - `synapses.U[synapses.U < 0] = U` and the D and F lines intend a mean replacement. Brian2 reads the boolean array as the integer indices 0 and 1, so only synapses 0 and 1 of each group get the mean, and every negative draw stays. That is about 2.3% of synapses each for U, D and F.
   - A negative D or F makes x or u grow without bound. In a test run, one neuron's voltage reached −7·10^15 within 25 ms.
   - `D_REPLACEMENT = "fixed"` (the default) patches the three lines in memory so they do what (d) and the plan intend. `"as_written"` keeps (d) unchanged, and our script version copies its behaviour.
   - 1e and the Maass check's script version follow this switch.

B. **(a)'s code is not exactly (a)'s Eqs. 17–19.**
   - snnTorch's `Leaky(reset_delay=False)` spikes when βu + i − [u_prev > 1] > 1 (strict). So while the stored voltage is above 1 (after a step that went above 2, or a blocked spike), the threshold test subtracts 1 more.
   - On synthetic Multi input at the fixed liquid's drive, this changes about 1% of window counts (20,000 of 2.0 million, 3 seeds); the strict versus non-strict threshold changes none.
   - `X_UPDATE = "equations"` (the default, as the plan says) follows Eqs. 17–19, so check 1f will most likely fail and the 39 partner simulations will be redone. That is the plan's own fallback.
   - `X_UPDATE = "authors_code"` reproduces the authors' loop exactly: 0 of 2.0 million window counts differ (`tests/test_logic.py`). With it, 1f can pass and the cached runs are reused. It applies to every liquid that resets by subtraction (X and the candidates without factor 5).

C. **(d)'s Brian2 schedule.** In (d)'s network, the synapse state update runs before the summed current, so the current used in a step has already decayed by one step. Our script version follows the schedule Brian2 actually builds for (d)'s objects (checked spike for spike in `tests/test_script_mode.py`).

D. **Counts.** The stage-2 plan produces 504 new simulations and 2,301 runs, as the plan's table says (234 of the runs are hybrids). If 1f fails, the 39 partner simulations and runs come on top. The plan's "about 400 units" is 252 here: 6 robust front ends × 3 seeds × 13 splits, plus 6 standard ones × 3 seeds. Units are a sharing granularity only; they do not change any result.

E. **NB variance factor.** The plan's 0.336 (the actual test/train ratio) is used. The replication's analysis used 0.25/0.75 = 0.333.
