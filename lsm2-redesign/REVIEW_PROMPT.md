I need you to check that the code for my master's thesis does what its plan says. The plan is the source of truth. It has already had three external reviews, so don't review the plan itself; report a plan problem only if the code exposes one.

**What you have**

- **The plan:** https://claude.ai/artifact/FkJKAHuxYF5JcdyALMdqae ("LSM-ECG thesis: redesign plan, design and criteria"). Read all of it, including the tables and the decisions table.
- **The code:** the attached zip `lsm2-redesign.zip`. Start with `README.md` and `NOTES.md`. NOTES.md maps every plan section to a file, lists every number the plan does not fix (each with the reason it was chosen), and lists five findings (A to E) made while building.
- **Source labels** used in the plan and the code:
  - (a) Perez-Lopez et al. 2026 and the authors' code, commit 02e45c4
  - (b) Huang et al. 2014
  - (c) Maass, Natschlager & Markram 2002
  - (d) Schmitt 2022, the Brian2 reproduction of (c)'s Figure 2: https://raw.githubusercontent.com/brian-team/brian2/master/examples/frompapers/Maass_Natschlaeger_Markram_2002.py
  - (e) Legenstein & Maass 2007
- **How the code is built:**
  - `lsm2/` is a package with the simulator, liquids, levels 1 to 4, gain rule, Maass check, ladder and statistics.
  - `nb/` holds the notebook's new cells: stage 1, stage-2 plan, executors and analysis.
  - `build_notebook.py` assembles `lsm_ecg_redesign.ipynb` from the replication notebook's existing cells plus these files. The replication cells are reused unchanged except for the edits listed in `build_notebook.py`.
  - `tests/` holds the level-1a closed-form tests, a spike-for-spike test of the script version against (d)'s Brian2 script, and tests of the decision logic and statistics.
- **If you can run code:** run `PYTHONPATH=. MAASS_SCRIPT=<path to (d)'s script> python -m pytest -q tests/`. It needs torch, numpy, scipy, pandas, brian2 and pytest. One test also needs the authors' repository; it skips itself without it.

**What to check, in this order**

1. **Simulation math** (`lsm2/sim.py`, `lsm2/liquid.py`), against the plan's "Simulation details". This matters most. Check:
   - the order within a sub-step;
   - the neuron update, tap, both resets, the minimum gap with and without the clamp;
   - every ms-to-grid conversion: leak, tap, delays, gaps, current decay, D and F;
   - currents scaled so their total equals the push;
   - input acting in its own sample's first sub-step;
   - the dynamic synapse in (d)'s update order with w / U-bar;
   - synapse state at the start of a beat;
   - the script version's six settings from level 1e, and the schedule Brian2 actually builds for (d)'s network.

   Work at least two cases by hand (for example one isolated EE spike through factor 7 on 4 sub-steps, and one facilitating EI train) and compare with the code.
2. **The 11 factors.** For each factor, does `candidate_settings` / `build_candidate` switch exactly what the plan's factor table says, as a whole, and nothing else? Is X with all 11 factors exactly Y? Check every number in `lsm2/config.py` against its stated source, including the factor-3 ratios, factor 9's 0.12 / 0.06 and factor 10's 0.1.
3. **Levels 1a to 4 and the Maass check** (`lsm2/level1.py`, `level2.py`, `measure.py`, `maass_check.py`, `tests/`). For each level, does the test and its pass rule match the plan's criteria table word for word? That includes 1e's 4.2 SE at 95% of points, level 2's 4.5 SE with targets from 10 million draws, level 3's band, starting-state rule and gap peak, and level 4's 0.05 R² gain at 100 ms on all seeds.

   For the Maass check, confirm:
   - both versions are run at g = 1 and at the rule's g;
   - the "must also reject something" condition is applied;
   - the rank-measure keep rule is "a peak inside the band above the smallest g, and a lower gap at the next higher g".
4. **The gain rule** (`lsm2/gain.py`). Check:
   - the grid, the order of the passes and the extensions (3x, at most 3);
   - the fine pass;
   - that g = 0 is excluded;
   - the tie rule (range of the best setting's seed values, smallest tied g);
   - the band-edge rule and the "limited by the band" flag;
   - that factor-10 candidates are swept but use Maass's scale;
   - that no in-band setting means level 3 fails.
5. **Input size** (`lsm2/calib.py`, and its use in the ladder and stage 1). Check:
   - the paper's rule always uses X's leak and the candidate's P_in (1 with factor 6);
   - factor 5's bisection, with recurrence off and the input-receiving neurons matched to the Maass check's paper-version rate, with the 2:1 split;
   - factor 9 replaces either rule;
   - sizes are measured once per front end on the search half.
6. **The ladder** (`lsm2/ladder.py`). Check:
   - X and Y are tested first;
   - forward runs in order 1 to 11 and stops at the first pass;
   - the backward loop removes newest first and repeats full passes;
   - confirmation uses the other half of DS1, seeds 101/202/303, frozen g and input size, and level 3 without the gap-peak part;
   - a failed confirmation means one retry, resuming forward from the last forward candidate, then backward again, then seeds 404/505/606;
   - the stop conditions are right;
   - Y always runs on 28 sub-steps, and is also tested on n if it is the final liquid;
   - every candidate's record holds firing, gap, twin runs and the memory curve.
7. **Stage 2** (`nb/c11_plan2.py`, `nb/c12_exec2.py`, against "Readouts" and "Experiments and controls"). Count the specs per experiment and compare with the plan's table (about 504 new simulations, 2,301 classification runs, the E1/E2 overlap of 39). Check:
   - the film layout and the 30 ms snapshots at samples 288 / 342 / 396;
   - the memory-ablation input, and that the mean beat comes from training rows only;
   - the three snapshot controls;
   - the RR variants and their per-spike size;
   - the hybrid bases;
   - the paper's liquid X at g = 1;
   - that the readout settings are the replication's;
   - that run ids cannot collide with the cached runs and change when the calibration changes;
   - that unit sharing across machines assigns every unit to exactly one machine;
   - determinism: per-beat random numbers come from (seed, beat index) on the CPU, the batch size is fixed, and deterministic algorithms are on.
8. **Statistics and claims** (`lsm2/stats.py`, `nb/c13_analysis2.py`). Check:
   - the Nadeau–Bengio test (seeds averaged per split, factor 1/12 + 0.336, df 11, its 95% interval);
   - Holm within each family;
   - the DS2 difference paired by seed;
   - the verdict rules, including F1's same sign in all 3 seeds;
   - the comparisons that make up F1 to F7;
   - F4's Friedman / Iman-Davenport / Kendall's W per seed;
   - claims A to E and the narrow wording, exactly as the plan states them;
   - the DS2-without-record-232 and per-class reports.
9. **Numbers without a source or reason.** Go through `NOTES.md`'s list of code choices. Is each one reasonable and consistent with the plan, and does each carry a reason? Then look for any number in the code that has neither a source nor a reason and is not on that list. Treat those as must-fix: it's the plan's ground rule.
10. **Findings A to E** in NOTES.md. Confirm or refute each from the code and the sources:
    - (A) (d)'s mean replacement of bad U, D, F draws does nothing in Brian2;
    - (B) (a)'s snnTorch neuron differs from its Eqs. 17–19 through an extra subtraction;
    - (C) (d)'s schedule;
    - (D) the run, simulation and unit counts;
    - (E) the 0.336 factor.

**Rules**

- For every finding, quote the plan sentence and the code (file and line), say what goes wrong in a concrete case, and give the smallest fix.
- Rank each finding must-fix, should-fix or minor:
  - **must-fix:** the code contradicts the plan, gives wrong numbers, or a number has no source or reason;
  - **should-fix:** the code is correct but fragile, or the plan leaves a gap the code fills silently;
  - **minor:** everything else that is still worth a line.
- Verify by computation wherever you can. Don't report style, naming or speed unless speed changes the plan's compute estimate by a lot.

**Output**

1. A table of findings, most severe first. Columns: #, rank, plan section, file:line, finding, smallest fix.
2. Notes for each of the 10 checks above, short where everything matches.
3. A list of what you verified as correct and how (by hand, by running a test, by reading).
