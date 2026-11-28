# lsm2: the liquid simulator used by the thesis notebooks

Written in this project (not the authors' code). The notebooks carry these files as text (`LSM2_FILES`) and write
them to disk when they run; this folder is the source, for reference and for running the tests.

Used by: `lsm_ecg_ytest.ipynb` (13 files), the task notebooks t1-t4 (config, liquid, sim, stats) and
`lsm_ecg_stream.ipynb` (config, liquid, sim, stats, measure, gain, calib, data, stream). The replication notebook
does not use it (it runs the authors' snnTorch code). With the authors' reset (`snn_quirk=True`), lsm2 reproduces
the stored replication runs exactly (task 1's `check.json`).

| file | what it is |
|---|---|
| config.py | constants (time steps, Maass 2002's parameters, grids) |
| liquid.py | liquid builders: Maass's recipe (`build_candidate`), the authors' liquid (`from_authors_model`), adaptation (`add_adaptation`) |
| sim.py | the simulator (plain loop and fast loop; streaming with carried state) |
| stream.py | streaming whole records: stacking liquids, chunk plans, window cutting, frozen thresholds |
| stats.py | Nadeau-Bengio corrected t-test, Holm, verdicts |
| measure.py, gain.py, calib.py, data.py | measures, the gain rule, input-size calibration, input data (ytest) |
| ladder.py, level1.py, level2.py, maass_check.py, brian2_ref.py | test and reference code (ytest) |

## Tests

    python -m pytest tests -k "not script"     # 71 tests, seconds
    MAASS_SCRIPT=/path/to/maass2002.py python -m pytest tests -k script   # 3 tests against Schmitt 2022's Brian2
                                                                          # reproduction of Maass 2002 (needs brian2)
    python stream_checks.py                    # the streaming code on CPU, a few minutes

The older `lsm2-redesign.zip` (2026-11-22) predates changes to config.py, liquid.py, sim.py and stats.py and has no
stream.py or tests/test_ytest.py.
