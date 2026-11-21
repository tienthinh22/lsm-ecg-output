# LSM-ECG replication: results so far
_Updated 2026-11-21T09:21:44 UTC, session 20261121_065408. Runner stopped: all planned runs are done._

## Environment
| torch | torch_pinned_2_4_0 | gpu | cuda | python | snntorch | sklearn | smoke_test |
|---|---|---|---|---|---|---|---|
| 2.14.1+cu126 | False | Tesla P40 | 12.6 | 3.13.13 | 0.9.4 | 1.7.2 | False |

## Checks
- DS1 beat order identical to the authors' file: **True**; class counts {'DS1': [45827, 943, 3787], 'DS2': [44216, 1837, 3219]}
- Authors' fold indices regenerate exactly: **True**
- Checksums: 704 files checked, 0 mismatches; missing from the input: none (none of them used by the pipeline)
- Brian2 vs PyTorch (fixed liquid, 10 beats, 2,229,500 neuron-steps): 0 mismatches vs float64 equations, 0 vs the authors' float32 snnTorch neuron; equations vs snnTorch (float32): 0
- Brian2 vs PyTorch (published liquid, 10 beats, 2,229,500 neuron-steps): 0 mismatches vs float64 equations, 0 vs the authors' float32 snnTorch neuron; equations vs snnTorch (float32): 0

## Progress by group
| group | done | pending |
|---|---|---|
| checks | 1 | 0 |
| brian2 | 1 | 0 |
| main_final | 288 | 0 |
| ablation_final | 50 | 0 |
| rr_final | 39 | 0 |
| baseline_final | 13 | 0 |
| gain_final | 63 | 0 |
| main_cv | 3456 | 0 |
| gain_cv | 756 | 0 |
| ablation_cv | 600 | 0 |
| rr_cv | 468 | 0 |
| baseline_cv | 156 | 0 |
| nn_tune | 636 | 0 |
| retune | 143 | 0 |
| kstale_final | 16 | 0 |
| kstale_cv | 192 | 0 |
| huang_final | 6 | 0 |
| huang_cv | 72 | 0 |
| meanrr_final | 2 | 0 |
| meanrr_cv | 24 | 0 |
| elmrr_final | 9 | 0 |
| elmrr_cv | 108 | 0 |
| rrliqmean_final | 18 | 0 |
| rrliqmean_cv | 216 | 0 |
| meanrr_nn | 78 | 0 |
| hybrid | 390 | 0 |
| working_diag | 1 | 0 |
| working_final | 12 | 0 |
| working_cv | 144 | 0 |

## Self-test (session 20261121_065408): all checks passed
- PASS packages: python 3.13.13, torch 2.14.1+cu126, numpy 2.1.3, sklearn 1.7.2, snntorch 0.9.4, wfdb 4.3.1, brian2 2.10.1
- NOTE torch 2.4.0 (authors'): 2.14.1+cu126 (fine; only matters for the exact-replication attempt)
- PASS brian2: 2.10.1
- PASS GPU: Tesla P40, 24.0 GB, CUDA 12.6; 650 liquid-sized steps in 0.14 s
- PASS cuDNN: convolution forward + backward ok
- PASS authors' code: commit 02e45c4f3232886aed16a40d77de1f851ef4fcdd
- PASS MIT-BIH checksums: 704 files checked, 0 mismatches, missing (unused) 0
- PASS DS1 = authors' order: class counts {'DS1': [45827, 943, 3787], 'DS2': [44216, 1837, 3219]}
- PASS authors' folds: fold indices regenerate exactly
- PASS liquid + readout (published): 64 beats: firing 0.4849/step, drive/target 23.86
- PASS liquid + readout (fixed): 64 beats: firing 0.0268/step, drive/target 1.00
- PASS GPU simulator vs reference: fixed: 0 of 109,760 window counts differ; published: 0 of 109,760 window counts differ (simulate() on cuda vs the float64 equations Brian2 matched, 10 DS2 beats)
- PASS extensions (Huang, hybrid, ELM+RR): Huang 2-matrix ensemble on 200 beats: Macro-F1 0.793; hybrid rule ok; RR window counts ok; Huang beats (50557, 200), mean-RR ratio median 1.000
- PASS working liquid (g_rec, raster): g_rec scales recurrent weights only: True; raster == simulate(): True; lambda 2: 2666 recurrent synapses
- PASS disk: 763 GB free at /root/lsm_ecg/output (all outputs together grow to ~12 GB)
- PASS RAM / cores: 113.3 GB RAM, 38 cores -> 1 parallel fits
- PASS log file: /root/lsm_ecg/output/logs/session_20261121_065408.log (tail -f /root/lsm_ecg/output/logs/latest.log)

## Resources this session
- runner: pipeline, 1 parallel fits; CPU cores 38; RAM 113.3 GB; GPU Tesla P40
- process peak RAM 8.0 GB; lowest available RAM 103.1 GB; peak GPU memory 4.69 GB (details: logs/resources_20261121_065408.csv)

## Projected time remaining (from measured speeds)
```

===== Projected time for everything that is left =====
 group  runs  jobs  hours
retune    35    35   0.21
TOTAL remaining: 0.2 h  ->  one continuous run (safety stop at 240.0 h)
runner: pipeline: 38 parallel fits (CPU cores 38, RAM allows 39), 222 s per Gcell per fit under load
machine: local; device: Tesla P40; CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz
measured speeds: ['encode_beats_per_s:latency', 'encode_beats_per_s:multi', 'encode_beats_per_s:rate', 'fit_s_per_Gcell:pipe', 'lsm_job_overhead_s', 'sim_beats_per_s:latency', 'sim_beats_per_s:multi', 'sim_beats_per_s:rate']
still default guesses (less reliable): nn_s_per_epoch_kbeat:cnn, nn_s_per_epoch_kbeat:gru, nn_unit_s_per_epoch_kbeat:cnn, nn_unit_s_per_epoch_kbeat:gru
```

## Config 11 on DS2 (paper: acc 93.51, Macro-F1 63.92)
| spec_mode | config_id | macro_f1_mean | accuracy_mean | recall_SVEB_mean | recall_VEB_mean | macro_f1_sd | accuracy_sd | recall_SVEB_sd | recall_VEB_sd | n |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_ain | 11 | 0.6453 | 0.9304 | 0.0798 | 0.9450 | 0.0049 | 0.0029 | 0.0081 | 0.0025 | 3 |
| fix_gamma | 11 | 0.6522 | 0.9352 | 0.0764 | 0.9399 | 0.0012 | 0.0016 | 0.0046 | 0.0023 | 3 |
| fixed | 11 | 0.6199 | 0.9163 | 0.0806 | 0.9323 | 0.0028 | 0.0008 | 0.0025 | 0.0030 | 3 |
| published | 11 | 0.6525 | 0.9367 | 0.0680 | 0.9332 | 0.0051 | 0.0036 | 0.0079 | 0.0040 | 3 |

## Friedman / Iman-Davenport / Kendall's W on the CV grid (paper: chi2 169.63, F 17.54, W 0.615)
| mode | seed | n_blocks | n_configs | chi2 | iman_davenport_F | kendall_W | best_by_rank |
|---|---|---|---|---|---|---|---|
| fix_ain | 69 | 12 | 24 | 168.0367 | 17.1207 | 0.6088 | 23 |
| fix_ain | 420 | 12 | 24 | 147.3200 | 12.5934 | 0.5338 | 11 |
| fix_ain | 2026 | 12 | 24 | 147.3667 | 12.6020 | 0.5339 | 12 |
| fix_gamma | 69 | 12 | 24 | 185.5733 | 22.5742 | 0.6724 | 12 |
| fix_gamma | 420 | 12 | 24 | 199.0700 | 28.4644 | 0.7213 | 12 |
| fix_gamma | 2026 | 12 | 24 | 181.0333 | 20.9691 | 0.6559 | 12 |
| fixed | 69 | 12 | 24 | 191.3567 | 24.8682 | 0.6933 | 11 |
| fixed | 420 | 12 | 24 | 188.7100 | 23.7806 | 0.6837 | 23 |
| fixed | 2026 | 12 | 24 | 179.1733 | 20.3550 | 0.6492 | 12 |
| published | 69 | 12 | 24 | 158.1333 | 14.7579 | 0.5729 | 12 |
| published | 420 | 12 | 24 | 200.2633 | 29.0863 | 0.7256 | 11 |
| published | 2026 | 12 | 24 | 165.6833 | 16.5208 | 0.6003 | 12 |

## Exact-replication attempt (published, seed 2026) vs the authors' per-fold CSVs
torch 2.4.0 active: False; GPU: Tesla P40 (authors: RTX 4060 Laptop, CUDA 11.8).

| encoder | folds | identical | mean_abs_diff |
|---|---|---|---|
| latency | 96 | 0 | 0.0178 |
| multi | 96 | 0 | 0.0162 |
| rate | 96 | 0 | 0.0114 |

## Encoders (morph + robust + SVM), published vs fixed
| mode | encoder | part | macro_f1_mean | macro_f1_sd_over_seeds | n_seeds |
|---|---|---|---|---|---|
| fix_ain | latency | cv | 0.5287 | 0.0107 | 3 |
| fix_ain | latency | final | 0.5752 | 0.0025 | 3 |
| fix_ain | multi | cv | 0.6228 | 0.0045 | 3 |
| fix_ain | multi | final | 0.6453 | 0.0049 | 3 |
| fix_ain | rate | cv | 0.5045 | 0.0079 | 3 |
| fix_ain | rate | final | 0.5794 | 0.0087 | 3 |
| fix_gamma | latency | cv | 0.5054 | 0.0035 | 3 |
| fix_gamma | latency | final | 0.5505 | 0.0085 | 3 |
| fix_gamma | multi | cv | 0.6104 | 0.0013 | 3 |
| fix_gamma | multi | final | 0.6522 | 0.0012 | 3 |
| fix_gamma | rate | cv | 0.5033 | 0.0010 | 3 |
| fix_gamma | rate | final | 0.5591 | 0.0054 | 3 |
| fixed | latency | cv | 0.5030 | 0.0068 | 3 |
| fixed | latency | final | 0.5874 | 0.0055 | 3 |
| fixed | multi | cv | 0.6165 | 0.0018 | 3 |
| fixed | multi | final | 0.6199 | 0.0028 | 3 |
| fixed | rate | cv | 0.5046 | 0.0029 | 3 |
| fixed | rate | final | 0.5522 | 0.0057 | 3 |
| published | latency | cv | 0.5373 | 0.0041 | 3 |
| published | latency | final | 0.5529 | 0.0038 | 3 |
| published | multi | cv | 0.6210 | 0.0109 | 3 |
| published | multi | final | 0.6525 | 0.0051 | 3 |
| published | rate | cv | 0.5057 | 0.0051 | 3 |
| published | rate | final | 0.5705 | 0.0088 | 3 |

## Input-gain sweep (fixed liquid; gain 1 = as the paper describes)
| spec_encoder | spec_gain | part | macro_f1 | macro_f1_sd | firing | drive | n |
|---|---|---|---|---|---|---|---|
| latency | 1 | cv | 0.5030 | 0.0825 | 0.0268 | 0.9794 | 36 |
| latency | 1 | final | 0.5874 | 0.0055 | 0.0268 | 0.9794 | 3 |
| latency | 2 | cv | 0.4892 | 0.0768 | 0.0729 | 1.9589 | 36 |
| latency | 2 | final | 0.5442 | 0.0036 | 0.0735 | 1.9589 | 3 |
| latency | 3 | cv | 0.4857 | 0.0691 | 0.0926 | 2.9383 | 36 |
| latency | 3 | final | 0.5381 | 0.0019 | 0.0926 | 2.9383 | 3 |
| latency | 5 | cv | 0.4971 | 0.0781 | 0.1595 | 4.8972 | 36 |
| latency | 5 | final | 0.5457 | 0.0090 | 0.1595 | 4.8972 | 3 |
| latency | 8 | cv | 0.5122 | 0.0842 | 0.2498 | 7.8355 | 36 |
| latency | 8 | final | 0.5632 | 0.0045 | 0.2498 | 7.8355 | 3 |
| latency | 13 | cv | 0.5307 | 0.0839 | 0.3656 | 12.7326 | 36 |
| latency | 13 | final | 0.5435 | 0.0042 | 0.3656 | 12.7327 | 3 |
| latency | 21 | cv | 0.5127 | 0.0779 | 0.4748 | 20.5681 | 36 |
| latency | 21 | final | 0.5737 | 0.0052 | 0.4749 | 20.5682 | 3 |
| latency | 34 | cv | 0.5142 | 0.0885 | 0.5463 | 33.3008 | 36 |
| latency | 34 | final | 0.5895 | 0.0192 | 0.5464 | 33.3009 | 3 |
| multi | 1 | cv | 0.6165 | 0.0954 | 0.0270 | 0.9967 | 36 |
| multi | 1 | final | 0.6199 | 0.0028 | 0.0276 | 1.0113 | 3 |
| multi | 2 | cv | 0.6085 | 0.0983 | 0.0582 | 1.9934 | 36 |
| multi | 2 | final | 0.6223 | 0.0008 | 0.0591 | 2.0227 | 3 |
| multi | 3 | cv | 0.6045 | 0.0997 | 0.0864 | 2.9901 | 36 |
| multi | 3 | final | 0.6265 | 0.0027 | 0.0876 | 3.0340 | 3 |
| multi | 5 | cv | 0.5987 | 0.1020 | 0.1340 | 4.9834 | 36 |
| multi | 5 | final | 0.6293 | 0.0059 | 0.1356 | 5.0567 | 3 |
| multi | 8 | cv | 0.6037 | 0.1027 | 0.1882 | 7.9735 | 36 |
| multi | 8 | final | 0.6459 | 0.0007 | 0.1900 | 8.0908 | 3 |
| multi | 13 | cv | 0.6073 | 0.0979 | 0.2508 | 12.9569 | 36 |
| multi | 13 | final | 0.6497 | 0.0024 | 0.2527 | 13.1475 | 3 |
| multi | 21 | cv | 0.6130 | 0.0907 | 0.3163 | 20.9305 | 36 |
| multi | 21 | final | 0.6487 | 0.0067 | 0.3180 | 21.2383 | 3 |
| multi | 34 | cv | 0.6130 | 0.0877 | 0.3817 | 33.8874 | 36 |
| multi | 34 | final | 0.6397 | 0.0057 | 0.3831 | 34.3858 | 3 |
| rate | 1 | cv | 0.5046 | 0.0731 | 0.0098 | 0.9950 | 36 |
| rate | 1 | final | 0.5522 | 0.0057 | 0.0098 | 0.9951 | 3 |
| rate | 2 | cv | 0.5028 | 0.0814 | 0.0444 | 1.9899 | 36 |
| rate | 2 | final | 0.5384 | 0.0038 | 0.0444 | 1.9902 | 3 |
| rate | 3 | cv | 0.4946 | 0.0859 | 0.0826 | 2.9849 | 36 |
| rate | 3 | final | 0.5200 | 0.0004 | 0.0826 | 2.9852 | 3 |
| rate | 5 | cv | 0.4528 | 0.0654 | 0.1638 | 4.9748 | 36 |
| rate | 5 | final | 0.5309 | 0.0089 | 0.1639 | 4.9754 | 3 |

## Liquid ablations, RR timing, baselines (morph + robust front end)
| variant | enc | part | macro_f1 | macro_f1_sd | recall_SVEB | f1_SVEB | recall_VEB | accuracy | firing | n |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN tuned + RR + mean-RR ratio | - | final | 0.6941 | 0.0500 | 0.4409 | 0.3937 | 0.9171 | 0.9083 |  | 3 |
| liquid published, RR in liquid | multi | final | 0.6911 | 0.0187 | 0.1744 | 0.1945 | 0.9254 | 0.9396 | 0.7496 | 3 |
| liquid fixed, RR in liquid | multi | final | 0.6791 | 0.0059 | 0.1040 | 0.1381 | 0.9335 | 0.9447 | 0.0191 | 3 |
| liquid fixed, RR in liquid (+ mean-RR ratio) | multi | final | 0.6762 | 0.0077 | 0.1092 | 0.1441 | 0.9233 | 0.9442 | 0.0179 | 3 |
| liquid published, RR in liquid | rate | final | 0.6759 | 0.0016 | 0.1210 | 0.1579 | 0.9132 | 0.9413 | 0.1517 | 3 |
| liquid published, RR in liquid (+ mean-RR ratio) | multi | final | 0.6741 | 0.0123 | 0.1428 | 0.1628 | 0.9203 | 0.9375 | 0.7886 | 3 |
| GRU tuned + RR | - | final | 0.6722 | 0.0064 | 0.4070 | 0.3522 | 0.9150 | 0.9038 |  | 3 |
| liquid published, RR in liquid (+ mean-RR ratio) | rate | final | 0.6649 | 0.0037 | 0.1109 | 0.1439 | 0.9102 | 0.9400 | 0.1573 | 3 |
| liquid fixed, RR in liquid (+ mean-RR ratio) | rate | final | 0.6638 | 0.0006 | 0.1065 | 0.1414 | 0.9021 | 0.9397 | 0.0096 | 3 |
| liquid fixed, RR in liquid | latency | final | 0.6636 | 0.0035 | 0.1464 | 0.1599 | 0.9094 | 0.9299 | 0.0185 | 3 |
| liquid fixed, RR in liquid | rate | final | 0.6625 | 0.0068 | 0.1105 | 0.1463 | 0.8927 | 0.9377 | 0.0097 | 3 |
| random features (ELM) + RR channels | multi | final | 0.6590 | 0.0046 | 0.0934 | 0.1142 | 0.9316 | 0.9376 |  | 3 |
| CNN + RR | - | final | 0.6588 | 0.0063 | 0.2642 | 0.2670 | 0.8330 | 0.9179 |  | 3 |
| liquid published, RR in readout | multi | final | 0.6537 | 0.0036 | 0.0668 | 0.0882 | 0.9332 | 0.9378 | 0.4381 | 3 |
| liquid published | multi | final | 0.6525 | 0.0051 | 0.0680 | 0.0888 | 0.9332 | 0.9367 | 0.4381 | 3 |
| liquid fix_gamma | multi | final | 0.6522 | 0.0012 | 0.0764 | 0.0988 | 0.9399 | 0.9352 | 0.3165 | 3 |
| GRU tuned + RR + mean-RR ratio | - | final | 0.6507 | 0.0272 | 0.4183 | 0.3784 | 0.9057 | 0.8869 |  | 3 |
| liquid published, recurrence off | multi | final | 0.6489 | 0.0024 | 0.0751 | 0.0972 | 0.9394 | 0.9340 | 0.3162 | 3 |
| random features (ELM) + RR channels | rate | final | 0.6482 | 0.0009 | 0.1238 | 0.1533 | 0.9127 | 0.9290 |  | 3 |
| liquid published, RR in liquid | latency | final | 0.6477 | 0.0074 | 0.1316 | 0.1403 | 0.9354 | 0.9244 | 0.3216 | 3 |
| liquid fix_ain | multi | final | 0.6453 | 0.0049 | 0.0798 | 0.1006 | 0.9450 | 0.9304 | 0.1420 | 3 |
| RR + beat shape + mean-RR ratio, linear | - | final | 0.6408 |  | 0.1040 | 0.1328 | 0.7645 | 0.9413 |  | 1 |
| CNN tuned + RR | - | final | 0.6388 | 0.0357 | 0.3488 | 0.3014 | 0.9293 | 0.8852 |  | 3 |
| RR + beat shape, linear | - | final | 0.6376 |  | 0.1034 | 0.1316 | 0.7552 | 0.9409 |  | 1 |
| random features (ELM) + RR channels | latency | final | 0.6328 | 0.0016 | 0.1190 | 0.1375 | 0.9185 | 0.9237 |  | 3 |
| liquid fixed, lambda 2.0, g_rec 30 | multi | final | 0.6294 | 0.0012 | 0.0838 | 0.0989 | 0.9340 | 0.9211 | 0.0331 | 3 |
| liquid published, RR in liquid (+ mean-RR ratio) | latency | final | 0.6275 | 0.0135 | 0.1504 | 0.1595 | 0.9074 | 0.9198 | 0.3573 | 3 |
| liquid fixed, RR in liquid (+ mean-RR ratio) | latency | final | 0.6254 | 0.0078 | 0.1363 | 0.1495 | 0.9099 | 0.9246 | 0.0175 | 3 |
| liquid fixed, RR in readout | multi | final | 0.6216 | 0.0025 | 0.0818 | 0.0953 | 0.9334 | 0.9169 | 0.0276 | 3 |
| liquid fixed | multi | final | 0.6199 | 0.0028 | 0.0806 | 0.0939 | 0.9323 | 0.9163 | 0.0276 | 3 |
| liquid fixed, recurrence off | multi | final | 0.6188 | 0.0005 | 0.0813 | 0.0935 | 0.9332 | 0.9152 | 0.0276 | 3 |
| liquid fixed, recurrence off, lambda 2.0, g_rec 30 | multi | final | 0.6188 | 0.0005 | 0.0813 | 0.0935 | 0.9332 | 0.9152 | 0.0276 | 3 |
| liquid fixed, lambda 1.5 | multi | final | 0.6188 | 0.0028 | 0.0820 | 0.0958 | 0.9342 | 0.9156 | 0.0277 | 3 |
| liquid fixed, lambda 2.5 | multi | final | 0.6187 | 0.0018 | 0.0818 | 0.0956 | 0.9329 | 0.9157 | 0.0277 | 3 |
| no liquid (encoder counts) | multi | final | 0.6092 |  | 0.0849 | 0.1075 | 0.9124 | 0.9144 |  | 1 |
| liquid published, RR in readout | rate | final | 0.6035 | 0.0119 | 0.0501 | 0.0821 | 0.8706 | 0.9256 | 0.1255 | 3 |
| liquid fixed, RR in readout | latency | final | 0.6027 | 0.0053 | 0.0348 | 0.0534 | 0.8769 | 0.9257 | 0.0268 | 3 |
| random features (ELM) | multi | final | 0.5950 | 0.0022 | 0.0760 | 0.0885 | 0.9209 | 0.9043 |  | 3 |
| GRU tuned | - | final | 0.5940 | 0.0408 | 0.2892 | 0.2375 | 0.9126 | 0.8490 |  | 3 |
| liquid fixed, recurrence off | latency | final | 0.5879 | 0.0023 | 0.0272 | 0.0394 | 0.8598 | 0.9180 | 0.0267 | 3 |
| liquid fixed | latency | final | 0.5874 | 0.0055 | 0.0274 | 0.0395 | 0.8668 | 0.9173 | 0.0268 | 3 |
| liquid fixed, RR in readout | rate | final | 0.5861 | 0.0050 | 0.0405 | 0.0687 | 0.8580 | 0.9198 | 0.0098 | 3 |
| liquid fixed, lambda 2.5 | latency | final | 0.5856 | 0.0068 | 0.0259 | 0.0380 | 0.8723 | 0.9170 | 0.0271 | 3 |
| liquid fix_ain | rate | final | 0.5794 | 0.0087 | 0.0287 | 0.0463 | 0.8538 | 0.9172 | 0.1192 | 3 |
| CNN tuned | - | final | 0.5758 | 0.0287 | 0.1392 | 0.1589 | 0.9226 | 0.8777 |  | 3 |
| liquid fixed, lambda 1.5 | latency | final | 0.5758 | 0.0075 | 0.0274 | 0.0407 | 0.8750 | 0.9119 | 0.0271 | 3 |
| liquid fix_ain | latency | final | 0.5752 | 0.0025 | 0.0406 | 0.0603 | 0.8715 | 0.9088 | 0.1604 | 3 |
| liquid published | rate | final | 0.5705 | 0.0088 | 0.0256 | 0.0407 | 0.8590 | 0.9123 | 0.1255 | 3 |
| RR only, linear | - | final | 0.5626 |  | 0.0348 | 0.0501 | 0.6968 | 0.9275 |  | 1 |
| liquid published, RR in readout | latency | final | 0.5601 | 0.0033 | 0.0419 | 0.0638 | 0.8741 | 0.9000 | 0.1543 | 3 |
| liquid fix_gamma | rate | final | 0.5591 | 0.0054 | 0.0212 | 0.0352 | 0.8270 | 0.9097 | 0.0108 | 3 |
| liquid fixed, lambda 1.5 | rate | final | 0.5574 | 0.0021 | 0.0201 | 0.0337 | 0.8351 | 0.9088 | 0.0099 | 3 |
| RR only + mean-RR ratio, linear | - | final | 0.5558 |  | 0.0316 | 0.0463 | 0.7167 | 0.9284 |  | 1 |
| liquid fixed, lambda 2.5 | rate | final | 0.5555 | 0.0108 | 0.0216 | 0.0355 | 0.8310 | 0.9069 | 0.0099 | 3 |
| liquid fixed, recurrence off | rate | final | 0.5544 | 0.0038 | 0.0203 | 0.0337 | 0.8271 | 0.9072 | 0.0098 | 3 |
| liquid published | latency | final | 0.5529 | 0.0038 | 0.0363 | 0.0546 | 0.8657 | 0.8968 | 0.1543 | 3 |
| liquid published, recurrence off | rate | final | 0.5523 | 0.0046 | 0.0210 | 0.0350 | 0.8302 | 0.9055 | 0.0108 | 3 |
| liquid fixed | rate | final | 0.5522 | 0.0057 | 0.0220 | 0.0366 | 0.8348 | 0.9048 | 0.0098 | 3 |
| RR + beat shape, LDA | - | final | 0.5512 |  | 0.0338 | 0.0428 | 0.6033 | 0.9195 |  | 1 |
| liquid fix_gamma | latency | final | 0.5505 | 0.0085 | 0.0254 | 0.0384 | 0.8598 | 0.8985 | 0.0241 | 3 |

## Corrected resampled t-tests on DS1 CV (Nadeau & Bengio; seeds averaged per split)
| comparison | n_splits | mean_diff | t | p |
|---|---|---|---|---|
| published vs fixed (config 11) | 12 | 0.0045 | 0.2186 | 0.8310 |
| fixed: multi vs rate | 12 | 0.1119 | 2.4600 | 0.0317 |
| fixed: multi vs latency | 12 | 0.1135 | 3.0185 | 0.0117 |
| fixed liquid vs no liquid | 12 | 0.0522 | 2.4413 | 0.0327 |
| fixed liquid vs random features | 12 | -0.0089 | -0.4529 | 0.6594 |
| fixed liquid vs recurrence off | 12 | 0.0009 | 0.2915 | 0.7761 |
| fixed + RR readout vs fixed | 12 | 0.0001 | 0.0110 | 0.9914 |
| fixed + RR in liquid vs fixed | 12 | 0.0569 | 1.0122 | 0.3332 |
| CNN + RR vs fixed liquid | 12 | -0.0043 | -0.1058 | 0.9177 |
| tuned CNN vs fixed liquid | 12 | -0.0337 | -1.0112 | 0.3336 |
| tuned GRU vs fixed liquid | 12 | -0.0556 | -1.2855 | 0.2250 |
| tuned CNN + RR vs fixed liquid + RR readout | 12 | 0.0114 | 0.2195 | 0.8303 |
| tuned GRU + RR vs fixed liquid + RR readout | 12 | 0.0035 | 0.0445 | 0.9653 |
| tuned GRU + RR vs published liquid + RR readout | 12 | -0.0039 | -0.0562 | 0.9562 |

## Re-tuning the fixed liquid (stage 1, DS1 CV rep 1)
| gain | lambda_val | num_levels | cv_macro_f1 |
|---|---|---|---|
| 1 | 2.5000 | 400 | 0.6596 |
| 1 | 2.5000 | 200 | 0.6471 |
| 1 | 0.8000 | 200 | 0.6422 |
| 2 | 2.5000 | 400 | 0.6410 |
| 2 | 1.5000 | 400 | 0.6393 |
| 1 | 1.5000 | 100 | 0.6288 |
| 3 | 2.5000 | 400 | 0.6267 |
| 3 | 0.8000 | 400 | 0.6266 |
| 5 | 2.5000 | 400 | 0.6256 |
| 5 | 1.5000 | 200 | 0.6255 |

## Tuned CNN / GRU baselines (stage 1: DS1 CV rep 1 only; DS2 untouched)
| model | best_trial | epochs | n_params | cv_macro_f1 | untuned_arch_best_cv_macro_f1 |
|---|---|---|---|---|---|
| CNN | 21 | 20 | 7883 | 0.6447 | 0.4580 |
| CNN + RR | 24 | 10 | 4563 | 0.7064 | 0.6968 |
| GRU | 22 | 20 | 25299 | 0.6375 | 0.4659 |
| GRU + RR | 16 | 15 | 18147 | 0.6915 | 0.6908 |

Chosen architectures:

- CNN: `{"dropout": 0.1, "head": 32, "kernels": [5, 5], "layers": 2, "lr": 0.001, "pool": "flat8", "wd": 0.0, "width": 12}`
- CNN + RR: `{"dropout": 0.0, "head": 32, "kernels": [11, 11], "layers": 2, "lr": 0.001, "pool": "avg", "wd": 0.001, "width": 12}`
- GRU: `{"bidir": false, "dropout": 0.0, "ds": 8, "head": 64, "hidden": 80, "layers": 1, "lr": 0.003, "readout": "mean", "wd": 0.0}`
- GRU + RR: `{"bidir": true, "dropout": 0.1, "ds": 4, "head": 32, "hidden": 48, "layers": 1, "lr": 0.001, "readout": "last", "wd": 0.0}`

Note: stage-2 CV numbers include the rep-1 folds used for the choice (slightly optimistic, as for the liquid re-tuning); the DS2 numbers are clean.

## Which A_in did the authors' Rate/Latency runs use? (CV mean Macro-F1, published code, seed 2026)
| config_id | encoder | authors_cv_macro_f1 | ours_A_in 0.05892 | ours_paper A_in |
|---|---|---|---|---|
| 1 | rate | 0.4847 | 0.4731 | 0.4979 |
| 2 | rate | 0.4853 | 0.4710 | 0.4990 |
| 3 | latency | 0.4830 | 0.4934 | 0.5046 |
| 4 | latency | 0.4962 | 0.4942 | 0.5041 |
| 7 | rate | 0.5062 | 0.4923 | 0.5629 |
| 8 | rate | 0.5074 | 0.4918 | 0.5122 |
| 9 | latency | 0.5260 | 0.5279 | 0.5884 |
| 10 | latency | 0.5367 | 0.5273 | 0.5325 |
| 13 | rate | 0.4941 | 0.4889 | 0.4942 |
| 14 | rate | 0.4898 | 0.4911 | 0.4916 |
| 15 | latency | 0.4904 | 0.5005 | 0.5081 |
| 16 | latency | 0.4961 | 0.5008 | 0.5070 |
| 19 | rate | 0.5191 | 0.5078 | 0.5217 |
| 20 | rate | 0.5262 | 0.5083 | 0.5210 |
| 21 | latency | 0.5322 | 0.5320 | 0.5241 |
| 22 | latency | 0.5338 | 0.5301 | 0.5235 |

## Working liquid: perturbation diagnostic (chosen g_rec = 30)
| label | firing_per_step | last_window_firing_per_step | recurrent_over_input_current | perturbation_early | perturbation_late | damage_ratio | n_recurrent_synapses | chosen |
|---|---|---|---|---|---|---|---|---|
| lambda 2, g_rec 1 | 0.0289 | 0.0000 | 0.0124 | 0.4243 | 0.0906 | 0.2135 | 2666 | False |
| lambda 2, g_rec 3 | 0.0292 | 0.0000 | 0.0376 | 0.4507 | 0.0983 | 0.2182 | 2666 | False |
| lambda 2, g_rec 10 | 0.0303 | 0.0000 | 0.1299 | 0.5627 | 0.1456 | 0.2587 | 2666 | False |
| lambda 2, g_rec 30 | 0.0345 | 0.0000 | 0.4498 | 0.9780 | 0.5248 | 0.5366 | 2666 | True |
| lambda 2, g_rec 100 | 0.1170 | 0.0111 | 5.9407 | 18.6780 | 35.7062 | 1.9117 | 2666 | False |
| lambda 2, g_rec 300 | 0.4818 | 0.4810 | 58.3022 | 5.5867 | 11.5983 | 2.0761 | 2666 | False |
| lambda 2, g_rec 1000 | 0.5434 | 0.5490 | 219.8698 | 0.4527 | 0.3112 | 0.6874 | 2666 | False |
| lambda 2, recurrence off | 0.0288 | 0.0000 | 0.0000 | 0.4133 | 0.0835 | 0.2020 | 0 | False |
| paper as described (lambda 0.8, g_rec 1) | 0.0289 | 0.0000 | 0.0058 | 0.4177 | 0.0850 | 0.2035 | 163 | False |
| paper as published (both bugs) | 0.4765 | 0.1987 | 9.1759 | 13.4353 | 1.2751 | 0.0949 | 163 | False |

## Extensions: Huang baseline, hybrid SVEB rule, mean-RR feature, ELM + RR (descriptive; verdicts come later)
| variant | encoder | cv_macro_f1 | cv_splits | cv_reps23 | ds2_macro_f1 | ds2_sd | ds2_runs | ds2_recall_SVEB | ds2_precision_SVEB | ds2_recall_VEB | ds2_precision_VEB | rep1_vote_ave_N_VEB | huang_primary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid (replace): LSM fixed, Multi + RR in liquid | - | 0.7046 | 12 | 0.6860 | 0.8265 | 0.0050 | 3 | 0.9027 | 0.4274 | 0.9335 | 0.9234 |  | False |
| hybrid (replace): LSM published, config 11 | - | 0.6935 | 12 | 0.6798 | 0.8205 | 0.0023 | 3 | 0.9291 | 0.4314 | 0.9332 | 0.8746 |  | False |
| hybrid (add): LSM fixed, Multi + RR in liquid | - | 0.7095 | 12 | 0.6923 | 0.8193 | 0.0044 | 3 | 0.9114 | 0.4054 | 0.9335 | 0.9234 |  | False |
| hybrid (replace): LSM published, Multi + RR in liquid | - | 0.6857 | 12 | 0.6717 | 0.8110 | 0.0072 | 3 | 0.8485 | 0.4102 | 0.9254 | 0.8939 |  | False |
| Huang 2014, gamma 0.296 | - | 0.6418 | 12 | 0.6226 | 0.8087 | 0.0025 | 3 | 0.9174 | 0.4031 | 0.8605 | 0.9328 | 0.8258 | False |
| hybrid (add): LSM published, config 11 | - | 0.6778 | 12 | 0.6646 | 0.8053 | 0.0078 | 3 | 0.9325 | 0.3889 | 0.9332 | 0.8746 |  | False |
| hybrid (add): LSM published, Multi + RR in liquid | - | 0.6913 | 12 | 0.6773 | 0.7996 | 0.0085 | 3 | 0.8621 | 0.3757 | 0.9254 | 0.8939 |  | False |
| Huang 2014, gamma 1.3 | - | 0.6435 | 12 | 0.6274 | 0.7914 | 0.0030 | 3 | 0.9156 | 0.4279 | 0.9345 | 0.7447 | 0.8377 | True |
| hybrid (replace): GRU tuned + RR | - | 0.6511 | 12 | 0.6311 | 0.7336 | 0.0079 | 3 | 0.7986 | 0.4016 | 0.9150 | 0.5874 |  | False |
| hybrid (add): GRU tuned + RR | - | 0.6547 | 12 | 0.6357 | 0.7275 | 0.0066 | 3 | 0.8193 | 0.3795 | 0.9150 | 0.5874 |  | False |
| hybrid (replace): CNN tuned + RR | - | 0.6623 | 12 | 0.6442 | 0.7252 | 0.0549 | 3 | 0.8138 | 0.4217 | 0.9293 | 0.5481 |  | False |
| hybrid (add): CNN tuned + RR | - | 0.6663 | 12 | 0.6486 | 0.7178 | 0.0618 | 3 | 0.8318 | 0.3961 | 0.9293 | 0.5481 |  | False |
| CNN tuned + RR + mean-RR ratio | - | 0.6234 | 12 | 0.6113 | 0.6941 | 0.0500 | 3 | 0.4409 | 0.3566 | 0.9171 | 0.6369 |  | False |
| liquid fixed, RR in liquid (+ mean-RR ratio) | multi | 0.6602 | 12 | 0.6525 | 0.6762 | 0.0077 | 3 | 0.1092 | 0.2129 | 0.9233 | 0.9031 |  | False |
| liquid published, RR in liquid (+ mean-RR ratio) | multi | 0.6598 | 12 | 0.6527 | 0.6741 | 0.0123 | 3 | 0.1428 | 0.1896 | 0.9203 | 0.8625 |  | False |
| liquid published, RR in liquid (+ mean-RR ratio) | rate | 0.6265 | 12 | 0.6117 | 0.6649 | 0.0037 | 3 | 0.1109 | 0.2049 | 0.9102 | 0.8542 |  | False |
| liquid fixed, RR in liquid (+ mean-RR ratio) | rate | 0.6266 | 12 | 0.6144 | 0.6638 | 0.0006 | 3 | 0.1065 | 0.2105 | 0.9021 | 0.8611 |  | False |
| random features (ELM) + RR channels | multi | 0.6877 | 12 | 0.6836 | 0.6590 | 0.0046 | 3 | 0.0934 | 0.1467 | 0.9316 | 0.8594 |  | False |
| GRU tuned + RR + mean-RR ratio | - | 0.6361 | 12 | 0.6214 | 0.6507 | 0.0272 | 3 | 0.4183 | 0.3470 | 0.9057 | 0.4959 |  | False |
| random features (ELM) + RR channels | rate | 0.6558 | 12 | 0.6469 | 0.6482 | 0.0009 | 3 | 0.1238 | 0.2015 | 0.9127 | 0.7580 |  | False |
| RR + beat shape + mean-RR ratio, linear | - | 0.5258 | 12 | 0.4985 | 0.6408 |  | 1 | 0.1040 | 0.1837 | 0.7645 | 0.8699 |  | False |
| random features (ELM) + RR channels | latency | 0.6523 | 12 | 0.6407 | 0.6328 | 0.0016 | 3 | 0.1190 | 0.1633 | 0.9185 | 0.7055 |  | False |
| liquid fixed, lambda 2.0, g_rec 30 | multi | 0.6131 | 12 | 0.6003 | 0.6294 | 0.0012 | 3 | 0.0838 | 0.1207 | 0.9340 | 0.7498 |  | False |
| liquid published, RR in liquid (+ mean-RR ratio) | latency | 0.6619 | 12 | 0.6373 | 0.6275 | 0.0135 | 3 | 0.1504 | 0.1704 | 0.9074 | 0.6526 |  | False |
| liquid fixed, RR in liquid (+ mean-RR ratio) | latency | 0.6360 | 12 | 0.6133 | 0.6254 | 0.0078 | 3 | 0.1363 | 0.1657 | 0.9099 | 0.6496 |  | False |
| liquid fixed, recurrence off, lambda 2.0, g_rec 30 | multi | 0.6156 | 12 | 0.6004 | 0.6188 | 0.0005 | 3 | 0.0813 | 0.1102 | 0.9332 | 0.7139 |  | False |
| RR only + mean-RR ratio, linear | - | 0.5393 | 12 | 0.5294 | 0.5558 |  | 1 | 0.0316 | 0.0866 | 0.7167 | 0.5866 |  | False |
| liquid fixed, lambda 2.0, g_rec 30, last-window readout | multi | 0.3170 | 12 | 0.3169 | 0.3153 | 0.0000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | False |
| liquid fixed, recurrence off, lambda 2.0, g_rec 30, last-window readout | multi | 0.3170 | 12 | 0.3169 | 0.3153 | 0.0000 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | False |
