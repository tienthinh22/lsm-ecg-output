# Config 11 vs continuous liquids with memory

Mode BENCHMARK. Seeds 2026, 69, 420. DS2 = train on all DS1, test on DS2; exact CV = the authors' 12 DS1 splits (each with its own normalisation and A_in).

**Rule (pre-declared):** a paper's-liquid streaming row is better than B1 if its DS2 SVEB F1 beats B1's by at least 0.01 on all three seeds and the exact-CV test on SVEB F1 (Nadeau-Bengio, Holm over {P1, P3}) calls it meaningful. Maass rows are DS2 only and descriptive.

## Headline: SVEB F1 against B1

| label | ds2_f1_SVEB_mean | B1_f1_SVEB_mean | diff_by_seed | beats_B1_by_margin_all_seeds | n_splits | cv_diff | ci_low | ci_high | p | p_holm | cv_verdict | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 no liquid | 0.1075 | 0.0957 | 69: +0.0085, 420: +0.0091, 2026: +0.0179 | False | 12 | -0.0431 | -0.1816 | 0.0954 | 0.5076 |  | inconclusive | outside the family (reported) |

p: unadjusted; p_holm: Holm-adjusted within the family {P1, P3} only (empty for the other rows, whose verdict uses p). n_splits: CV splits in the test (12 when complete).

## The same test on macro F1 (as in t1-t4; reported)

| label | n_splits | cv_diff | ci_low | ci_high | p | p_holm | cv_verdict | ds2_diff_mean |
|---|---|---|---|---|---|---|---|---|
| B2 no liquid | 12 | -0.0549 | -0.1052 | -0.0046 | 0.0350 |  | meaningful | -0.0078 |

## DS2 means over seeds (and CV means)

| row | seeds | macro_f1 | f1_SVEB | recall_SVEB | precision_SVEB | auc_SVEB | f1_VEB | recall_VEB | precision_VEB | recall_N | precision_N | f1_SVEB_no232 | recall_SVEB_no232 | cv_f1_SVEB | cv_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 config 11 (per beat) | 3 | 0.6170 | 0.0957 | 0.0826 | 0.1139 | 0.6953 | 0.8018 | 0.9322 | 0.7034 | 0.9478 | 0.9596 | 0.1666 | 0.3267 | 0.2274 | 0.6192 |
| B2 no liquid | 1 | 0.6092 | 0.1075 | 0.0849 | 0.1463 | 0.6450 | 0.7662 | 0.9124 | 0.6604 | 0.9490 | 0.9589 | 0.2039 | 0.3407 | 0.1843 | 0.5643 |

Figure: figures/ds2_sveb_f1.png

## Keep vs drop: without each record's first 30 s (refit, DS2 only)

The streaming liquids start from rest at each record's start. Drop: every row refit on the DS1 beats and tested on the DS2 beats at least 30 s into their record; the DS2 part of the rule is applied against B1's drop fit (no CV on this set).

| row | keep_f1_SVEB | drop_f1_SVEB |
|---|---|---|
| B1 config 11 (per beat) | 0.0957 |  |
| B2 no liquid | 0.1075 |  |

## Where a difference comes from (SVEB F1)

|  |
|

## DS2 SVEB recall per record (mean over seeds)

| record | SVEB beats | B1 | B2 |
|---|---|---|---|
| 100 | 33 | 0.5152 | 0.4242 |
| 103 | 2 | 0.0000 | 0.0000 |
| 113 | 6 | 0.0000 | 0.0000 |
| 117 | 1 | 1.0000 | 1.0000 |
| 121 | 1 | 0.0000 | 0.0000 |
| 200 | 30 | 0.3000 | 0.3000 |
| 202 | 55 | 0.6545 | 0.6909 |
| 210 | 22 | 0.6212 | 0.7273 |
| 213 | 28 | 0.0476 | 0.0000 |
| 219 | 7 | 0.0000 | 0.1429 |
| 222 | 209 | 0.2935 | 0.3301 |
| 228 | 3 | 0.0000 | 0.0000 |
| 231 | 1 | 0.0000 | 0.0000 |
| 232 | 1382 | 0.0022 | 0.0007 |
| 233 | 7 | 0.2381 | 0.1429 |
| 234 | 50 | 0.1533 | 0.1200 |

## Checks

* fast_path: pass; graphs True
* beat_order_labels: pass; stored lsm__fixed_s2026_morph-robust-multi_svm__final__fee7afb5; beats 49272
* machinery: pass; same_predictions 1.0; largest_score_difference 0.0; macro_f1 0.6197253684933735; stored_macro_f1 0.6197253684933735; stored lsm__fixed_s2026_morph-robust-multi_svm__final__fee7afb5
* adaptation_one_neuron: pass; threshold_at_10Hz 1.5667; u_after_spike 2.0; second_spike_threshold 1.55887; u_after_second_spike 1.44113; want_after_second 1.44113; rise_tau_1s 0.05659; want_tau_1s 0.0567; rise_tau_6s 0.00944; want_tau_6s 0.00945
* encoding_alignment: pass; final/DS1 0 of 3000 beats differ; final/DS2 0 of 3000 beats differ; cv-r1f1/DS1 0 of 3000 beats differ
* window_alignment: pass; beats 500; differ 0; agrees_with_existing_simulator one beat's 650-step input from rest
* segments_vs_one_run: pass; beats 280; differ 0; segment_steps 998; adaptation_sum_rel_diff 1.0653833534700778e-15
* end_of_record: pass; steps_after_end 700
* B1_vs_t1: pass; compared 39; differ 0; missing 0
* B2_vs_t1: pass; compared 13; differ 0; missing 0
* paper_stack_structure: pass; liquids 15
* paper_stack_short_run: pass; steps_compared 2000; first_spike_difference_at_step None; spikes_matched 1061032; max_voltage_difference 0.0; tolerance 0.0001
* paper_stacked_vs_alone_spikes: reported; beats_differ 0; liquids 15; beats_compared 42945; differ_first_half 0; differ_second_half 0
* paper_chunked_vs_unbroken: FAIL; beats 2863; beats_differ 15; share_beats_differ 0.005239259517988124; counts_differ 19; differ_first_half 0; differ_second_half 15; note the paper's rows run unchunked anyway
* thresholds_of_1_are_P1: pass; beats_differ 0; beats_compared 2863
* adaptation_on_P3: pass; mean_threshold_adaptive 1.3664577390416273; n_adaptive 206; n_adaptive_inhibitory 46
* frozen_P3f: FAIL; median_abs_rate_ratio 0.0502; p90_abs_rate_ratio 0.1002; qrs_rate_diff 0.0032; between_rate_diff 0.0114; adaptive_neurons 206; averaged_over DS1 records only
* maass_stack_structure: pass; liquids 2
* maass_stack_short_run: pass; steps_compared 2000; first_spike_difference_at_step None; spikes_matched 12419; max_voltage_difference 0.0; tolerance 0.0001
* maass_stacked_vs_plain_spikes: reported; beats_differ 0; beats_compared 5726; differ_first_half 0; differ_second_half 0
* maass_chunked_vs_unbroken: pass; beats_differ 0; beats_compared 5726; note chunks are used only if identical
* maass_stacked_chunked_vs_stacked: pass; beats_differ 0; beats_compared 5726
* maass_segment_task: pass; seed_2026 older segments on 0.876 vs off 0.488; seed_69 older segments on 0.794 vs off 0.624; seed_420 older segments on 0.787 vs off 0.551; note the ytest notebook's build check; reported, the run goes on
* frozen_M3f: FAIL; note outside the bounds: M3f is reported as approximate (the plan), no tuning; median_abs_rate_ratio 0.0382; p90_abs_rate_ratio 0.6795; qrs_rate_diff 0.0009; between_rate_diff 0.0188; adaptive_neurons 54; averaged_over DS1 records only
