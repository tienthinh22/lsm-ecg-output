# Benchmark

Device: Tesla P40. 300 s of every record (BENCHMARK). Benchmark time 108.8 min.

## Projected FULL (3 seeds): what FULL still has to run

| block | GPU time |
|---|---|
| P1-P5 final | 21.2 min |
| P3f final | 2.6 min |
| P1-P5 CV (12 splits) | 180.3 min |
| P3f CV (12 splits) | 30.9 min |
| M1, M3 final | 87.1 min |
| M3f final | 41.9 min |
| M0 per beat | 65.1 min |
| machinery check at FULL's start (one B1 seed, DS1 + DS2) | 0.6 min |
| **total** | **7.16 h ($2.29 at $0.32/h)** |

Readout fits left for FULL on the CPU: 280, about 0.8 h with 8 threads (they overlap the GPU work).

Already done here, not repeated by FULL: Maass calibration (saved, reused by FULL): 21.2 min; B1 and B2 on DS2 and the CV splits (kept): 13.8 min.

## Switches chosen

* paper's liquid: {'stack': True, 'chunks': False, 'why': 'stacking test passed and faster'}
* Maass's liquid: {'stack': True, 'chunks': True, 'option': 'both', 'projected_seconds': {'plain': 21605.01132928286, 'stacked': 14457.312400306351, 'chunked': 7052.707123655038, 'both': 5226.393295910237}, 'allowed': {'plain': True, 'stacked': True, 'chunked': True, 'both': True}}

## Checks

| check | result | details |
|---|---|---|
| fast_path | pass | graphs True |
| beat_order_labels | pass | stored lsm__fixed_s2026_morph-robust-multi_svm__final__fee7afb5, beats 49272 |
| machinery | pass | same_predictions 1.0, largest_score_difference 0.0, macro_f1 0.6197253684933735, stored_macro_f1 0.6197253684933735, stored lsm__fixed_s2026_morph-robust-multi_svm__final__fee7afb5 |
| adaptation_one_neuron | pass | threshold_at_10Hz 1.5667, u_after_spike 2.0, second_spike_threshold 1.55887, u_after_second_spike 1.44113, want_after_second 1.44113, rise_tau_1s 0.05659, want_tau_1s 0.0567, rise_tau_6s 0.00944, want_tau_6s 0.00945 |
| encoding_alignment | pass | final/DS1 0 of 3000 beats differ, final/DS2 0 of 3000 beats differ, cv-r1f1/DS1 0 of 3000 beats differ |
| window_alignment | pass | beats 500, differ 0, agrees_with_existing_simulator one beat's 650-step input from rest |
| segments_vs_one_run | pass | beats 280, differ 0, segment_steps 998, adaptation_sum_rel_diff 1.0653833534700778e-15 |
| end_of_record | pass | steps_after_end 700 |
| B1_vs_t1 | not run | note task 1's rows.json not found at ~/lsm_ecg/output_t1_rrcheck/t1_rrcheck/rows.json |
| B2_vs_t1 | not run | note task 1's rows.json not found |
| paper_stack_structure | pass | liquids 15 |
| paper_stack_short_run | pass | steps_compared 2000, first_spike_difference_at_step None, spikes_matched 1061032, max_voltage_difference 0.0, tolerance 0.0001 |
| paper_stacked_vs_alone_spikes | reported | beats_differ 0, liquids 15, beats_compared 42945, differ_first_half 0, differ_second_half 0 |
| paper_chunked_vs_unbroken | FAIL | beats 2863, beats_differ 15, share_beats_differ 0.005239259517988124, counts_differ 19, differ_first_half 0, differ_second_half 15, note the paper's rows run unchunked anyway |
| thresholds_of_1_are_P1 | pass | beats_differ 0, beats_compared 2863 |
| adaptation_on_P3 | pass | mean_threshold_adaptive 1.3664577390416273, n_adaptive 206, n_adaptive_inhibitory 46 |
| frozen_P3f | FAIL | median_abs_rate_ratio 0.0502, p90_abs_rate_ratio 0.1002, qrs_rate_diff 0.0032, between_rate_diff 0.0114, adaptive_neurons 206, averaged_over DS1 records only |
| maass_stack_structure | pass | liquids 2 |
| maass_stack_short_run | pass | steps_compared 2000, first_spike_difference_at_step None, spikes_matched 12419, max_voltage_difference 0.0, tolerance 0.0001 |
| maass_stacked_vs_plain_spikes | reported | beats_differ 0, beats_compared 5726, differ_first_half 0, differ_second_half 0 |
| maass_chunked_vs_unbroken | pass | beats_differ 0, beats_compared 5726, note chunks are used only if identical |
| maass_stacked_chunked_vs_stacked | pass | beats_differ 0, beats_compared 5726 |
| maass_segment_task | pass | seed_2026 older segments on 0.876 vs off 0.488, seed_69 older segments on 0.794 vs off 0.624, seed_420 older segments on 0.787 vs off 0.551, note the ytest notebook's build check; reported, the run goes on |
| frozen_M3f | FAIL | note outside the bounds: M3f is reported as approximate (the plan), no tuning, median_abs_rate_ratio 0.0382, p90_abs_rate_ratio 0.6795, qrs_rate_diff 0.0009, between_rate_diff 0.0188, adaptive_neurons 54, averaged_over DS1 records only |
