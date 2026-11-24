# Task 1: is the RR gain the liquid's own work?

Config 11, the paper's liquid with the three fixes. Seeds 2026, 69, 420; DS2 (train on DS1) and the authors' 12 CV splits. RR = 3 RR ratios x 10 thresholds = 30 extra input channels (an active channel spikes every 5th step).

* **liquid + RR**: the RR channels go into the liquid with the encoder's spikes.
* **random features + RR**: the encoder's spikes and the RR channels counted in the 32 windows, times the same liquid's input weights, centred, ReLU. No time dynamics, no recurrence.
* **no liquid + RR**: the same counts straight to the readout.

Check against the replication's stored runs: **all match**

## Means

| model | cv_runs | cv_macro_f1 | cv_recall_SVEB | cv_f1_SVEB | cv_recall_VEB | ds2_runs | ds2_macro_f1 | ds2_recall_SVEB | ds2_f1_SVEB | ds2_precision_SVEB | ds2_recall_VEB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| liquid | 36 | 0.6192 | 0.2780 | 0.2274 | 0.8304 | 3 | 0.6170 | 0.0826 | 0.0957 | 0.1139 | 0.9322 |
| liquid + RR | 36 | 0.6777 | 0.2974 | 0.3116 | 0.7987 | 3 | 0.6776 | 0.1034 | 0.1379 | 0.2075 | 0.9297 |
| random features | 36 | 0.6254 | 0.3058 | 0.2500 | 0.8246 | 3 | 0.5950 | 0.0760 | 0.0885 | 0.1061 | 0.9209 |
| random features + RR | 36 | 0.6877 | 0.3377 | 0.3438 | 0.8201 | 3 | 0.6590 | 0.0934 | 0.1142 | 0.1467 | 0.9316 |
| no liquid | 12 | 0.5643 | 0.2365 | 0.1843 | 0.6733 | 1 | 0.6092 | 0.0849 | 0.1075 | 0.1463 | 0.9124 |
| no liquid + RR | 12 | 0.6172 | 0.3108 | 0.2656 | 0.6899 | 1 | 0.6631 | 0.1219 | 0.1420 | 0.1698 | 0.9338 |

## Paired comparisons (Macro-F1)

Nadeau-Bengio corrected t-test over the CV splits (seeds averaged per split), Holm over the family; DS2 difference averaged over the seeds; verdicts meaningful / comparable / inconclusive.

| comparison | n_splits | cv_diff | ci_low | ci_high | p_holm | ds2_diff | ds2_diff_recall_SVEB | verdict |
|---|---|---|---|---|---|---|---|---|
| liquid + RR - random features + RR (THE QUESTION: does the liquid do more with RR than a random layer?) | 12 | -0.0100 | -0.0593 | 0.0393 | 1.0000 | 0.0186 | 0.0100 | inconclusive |
| liquid - random features (the same question without RR) | 12 | -0.0062 | -0.0513 | 0.0390 | 1.0000 | 0.0220 | 0.0065 | inconclusive |
| liquid + RR - liquid (RR into the liquid) | 12 | 0.0585 | -0.0645 | 0.1815 | 1.0000 | 0.0606 | 0.0209 | inconclusive |
| random features + RR - random features (RR into a random layer) | 12 | 0.0623 | -0.0418 | 0.1665 | 1.0000 | 0.0639 | 0.0174 | inconclusive |
| no liquid + RR - no liquid (RR channels, no layer) | 12 | 0.0529 | -0.0482 | 0.1540 | 1.0000 | 0.0539 | 0.0370 | inconclusive |
| random features + RR - no liquid + RR (a random layer, given RR) | 12 | 0.0705 | -0.0079 | 0.1488 | 0.5128 | -0.0041 | -0.0285 | inconclusive |
| liquid + RR - no liquid + RR (the liquid, given RR) | 12 | 0.0605 | -0.0172 | 0.1382 | 0.6866 | 0.0145 | -0.0185 | inconclusive |
