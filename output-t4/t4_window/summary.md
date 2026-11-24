# Task 4: a longer window (the neighbouring beats) and the ladder

Config 11, the paper's liquid with the three fixes. Windows R +- 0.8, 1.6 s, cut from the same filtered records; the same 50510 DS1 and 49231 DS2 beats for every size (those whose longest window fits in the record), the authors' folds restricted to them. Seeds 2026, 69, 420; DS2 and the 12 CV splits.

* **no liquid**: the encoder's spikes counted in windows of 20 steps, straight to the readout.
* **random features**: those counts times the liquid's input weights, centred, ReLU.
* **liquid, no internal connections**: the spiking liquid with its recurrent connections switched off.
* **liquid**: the full liquid.

Check against the replication's stored runs: **all match**

## Means

| model | cv_runs | cv_macro_f1 | cv_recall_SVEB | cv_f1_SVEB | cv_recall_VEB | ds2_runs | ds2_macro_f1 | ds2_recall_SVEB | ds2_f1_SVEB | ds2_precision_SVEB | ds2_recall_VEB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no liquid, +-0.8 s | 12 | 0.5613 | 0.2460 | 0.1950 | 0.6398 | 1 | 0.6110 | 0.0833 | 0.1079 | 0.1533 | 0.9071 |
| random features, +-0.8 s | 36 | 0.6213 | 0.2870 | 0.2418 | 0.8238 | 3 | 0.5971 | 0.0826 | 0.0931 | 0.1069 | 0.9181 |
| liquid, no internal connections, +-0.8 s | 36 | 0.6201 | 0.2800 | 0.2325 | 0.8317 | 3 | 0.6211 | 0.0826 | 0.0947 | 0.1114 | 0.9321 |
| liquid, +-0.8 s | 36 | 0.6208 | 0.2798 | 0.2321 | 0.8317 | 3 | 0.6202 | 0.0813 | 0.0926 | 0.1078 | 0.9347 |
| no liquid, +-1.6 s | 12 | 0.5736 | 0.1447 | 0.1288 | 0.7640 | 1 | 0.5907 | 0.0425 | 0.0559 | 0.0817 | 0.8909 |
| random features, +-1.6 s | 36 | 0.5700 | 0.1387 | 0.1221 | 0.8364 | 3 | 0.5903 | 0.0572 | 0.0675 | 0.0829 | 0.9002 |
| liquid, no internal connections, +-1.6 s | 36 | 0.5852 | 0.1485 | 0.1449 | 0.8230 | 3 | 0.6206 | 0.0517 | 0.0604 | 0.0726 | 0.9189 |
| liquid, +-1.6 s | 36 | 0.5811 | 0.1417 | 0.1335 | 0.8190 | 3 | 0.6199 | 0.0532 | 0.0623 | 0.0754 | 0.9176 |

## Paired comparisons (Macro-F1)

Nadeau-Bengio corrected t-test over the CV splits, Holm over the family; DS2 difference averaged over seeds. The last rows compare how much each layer adds over no liquid at the long window with how much it adds at the short one.

| comparison | n_splits | cv_diff | ci_low | ci_high | p_holm | ds2_diff | ds2_diff_recall_SVEB | verdict |
|---|---|---|---|---|---|---|---|---|
| no liquid: +-1.6 s - +-0.8 s (does the longer window help?) | 12 | 0.0123 | -0.0818 | 0.1063 | 1.0000 | -0.0202 | -0.0408 | inconclusive |
| random features: +-1.6 s - +-0.8 s (does the longer window help?) | 12 | -0.0512 | -0.1280 | 0.0255 | 1.0000 | -0.0068 | -0.0254 | inconclusive |
| liquid, no internal connections: +-1.6 s - +-0.8 s (does the longer window help?) | 12 | -0.0348 | -0.0986 | 0.0290 | 1.0000 | -0.0004 | -0.0308 | inconclusive |
| liquid: +-1.6 s - +-0.8 s (does the longer window help?) | 12 | -0.0397 | -0.1031 | 0.0237 | 1.0000 | -0.0003 | -0.0281 | inconclusive |
| +-0.8 s: liquid - random features | 12 | -0.0005 | -0.0367 | 0.0356 | 1.0000 | 0.0231 | -0.0013 | inconclusive |
| +-0.8 s: liquid - no internal connections | 12 | 0.0007 | -0.0049 | 0.0063 | 1.0000 | -0.0009 | -0.0013 | comparable |
| +-0.8 s: random features - no liquid | 12 | 0.0600 | -0.0073 | 0.1273 | 0.9069 | -0.0139 | -0.0007 | inconclusive |
| +-1.6 s: liquid - random features | 12 | 0.0110 | -0.0373 | 0.0594 | 1.0000 | 0.0296 | -0.0040 | inconclusive |
| +-1.6 s: liquid - no internal connections | 12 | -0.0042 | -0.0111 | 0.0028 | 1.0000 | -0.0008 | 0.0015 | comparable |
| +-1.6 s: random features - no liquid | 12 | -0.0035 | -0.0940 | 0.0869 | 1.0000 | -0.0004 | 0.0147 | inconclusive |
| (liquid - no liquid) at +-1.6 s - the same at +-0.8 s (does the layer use the extra timing?) | 12 | -0.0520 | -0.1578 | 0.0539 | 1.0000 | 0.0199 | 0.0127 | inconclusive |
| (random features - no liquid) at +-1.6 s - the same at +-0.8 s (does the layer use the extra timing?) | 12 | -0.0635 | -0.1769 | 0.0498 | 1.0000 | 0.0135 | 0.0154 | inconclusive |
