# Task 3: is class imbalance the bottleneck?

Config 11, the paper's liquid with the three fixes. The readout's class weights are (balanced weight) ** power; powers 0, 0.3, 0.6, 1 (the paper: 0.6). Seeds 2026, 69, 420; DS2 and the 12 CV splits. Each power uses the same states and the same scaler.

Check against the replication's stored runs: **all match**

## Means

| model | cv_runs | cv_macro_f1 | cv_recall_SVEB | cv_f1_SVEB | cv_recall_VEB | ds2_runs | ds2_macro_f1 | ds2_recall_SVEB | ds2_f1_SVEB | ds2_precision_SVEB | ds2_recall_VEB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| liquid, power 0 | 36 | 0.6017 | 0.1554 | 0.1586 | 0.7991 | 3 | 0.6251 | 0.0443 | 0.0636 | 0.1132 | 0.9296 |
| liquid, power 0.3 | 36 | 0.6051 | 0.1934 | 0.1777 | 0.8143 | 3 | 0.6200 | 0.0595 | 0.0779 | 0.1129 | 0.9296 |
| liquid, power 0.6 | 36 | 0.6192 | 0.2780 | 0.2274 | 0.8304 | 3 | 0.6170 | 0.0826 | 0.0957 | 0.1139 | 0.9322 |
| liquid, power 1 | 36 | 0.6166 | 0.5252 | 0.2651 | 0.8139 | 3 | 0.5878 | 0.1301 | 0.0820 | 0.0599 | 0.9104 |
| no_liquid, power 0 | 12 | 0.5349 | 0.1109 | 0.1168 | 0.5950 | 1 | 0.6125 | 0.0338 | 0.0560 | 0.1649 | 0.9118 |
| no_liquid, power 0.3 | 12 | 0.5469 | 0.1675 | 0.1504 | 0.6160 | 1 | 0.6102 | 0.0490 | 0.0722 | 0.1374 | 0.9118 |
| no_liquid, power 0.6 | 12 | 0.5643 | 0.2365 | 0.1843 | 0.6733 | 1 | 0.6092 | 0.0849 | 0.1075 | 0.1463 | 0.9124 |
| no_liquid, power 1 | 12 | 0.5655 | 0.4321 | 0.2311 | 0.6524 | 1 | 0.5758 | 0.1361 | 0.0876 | 0.0645 | 0.8993 |

## Each power against the paper's 0.6 (Macro-F1)

Nadeau-Bengio corrected t-test over the CV splits, Holm within each model; DS2 difference averaged over seeds. Look at SVEB recall and F1 in the means: more weight trades SVEB precision for recall; if SVEB F1 stays low across the powers, imbalance is not what limits SVEB.

| comparison | n_splits | cv_diff | ci_low | ci_high | p_holm | ds2_diff | ds2_diff_recall_SVEB | verdict |
|---|---|---|---|---|---|---|---|---|
| liquid: power 0 - power 0.6 | 12 | -0.0175 | -0.0681 | 0.0331 | 1.0000 | 0.0081 | -0.0383 | inconclusive |
| liquid: power 0.3 - power 0.6 | 12 | -0.0141 | -0.0529 | 0.0246 | 1.0000 | 0.0030 | -0.0230 | inconclusive |
| liquid: power 1 - power 0.6 | 12 | -0.0026 | -0.0656 | 0.0604 | 1.0000 | -0.0292 | 0.0475 | inconclusive |
| no_liquid: power 0 - power 0.6 | 12 | -0.0294 | -0.0874 | 0.0286 | 0.8645 | 0.0033 | -0.0512 | inconclusive |
| no_liquid: power 0.3 - power 0.6 | 12 | -0.0174 | -0.0555 | 0.0206 | 0.8645 | 0.0010 | -0.0359 | inconclusive |
| no_liquid: power 1 - power 0.6 | 12 | 0.0013 | -0.0739 | 0.0765 | 0.9712 | -0.0334 | 0.0512 | inconclusive |
