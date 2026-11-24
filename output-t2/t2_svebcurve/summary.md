# Task 2: would more SVEB training data help?

Config 11, the paper's liquid with the three fixes, trained on DS1, tested on DS2. Every N and VEB beat is kept; the SVEB beats of only part of the 16 DS1 patients with SVEB are kept (5 random draws per fraction). The liquid is the same for every subset; seeds 2026, 69, 420.

Check against the replication's stored runs: **all match**

## The curve (means over draws and seeds; sd in the csv)

| model | fraction | runs | sveb_train_beats | macro_f1 | recall_SVEB | f1_SVEB | precision_SVEB | recall_VEB |
|---|---|---|---|---|---|---|---|---|
| liquid | 0.25 | 15 | 263 | 0.6109 | 0.0128 | 0.0225 | 0.1892 | 0.9579 |
| liquid | 0.50 | 15 | 437 | 0.6102 | 0.0292 | 0.0447 | 0.1000 | 0.9497 |
| liquid | 0.75 | 15 | 656 | 0.6174 | 0.0572 | 0.0717 | 0.0996 | 0.9388 |
| liquid | 1.00 | 3 | 943 | 0.6170 | 0.0826 | 0.0957 | 0.1139 | 0.9322 |
| no_liquid | 0.25 | 5 | 263 | 0.5886 | 0.0107 | 0.0188 | 0.1991 | 0.9447 |
| no_liquid | 0.50 | 5 | 437 | 0.5959 | 0.0277 | 0.0429 | 0.1188 | 0.9351 |
| no_liquid | 0.75 | 5 | 656 | 0.6047 | 0.0568 | 0.0780 | 0.1445 | 0.9264 |
| no_liquid | 1.00 | 1 | 943 | 0.6092 | 0.0849 | 0.1075 | 0.1463 | 0.9124 |

## Trend (Spearman correlation with the number of SVEB training beats, over all runs)

| model | metric | spearman_vs_sveb_beats | p | runs |
|---|---|---|---|---|
| liquid | f1_SVEB | 0.9006 | 0.0000 | 48 |
| liquid | recall_SVEB | 0.9195 | 0.0000 | 48 |
| liquid | macro_f1 | 0.6694 | 0.0000 | 48 |
| no_liquid | f1_SVEB | 0.8933 | 0.0000 | 16 |
| no_liquid | recall_SVEB | 0.9154 | 0.0000 | 16 |
| no_liquid | macro_f1 | 0.7697 | 0.0005 | 16 |

A curve still rising at fraction 1.0 says more SVEB data of this kind would help; a flat one says the limit is elsewhere.
