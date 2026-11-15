# Backend ablation - results (2026-11-15)

Generated: 2026-11-15T08:05:10+00:00

Offline summary only. Sensor-origin ATE is recomputed from `trajectory.csv` + truth CSV + `T_BS` read from each run's manifest via `benchmark_euroc.trajectory_metrics`. Common-timestamp ATE uses the intersection of output timestamps for each candidate/default pair. Groups are separated by platform, rate and GT extrinsic source; CPU is never compared across rates. ARM inputs are discovered dynamically from `metrics.json` + nearest `manifest.json`. `actual gap` is the maximum consecutive timestamp gap over all rows of the original `trajectory.csv`; `GT-matched gap` is the original benchmark `max_output_gap_s` after truth-valid filtering. `matched/output` is a matching count, not a dropped-output count (truth outages/window can exclude poses).

The 5% ATE line was **not pre-agreed** with the user. It is an engineering screening line for triage only, not user acceptance. `screening_regression` labels one run outside that line and does **not** assert a global algorithm failure.

## Run summary

| stage | variant | dataset | status | ATE m | ATE common m | CPU s | CPU % | lat p95 ms | RSS MiB | actual gap s | GT-matched gap s | matched/output | restarts req/reset | drops | usedobs | budget def/exp |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline2 | baseline | V1_02_medium | success | 0.15115 | - | 13.77 | 27.3 | 12.44 | 88.2 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| baseline2 | baseline | MH_05_difficult | success | 0.75244 | - | 17.16 | 28.0 | 14.16 | 90.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| candidate2 | candidate_control | V1_02_medium | success | 0.15115 | - | 13.60 | 27.0 | 12.72 | 88.9 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| candidate2 | candidate_control | MH_05_difficult | success | 0.75244 | - | 16.98 | 27.7 | 14.07 | 90.5 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| screen2 | qr | V1_02_medium | success | 0.15115 | 0.15115 | 13.57 | 26.9 | 12.74 | 88.8 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| screen2 | qr | MH_05_difficult | failed (not_algorithm_failure; partial excluded) | - | - | - | - | - | - | - | - | - | -/- | - | - | -/- |
| screen2 | qr | MH_05_difficult | success | 0.75244 | 0.75244 | 15.66 | 25.5 | 13.45 | 90.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| screen2 | batch | V1_02_medium | success | 0.15115 | 0.15115 | 12.80 | 25.4 | 11.98 | 89.1 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| screen2 | batch | MH_05_difficult | success | 0.75244 | 0.75244 | 15.46 | 25.2 | 13.12 | 90.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| screen2 | obs240 | V1_02_medium | success | 0.15707 | 0.15707 | 11.99 | 23.8 | 11.50 | 89.0 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | 1003/50 |
| screen2 | obs240 | MH_05_difficult | success | 0.76945 | 0.76945 | 15.08 | 24.6 | 12.83 | 90.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | 77/8 |
| screen2 | obs480 | V1_02_medium | success | 0.16742 | 0.16742 | 12.07 | 24.0 | 11.36 | 89.1 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | 6/0 |
| screen2 | obs480 | MH_05_difficult | success | 0.75104 | 0.75104 | 15.49 | 25.2 | 12.98 | 90.4 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | 0/15 |
| baseline8 | baseline | V1_01_easy | success | 0.12955 | - | 20.98 | 26.8 | 11.39 | 89.4 | 0.100 | 0.100 | 1437/1446 | 0/0 | 0 | - | -/- |
| baseline8 | baseline | V1_03_difficult | success | 0.24236 | - | 15.99 | 27.4 | 13.90 | 90.0 | 0.100 | 0.100 | 1056/1064 | 0/0 | 0 | - | -/- |
| baseline8 | baseline | MH_01_easy | success | 0.18070 | - | 26.75 | 27.6 | 12.71 | 99.6 | 0.100 | 0.700 | 1547/1831 | 0/0 | 0 | - | -/- |
| baseline8 | baseline | MH_02_easy | success | 6.77604 | - | 22.42 | 27.5 | 13.27 | 92.6 | 1.200 | 1.400 | 1219/1499 | 1/1 | 0 | - | -/- |
| baseline8 | baseline | MH_03_medium | success | 0.56238 | - | 19.22 | 26.7 | 13.20 | 91.7 | 0.100 | 3.200 | 897/1340 | 0/0 | 0 | - | -/- |
| baseline8 | baseline | MH_04_difficult | success | 0.58275 | - | 13.76 | 24.9 | 14.64 | 91.1 | 0.100 | 1.800 | 713/1006 | 0/0 | 0 | - | -/- |
| stable6 | candidate_control | V1_01_easy | success | 0.12955 | - | 19.93 | 25.5 | 10.70 | 89.2 | 0.100 | 0.100 | 1437/1446 | 0/0 | 0 | - | -/- |
| stable6 | candidate_control | V1_03_difficult | success | 0.24236 | - | 16.03 | 27.5 | 13.31 | 89.9 | 0.100 | 0.100 | 1056/1064 | 0/0 | 0 | - | -/- |
| stable6 | candidate_control | MH_01_easy | success | 0.18070 | - | 27.78 | 28.6 | 13.05 | 99.6 | 0.100 | 0.700 | 1547/1831 | 0/0 | 0 | - | -/- |
| stable6 | candidate_control | MH_02_easy | success | 6.77604 | - | 24.01 | 29.5 | 13.78 | 93.1 | 1.200 | 1.400 | 1219/1499 | 1/1 | 0 | - | -/- |
| stable6 | candidate_control | MH_03_medium | success | 0.56238 | - | 19.46 | 27.0 | 13.43 | 91.3 | 0.100 | 3.200 | 897/1340 | 0/0 | 0 | - | -/- |
| stable6 | candidate_control | MH_04_difficult | success | 0.58275 | - | 14.47 | 26.1 | 14.90 | 91.3 | 0.100 | 1.800 | 713/1006 | 0/0 | 0 | - | -/- |
| stable6 | qr | V1_01_easy | success | 0.12955 | 0.12955 | 20.19 | 25.8 | 10.99 | 89.2 | 0.100 | 0.100 | 1437/1446 | 0/0 | 0 | - | -/- |
| stable6 | qr | V1_03_difficult | success | 0.24236 | 0.24236 | 15.44 | 26.5 | 13.19 | 89.7 | 0.100 | 0.100 | 1056/1064 | 0/0 | 0 | - | -/- |
| stable6 | qr | MH_01_easy | success | 0.18070 | 0.18070 | 26.36 | 27.2 | 12.40 | 99.4 | 0.100 | 0.700 | 1547/1831 | 0/0 | 0 | - | -/- |
| stable6 | qr | MH_02_easy | success | 6.77604 | 6.77604 | 21.89 | 26.9 | 13.27 | 92.8 | 1.200 | 1.400 | 1219/1499 | 1/1 | 0 | - | -/- |
| stable6 | qr | MH_03_medium | success | 0.56238 | 0.56238 | 18.93 | 26.3 | 13.24 | 90.9 | 0.100 | 3.200 | 897/1340 | 0/0 | 0 | - | -/- |
| stable6 | qr | MH_04_difficult | success | 0.58275 | 0.58275 | 14.11 | 25.5 | 14.40 | 91.0 | 0.100 | 1.800 | 713/1006 | 0/0 | 0 | - | -/- |
| stable6 | batch | V1_01_easy | success | 0.12955 | 0.12955 | 20.73 | 26.5 | 10.73 | 89.3 | 0.100 | 0.100 | 1437/1446 | 0/0 | 0 | - | -/- |
| stable6 | batch | V1_03_difficult | success | 0.24236 | 0.24236 | 16.77 | 28.8 | 13.83 | 89.9 | 0.100 | 0.100 | 1056/1064 | 0/0 | 0 | - | -/- |
| stable6 | batch | MH_01_easy | success | 0.18070 | 0.18070 | 27.92 | 28.8 | 13.38 | 99.5 | 0.100 | 0.700 | 1547/1831 | 0/0 | 0 | - | -/- |
| stable6 | batch | MH_02_easy | success | 6.77604 | 6.77604 | 22.88 | 28.1 | 13.83 | 92.7 | 1.200 | 1.400 | 1219/1499 | 1/1 | 0 | - | -/- |
| stable6 | batch | MH_03_medium | success | 0.56238 | 0.56238 | 18.15 | 25.2 | 12.63 | 91.0 | 0.100 | 3.200 | 897/1340 | 0/0 | 0 | - | -/- |
| stable6 | batch | MH_04_difficult | success | 0.58275 | 0.58275 | 13.67 | 24.7 | 14.38 | 90.9 | 0.100 | 1.800 | 713/1006 | 0/0 | 0 | - | -/- |
| stable6 | obs240 | V1_01_easy | success | 0.16487 | 0.16487 | 19.71 | 25.2 | 10.74 | 89.3 | 0.100 | 0.100 | 1437/1446 | 0/0 | 0 | - | 158/26 |
| stable6 | obs240 | V1_03_difficult | success | 0.26708 | 0.26708 | 14.62 | 25.1 | 12.86 | 89.9 | 0.100 | 0.100 | 1056/1064 | 0/0 | 0 | - | 3385/513 |
| stable6 | obs240 | MH_01_easy | success | 0.17165 | 0.17165 | 25.14 | 25.9 | 12.15 | 99.5 | 0.100 | 0.700 | 1547/1831 | 0/0 | 0 | - | 4/19 |
| stable6 | obs240 | MH_02_easy | success | 3.50060 | 3.51256 | 21.36 | 26.2 | 13.15 | 94.4 | 1.200 | 1.400 | 1208/1488 | 2/2 | 0 | - | 267/212 |
| stable6 | obs240 | MH_03_medium | success | 0.55421 | 0.55421 | 18.99 | 26.3 | 13.07 | 91.0 | 0.100 | 3.200 | 897/1340 | 0/0 | 0 | - | 169/8 |
| stable6 | obs240 | MH_04_difficult | success | 0.60221 | 0.60221 | 14.64 | 26.4 | 15.75 | 90.8 | 0.100 | 1.800 | 713/1006 | 0/0 | 0 | - | 142/0 |
| combo2 | qr_batch | V1_02_medium | success | 0.15115 | 0.15115 | 12.72 | 25.3 | 12.01 | 89.0 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| combo2 | qr_batch | MH_05_difficult | success | 0.75244 | 0.75244 | 15.47 | 25.2 | 12.93 | 90.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | baseline | MH_05_difficult | success | 0.64241 | - | 82.59 | 69.4 | 49.75 | 99.1 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | baseline | V1_02_medium | success | 0.19419 | - | 63.73 | 66.0 | 46.14 | 96.9 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | batch | MH_05_difficult | success | 0.64241 | 0.64241 | 82.56 | 69.6 | 49.51 | 99.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | batch | V1_02_medium | success | 0.17239 | 0.17239 | 63.35 | 65.6 | 45.97 | 98.6 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | candidate_control | MH_05_difficult | success | 0.64241 | - | 82.86 | 69.3 | 49.45 | 100.0 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | candidate_control | V1_02_medium | success | 0.19419 | - | 63.52 | 65.0 | 45.89 | 98.5 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | obs240 | MH_05_difficult | success | 0.62808 | 0.62808 | 82.40 | 69.4 | 49.51 | 99.6 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | 15/15 |
| arm | obs240 | V1_02_medium | success | 0.17681 | 0.17681 | 63.37 | 65.6 | 45.59 | 98.4 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | 1414/58 |
| arm | obs480 | MH_05_difficult | success | 0.64956 | 0.64956 | 82.84 | 69.8 | 49.64 | 100.2 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | obs480 | V1_02_medium | success | 0.18251 | 0.18251 | 63.51 | 65.7 | 45.75 | 98.5 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | qr | MH_05_difficult | success | 0.64241 | 0.64241 | 82.54 | 69.6 | 48.96 | 99.7 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | qr | V1_02_medium | success | 0.17239 | 0.17239 | 63.41 | 65.6 | 45.62 | 98.5 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | qr_batch | MH_05_difficult | success | 0.64241 | 0.64241 | 82.20 | 69.3 | 49.20 | 100.0 | 0.100 | 2.800 | 807/1126 | 0/0 | 0 | - | -/- |
| arm | qr_batch | V1_02_medium | success | 0.17239 | 0.17239 | 63.38 | 65.5 | 46.20 | 98.5 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | qr_batch | V1_02_medium | success | 0.17239 | 0.17239 | 63.45 | 65.7 | 46.17 | 98.6 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | candidate_control | V1_02_medium | success | 0.17239 | - | 63.64 | 65.9 | 45.19 | 98.5 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | candidate_control | V1_02_medium | success | 0.17239 | - | 63.47 | 65.7 | 45.98 | 98.6 | 0.100 | 0.100 | 845/845 | 0/0 | 0 | - | -/- |
| arm | obs240 | V1_02_medium | success | 0.15299 | - | 18.36 | 52.2 | 47.10 | 98.1 | 0.100 | 0.100 | 229/229 | 0/0 | 0 | 167.5 | 3212/1006 |

## Truth matching / coverage

`matched/output` is preserved in full. An unmatched pose is not evidence of missing output: it may be outside the truth-valid window or inside a truth outage.

| dataset | variant | run | matched/output | unmatched | matched fraction | truth rows | truth time range s | bag truth coverage |
|---|---|---|---:|---:|---:|---:|---|---:|
| V1_02_medium | baseline | 20261115T145328_baseline2_01_baseline_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | baseline | 20261115T145328_baseline2_02_baseline_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_02_medium | candidate_control | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | candidate_control | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_02_medium | qr | 20261115T150833_screen2_01_qr_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | qr | 20261115T151220_screen2_04_qr_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_02_medium | batch | 20261115T151220_screen2_05_batch_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | batch | 20261115T151220_screen2_06_batch_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_02_medium | obs240 | 20261115T151220_screen2_07_obs240_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | obs240 | 20261115T151220_screen2_08_obs240_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_02_medium | obs480 | 20261115T151220_screen2_09_obs480_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | obs480 | 20261115T151220_screen2_10_obs480_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| V1_01_easy | baseline | 20261115T151921_baseline8_01_baseline_V1_01_easy | 1437/1446 | 9 | 0.9938 | 14629 | [1403715271.7051797, 1403715417.9881175] | 0.9939 |
| V1_03_difficult | baseline | 20261115T151921_baseline8_03_baseline_V1_03_difficult | 1056/1064 | 8 | 0.9925 | 10807 | [1403715888.4164658, 1403715996.4784107] | 0.9833 |
| MH_01_easy | baseline | 20261115T151921_baseline8_04_baseline_MH_01_easy | 1547/1831 | 284 | 0.8449 | 3099 | [1403636578.9228814, 1403636766.6788812] | 0.9999 |
| MH_02_easy | baseline | 20261115T151921_baseline8_05_baseline_MH_02_easy | 1219/1499 | 280 | 0.8132 | 2487 | [1403636856.8298812, 1403637010.0888813] | 0.9964 |
| MH_03_medium | baseline | 20261115T151921_baseline8_06_baseline_MH_03_medium | 897/1340 | 443 | 0.6694 | 1876 | [1403637133.0408814, 1403637266.4448812] | 0.9815 |
| MH_04_difficult | baseline | 20261115T151921_baseline8_07_baseline_MH_04_difficult | 713/1006 | 293 | 0.7087 | 1470 | [1403638129.1412687, 1403638229.9002688] | 0.9822 |
| V1_01_easy | candidate_control | 20261115T152712_stable6_01_candidate_control_V1_01_easy | 1437/1446 | 9 | 0.9938 | 14629 | [1403715271.7051797, 1403715417.9881175] | 0.9939 |
| V1_03_difficult | candidate_control | 20261115T152712_stable6_02_candidate_control_V1_03_difficult | 1056/1064 | 8 | 0.9925 | 10807 | [1403715888.4164658, 1403715996.4784107] | 0.9833 |
| MH_01_easy | candidate_control | 20261115T152712_stable6_03_candidate_control_MH_01_easy | 1547/1831 | 284 | 0.8449 | 3099 | [1403636578.9228814, 1403636766.6788812] | 0.9999 |
| MH_02_easy | candidate_control | 20261115T152712_stable6_04_candidate_control_MH_02_easy | 1219/1499 | 280 | 0.8132 | 2487 | [1403636856.8298812, 1403637010.0888813] | 0.9964 |
| MH_03_medium | candidate_control | 20261115T152712_stable6_05_candidate_control_MH_03_medium | 897/1340 | 443 | 0.6694 | 1876 | [1403637133.0408814, 1403637266.4448812] | 0.9815 |
| MH_04_difficult | candidate_control | 20261115T152712_stable6_06_candidate_control_MH_04_difficult | 713/1006 | 293 | 0.7087 | 1470 | [1403638129.1412687, 1403638229.9002688] | 0.9822 |
| V1_01_easy | qr | 20261115T153457_stable6_01_qr_V1_01_easy | 1437/1446 | 9 | 0.9938 | 14629 | [1403715271.7051797, 1403715417.9881175] | 0.9939 |
| V1_03_difficult | qr | 20261115T153457_stable6_02_qr_V1_03_difficult | 1056/1064 | 8 | 0.9925 | 10807 | [1403715888.4164658, 1403715996.4784107] | 0.9833 |
| MH_01_easy | qr | 20261115T153457_stable6_03_qr_MH_01_easy | 1547/1831 | 284 | 0.8449 | 3099 | [1403636578.9228814, 1403636766.6788812] | 0.9999 |
| MH_02_easy | qr | 20261115T153457_stable6_04_qr_MH_02_easy | 1219/1499 | 280 | 0.8132 | 2487 | [1403636856.8298812, 1403637010.0888813] | 0.9964 |
| MH_03_medium | qr | 20261115T153457_stable6_05_qr_MH_03_medium | 897/1340 | 443 | 0.6694 | 1876 | [1403637133.0408814, 1403637266.4448812] | 0.9815 |
| MH_04_difficult | qr | 20261115T153457_stable6_06_qr_MH_04_difficult | 713/1006 | 293 | 0.7087 | 1470 | [1403638129.1412687, 1403638229.9002688] | 0.9822 |
| V1_01_easy | batch | 20261115T154242_stable6_01_batch_V1_01_easy | 1437/1446 | 9 | 0.9938 | 14629 | [1403715271.7051797, 1403715417.9881175] | 0.9939 |
| V1_03_difficult | batch | 20261115T154242_stable6_02_batch_V1_03_difficult | 1056/1064 | 8 | 0.9925 | 10807 | [1403715888.4164658, 1403715996.4784107] | 0.9833 |
| MH_01_easy | batch | 20261115T154242_stable6_03_batch_MH_01_easy | 1547/1831 | 284 | 0.8449 | 3099 | [1403636578.9228814, 1403636766.6788812] | 0.9999 |
| MH_02_easy | batch | 20261115T154242_stable6_04_batch_MH_02_easy | 1219/1499 | 280 | 0.8132 | 2487 | [1403636856.8298812, 1403637010.0888813] | 0.9964 |
| MH_03_medium | batch | 20261115T154242_stable6_05_batch_MH_03_medium | 897/1340 | 443 | 0.6694 | 1876 | [1403637133.0408814, 1403637266.4448812] | 0.9815 |
| MH_04_difficult | batch | 20261115T154242_stable6_06_batch_MH_04_difficult | 713/1006 | 293 | 0.7087 | 1470 | [1403638129.1412687, 1403638229.9002688] | 0.9822 |
| V1_01_easy | obs240 | 20261115T155025_stable6_01_obs240_V1_01_easy | 1437/1446 | 9 | 0.9938 | 14629 | [1403715271.7051797, 1403715417.9881175] | 0.9939 |
| V1_03_difficult | obs240 | 20261115T155025_stable6_02_obs240_V1_03_difficult | 1056/1064 | 8 | 0.9925 | 10807 | [1403715888.4164658, 1403715996.4784107] | 0.9833 |
| MH_01_easy | obs240 | 20261115T155025_stable6_03_obs240_MH_01_easy | 1547/1831 | 284 | 0.8449 | 3099 | [1403636578.9228814, 1403636766.6788812] | 0.9999 |
| MH_02_easy | obs240 | 20261115T155025_stable6_04_obs240_MH_02_easy | 1208/1488 | 280 | 0.8118 | 2487 | [1403636856.8298812, 1403637010.0888813] | 0.9964 |
| MH_03_medium | obs240 | 20261115T155025_stable6_05_obs240_MH_03_medium | 897/1340 | 443 | 0.6694 | 1876 | [1403637133.0408814, 1403637266.4448812] | 0.9815 |
| MH_04_difficult | obs240 | 20261115T155025_stable6_06_obs240_MH_04_difficult | 713/1006 | 293 | 0.7087 | 1470 | [1403638129.1412687, 1403638229.9002688] | 0.9822 |
| V1_02_medium | qr_batch | 20261115T155819_combo2_01_qr_batch_V1_02_medium | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | 0.9999 |
| MH_05_difficult | qr_batch | 20261115T155819_combo2_02_qr_batch_MH_05_difficult | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | 0.9859 |
| MH_05_difficult | baseline | arm:baseline_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | baseline | arm:baseline_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | batch | arm:cand_batch_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | batch | arm:cand_batch_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | candidate_control | arm:cand_default_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | candidate_control | arm:cand_default_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | obs240 | arm:cand_obs240_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | obs240 | arm:cand_obs240_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | obs480 | arm:cand_obs480_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | obs480 | arm:cand_obs480_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | qr | arm:cand_qr_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | qr | arm:cand_qr_v1:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| MH_05_difficult | qr_batch | arm:cand_qrbatch_mh:MH_05_difficult_r1 | 807/1126 | 319 | 0.7167 | 1665 | [1403638519.7068303, 1403638632.8568304] | - |
| V1_02_medium | qr_batch | arm:cand_qrbatch_v1_r2:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| V1_02_medium | qr_batch | arm:cand_qrbatch_v1_r3:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| V1_02_medium | candidate_control | arm:candidate_control_v1_r2:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| V1_02_medium | candidate_control | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845/845 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |
| V1_02_medium | obs240 | arm:diag_obs240_v1:V1_02_medium_r1 | 229/229 | 0 | 1.0000 | 9418 | [1403715517.8697436, 1403715612.0414774] | - |

## Candidate vs candidate_default

Each row is one run; screening sessions and repeats are never merged. The label is `screening_within_5pct` or `screening_regression` under the engineering line above.

| dataset | variant | run | default | common ts | ATE cand | ATE default | delta % | engineering screen | CPU delta % | same group |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| V1_02_medium | qr | 20261115T150833_screen2_01_qr_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.15115 | 0.15115 | 0.000 | screening_within_5pct | -0.2 | True |
| V1_02_medium | batch | 20261115T151220_screen2_05_batch_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.15115 | 0.15115 | 0.000 | screening_within_5pct | -5.9 | True |
| V1_02_medium | obs240 | 20261115T151220_screen2_07_obs240_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.15707 | 0.15115 | 3.913 | screening_within_5pct | -11.8 | True |
| V1_02_medium | obs480 | 20261115T151220_screen2_09_obs480_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.16742 | 0.15115 | 10.762 | screening_regression | -11.2 | True |
| V1_02_medium | qr_batch | 20261115T155819_combo2_01_qr_batch_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.15115 | 0.15115 | 0.000 | screening_within_5pct | -6.5 | True |
| MH_05_difficult | qr | 20261115T151220_screen2_04_qr_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.75244 | 0.75244 | 0.000 | screening_within_5pct | -7.8 | True |
| MH_05_difficult | batch | 20261115T151220_screen2_06_batch_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.75244 | 0.75244 | 0.000 | screening_within_5pct | -9.0 | True |
| MH_05_difficult | obs240 | 20261115T151220_screen2_08_obs240_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.76945 | 0.75244 | 2.261 | screening_within_5pct | -11.2 | True |
| MH_05_difficult | obs480 | 20261115T151220_screen2_10_obs480_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.75104 | 0.75244 | -0.185 | screening_within_5pct | -8.8 | True |
| MH_05_difficult | qr_batch | 20261115T155819_combo2_02_qr_batch_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.75244 | 0.75244 | 0.000 | screening_within_5pct | -8.9 | True |
| V1_01_easy | qr | 20261115T153457_stable6_01_qr_V1_01_easy | 20261115T152712_stable6_01_candidate_control_V1_01_easy | 1446 | 0.12955 | 0.12955 | 0.000 | screening_within_5pct | 1.3 | True |
| V1_01_easy | batch | 20261115T154242_stable6_01_batch_V1_01_easy | 20261115T152712_stable6_01_candidate_control_V1_01_easy | 1446 | 0.12955 | 0.12955 | 0.000 | screening_within_5pct | 4.0 | True |
| V1_01_easy | obs240 | 20261115T155025_stable6_01_obs240_V1_01_easy | 20261115T152712_stable6_01_candidate_control_V1_01_easy | 1446 | 0.16487 | 0.12955 | 27.258 | screening_regression | -1.1 | True |
| V1_03_difficult | qr | 20261115T153457_stable6_02_qr_V1_03_difficult | 20261115T152712_stable6_02_candidate_control_V1_03_difficult | 1064 | 0.24236 | 0.24236 | 0.000 | screening_within_5pct | -3.7 | True |
| V1_03_difficult | batch | 20261115T154242_stable6_02_batch_V1_03_difficult | 20261115T152712_stable6_02_candidate_control_V1_03_difficult | 1064 | 0.24236 | 0.24236 | -0.000 | screening_within_5pct | 4.6 | True |
| V1_03_difficult | obs240 | 20261115T155025_stable6_02_obs240_V1_03_difficult | 20261115T152712_stable6_02_candidate_control_V1_03_difficult | 1064 | 0.26708 | 0.24236 | 10.200 | screening_regression | -8.8 | True |
| MH_01_easy | qr | 20261115T153457_stable6_03_qr_MH_01_easy | 20261115T152712_stable6_03_candidate_control_MH_01_easy | 1831 | 0.18070 | 0.18070 | 0.000 | screening_within_5pct | -5.1 | True |
| MH_01_easy | batch | 20261115T154242_stable6_03_batch_MH_01_easy | 20261115T152712_stable6_03_candidate_control_MH_01_easy | 1831 | 0.18070 | 0.18070 | -0.000 | screening_within_5pct | 0.5 | True |
| MH_01_easy | obs240 | 20261115T155025_stable6_03_obs240_MH_01_easy | 20261115T152712_stable6_03_candidate_control_MH_01_easy | 1831 | 0.17165 | 0.18070 | -5.007 | screening_within_5pct | -9.5 | True |
| MH_02_easy | qr | 20261115T153457_stable6_04_qr_MH_02_easy | 20261115T152712_stable6_04_candidate_control_MH_02_easy | 1499 | 6.77604 | 6.77604 | 0.000 | screening_within_5pct | -8.8 | True |
| MH_02_easy | batch | 20261115T154242_stable6_04_batch_MH_02_easy | 20261115T152712_stable6_04_candidate_control_MH_02_easy | 1499 | 6.77604 | 6.77604 | 0.000 | screening_within_5pct | -4.7 | True |
| MH_02_easy | obs240 | 20261115T155025_stable6_04_obs240_MH_02_easy | 20261115T152712_stable6_04_candidate_control_MH_02_easy | 1477 | 3.51256 | 6.70435 | -47.608 | screening_within_5pct | -11.0 | True |
| MH_03_medium | qr | 20261115T153457_stable6_05_qr_MH_03_medium | 20261115T152712_stable6_05_candidate_control_MH_03_medium | 1340 | 0.56238 | 0.56238 | 0.000 | screening_within_5pct | -2.7 | True |
| MH_03_medium | batch | 20261115T154242_stable6_05_batch_MH_03_medium | 20261115T152712_stable6_05_candidate_control_MH_03_medium | 1340 | 0.56238 | 0.56238 | -0.000 | screening_within_5pct | -6.7 | True |
| MH_03_medium | obs240 | 20261115T155025_stable6_05_obs240_MH_03_medium | 20261115T152712_stable6_05_candidate_control_MH_03_medium | 1340 | 0.55421 | 0.56238 | -1.454 | screening_within_5pct | -2.4 | True |
| MH_04_difficult | qr | 20261115T153457_stable6_06_qr_MH_04_difficult | 20261115T152712_stable6_06_candidate_control_MH_04_difficult | 1006 | 0.58275 | 0.58275 | 0.000 | screening_within_5pct | -2.5 | True |
| MH_04_difficult | batch | 20261115T154242_stable6_06_batch_MH_04_difficult | 20261115T152712_stable6_06_candidate_control_MH_04_difficult | 1006 | 0.58275 | 0.58275 | -0.000 | screening_within_5pct | -5.5 | True |
| MH_04_difficult | obs240 | 20261115T155025_stable6_06_obs240_MH_04_difficult | 20261115T152712_stable6_06_candidate_control_MH_04_difficult | 1006 | 0.60221 | 0.58275 | 3.338 | screening_within_5pct | 1.2 | True |
| MH_05_difficult | batch | arm:cand_batch_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.64241 | 0.64241 | 0.000 | screening_within_5pct | -0.4 | True |
| MH_05_difficult | obs240 | arm:cand_obs240_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.62808 | 0.64241 | -2.231 | screening_within_5pct | -0.6 | True |
| MH_05_difficult | obs480 | arm:cand_obs480_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.64956 | 0.64241 | 1.112 | screening_within_5pct | -0.0 | True |
| MH_05_difficult | qr | arm:cand_qr_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.64241 | 0.64241 | 0.000 | screening_within_5pct | -0.4 | True |
| MH_05_difficult | qr_batch | arm:cand_qrbatch_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.64241 | 0.64241 | 0.000 | screening_within_5pct | -0.8 | True |
| V1_02_medium | batch | arm:cand_batch_v1:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.17239 | 0.17239 | -0.000 | screening_within_5pct | -0.2 | True |
| V1_02_medium | obs240 | arm:cand_obs240_v1:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.17681 | 0.17239 | 2.563 | screening_within_5pct | -0.2 | True |
| V1_02_medium | obs480 | arm:cand_obs480_v1:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.18251 | 0.17239 | 5.870 | screening_regression | 0.1 | True |
| V1_02_medium | qr | arm:cand_qr_v1:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.17239 | 0.17239 | 0.000 | screening_within_5pct | -0.1 | True |
| V1_02_medium | qr_batch | arm:cand_qrbatch_v1_r2:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.17239 | 0.17239 | -0.000 | screening_within_5pct | -0.1 | True |
| V1_02_medium | qr_batch | arm:cand_qrbatch_v1_r3:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.17239 | 0.17239 | -0.000 | screening_within_5pct | -0.0 | True |

## Baseline equivalence (baseline vs candidate_control)

Candidate comparisons are only attributable after this check passes.

| dataset | baseline | candidate_control | common ts | baseline ATE | control ATE | delta % | pass |
|---|---|---|---:|---:|---:|---:|---|
| V1_02_medium | 20261115T145328_baseline2_01_baseline_V1_02_medium | 20261115T150629_candidate2_01_candidate_control_V1_02_medium | 845 | 0.15115 | 0.15115 | 0.0000 | equivalent |
| MH_05_difficult | 20261115T145328_baseline2_02_baseline_MH_05_difficult | 20261115T150629_candidate2_02_candidate_control_MH_05_difficult | 1126 | 0.75244 | 0.75244 | 0.0000 | equivalent |
| V1_01_easy | 20261115T151921_baseline8_01_baseline_V1_01_easy | 20261115T152712_stable6_01_candidate_control_V1_01_easy | 1446 | 0.12955 | 0.12955 | 0.0000 | equivalent |
| V1_03_difficult | 20261115T151921_baseline8_03_baseline_V1_03_difficult | 20261115T152712_stable6_02_candidate_control_V1_03_difficult | 1064 | 0.24236 | 0.24236 | 0.0000 | equivalent |
| MH_01_easy | 20261115T151921_baseline8_04_baseline_MH_01_easy | 20261115T152712_stable6_03_candidate_control_MH_01_easy | 1831 | 0.18070 | 0.18070 | 0.0000 | equivalent |
| MH_02_easy | 20261115T151921_baseline8_05_baseline_MH_02_easy | 20261115T152712_stable6_04_candidate_control_MH_02_easy | 1499 | 6.77604 | 6.77604 | 0.0000 | equivalent |
| MH_03_medium | 20261115T151921_baseline8_06_baseline_MH_03_medium | 20261115T152712_stable6_05_candidate_control_MH_03_medium | 1340 | 0.56238 | 0.56238 | 0.0000 | equivalent |
| MH_04_difficult | 20261115T151921_baseline8_07_baseline_MH_04_difficult | 20261115T152712_stable6_06_candidate_control_MH_04_difficult | 1006 | 0.58275 | 0.58275 | 0.0000 | equivalent |
| MH_05_difficult | arm:baseline_mh:MH_05_difficult_r1 | arm:cand_default_mh:MH_05_difficult_r1 | 1126 | 0.64241 | 0.64241 | 0.0000 | equivalent |
| V1_02_medium | arm:baseline_v1:V1_02_medium_r1 | arm:candidate_control_v1_r3:V1_02_medium_r1 | 845 | 0.19419 | 0.17239 | -11.2219 | not_equivalent |

## Excluded partial runs

| run | dataset | variant | outcome | reason | excluded from comparisons |
|---|---|---|---|---|---|
| 20261115T150833_screen2_02_qr_MH_05_difficult | MH_05_difficult | qr | not_algorithm_failure | interrupted; exit_code=-2; KeyboardInterrupt/SIGINT evidence; partial run excluded from metrics and comparisons | True |

## Microbench (evidence/microbench_local.json)

- reduce svd: 12.097 us/call, checksum 4366410614.159263
- reduce qr: 11.492 us/call, checksum 4366410614.159263
- validation: valid 120/120, invalid_mismatch=0, gate_dof_mismatch=0, finite_violations=0
- Scope: production calls only; excludes scene build, state copy, track copy, warmup, validation.

## Research: math / implementation risks

1. **QR nullspace projection.** `qr` replaces the historical full-U SVD with a Givens QR of the point Jacobian. The microbenchmark shows identical checksums and zero validation mismatches, but that is a local algebra check, not end-to-end trajectory equivalence; rank tolerance, pivot order and chi2-gate degrees of freedom remain the places where a silent divergence could appear.
2. **Observation budget.** `obs240`/`obs480` select complete tracks before triangulation, deferring or expiring tracks atomically. The risk is a changed effective information set and selection bias: deferred tracks are retried, margin/oversized tracks are expired, and budget-selected observations remain an upper bound on observations actually used after triangulation/gating.
3. **Batch IMU covariance.** `batch` switches image-time IMU covariance accumulation. The risk is changed cross-correlation/conditioning and a default-path regression; it must first pass the baseline-equivalence gate before any speed result is attributed to batching.

Parameters stay default-off: `schur_nullspace_method=svd`, `schur_batch_imu_covariance=false`, `schur_max_update_observations=0` (unlimited).

## Phase-gate audit

- baseline2: complete (V1_02 0.15115 m, MH_05 0.75244 m). baseline8: in progress.
- candidate build and candidate_control: complete; baseline-equivalence table above is the gate.
- screen2: qr/batch/obs240/obs480 runs are reported individually, including the interrupted QR MH_05 attempt (exit -2, KeyboardInterrupt), labeled `not_algorithm_failure` and excluded as partial; the successful rerun is the screened value. No repeats or sessions are merged.
- stable6 and `qr_batch` combo remain gated; ARM results are a separate platform group and must not share CPU claims with local x86 results.

ARM discovery root: `/home/sprout_stars/rmcode/VINS-Fusion/output/qr_imu_budget_20261115/arm_results` (missing or empty is reported, never substituted).
