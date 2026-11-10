# Pre-training & post-training results (Cora, 5 seeds, mean ± std test accuracy %)

## 1. Self-supervised pre-training (frozen encoder)

| Method | Linear probe acc | k-means NMI | Pre-train time |
|---|:---:|:---:|:---:|
| Random init (no pre-training) | 71.5 ± 0.7 | 0.282 ± 0.035 | 0.0s |
| DGI | 81.7 ± 0.5 | 0.551 ± 0.005 | 58.4s |
| GRACE | 81.2 ± 1.5 | 0.511 ± 0.018 | 250.2s |
| BGRL | 78.1 ± 1.4 | 0.467 ± 0.022 | 204.7s |
| GraphMAE | 74.6 ± 0.4 | 0.293 ± 0.046 | 374.9s |
| GAE | 78.0 ± 1.2 | 0.356 ± 0.009 | 48.5s |

## 2. Fine-tuning by label budget (labels per class)

| Init | 1 | 3 | 5 | 10 | 20 (public) |
|---|:---:|:---:|:---:|:---:|:---:|
| Scratch | 49.1 ± 10.3 | 60.6 ± 7.1 | 71.4 ± 2.0 | 76.1 ± 2.5 | 81.3 ± 0.5 |
| DGI | **65.4 ± 7.8** | **73.1 ± 3.3** | **75.6 ± 1.7** | 77.2 ± 2.5 | 81.5 ± 0.7 |
| GRACE | 54.2 ± 8.7 | 67.7 ± 4.2 | 74.2 ± 2.5 | 77.1 ± 3.1 | **82.3 ± 1.1** |
| BGRL | 50.5 ± 7.6 | 60.3 ± 4.2 | 68.9 ± 2.7 | 73.8 ± 2.4 | 79.0 ± 1.2 |
| GraphMAE | 46.8 ± 8.7 | 60.8 ± 3.3 | 68.9 ± 1.9 | 74.2 ± 1.2 | 79.8 ± 0.7 |
| GAE | 55.1 ± 7.8 | 66.2 ± 4.7 | 75.4 ± 3.8 | **78.4 ± 1.7** | 81.6 ± 0.4 |

Post-training builds on **DGI** (chosen by mean validation accuracy).

## 3. Post-training stack

| Stage | 1 | 3 | 5 | 10 | 20 (public) |
|---|:---:|:---:|:---:|:---:|:---:|
| Scratch | 49.1 ± 10.3 | 60.6 ± 7.1 | 71.4 ± 2.0 | 76.1 ± 2.5 | 81.3 ± 0.5 |
| Pre-trained | 65.4 ± 7.8 | 73.1 ± 3.3 | 75.6 ± 1.7 | 77.2 ± 2.5 | 81.5 ± 0.7 |
| + Self-training | 65.4 ± 7.8 | 73.1 ± 3.3 | 75.6 ± 1.7 | 78.7 ± 1.8 | 82.4 ± 0.7 |
| + Correct&Smooth | 65.3 ± 8.0 | 73.1 ± 3.3 | 75.6 ± 1.7 | 79.4 ± 2.4 | 83.1 ± 0.8 |
| Ensemble (val-selected) | 65.4 ± 7.8 | 73.8 ± 2.1 | 77.5 ± 3.7 | 79.9 ± 1.1 | 83.0 ± 0.4 |
| Ensemble + C&S | 65.3 ± 8.0 | 73.8 ± 2.1 | 77.5 ± 3.7 | 80.2 ± 1.0 | 83.5 ± 0.4 |
| Self-ensembling self-training | 73.7 ± 6.6 | 79.8 ± 1.4 | 80.6 ± 2.2 | 82.9 ± 1.1 | 83.9 ± 0.6 |

## 4. Calibration (temperature scaling on the final single-model pipeline)

| Budget | ECE before | ECE after | NLL before | NLL after | T |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.5 ± 13.1 | 4.3 ± 1.6 | 1.021 | 0.884 | 0.68 |
| 3 | 37.2 ± 16.7 | 11.0 ± 13.8 | 1.142 | 0.752 | 0.38 |
| 5 | 18.6 ± 9.3 | 4.5 ± 1.0 | 0.758 | 0.618 | 0.69 |
| 10 | 12.6 ± 5.3 | 4.2 ± 1.3 | 0.655 | 0.600 | 0.79 |
| 20 (public) | 12.6 ± 7.4 | 3.4 ± 0.8 | 0.582 | 0.498 | 0.77 |

## 5. Distillation into a graph-free MLP

| Budget | Teacher (GNN pipeline) | MLP from labels | Distilled MLP |
|---|:---:|:---:|:---:|
| 1 | 73.7 ± 6.6 | 30.9 ± 4.1 | 72.5 ± 7.2 |
| 3 | 79.8 ± 1.4 | 34.2 ± 4.0 | 76.1 ± 4.0 |
| 5 | 80.6 ± 2.2 | 41.2 ± 3.1 | 79.6 ± 1.9 |
| 10 | 82.9 ± 1.1 | 49.4 ± 1.6 | 81.1 ± 1.3 |
| 20 (public) | 83.9 ± 0.6 | 59.3 ± 0.5 | 82.6 ± 0.7 |

Full-graph inference: GCN 16.1 ms vs MLP 3.5 ms (4.6x faster, and needs no edges at inference).

_Total runtime: 91.1 min on CPU._
