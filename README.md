# Graph Neural Networks for Social Network Analysis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.5+-3C2179?logo=pyg" alt="PyG">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="https://sugeerth.github.io/gnn/"><img src="https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue" alt="Demo"></a>
  <a href="https://colab.research.google.com/github/sugeerth/gnn/blob/main/GNN_Social_Network_Analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

**How GNNs learn node representations that reveal community structure in citation networks.**

This project trains **GCN**, **GAT**, and **GraphSAGE** on the Cora citation network and demonstrates that GNN-learned embeddings capture community structure **4.5x better** than raw features (NMI: 0.13 → 0.59).

## Results

| Model | Test Accuracy | Macro F1 | Cluster NMI | Parameters |
|-------|:---:|:---:|:---:|:---:|
| **GAT** | **81.8%** | **80.2%** | **0.59** | 92K |
| GCN | 78.5% | 77.4% | 0.57 | 188K |
| GraphSAGE | 72.6% | 72.7% | 0.58 | 192K |
| Raw Features | — | — | 0.13 | — |

### Key Findings

- **Message passing works**: GNNs exploit the 81% homophily in citation networks — papers cite papers in the same field
- **Attention matters**: GAT outperforms by learning *which* neighbors are most informative
- **Embeddings cluster cleanly**: GNN representations form tight, well-separated clusters useful for downstream tasks like link prediction and recommendation

## Pre-training & Post-training (label-scarce regime)

**[Results page →](https://sugeerth.github.io/gnn/pretrain.html)** · `python3 pretrain_posttrain.py` · full tables in [`docs/pretrain_posttrain.md`](docs/pretrain_posttrain.md)

Five self-supervised objectives (**DGI, GRACE, BGRL, GraphMAE, GAE**) pre-train the same 2-layer GCN encoder with no labels. Each encoder is then fine-tuned with 1, 3, 5, 10 or 20 labels per class, against an identical model trained from scratch. A post-training stack goes on top: self-training with pseudo-labels, Correct & Smooth, greedy ensemble selection, temperature scaling, and distillation into a graph-free MLP. Every hyperparameter and every post-training step is chosen on the validation set. The self-training design was compared with a script that prints validation accuracy only. All numbers are test accuracy, mean ± std over 5 seeds.

| Labels / class | Scratch | DGI pre-trained | Ensemble + C&S | **Self-ensembling self-training** | Post-training gain |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 49.1 ± 10.3 | 65.4 ± 7.8 | 65.3 ± 8.0 | **73.7 ± 6.6** | **+8.2** |
| 3 | 60.6 ± 7.1 | 73.1 ± 3.3 | 73.8 ± 2.1 | **79.8 ± 1.4** | **+6.8** |
| 5 | 71.4 ± 2.0 | 75.6 ± 1.7 | 77.5 ± 3.7 | **80.6 ± 2.2** | **+5.0** |
| 10 | 76.1 ± 2.5 | 77.2 ± 2.5 | 80.2 ± 1.0 | **82.9 ± 1.1** | **+5.7** |
| 20 (public) | 81.3 ± 0.5 | 81.5 ± 0.7 | 83.5 ± 0.4 | **83.9 ± 0.6** | **+2.4** |

**[Post-training deep dive →](https://sugeerth.github.io/gnn/posttrain.html)** plays the self-training loop round by round on the citation graph.

- **Pre-training pays off most when labels are scarce.** DGI adds +16.3 pts at 1 label per class and +12.5 at 3, but only +0.3 at 20. Frozen DGI embeddings already reach 81.7% with a linear probe, against 71.5% for a random GCN.
- **Self-ensembling self-training is the big post-training win.** Each round, a teacher is built: a validation-picked ensemble of every model so far, passed through Correct & Smooth and temperature-calibrated. It pseudo-labels the 50·r papers per class it is most sure about. Students restart from the top-3 pre-trained encoders and train on true labels plus confidence-weighted pseudo-labels, then join the pool. Validation picks the round. With 1 label per class this lifts accuracy from 65.4% to 73.7%, **+24.5 pts over scratch from seven labelled papers**. Pseudo-label precision stays at 76–85% while their number grows to about 1,700.
- **Calibration:** one validation-fitted temperature cuts the final teacher's expected calibration error from 12.6% to 3.4% on the public split, and from 18.5% to 4.3% at 1 label.
- **Distillation:** a graph-free MLP distilled from the self-trained teacher reaches 82.6% (public split) and 72.5% (1 label), against 59.3% and 30.9% for the same MLP trained on labels alone. It runs inference about 4.6× faster and needs no edges.
- **Negative results kept:** BGRL and GraphMAE fine-tune worse than scratch at most budgets. GraphMAE's embeddings collapsed until a BatchNorm encoder was added; the fix was chosen on validation only.

## Interactive Demo

**[Live Website →](https://sugeerth.github.io/gnn/)**

The interactive website features:
- Force-directed network visualization with hover interactions
- Switchable t-SNE embedding views (Raw Features → GCN → GAT → GraphSAGE)
- Training dynamics charts
- Community detection comparison
- Per-class confusion matrices

## Run in Google Colab

**[Open Notebook →](https://colab.research.google.com/github/sugeerth/gnn/blob/main/GNN_Social_Network_Analysis.ipynb)**

Full training pipeline with Plotly visualizations — runs on free T4 GPU.

## Local Setup

```bash
pip install -r requirements.txt
python3 train_gnn.py
# Open docs/index.html in your browser
```

## Project Structure

```
├── train_gnn.py                      # Training script — GCN, GAT, GraphSAGE
├── pretrain_posttrain.py             # Self-supervised pre-training + post-training study
├── GNN_Social_Network_Analysis.ipynb  # Google Colab notebook
├── docs/
│   ├── index.html                    # Interactive visualization website
│   ├── pretrain.html                 # Pre-/post-training results page
│   ├── posttrain.html                # Post-training deep dive (self-training, round by round)
│   ├── pretrain_posttrain.json/.md   # Results of pretrain_posttrain.py
│   └── results.json                  # Training results & embeddings
├── requirements.txt
└── README.md
```

## How It Works

### The Network
The **Cora citation network** contains 2,708 scientific papers classified into 7 research areas (Neural Networks, Genetic Algorithms, etc.), connected by 5,278 undirected citation links (5,429 directed citations). Each paper has a 1,433-dimensional bag-of-words feature vector.

### GNN Architectures

**GCN** (Graph Convolutional Network) — Spectral convolution that averages neighbor features with skip connections for gradient flow.

**GAT** (Graph Attention Network) — Learns attention weights per neighbor, so important citations contribute more than tangential ones. Uses 8 attention heads.

**GraphSAGE** — Inductive learning through sampling and aggregation. Can generalize to unseen nodes without retraining.

### Why GNNs Beat Feature-Only Methods

Raw bag-of-words features don't capture *who cites whom*. GNNs propagate information through the graph via message passing — each layer aggregates 1-hop neighborhood information, so a 2-layer GNN sees 2-hop context. Since 81% of citations connect same-class papers, this neighborhood signal is extremely informative.

## License

MIT
