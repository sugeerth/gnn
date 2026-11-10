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

Five self-supervised objectives (**DGI, GRACE, BGRL, GraphMAE, GAE**) pre-train the same 2-layer GCN encoder with no labels. Each encoder is then fine-tuned with 1, 3, 5, 10 or 20 labels per class, against an identical model trained from scratch. A post-training stack goes on top: self-training with pseudo-labels, Correct & Smooth, greedy ensemble selection, temperature scaling, and distillation into a graph-free MLP. Every hyperparameter and every post-training step is chosen on the validation set. All numbers are test accuracy, mean ± std over 5 seeds.

| Labels / class | Scratch | DGI pre-trained | + post-training stack |
|:---:|:---:|:---:|:---:|
| 1 | 49.1 ± 10.3 | **65.4 ± 7.8** | 65.3 ± 8.0 |
| 3 | 60.6 ± 7.1 | 73.1 ± 3.3 | **73.8 ± 2.1** |
| 5 | 71.4 ± 2.0 | 75.6 ± 1.7 | **77.5 ± 3.7** |
| 10 | 76.1 ± 2.5 | 77.2 ± 2.5 | **80.2 ± 1.0** |
| 20 (public) | 81.3 ± 0.5 | 81.5 ± 0.7 | **83.5 ± 0.4** |

- **Pre-training pays off most when labels are scarce.** DGI adds +16.3 pts at 1 label per class and +12.5 at 3, but only +0.3 at 20. Frozen DGI embeddings already reach 81.7% with a linear probe, against 71.5% for a random GCN.
- **Post-training needs labels to work with.** Gated on validation, the stack adds 0 to +3 pts. It helps most at 10–20 labels, where Correct & Smooth and ensembling have signal to propagate and select on.
- **Calibration:** one validation-fitted temperature cuts expected calibration error from 10.4% to 3.3% on the public split, and from 54% to 7% at 5 labels.
- **Distillation:** a graph-free MLP distilled from the GNN pipeline reaches 81.9%, against 59.3% for the same MLP trained on labels alone. It runs inference about 3× faster and needs no edges.
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
