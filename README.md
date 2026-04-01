# Graph Neural Networks for Social Network Analysis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.5+-3C2179?logo=pyg" alt="PyG">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="https://sugeerth.github.io/gnn/"><img src="https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue" alt="Demo"></a>
  <a href="https://colab.research.google.com/github/sugeerth/gnn/blob/main/GNN_Social_Network_Analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

**How GNNs learn powerful node representations that reveal hidden community structure in networks.**

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
- **Embeddings are powerful**: GNN representations form tight, well-separated clusters useful for downstream tasks like link prediction and recommendation

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
├── GNN_Social_Network_Analysis.ipynb  # Google Colab notebook
├── docs/
│   ├── index.html                    # Interactive visualization website
│   └── results.json                  # Training results & embeddings
├── requirements.txt
└── README.md
```

## How It Works

### The Network
The **Cora citation network** contains 2,708 scientific papers classified into 7 research areas (Neural Networks, Genetic Algorithms, etc.), connected by 5,429 citation links. Each paper has a 1,433-dimensional bag-of-words feature vector.

### GNN Architectures

**GCN** (Graph Convolutional Network) — Spectral convolution that averages neighbor features with skip connections for gradient flow.

**GAT** (Graph Attention Network) — Learns attention weights per neighbor, so important citations contribute more than tangential ones. Uses 8 attention heads.

**GraphSAGE** — Inductive learning through sampling and aggregation. Can generalize to unseen nodes without retraining.

### Why GNNs Beat Feature-Only Methods

Raw bag-of-words features don't capture *who cites whom*. GNNs propagate information through the graph via message passing — each layer aggregates 1-hop neighborhood information, so a 2-layer GNN sees 2-hop context. Since 81% of citations connect same-class papers, this neighborhood signal is extremely informative.

## License

MIT
