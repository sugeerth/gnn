#!/usr/bin/env python3
"""
GNN Social Network Analysis — Representation Learning & Community Detection
============================================================================
Trains GCN, GAT, and GraphSAGE on real-world networks.
Demonstrates how GNNs learn powerful node representations that capture
community structure, influence patterns, and latent social dynamics.

Outputs: docs/results.json (consumed by the interactive website)
"""

import json
import os
import ssl
import time
import certifi
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import to_networkx, degree
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
import networkx as nx

# Fix SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Load Dataset — Cora Citation Network (a classic social/citation network)
# ---------------------------------------------------------------------------
dataset = Planetoid(root="./data/Cora", name="Cora")
data = dataset[0]
NUM_CLASSES = dataset.num_classes
CLASS_NAMES = [
    "Case Based", "Genetic Algorithms", "Neural Networks",
    "Probabilistic Methods", "Reinforcement Learning",
    "Rule Learning", "Theory"
]

print(f"\n{'='*60}")
print(f"Cora Citation Network")
print(f"{'='*60}")
print(f"Nodes (papers): {data.num_nodes:,}")
print(f"Edges (citations): {data.num_edges:,}")
print(f"Features per node: {data.num_node_features}")
print(f"Classes: {NUM_CLASSES} — {CLASS_NAMES}")
print(f"{'='*60}\n")

data = data.to(DEVICE)

# ---------------------------------------------------------------------------
# 2. Model Definitions — Three powerful GNN architectures
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """Graph Convolutional Network with skip connections."""
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid)
        self.bn1 = torch.nn.BatchNorm1d(hid)
        self.conv2 = GCNConv(hid, hid)
        self.bn2 = torch.nn.BatchNorm1d(hid)
        self.conv3 = GCNConv(hid, out_ch)
        self.skip = torch.nn.Linear(in_ch, hid)

    def forward(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = h + F.elu(self.skip(x))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv3(h, edge_index)

    def get_embeddings(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = h + F.elu(self.skip(x))
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        return h


class GAT(torch.nn.Module):
    """Graph Attention Network — learns importance weights per neighbor."""
    def __init__(self, in_ch, hid, out_ch, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hid * heads, out_ch, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(F.dropout(x, p=0.6, training=self.training), edge_index))
        h = self.conv2(F.dropout(h, p=0.6, training=self.training), edge_index)
        return h

    def get_embeddings(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        return h


class GraphSAGE(torch.nn.Module):
    """GraphSAGE — inductive representation learning via sampling & aggregation."""
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.bn1 = torch.nn.BatchNorm1d(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = torch.nn.BatchNorm1d(hid)
        self.linear = torch.nn.Linear(hid, out_ch)

    def forward(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.linear(h)

    def get_embeddings(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        return h


# ---------------------------------------------------------------------------
# 3. Training Loop with early stopping
# ---------------------------------------------------------------------------
def train_model(model, data, epochs=300, lr=0.005, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_pred = logits[data.train_mask].argmax(dim=1).cpu()
            val_pred = logits[data.val_mask].argmax(dim=1).cpu()
            train_acc = accuracy_score(data.y[data.train_mask].cpu(), train_pred)
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), val_pred)
            val_f1 = f1_score(data.y[data.val_mask].cpu(), val_pred, average="macro")

        history["train_loss"].append(float(loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Train {train_acc:.4f} | Val {val_acc:.4f} | F1 {val_f1:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return history


def evaluate(model, data, mask, label="Test"):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].argmax(dim=1).cpu()
        y = data.y[mask].cpu()
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    report = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y, pred).tolist()
    print(f"\n  {label} Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    return {"accuracy": float(acc), "f1_macro": float(f1),
            "report": report, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# 4. Train all models
# ---------------------------------------------------------------------------
HIDDEN = 64
results = {}
all_embeddings = {}

models_config = {
    "GCN": GCN(data.num_node_features, HIDDEN, NUM_CLASSES),
    "GAT": GAT(data.num_node_features, 8, NUM_CLASSES, heads=8),
    "GraphSAGE": GraphSAGE(data.num_node_features, HIDDEN, NUM_CLASSES),
}

for name, model in models_config.items():
    print(f"\n{'─'*60}")
    print(f"Training {name}...")
    print(f"{'─'*60}")
    model = model.to(DEVICE)
    t0 = time.time()
    history = train_model(model, data, epochs=300)
    elapsed = time.time() - t0
    test_metrics = evaluate(model, data, data.test_mask)

    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    all_embeddings[name] = emb

    results[name] = {
        "history": history,
        "test": test_metrics,
        "training_time": round(elapsed, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }

# ---------------------------------------------------------------------------
# 5. Embedding Analysis — t-SNE + Clustering
# ---------------------------------------------------------------------------
print("\n\nComputing t-SNE embeddings...")

# Also compute raw feature t-SNE for comparison
raw_features = data.x.cpu().numpy()
print("  t-SNE for raw features (no GNN)...")
tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
raw_2d = tsne_raw.fit_transform(raw_features)

tsne_results = {"Raw Features": raw_2d.tolist()}
cluster_results = {}

# Cluster raw features
kmeans_raw = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10)
raw_clusters = kmeans_raw.fit_predict(raw_features)
raw_nmi = normalized_mutual_info_score(data.y.cpu().numpy(), raw_clusters)
raw_sil = silhouette_score(raw_features, raw_clusters, sample_size=min(2000, len(raw_features)), random_state=42)
cluster_results["Raw Features"] = {
    "nmi": round(float(raw_nmi), 4),
    "silhouette": round(float(raw_sil), 4),
}
print(f"    Raw — NMI: {raw_nmi:.4f} | Silhouette: {raw_sil:.4f}")

for name, emb in all_embeddings.items():
    print(f"  t-SNE for {name}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(emb)
    tsne_results[name] = emb_2d.tolist()

    kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(emb)
    nmi = normalized_mutual_info_score(data.y.cpu().numpy(), cluster_labels)
    sil = silhouette_score(emb, cluster_labels, sample_size=min(2000, len(emb)), random_state=42)
    cluster_results[name] = {
        "nmi": round(float(nmi), 4),
        "silhouette": round(float(sil), 4),
    }
    print(f"    {name} — NMI: {nmi:.4f} | Silhouette: {sil:.4f}")

# ---------------------------------------------------------------------------
# 6. Graph-level statistics
# ---------------------------------------------------------------------------
print("\nComputing graph statistics...")
data_cpu = data.cpu()
G = to_networkx(data_cpu, to_undirected=True)
degree_seq = [d for _, d in G.degree()]
labels_cpu = data_cpu.y.numpy()

graph_stats = {
    "num_nodes": G.number_of_nodes(),
    "num_edges": G.number_of_edges(),
    "avg_degree": round(float(np.mean(degree_seq)), 2),
    "median_degree": int(np.median(degree_seq)),
    "max_degree": int(np.max(degree_seq)),
    "min_degree": int(np.min(degree_seq)),
    "density": round(float(nx.density(G)), 6),
    "num_components": nx.number_connected_components(G),
    "class_distribution": {
        CLASS_NAMES[i]: int((labels_cpu == i).sum())
        for i in range(NUM_CLASSES)
    },
}

# Degree distribution
hist, bin_edges = np.histogram(degree_seq, bins=50)
graph_stats["degree_hist"] = {
    "counts": hist.tolist(),
    "bin_edges": [round(float(b), 2) for b in bin_edges],
}

# Homophily ratio (fraction of edges connecting same-class nodes)
edge_index = data_cpu.edge_index.numpy()
same_class = (labels_cpu[edge_index[0]] == labels_cpu[edge_index[1]]).sum()
homophily = float(same_class) / edge_index.shape[1]
graph_stats["homophily"] = round(homophily, 4)
print(f"  Homophily: {homophily:.4f}")

# Per-class avg degree
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_nodes = np.where(labels_cpu == cls_idx)[0]
    cls_degrees = [degree_seq[n] for n in cls_nodes]
    graph_stats.setdefault("class_avg_degree", {})[cls_name] = round(float(np.mean(cls_degrees)), 2)

# ---------------------------------------------------------------------------
# 7. Community Detection (Louvain) — compare with GNN clusters
# ---------------------------------------------------------------------------
print("Running Louvain community detection...")
communities = nx.community.louvain_communities(G, seed=42)
louvain_labels = np.zeros(G.number_of_nodes(), dtype=int)
for i, comm in enumerate(communities):
    for node in comm:
        louvain_labels[node] = i

louvain_nmi = normalized_mutual_info_score(labels_cpu, louvain_labels)
print(f"  Louvain communities: {len(communities)} | NMI vs true labels: {louvain_nmi:.4f}")
graph_stats["louvain_communities"] = len(communities)
graph_stats["louvain_nmi"] = round(float(louvain_nmi), 4)

# ---------------------------------------------------------------------------
# 8. Neighborhood analysis — what makes GNN powerful
# ---------------------------------------------------------------------------
print("Analyzing neighborhood label consistency...")

# For each node, what fraction of neighbors share the same label?
neighbor_consistency = []
for node in range(G.number_of_nodes()):
    neighbors = list(G.neighbors(node))
    if len(neighbors) > 0:
        same = sum(1 for n in neighbors if labels_cpu[n] == labels_cpu[node])
        neighbor_consistency.append(same / len(neighbors))
    else:
        neighbor_consistency.append(0.0)

graph_stats["avg_neighbor_consistency"] = round(float(np.mean(neighbor_consistency)), 4)

# Per-class neighbor consistency
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_nodes = np.where(labels_cpu == cls_idx)[0]
    cls_cons = [neighbor_consistency[n] for n in cls_nodes]
    graph_stats.setdefault("class_neighbor_consistency", {})[cls_name] = round(float(np.mean(cls_cons)), 4)

# ---------------------------------------------------------------------------
# 9. Build subgraph for interactive visualization
# ---------------------------------------------------------------------------
print("\nExtracting subgraph for interactive visualization...")

# Sample connected component centered on high-degree nodes from each class
sample_nodes = set()
for cls in range(NUM_CLASSES):
    cls_nodes = np.where(labels_cpu == cls)[0]
    cls_degrees = [(n, degree_seq[n]) for n in cls_nodes]
    cls_degrees.sort(key=lambda x: -x[1])
    seeds = [n for n, _ in cls_degrees[:3]]
    for seed in seeds:
        ego = nx.ego_graph(G, seed, radius=1)
        sample_nodes.update(ego.nodes())

sample_nodes = sorted(list(sample_nodes))[:1500]
subG = G.subgraph(sample_nodes).copy()

# Use best model for visualization
best_model = max(results, key=lambda k: results[k]["test"]["accuracy"])
best_tsne = tsne_results[best_model]

node_map = {old: new for new, old in enumerate(sample_nodes)}
sub_nodes = []
for old_id in sample_nodes:
    if old_id in subG:
        sub_nodes.append({
            "id": node_map[old_id],
            "original_id": int(old_id),
            "label": int(labels_cpu[old_id]),
            "class": CLASS_NAMES[int(labels_cpu[old_id])],
            "degree": degree_seq[old_id],
            "tsne_x": float(best_tsne[old_id][0]),
            "tsne_y": float(best_tsne[old_id][1]),
            "raw_tsne_x": float(tsne_results["Raw Features"][old_id][0]),
            "raw_tsne_y": float(tsne_results["Raw Features"][old_id][1]),
            "neighbor_consistency": round(neighbor_consistency[old_id], 3),
        })

sub_edges = []
for u, v in subG.edges():
    if u in node_map and v in node_map:
        sub_edges.append({"source": node_map[u], "target": node_map[v]})

print(f"  Subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")

# ---------------------------------------------------------------------------
# 10. Save all results
# ---------------------------------------------------------------------------
os.makedirs("docs", exist_ok=True)

output = {
    "dataset": "Cora Citation Network",
    "description": "2,708 scientific papers classified into 7 research areas, connected by 5,429 citation links. Each paper has a 1,433-dim bag-of-words feature vector.",
    "graph_stats": graph_stats,
    "class_names": CLASS_NAMES,
    "models": {},
    "cluster_analysis": cluster_results,
    "best_model": best_model,
    "subgraph": {"nodes": sub_nodes, "edges": sub_edges},
}

for name in results:
    m = results[name]
    output["models"][name] = {
        "history": m["history"],
        "test": m["test"],
        "training_time": m["training_time"],
        "params": m["params"],
        "tsne_sample": [tsne_results[name][n["original_id"]] for n in sub_nodes],
    }

# Raw features t-SNE for comparison
output["models"]["Raw Features"] = {
    "history": None,
    "test": None,
    "training_time": 0,
    "params": 0,
    "tsne_sample": [tsne_results["Raw Features"][n["original_id"]] for n in sub_nodes],
}

with open("docs/results.json", "w") as f:
    json.dump(output, f)

file_size = os.path.getsize("docs/results.json") / (1024 * 1024)
print(f"\n{'='*60}")
print(f"Results saved to docs/results.json ({file_size:.1f} MB)")
print(f"Best model: {best_model} — Test Acc: {results[best_model]['test']['accuracy']:.4f}")
print(f"\nModel Comparison:")
print(f"{'Model':<12} {'Acc':>8} {'F1':>8} {'NMI':>8} {'Params':>10} {'Time':>8}")
print(f"{'─'*56}")
for name in results:
    r = results[name]
    c = cluster_results[name]
    print(f"{name:<12} {r['test']['accuracy']:>8.4f} {r['test']['f1_macro']:>8.4f} "
          f"{c['nmi']:>8.4f} {r['params']:>10,} {r['training_time']:>7.1f}s")
print(f"\nRaw Features — NMI: {cluster_results['Raw Features']['nmi']:.4f} "
      f"(GNNs improve clustering by learning graph structure!)")
print(f"{'='*60}")
