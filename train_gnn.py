#!/usr/bin/env python3
"""
GNN Social Network Analysis — Representation Learning & Community Detection
============================================================================
Trains GCN, GAT, and GraphSAGE on the Cora citation network.
Shows that GNN-learned node embeddings capture community structure far better
than raw bag-of-words features (clustering NMI 0.13 -> 0.59).

Outputs: docs/results.json (consumed by the interactive website)
"""

import json
import os
import ssl  # noqa: F401  (imported for its platform cert-handling side effects)
import time
import certifi
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
import networkx as nx

# Fix SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()

# Hyperparameters / run configuration (named so intent is obvious at the call site).
HIDDEN = 64                 # hidden dimension for GCN / GraphSAGE
GAT_HIDDEN = 8              # per-head hidden dimension for GAT (8 heads * 8 = 64)
GAT_HEADS = 8
EPOCHS = 300
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 50    # epochs without val-acc improvement before stopping
DROPOUT = 0.5
GAT_DROPOUT = 0.6
RANDOM_SEED = 42
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
DEGREE_HIST_BINS = 50
SILHOUETTE_SAMPLE_CAP = 2000   # cap silhouette sampling so it stays cheap on large graphs
SUBGRAPH_NODE_CAP = 1500       # max nodes kept in the interactive-viz subgraph
SEEDS_PER_CLASS = 3            # highest-degree nodes per class used as ego-graph seeds

CLASS_NAMES = [
    "Case Based", "Genetic Algorithms", "Neural Networks",
    "Probabilistic Methods", "Reinforcement Learning",
    "Rule Learning", "Theory"
]


# ---------------------------------------------------------------------------
# Pure graph/label analytics (no torch — operate on plain numpy arrays and a
# networkx graph). Kept dependency-light and side-effect-free so they can be
# unit-tested in isolation from the heavy training pipeline.
# ---------------------------------------------------------------------------
def compute_homophily(labels, edge_index):
    """Fraction of edges whose endpoints share the same class label.

    ``edge_index`` is a (2, num_edges) array of node indices, matching the
    PyG edge-index layout. Returns a float in [0, 1].
    """
    same_class = (labels[edge_index[0]] == labels[edge_index[1]]).sum()
    return float(same_class) / edge_index.shape[1]


def compute_neighbor_consistency(graph, labels):
    """Per-node fraction of neighbors sharing the node's label.

    Isolated nodes (no neighbors) contribute 0.0 so the list stays aligned
    with node indices 0..N-1.
    """
    consistency = []
    for node in range(graph.number_of_nodes()):
        neighbors = list(graph.neighbors(node))
        if neighbors:
            same = sum(1 for n in neighbors if labels[n] == labels[node])
            consistency.append(same / len(neighbors))
        else:
            consistency.append(0.0)
    return consistency


def per_class_mean(values, labels, num_classes, class_names, ndigits=2):
    """Mean of ``values`` grouped by class, keyed by class name.

    ``values`` is indexable by node id (list or array); ``labels`` is the
    per-node class array. Replaces the repeated setdefault/np.mean idiom.
    """
    result = {}
    for cls_idx in range(num_classes):
        cls_nodes = np.where(labels == cls_idx)[0]
        cls_values = [values[n] for n in cls_nodes]
        result[class_names[cls_idx]] = round(float(np.mean(cls_values)), ndigits)
    return result


def flatten_communities(communities, num_nodes):
    """Convert a list of node-sets (community partition) into a per-node label array."""
    labels = np.zeros(num_nodes, dtype=int)
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    return labels


DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ---------------------------------------------------------------------------
# 2. Model Definitions — GCN, GAT, GraphSAGE
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
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.conv3(h, edge_index)

    def get_embeddings(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = h + F.elu(self.skip(x))
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        return h


class GAT(torch.nn.Module):
    """Graph Attention Network — learns importance weights per neighbor."""
    def __init__(self, in_ch, hid, out_ch, heads=GAT_HEADS):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid, heads=heads, dropout=GAT_DROPOUT)
        self.conv2 = GATConv(hid * heads, out_ch, heads=1, concat=False, dropout=GAT_DROPOUT)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(F.dropout(x, p=GAT_DROPOUT, training=self.training), edge_index))
        h = self.conv2(F.dropout(h, p=GAT_DROPOUT, training=self.training), edge_index)
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
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.linear(h)

    def get_embeddings(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        return h


# ---------------------------------------------------------------------------
# 3. Training Loop with early stopping
# ---------------------------------------------------------------------------
def train_model(model, data, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0
    best_state = None
    patience = EARLY_STOP_PATIENCE
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


def main():
    """Load Cora, train all models, run the analyses, and write docs/results.json.

    Wrapped in a function (instead of running at import) so that ``import
    train_gnn`` does not download data or train models — only ``python3
    train_gnn.py`` (or calling ``main()``) triggers the pipeline. Behavior of a
    direct run is unchanged.
    """
    # -----------------------------------------------------------------------
    # 1. Load Dataset — Cora Citation Network (a classic social/citation network)
    # -----------------------------------------------------------------------
    dataset = Planetoid(root="./data/Cora", name="Cora")
    data = dataset[0]
    num_classes = dataset.num_classes

    print(f"Using device: {DEVICE}")
    print(f"\n{'='*60}")
    print(f"Cora Citation Network")
    print(f"{'='*60}")
    print(f"Nodes (papers): {data.num_nodes:,}")
    print(f"Edges (citations): {data.num_edges:,}")
    print(f"Features per node: {data.num_node_features}")
    print(f"Classes: {num_classes} — {CLASS_NAMES}")
    print(f"{'='*60}\n")

    data = data.to(DEVICE)

    # -----------------------------------------------------------------------
    # 4. Train all models
    # -----------------------------------------------------------------------
    results = {}
    all_embeddings = {}

    models_config = {
        "GCN": GCN(data.num_node_features, HIDDEN, num_classes),
        "GAT": GAT(data.num_node_features, GAT_HIDDEN, num_classes, heads=GAT_HEADS),
        "GraphSAGE": GraphSAGE(data.num_node_features, HIDDEN, num_classes),
    }

    for name, model in models_config.items():
        print(f"\n{'─'*60}")
        print(f"Training {name}...")
        print(f"{'─'*60}")
        model = model.to(DEVICE)
        t0 = time.time()
        history = train_model(model, data, epochs=EPOCHS)
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

    # -----------------------------------------------------------------------
    # 5. Embedding Analysis — t-SNE + Clustering
    # -----------------------------------------------------------------------
    print("\n\nComputing t-SNE embeddings...")

    def _tsne():
        return TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                    random_state=RANDOM_SEED, n_iter=TSNE_ITERATIONS)

    def _kmeans():
        return KMeans(n_clusters=num_classes, random_state=RANDOM_SEED, n_init=10)

    def _silhouette(features, cluster_labels):
        return silhouette_score(
            features, cluster_labels,
            sample_size=min(SILHOUETTE_SAMPLE_CAP, len(features)),
            random_state=RANDOM_SEED,
        )

    true_labels = data.y.cpu().numpy()

    # Also compute raw feature t-SNE for comparison
    raw_features = data.x.cpu().numpy()
    print("  t-SNE for raw features (no GNN)...")
    raw_2d = _tsne().fit_transform(raw_features)

    tsne_results = {"Raw Features": raw_2d.tolist()}
    cluster_results = {}

    # Cluster raw features
    raw_clusters = _kmeans().fit_predict(raw_features)
    raw_nmi = normalized_mutual_info_score(true_labels, raw_clusters)
    raw_sil = _silhouette(raw_features, raw_clusters)
    cluster_results["Raw Features"] = {
        "nmi": round(float(raw_nmi), 4),
        "silhouette": round(float(raw_sil), 4),
    }
    print(f"    Raw — NMI: {raw_nmi:.4f} | Silhouette: {raw_sil:.4f}")

    for name, emb in all_embeddings.items():
        print(f"  t-SNE for {name}...")
        emb_2d = _tsne().fit_transform(emb)
        tsne_results[name] = emb_2d.tolist()

        cluster_labels = _kmeans().fit_predict(emb)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        sil = _silhouette(emb, cluster_labels)
        cluster_results[name] = {
            "nmi": round(float(nmi), 4),
            "silhouette": round(float(sil), 4),
        }
        print(f"    {name} — NMI: {nmi:.4f} | Silhouette: {sil:.4f}")

    # -----------------------------------------------------------------------
    # 6. Graph-level statistics
    # -----------------------------------------------------------------------
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
            for i in range(num_classes)
        },
    }

    # Degree distribution
    hist, bin_edges = np.histogram(degree_seq, bins=DEGREE_HIST_BINS)
    graph_stats["degree_hist"] = {
        "counts": hist.tolist(),
        "bin_edges": [round(float(b), 2) for b in bin_edges],
    }

    # Homophily ratio (fraction of edges connecting same-class nodes)
    edge_index = data_cpu.edge_index.numpy()
    homophily = compute_homophily(labels_cpu, edge_index)
    graph_stats["homophily"] = round(homophily, 4)
    print(f"  Homophily: {homophily:.4f}")

    # Per-class avg degree
    graph_stats["class_avg_degree"] = per_class_mean(
        degree_seq, labels_cpu, num_classes, CLASS_NAMES, ndigits=2)

    # -----------------------------------------------------------------------
    # 7. Community Detection (Louvain) — compare with GNN clusters
    # -----------------------------------------------------------------------
    print("Running Louvain community detection...")
    communities = nx.community.louvain_communities(G, seed=RANDOM_SEED)
    louvain_labels = flatten_communities(communities, G.number_of_nodes())

    louvain_nmi = normalized_mutual_info_score(labels_cpu, louvain_labels)
    print(f"  Louvain communities: {len(communities)} | NMI vs true labels: {louvain_nmi:.4f}")
    graph_stats["louvain_communities"] = len(communities)
    graph_stats["louvain_nmi"] = round(float(louvain_nmi), 4)

    # -----------------------------------------------------------------------
    # 8. Neighborhood analysis — label consistency across hops
    # -----------------------------------------------------------------------
    print("Analyzing neighborhood label consistency...")

    # For each node, what fraction of neighbors share the same label?
    neighbor_consistency = compute_neighbor_consistency(G, labels_cpu)

    graph_stats["avg_neighbor_consistency"] = round(float(np.mean(neighbor_consistency)), 4)

    # Per-class neighbor consistency
    graph_stats["class_neighbor_consistency"] = per_class_mean(
        neighbor_consistency, labels_cpu, num_classes, CLASS_NAMES, ndigits=4)

    # -----------------------------------------------------------------------
    # 9. Build subgraph for interactive visualization
    # -----------------------------------------------------------------------
    print("\nExtracting subgraph for interactive visualization...")

    # Sample connected component centered on high-degree nodes from each class
    sample_nodes = set()
    for cls in range(num_classes):
        cls_nodes = np.where(labels_cpu == cls)[0]
        cls_degrees = [(n, degree_seq[n]) for n in cls_nodes]
        cls_degrees.sort(key=lambda x: -x[1])
        seeds = [n for n, _ in cls_degrees[:SEEDS_PER_CLASS]]
        for seed in seeds:
            ego = nx.ego_graph(G, seed, radius=1)
            sample_nodes.update(ego.nodes())

    sample_nodes = sorted(list(sample_nodes))[:SUBGRAPH_NODE_CAP]
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

    # -----------------------------------------------------------------------
    # 10. Save all results
    # -----------------------------------------------------------------------
    os.makedirs("docs", exist_ok=True)

    output = {
        "dataset": "Cora Citation Network",
        "description": "2,708 scientific papers classified into 7 research areas, connected by 5,278 undirected citation links (5,429 directed citations). Each paper has a 1,433-dim bag-of-words feature vector.",
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
          f"(GNNs improve clustering by learning graph structure)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
