"""Unit tests for the pure graph/label analytics helpers in train_gnn.

These cover the side-effect-free functions that were extracted out of the
top-level training script so that graph statistics can be verified without
running (or even importing the weights of) the GNN pipeline.

``train_gnn`` imports torch/torch_geometric at module load time, so the whole
module is skipped if those heavy deps are unavailable. The functions under test
themselves only need numpy + networkx.
"""
import numpy as np
import pytest

# train_gnn imports torch at module scope; skip the whole module if it (or the
# networkx dependency the helpers need) cannot be imported.
train_gnn = pytest.importorskip("train_gnn")
nx = pytest.importorskip("networkx")

compute_homophily = train_gnn.compute_homophily
compute_neighbor_consistency = train_gnn.compute_neighbor_consistency
per_class_mean = train_gnn.per_class_mean
flatten_communities = train_gnn.flatten_communities


# --------------------------------------------------------------------------- #
# compute_homophily
# --------------------------------------------------------------------------- #
def test_homophily_all_same_class_is_one():
    labels = np.array([0, 0, 0])
    # edges 0-1, 1-2 (PyG (2, E) layout)
    edge_index = np.array([[0, 1], [1, 2]])
    assert compute_homophily(labels, edge_index) == 1.0


def test_homophily_all_cross_class_is_zero():
    labels = np.array([0, 1, 0, 1])
    edge_index = np.array([[0, 2], [1, 3]])  # 0-1 (0,1) and 2-3 (0,1): both cross-class
    assert compute_homophily(labels, edge_index) == 0.0


def test_homophily_mixed_fraction():
    labels = np.array([0, 0, 1])
    # edges: 0-1 (same), 1-2 (different) -> 1/2
    edge_index = np.array([[0, 1], [1, 2]])
    assert compute_homophily(labels, edge_index) == pytest.approx(0.5)


def test_homophily_returns_python_float():
    labels = np.array([0, 0])
    edge_index = np.array([[0], [1]])
    result = compute_homophily(labels, edge_index)
    assert isinstance(result, float)


# --------------------------------------------------------------------------- #
# compute_neighbor_consistency
# --------------------------------------------------------------------------- #
def test_neighbor_consistency_matches_labels():
    # Triangle of same-class nodes -> every node fully consistent.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    labels = np.array([5, 5, 5])
    assert compute_neighbor_consistency(G, labels) == [1.0, 1.0, 1.0]


def test_neighbor_consistency_mixed_neighbors():
    # Star: center node 0 has neighbors 1 (same) and 2 (different) -> 0.5.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    labels = np.array([0, 0, 1])
    cons = compute_neighbor_consistency(G, labels)
    assert cons[0] == pytest.approx(0.5)
    assert cons[1] == 1.0  # node 1's only neighbor (0) shares its label
    assert cons[2] == 0.0  # node 2's only neighbor (0) differs


def test_neighbor_consistency_isolated_node_is_zero():
    G = nx.Graph()
    G.add_nodes_from([0, 1])  # no edges
    labels = np.array([0, 1])
    assert compute_neighbor_consistency(G, labels) == [0.0, 0.0]


def test_neighbor_consistency_is_node_aligned():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    labels = np.array([0, 0, 1, 9])
    cons = compute_neighbor_consistency(G, labels)
    assert len(cons) == 4  # one entry per node id 0..3


# --------------------------------------------------------------------------- #
# per_class_mean
# --------------------------------------------------------------------------- #
def test_per_class_mean_groups_and_rounds():
    values = [10, 20, 1, 3]
    labels = np.array([0, 0, 1, 1])
    names = ["A", "B"]
    out = per_class_mean(values, labels, num_classes=2, class_names=names)
    assert out == {"A": 15.0, "B": 2.0}


def test_per_class_mean_respects_ndigits():
    values = [1, 2]            # class 0 mean = 1.5
    labels = np.array([0, 0])
    out4 = per_class_mean(values, labels, 1, ["X"], ndigits=4)
    # mean of [1,2] = 1.5; default rounding to 2 still 1.5
    assert out4 == {"X": 1.5}

    values2 = [1, 1, 2]        # mean = 1.3333...
    labels2 = np.array([0, 0, 0])
    assert per_class_mean(values2, labels2, 1, ["X"], ndigits=4) == {"X": 1.3333}
    assert per_class_mean(values2, labels2, 1, ["X"], ndigits=2) == {"X": 1.33}


def test_per_class_mean_values_are_python_floats():
    out = per_class_mean([1, 2], np.array([0, 0]), 1, ["X"])
    assert isinstance(out["X"], float)


# --------------------------------------------------------------------------- #
# flatten_communities
# --------------------------------------------------------------------------- #
def test_flatten_communities_assigns_community_ids():
    communities = [{0, 2}, {1, 3, 4}]
    labels = flatten_communities(communities, num_nodes=5)
    assert labels.tolist() == [0, 1, 0, 1, 1]


def test_flatten_communities_single_community():
    labels = flatten_communities([{0, 1, 2}], num_nodes=3)
    assert labels.tolist() == [0, 0, 0]


def test_flatten_communities_dtype_is_int():
    labels = flatten_communities([{0}], num_nodes=1)
    assert labels.dtype == np.int_ or np.issubdtype(labels.dtype, np.integer)
