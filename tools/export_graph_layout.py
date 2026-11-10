#!/usr/bin/env python3
"""Export a compact, pre-laid-out Cora subgraph for docs/pretrain.html's hero.

Reads the subgraph already stored in docs/results.json (written by train_gnn.py),
keeps its largest connected component, runs a seeded spring layout and writes
docs/cora_layout.json: {"nodes": [[x, y, label, degree, cora_node_id], ...], "edges": [[u, v], ...]}.
"""
import json
import networkx as nx

src = json.load(open("docs/results.json"))["subgraph"]
G = nx.Graph()
for n in src["nodes"]:
    G.add_node(n["id"], label=n["label"], degree=n["degree"], orig=n["original_id"])
G.add_edges_from((e["source"], e["target"]) for e in src["edges"])
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=7))
xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
idx = {n: i for i, n in enumerate(G.nodes())}
nodes = [[round((pos[n][0] - lo_x) / (hi_x - lo_x), 4), round((pos[n][1] - lo_y) / (hi_y - lo_y), 4),
          G.nodes[n]["label"], G.nodes[n]["degree"], G.nodes[n]["orig"]] for n in G.nodes()]
edges = [[idx[u], idx[v]] for u, v in G.edges()]
json.dump({"nodes": nodes, "edges": edges}, open("docs/cora_layout.json", "w"), separators=(",", ":"))
print(f"{len(nodes)} nodes, {len(edges)} edges -> docs/cora_layout.json")
