#!/usr/bin/env python3
"""
Self-supervised pre-training + post-training for GNNs on Cora
=============================================================

Pre-training (label-free, uses only the graph + node features):
  * DGI       - Deep Graph Infomax: maximise MI between node and graph summary
  * GRACE     - node-level contrastive learning over two augmented views
  * BGRL      - bootstrapped (negative-free) learning with an EMA target encoder
  * GraphMAE  - masked node-feature autoencoding with a scaled-cosine loss
  * GAE       - edge reconstruction (link prediction) with negative sampling

Every pre-trained encoder is evaluated three ways:
  1. linear probe on frozen embeddings,
  2. k-means NMI against the true research areas,
  3. full fine-tuning under several label budgets (1/3/5/10 labels per class
     plus the public 20-per-class split), against the same architecture
     trained from a random init.

Post-training (applied on top of fine-tuned models):
  * Self-training   - rounds of class-balanced confident pseudo-labels
  * Correct&Smooth  - propagate residual errors, then smooth predictions
  * Ensembling      - average all fine-tuned inits, then Correct&Smooth
  * Calibration     - temperature scaling fit on validation (ECE / NLL)
  * Distillation    - compress the final graph model into a graph-free MLP

All hyper-parameter choices that are searched (encoder LR, probe C,
self-training rounds, C&S alphas, temperature, distillation weight) are
selected on the validation set only. Results are averaged over seeds.

Outputs: docs/pretrain_posttrain.json and docs/pretrain_posttrain.md
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.utils import dropout_edge, negative_sampling

CLASS_NAMES = [
    "Case Based", "Genetic Algorithms", "Neural Networks",
    "Probabilistic Methods", "Reinforcement Learning",
    "Rule Learning", "Theory",
]
METHODS = ["DGI", "GRACE", "BGRL", "GraphMAE", "GAE"]
INITS = ["Scratch"] + METHODS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "std": float(xs.std()), "runs": [float(x) for x in xs]}


def fmt(ms, pct=True):
    if pct:
        return f"{ms['mean'] * 100:.1f} ± {ms['std'] * 100:.1f}"
    return f"{ms['mean']:.3f} ± {ms['std']:.3f}"


# ---------------------------------------------------------------------------
# Encoder shared by every pre-training method and by fine-tuning
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """2-layer GCN with PReLU. Dropout is off during pre-training and switched
    on for fine-tuning by setting the attributes."""

    def __init__(self, in_dim, hid, out, norm=False):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid), GCNConv(hid, out)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hid), nn.BatchNorm1d(out)]) if norm else None
        self.acts = nn.ModuleList([nn.PReLU(hid), nn.PReLU(out)])
        self.out_dim = out
        self.dropout = 0.0
        self.in_dropout = 0.0

    def forward(self, x, edge_index):
        x = F.dropout(x, self.in_dropout, self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.acts[i](x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, self.training)
        return x


def augment(x, edge_index, p_edge, p_feat):
    ei, _ = dropout_edge(edge_index, p=p_edge)
    keep = (torch.rand(x.size(1), device=x.device) >= p_feat).float()
    return x * keep, ei


class _Best:
    """Keeps the encoder state with the lowest pre-training loss."""

    def __init__(self):
        self.loss, self.state = math.inf, None

    def update(self, loss, module):
        if loss < self.loss:
            self.loss = loss
            self.state = copy.deepcopy(module.state_dict())


# ---------------------------------------------------------------------------
# Pre-training objectives
# ---------------------------------------------------------------------------
def pretrain_dgi(enc, data, cfg):
    d = enc.out_dim
    W = nn.Parameter(torch.empty(d, d))
    nn.init.xavier_uniform_(W)
    opt = torch.optim.Adam(list(enc.parameters()) + [W], lr=cfg["lr"])
    n = data.num_nodes
    ones, zeros = torch.ones(n), torch.zeros(n)
    hist, best = [], _Best()
    for _ in range(cfg["epochs"]):
        enc.train()
        opt.zero_grad()
        pos = enc(data.x, data.edge_index)
        neg = enc(data.x[torch.randperm(n)], data.edge_index)  # feature shuffling corruption
        summary = torch.sigmoid(pos.mean(0))
        v = W @ summary
        loss = (F.binary_cross_entropy_with_logits(pos @ v, ones)
                + F.binary_cross_entropy_with_logits(neg @ v, zeros))
        loss.backward()
        opt.step()
        hist.append(float(loss))
        best.update(float(loss), enc)
    enc.load_state_dict(best.state)
    return hist


def _nt_xent(h1, h2, tau):
    z1, z2 = F.normalize(h1, dim=1), F.normalize(h2, dim=1)

    def semi(a, b):
        refl = torch.exp(a @ a.t() / tau)
        btw = torch.exp(a @ b.t() / tau)
        return -torch.log(btw.diag() / (refl.sum(1) + btw.sum(1) - refl.diag()))

    return 0.5 * (semi(z1, z2) + semi(z2, z1)).mean()


def pretrain_grace(enc, data, cfg):
    d = enc.out_dim
    proj = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, d))
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()),
                           lr=cfg["lr"], weight_decay=cfg["wd"])
    hist, best = [], _Best()
    for _ in range(cfg["epochs"]):
        enc.train()
        opt.zero_grad()
        x1, e1 = augment(data.x, data.edge_index, cfg["pe1"], cfg["pf1"])
        x2, e2 = augment(data.x, data.edge_index, cfg["pe2"], cfg["pf2"])
        loss = _nt_xent(proj(enc(x1, e1)), proj(enc(x2, e2)), cfg["tau"])
        loss.backward()
        opt.step()
        hist.append(float(loss))
        best.update(float(loss), enc)
    enc.load_state_dict(best.state)
    return hist


def pretrain_bgrl(enc, data, cfg):
    d = enc.out_dim
    pred = nn.Sequential(nn.Linear(d, cfg["pred_hid"]), nn.BatchNorm1d(cfg["pred_hid"]),
                         nn.PReLU(), nn.Linear(cfg["pred_hid"], d))
    target = copy.deepcopy(enc)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["wd"])
    epochs = cfg["epochs"]
    warm = max(1, epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm
        else 0.5 * (1 + math.cos(math.pi * (e - warm) / (epochs - warm))))
    hist = []
    for ep in range(epochs):
        enc.train()
        target.train()
        opt.zero_grad()
        x1, e1 = augment(data.x, data.edge_index, cfg["pe1"], cfg["pf1"])
        x2, e2 = augment(data.x, data.edge_index, cfg["pe2"], cfg["pf2"])
        p1, p2 = pred(enc(x1, e1)), pred(enc(x2, e2))
        with torch.no_grad():
            t1, t2 = target(x1, e1), target(x2, e2)
        loss = (2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
                + 2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
        loss.backward()
        opt.step()
        sched.step()
        # EMA momentum anneals 0.99 -> 1.0 (cosine)
        m = 1 - (1 - cfg["ema"]) * (math.cos(math.pi * ep / epochs) + 1) / 2
        with torch.no_grad():
            for po, pt in zip(enc.parameters(), target.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1 - m)
        hist.append(float(loss))
    # BGRL's loss is not a reliable model-selection signal; keep the final online encoder.
    return hist


def pretrain_graphmae(enc, data, cfg):
    d, f = enc.out_dim, data.num_features
    mask_token = nn.Parameter(torch.zeros(1, f))
    enc2dec = nn.Linear(d, d, bias=False)
    decoder = GCNConv(d, f)
    params = list(enc.parameters()) + [mask_token] + list(enc2dec.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    n = data.num_nodes
    n_mask = int(cfg["mask_rate"] * n)
    hist, best = [], _Best()
    for _ in range(cfg["epochs"]):
        enc.train()
        opt.zero_grad()
        m = torch.randperm(n)[:n_mask]
        x_in = data.x.clone()
        x_in[m] = 0.0
        x_in[m] += mask_token
        h = enc2dec(enc(x_in, data.edge_index))
        h = h.clone()
        h[m] = 0.0  # re-mask before decoding
        rec = decoder(h, data.edge_index)
        loss = (1 - F.cosine_similarity(rec[m], data.x[m], dim=-1)).pow(cfg["gamma"]).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss))
        best.update(float(loss), enc)
    enc.load_state_dict(best.state)
    return hist


def pretrain_gae(enc, data, cfg):
    opt = torch.optim.Adam(enc.parameters(), lr=cfg["lr"])
    hist, best = [], _Best()
    ei = data.edge_index
    for _ in range(cfg["epochs"]):
        enc.train()
        opt.zero_grad()
        z = enc(data.x, ei)
        neg = negative_sampling(ei, num_nodes=data.num_nodes, num_neg_samples=ei.size(1))
        pos_l = (z[ei[0]] * z[ei[1]]).sum(-1)
        neg_l = (z[neg[0]] * z[neg[1]]).sum(-1)
        loss = (F.binary_cross_entropy_with_logits(pos_l, torch.ones_like(pos_l))
                + F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l)))
        loss.backward()
        opt.step()
        hist.append(float(loss))
        best.update(float(loss), enc)
    enc.load_state_dict(best.state)
    return hist


PRETRAINERS = {
    "DGI": (pretrain_dgi, False),
    "GRACE": (pretrain_grace, False),
    "BGRL": (pretrain_bgrl, True),
    "GraphMAE": (pretrain_graphmae, False),
    "GAE": (pretrain_gae, False),
}


def default_pretrain_cfg(scale):
    e = lambda n: max(5, int(n * scale))  # noqa: E731
    return {
        "DGI": {"epochs": e(300), "lr": 1e-3},
        "GRACE": {"epochs": e(300), "lr": 5e-4, "wd": 1e-5, "tau": 0.4,
                  "pe1": 0.2, "pe2": 0.4, "pf1": 0.3, "pf2": 0.4},
        "BGRL": {"epochs": e(500), "lr": 1e-3, "wd": 1e-5, "ema": 0.99, "pred_hid": 512,
                 "pe1": 0.3, "pe2": 0.2, "pf1": 0.3, "pf2": 0.2},
        "GraphMAE": {"epochs": e(800), "lr": 1e-3, "wd": 2e-4, "mask_rate": 0.5, "gamma": 3},
        "GAE": {"epochs": e(300), "lr": 1e-2},
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed(enc, data):
    enc.eval()
    return enc(data.x, data.edge_index).numpy()


def linear_probe(emb, y, train_idx, val_idx, test_idx):
    X = StandardScaler().fit_transform(emb)
    best = (-1, 0, None)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(X[train_idx], y[train_idx])
        va = clf.score(X[val_idx], y[val_idx])
        if va > best[0]:
            best = (va, clf.score(X[test_idx], y[test_idx]), C)
    return {"val": best[0], "test": best[1], "C": best[2]}


def kmeans_nmi(emb, y, seed):
    labels = KMeans(n_clusters=len(CLASS_NAMES), random_state=seed, n_init=10).fit_predict(
        StandardScaler().fit_transform(emb))
    return normalized_mutual_info_score(y, labels)


def acc(probs, y, idx):
    return float((probs[idx].argmax(1) == y[idx]).float().mean())


class Classifier(nn.Module):
    def __init__(self, enc, n_cls, dropout):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.out_dim, n_cls)
        self.drop = dropout

    def forward(self, x, ei):
        return self.head(F.dropout(self.enc(x, ei), self.drop, self.training))


def finetune(init_state, norm, data, train_idx, train_y, seed, enc_lr, a):
    """Supervised training of encoder + linear head. Model selection on val acc
    (ties broken by val loss). Returns full-graph log-probs of the best epoch."""
    set_seed(seed)
    enc = Encoder(data.num_features, a.hidden, a.out, norm=norm)
    if init_state is not None:
        enc.load_state_dict(init_state)
    enc.dropout, enc.in_dropout = 0.5, 0.5
    model = Classifier(enc, len(CLASS_NAMES), 0.5)
    opt = torch.optim.Adam([
        {"params": model.enc.parameters(), "lr": enc_lr},
        {"params": model.head.parameters(), "lr": 1e-2},
    ], weight_decay=5e-4)
    y, val_idx = data.y, data.val_idx
    best_key, best_logp, bad = (-1.0, -math.inf), None, 0
    for _ in range(a.ft_epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        F.cross_entropy(out[train_idx], train_y).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logp = F.log_softmax(model(data.x, data.edge_index), 1)
        va = float((logp[val_idx].argmax(1) == y[val_idx]).float().mean())
        vl = float(F.nll_loss(logp[val_idx], y[val_idx]))
        key = (va, -vl)
        if key > best_key:
            best_key, best_logp, bad = key, logp, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    return best_logp, best_key[0]


def finetune_best_lr(init_state, norm, data, train_idx, train_y, seed, a):
    best = None
    for lr in a.enc_lrs:
        logp, va = finetune(init_state, norm, data, train_idx, train_y, seed, lr, a)
        if best is None or va > best[1]:
            best = (logp, va, lr)
    return best


def make_split(data, k, seed):
    """k labelled nodes per class, drawn from nodes outside the public val/test sets."""
    if k == "public":
        return data.train_mask.nonzero().view(-1)
    g = torch.Generator().manual_seed(1000 + seed)
    pool = ~(data.val_mask | data.test_mask)
    idx = []
    for c in range(len(CLASS_NAMES)):
        cand = (pool & (data.y == c)).nonzero().view(-1)
        idx.append(cand[torch.randperm(cand.numel(), generator=g)[:k]])
    return torch.cat(idx)


# ---------------------------------------------------------------------------
# Post-training
# ---------------------------------------------------------------------------
def self_train(init_state, norm, data, train_idx, seed, enc_lr, base_logp, base_val, a):
    """Class-balanced pseudo-labelling. Each round retrains from the same init on
    labelled + pseudo-labelled nodes. The number of rounds is chosen on val."""
    y = data.y
    excluded = torch.zeros(data.num_nodes, dtype=torch.bool)
    excluded[train_idx] = True
    excluded[data.val_idx] = True  # never pseudo-label the selection set
    trace = [{"round": 0, "val": base_val, "test": acc(base_logp, y, data.test_idx), "n_pseudo": 0}]
    best = (base_val, base_logp, 0)
    cur = base_logp
    for r in range(1, a.st_rounds + 1):
        probs = cur.exp()
        conf, pred = probs.max(1)
        pseudo = []
        for c in range(len(CLASS_NAMES)):
            cand = ((pred == c) & ~excluded & (conf >= a.st_threshold)).nonzero().view(-1)
            top = cand[conf[cand].argsort(descending=True)[: a.st_per_class * r]]
            pseudo.append(top)
        pseudo = torch.cat(pseudo)
        idx = torch.cat([train_idx, pseudo])
        tgt = torch.cat([y[train_idx], pred[pseudo]])
        logp, va = finetune(init_state, norm, data, idx, tgt, seed, enc_lr, a)
        pseudo_acc = float((pred[pseudo] == y[pseudo]).float().mean()) if pseudo.numel() else 0.0
        trace.append({"round": r, "val": va, "test": acc(logp, y, data.test_idx),
                      "n_pseudo": int(pseudo.numel()), "pseudo_label_acc": pseudo_acc})
        if va > best[0]:
            best = (va, logp, r)
        cur = logp
    return best[1], best[0], best[2], trace


def correct_and_smooth(logp, data, train_idx):
    """Correct&Smooth with alphas chosen on val. Returns probabilities."""
    y = data.y
    probs = logp.exp()
    best = (-1, None, None)
    for a1 in [0.5, 0.8, 1.0]:
        for a2 in [0.5, 0.8, 0.9]:
            cs = CorrectAndSmooth(num_correction_layers=50, correction_alpha=a1,
                                  num_smoothing_layers=50, smoothing_alpha=a2, autoscale=True)
            out = cs.correct(probs, y[train_idx], train_idx, data.edge_index)
            out = cs.smooth(out, y[train_idx], train_idx, data.edge_index)
            out = out.clamp_min(1e-9)
            out = out / out.sum(1, keepdim=True)
            va = acc(out, y, data.val_idx)
            if va > best[0]:
                best = (va, out, (a1, a2))
    return best[1], best[0], best[2]


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(e)


def temperature_scale(logp, data):
    """Fit a single temperature on val log-probs (works for logits or C&S output)."""
    vl, vy = logp[data.val_idx], data.y[data.val_idx]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(vl / log_t.exp(), vy)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp())
    return F.log_softmax(logp / t, 1), t


def calib_report(logp, data):
    ti, y = data.test_idx, data.y
    p = logp.exp()
    return {"ece": ece(p[ti], y[ti]), "nll": float(F.nll_loss(logp[ti].clamp_min(-30), y[ti])),
            "acc": acc(p, y, ti)}


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.l1, self.l2 = nn.Linear(i, h), nn.Linear(h, o)

    def forward(self, x):
        x = F.dropout(x, 0.5, self.training)
        return self.l2(F.dropout(F.relu(self.l1(x)), 0.5, self.training))


def train_mlp(data, train_idx, seed, a, teacher=None, lam=1.0):
    """lam * CE(labels) + (1 - lam) * KL(teacher || student) over all nodes."""
    set_seed(seed)
    mlp = MLP(data.num_features, a.hidden, len(CLASS_NAMES))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2, weight_decay=5e-4)
    y = data.y
    best, bad = (-1.0, None), 0
    for _ in range(a.ft_epochs * 2):
        mlp.train()
        opt.zero_grad()
        out = F.log_softmax(mlp(data.x), 1)
        loss = lam * F.nll_loss(out[train_idx], y[train_idx])
        if teacher is not None and lam < 1:
            loss = loss + (1 - lam) * F.kl_div(out, teacher, reduction="batchmean", log_target=False)
        loss.backward()
        opt.step()
        mlp.eval()
        with torch.no_grad():
            lp = F.log_softmax(mlp(data.x), 1)
        va = acc(lp, y, data.val_idx)
        if va > best[0]:
            best, bad = (va, lp), 0
        else:
            bad += 1
            if bad >= a.patience * 2:
                break
    return best[1], best[0], mlp


def distill(teacher_probs, data, train_idx, seed, a):
    mlp_lp, mlp_va, _ = train_mlp(data, train_idx, seed, a)
    best = None
    for lam in [0.0, 0.1, 0.3, 0.5]:
        lp, va, model = train_mlp(data, train_idx, seed, a, teacher=teacher_probs, lam=lam)
        if best is None or va > best[1]:
            best = (lp, va, lam, model)
    return {"mlp_scratch": acc(mlp_lp, data.y, data.test_idx),
            "mlp_distilled": acc(best[0], data.y, data.test_idx),
            "lambda": best[2]}, best[3]


def inference_ms(fn, reps=50):
    with torch.no_grad():
        for _ in range(5):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
    return (time.perf_counter() - t0) / reps * 1000


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", default="1,3,5,10,public",
                    help="labels per class for fine-tuning; 'public' = Planetoid split")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", type=int, default=256)
    ap.add_argument("--ft-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--enc-lrs", default="0.001,0.01")
    ap.add_argument("--pretrain-scale", type=float, default=1.0,
                    help="multiplier on pre-training epochs")
    ap.add_argument("--st-rounds", type=int, default=3)
    ap.add_argument("--st-per-class", type=int, default=40)
    ap.add_argument("--st-threshold", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--out-prefix", default="docs/pretrain_posttrain")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.budgets, a.ft_epochs, a.patience = 1, "3,public", 40, 20
        a.pretrain_scale, a.st_rounds = 0.05, 1
    a.enc_lrs = [float(x) for x in a.enc_lrs.split(",")]
    budgets = [b if b == "public" else int(b) for b in a.budgets.split(",")]
    torch.set_num_threads(a.threads)

    dataset = Planetoid(root="./data/Cora", name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    data.val_idx = data.val_mask.nonzero().view(-1)
    data.test_idx = data.test_mask.nonzero().view(-1)
    y_np = data.y.numpy()
    pcfg = default_pretrain_cfg(a.pretrain_scale)
    t_start = time.time()
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} directed edges, "
          f"{data.num_features} features | seeds={a.seeds} budgets={budgets}")

    # ---------------- Phase 1: self-supervised pre-training ----------------
    print("\n=== Phase 1: self-supervised pre-training ===")
    states = {m: {} for m in METHODS}
    pre = {m: {"probe": [], "nmi": [], "time": [], "loss_curve": None} for m in INITS}
    pre_probe_split = make_split(data, "public", 0)
    for seed in range(a.seeds):
        # Random-init encoder: the "no pre-training" reference point for probes/NMI
        set_seed(seed)
        rnd = Encoder(data.num_features, a.hidden, a.out)
        emb = embed(rnd, data)
        pr = linear_probe(emb, y_np, pre_probe_split.numpy(), data.val_idx.numpy(), data.test_idx.numpy())
        pre["Scratch"]["probe"].append(pr["test"])
        pre["Scratch"]["nmi"].append(kmeans_nmi(emb, y_np, seed))
        pre["Scratch"]["time"].append(0.0)
        for m in METHODS:
            fn, norm = PRETRAINERS[m]
            set_seed(seed)
            enc = Encoder(data.num_features, a.hidden, a.out, norm=norm)
            t0 = time.time()
            hist = fn(enc, data, pcfg[m])
            dt = time.time() - t0
            states[m][seed] = copy.deepcopy(enc.state_dict())
            emb = embed(enc, data)
            pr = linear_probe(emb, y_np, pre_probe_split.numpy(), data.val_idx.numpy(), data.test_idx.numpy())
            nmi = kmeans_nmi(emb, y_np, seed)
            pre[m]["probe"].append(pr["test"])
            pre[m]["nmi"].append(nmi)
            pre[m]["time"].append(dt)
            if seed == 0:
                step = max(1, len(hist) // 100)
                pre[m]["loss_curve"] = [round(v, 5) for v in hist[::step]]
            print(f"  seed {seed} {m:<9} {dt:6.1f}s | linear probe {pr['test']:.4f} | NMI {nmi:.4f}")
    pretrain_summary = {
        m: {"linear_probe": mean_std(pre[m]["probe"]), "nmi": mean_std(pre[m]["nmi"]),
            "time_s": float(np.mean(pre[m]["time"])), "loss_curve": pre[m]["loss_curve"]}
        for m in INITS
    }

    # ---------------- Phase 2: fine-tuning under label budgets ----------------
    print("\n=== Phase 2: fine-tuning (scratch vs pre-trained) ===")
    ft = {str(b): {i: [] for i in INITS} for b in budgets}  # list of dicts per seed
    for b in budgets:
        for seed in range(a.seeds):
            tr = make_split(data, b, seed)
            for init in INITS:
                st = None if init == "Scratch" else states[init][seed]
                norm = init != "Scratch" and PRETRAINERS[init][1]
                logp, va, lr = finetune_best_lr(st, norm, data, tr, data.y[tr], seed, a)
                ft[str(b)][init].append({"logp": logp, "val": va, "lr": lr,
                                         "test": acc(logp, data.y, data.test_idx)})
            row = " ".join(f"{i}={ft[str(b)][i][-1]['test']:.3f}" for i in INITS)
            print(f"  budget {b!s:>6} seed {seed}: {row}")
    ft_summary = {
        b: {i: {"test": mean_std([r["test"] for r in ft[b][i]]),
                "val": mean_std([r["val"] for r in ft[b][i]]),
                "enc_lr": [r["lr"] for r in ft[b][i]]} for i in INITS}
        for b in ft
    }

    # Best pre-training method: highest mean *validation* accuracy across budgets
    best_init = max(METHODS, key=lambda m: np.mean([ft_summary[b][m]["val"]["mean"] for b in ft]))
    print(f"\n  Selected pre-training method (by mean val acc): {best_init}")

    # ---------------- Phase 3: post-training ----------------
    print("\n=== Phase 3: post-training ===")
    post = {}
    calib = {}
    distill_res = {}
    st_traces = {}
    for b in budgets:
        bk = str(b)
        stages = {k: [] for k in ["Scratch", "Pre-trained", "+ Self-training", "+ Correct&Smooth",
                                  "Ensemble (all inits)", "Ensemble + C&S"]}
        rounds_chosen, traces = [], []
        cal_rows = {"before": [], "after": [], "T": []}
        dist_rows = {"mlp_scratch": [], "mlp_distilled": [], "teacher": []}
        for seed in range(a.seeds):
            tr = make_split(data, b, seed)
            y = data.y
            base = ft[bk][best_init][seed]
            norm = PRETRAINERS[best_init][1]
            stages["Scratch"].append(ft[bk]["Scratch"][seed]["test"])
            stages["Pre-trained"].append(base["test"])

            st_logp, st_val, r_best, trace = self_train(
                states[best_init][seed], norm, data, tr, seed, base["lr"], base["logp"], base["val"], a)
            stages["+ Self-training"].append(acc(st_logp, y, data.test_idx))
            rounds_chosen.append(r_best)
            traces.append(trace)

            cs_probs, _, alphas = correct_and_smooth(st_logp, data, tr)
            stages["+ Correct&Smooth"].append(acc(cs_probs, y, data.test_idx))

            ens = torch.stack([ft[bk][i][seed]["logp"].exp() for i in INITS]).mean(0)
            stages["Ensemble (all inits)"].append(acc(ens, y, data.test_idx))
            ens_cs, _, _ = correct_and_smooth(ens.log(), data, tr)
            stages["Ensemble + C&S"].append(acc(ens_cs, y, data.test_idx))

            # Calibration of the final single-model pipeline (log of C&S probs)
            final_logp = cs_probs.log()
            scaled, t = temperature_scale(final_logp, data)
            cal_rows["before"].append(calib_report(final_logp, data))
            cal_rows["after"].append(calib_report(scaled, data))
            cal_rows["T"].append(t)

            # Distillation of the final pipeline into a graph-free MLP
            d, student = distill(scaled.exp(), data, tr, seed, a)
            dist_rows["mlp_scratch"].append(d["mlp_scratch"])
            dist_rows["mlp_distilled"].append(d["mlp_distilled"])
            dist_rows["teacher"].append(acc(cs_probs, y, data.test_idx))
            print(f"  budget {bk:>6} seed {seed}: scratch {stages['Scratch'][-1]:.3f} -> "
                  f"pre {base['test']:.3f} -> ST(r={r_best}) {stages['+ Self-training'][-1]:.3f} -> "
                  f"C&S{alphas} {stages['+ Correct&Smooth'][-1]:.3f} | ens+C&S "
                  f"{stages['Ensemble + C&S'][-1]:.3f} | MLP {d['mlp_scratch']:.3f} -> "
                  f"distilled {d['mlp_distilled']:.3f}")
        post[bk] = {k: mean_std(v) for k, v in stages.items()}
        post[bk]["self_training_rounds"] = rounds_chosen
        st_traces[bk] = traces
        calib[bk] = {
            "before": {k: mean_std([r[k] for r in cal_rows["before"]]) for k in ["ece", "nll", "acc"]},
            "after": {k: mean_std([r[k] for r in cal_rows["after"]]) for k in ["ece", "nll", "acc"]},
            "temperature": mean_std(cal_rows["T"]),
        }
        distill_res[bk] = {k: mean_std(v) for k, v in dist_rows.items()}

    # Inference cost: GCN teacher vs distilled MLP (single forward on full graph)
    enc = Encoder(data.num_features, a.hidden, a.out)
    gcn = Classifier(enc, len(CLASS_NAMES), 0.5).eval()
    mlp = MLP(data.num_features, a.hidden, len(CLASS_NAMES)).eval()
    speed = {"gcn_ms": inference_ms(lambda: gcn(data.x, data.edge_index)),
             "mlp_ms": inference_ms(lambda: mlp(data.x))}

    elapsed = time.time() - t_start
    out = {
        "config": {k: v for k, v in vars(a).items()},
        "pretrain_config": pcfg,
        "budgets": [str(b) for b in budgets],
        "inits": INITS,
        "best_pretrain_method": best_init,
        "pretrain": pretrain_summary,
        "finetune": ft_summary,
        "posttrain": post,
        "self_training_traces": st_traces,
        "calibration": calib,
        "distillation": distill_res,
        "inference_speed": speed,
        "runtime_s": round(elapsed, 1),
    }
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    with open(a.out_prefix + ".json", "w") as f:
        json.dump(out, f)
    md = render_markdown(out)
    with open(a.out_prefix + ".md", "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Done in {elapsed / 60:.1f} min -> {a.out_prefix}.json / .md")


def budget_label(b):
    return "20 (public)" if b == "public" else b


def render_markdown(o):
    L = []
    n = o["config"]["seeds"]
    L.append(f"# Pre-training & post-training results (Cora, {n} seeds, mean ± std test accuracy %)\n")
    L.append("## 1. Self-supervised pre-training (frozen encoder)\n")
    L.append("| Method | Linear probe acc | k-means NMI | Pre-train time |")
    L.append("|---|:---:|:---:|:---:|")
    for m in o["inits"]:
        p = o["pretrain"][m]
        name = "Random init (no pre-training)" if m == "Scratch" else m
        L.append(f"| {name} | {fmt(p['linear_probe'])} | {fmt(p['nmi'], False)} | "
                 f"{p['time_s']:.1f}s |")
    L.append("\n## 2. Fine-tuning by label budget (labels per class)\n")
    L.append("| Init | " + " | ".join(budget_label(b) for b in o["budgets"]) + " |")
    L.append("|---|" + ":---:|" * len(o["budgets"]))
    for i in o["inits"]:
        cells = []
        for b in o["budgets"]:
            col = {j: o["finetune"][b][j]["test"]["mean"] for j in o["inits"]}
            c = fmt(o["finetune"][b][i]["test"])
            cells.append(f"**{c}**" if col[i] == max(col.values()) else c)
        L.append(f"| {i} | " + " | ".join(cells) + " |")
    L.append(f"\nPost-training builds on **{o['best_pretrain_method']}** "
             "(chosen by mean validation accuracy).\n")
    L.append("## 3. Post-training stack\n")
    stages = ["Scratch", "Pre-trained", "+ Self-training", "+ Correct&Smooth",
              "Ensemble (all inits)", "Ensemble + C&S"]
    L.append("| Stage | " + " | ".join(budget_label(b) for b in o["budgets"]) + " |")
    L.append("|---|" + ":---:|" * len(o["budgets"]))
    for s in stages:
        L.append(f"| {s} | " + " | ".join(fmt(o["posttrain"][b][s]) for b in o["budgets"]) + " |")
    L.append("\n## 4. Calibration (temperature scaling on the final single-model pipeline)\n")
    L.append("| Budget | ECE before | ECE after | NLL before | NLL after | T |")
    L.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for b in o["budgets"]:
        c = o["calibration"][b]
        L.append(f"| {budget_label(b)} | {fmt(c['before']['ece'])} | {fmt(c['after']['ece'])} | "
                 f"{c['before']['nll']['mean']:.3f} | {c['after']['nll']['mean']:.3f} | "
                 f"{c['temperature']['mean']:.2f} |")
    L.append("\n## 5. Distillation into a graph-free MLP\n")
    L.append("| Budget | Teacher (GNN pipeline) | MLP from labels | Distilled MLP |")
    L.append("|---|:---:|:---:|:---:|")
    for b in o["budgets"]:
        d = o["distillation"][b]
        L.append(f"| {budget_label(b)} | {fmt(d['teacher'])} | {fmt(d['mlp_scratch'])} | "
                 f"{fmt(d['mlp_distilled'])} |")
    s = o["inference_speed"]
    L.append(f"\nFull-graph inference: GCN {s['gcn_ms']:.1f} ms vs MLP {s['mlp_ms']:.1f} ms "
             f"({s['gcn_ms'] / max(s['mlp_ms'], 1e-9):.1f}x faster, and needs no edges at inference).\n")
    L.append(f"_Total runtime: {o['runtime_s'] / 60:.1f} min on CPU._\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
