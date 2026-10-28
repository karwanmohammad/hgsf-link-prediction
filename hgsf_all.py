"""
HGSF+ -- ONE self-contained file (leakage-free link prediction).
Methods: CN, AA, JC, RA, PA | StructMLP | GraphSAGE | SAGE+Struct (concat) | HGSF (gated fusion)
All learned models receive the same 10 structural features; thresholds are chosen on validation only;
random and distance-matched hard negatives; paired t-tests; per-seed CSV output (tag *_v4).
Usage:
  pip install torch_geometric scikit-learn scipy
  python hgsf_all.py --dataset cora --seeds 5 --neg both --tune
  python hgsf_all.py --dataset facebook --edges facebook_combined.txt.gz --seeds 5 --neg both --tune
"""
import copy
import argparse
import gzip

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, to_undirected

HEURISTICS = ["CN", "AA", "JC", "RA", "PA"]
NEURAL = ["StructMLP", "GraphSAGE", "SAGE+Struct (concat)", "HGSF (gated fusion)"]
MODE = {"StructMLP": "struct", "GraphSAGE": "sage",
        "SAGE+Struct (concat)": "concat", "HGSF (gated fusion)": "hgsf"}
COLS = ["AUC", "AP", "Precision", "Recall", "F1"]


# ----------------------------------------------------------------- data ----
def load_edges(path):
    opener = gzip.open if path.endswith(".gz") else open
    edges = set()
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a, b = map(int, line.split()[:2])
            if a != b:
                edges.add((min(a, b), max(a, b)))
    edges = sorted(edges)
    return edges, max(max(e) for e in edges) + 1


def load_graph(args):
    """returns edges (sorted unique undirected pairs), n, base node features (or None)."""
    if args.dataset == "facebook":
        edges, n = load_edges(args.edges)
        return edges, n, None
    from torch_geometric.datasets import Planetoid
    name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[args.dataset]
    data = Planetoid(root=args.root, name=name)[0]
    edges = sorted({(min(a, b), max(a, b)) for a, b in data.edge_index.t().tolist() if a != b})
    base = None
    if not args.no_feats:
        base = data.x.clone()
        base = base / base.sum(1, keepdim=True).clamp(min=1e-8)
    return edges, data.num_nodes, base


def sample_neg(n, k, exclude, rng, adj_list, hard):
    out = set()
    while len(out) < k:
        if hard:
            u = int(rng.integers(n))
            if not adj_list[u]:
                continue
            w = adj_list[u][rng.integers(len(adj_list[u]))]
            v = adj_list[w][rng.integers(len(adj_list[w]))]
        else:
            u, v = (int(x) for x in rng.integers(n, size=2))
        if u == v:
            continue
        p = (min(u, v), max(u, v))
        if p in exclude or p in out:
            continue
        out.add(p)
    return np.array(sorted(out), dtype=np.int64)






# -------------------------------------------------------------- metrics ----
def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return t[int(np.argmax(f[:-1]))]


def metrics(y, s, thr):
    pred = s >= thr
    tp = float(np.sum(pred & (y == 1)))
    P = tp / max(pred.sum(), 1)
    R = tp / max((y == 1).sum(), 1)
    return {"AUC": roc_auc_score(y, s), "AP": average_precision_score(y, s),
            "Precision": P, "Recall": R, "F1": 2 * P * R / (P + R + 1e-12)}


def eval_heuristics(D):
    res = {}
    for j, name in enumerate(HEURISTICS):
        sv = np.r_[D["raw"]["val_pos"][:, j], D["raw"]["val_neg"][:, j]]
        st = np.r_[D["raw"]["test_pos"][:, j], D["raw"]["test_neg"][:, j]]
        thr = best_f1_threshold(D["val"]["y"], sv)
        res[name] = metrics(D["test"]["y"], st, thr)
    return res


# --------------------------------------------------------------- models ----


def node_x(ei, D):
    deg = degree(ei[0], D["n"])
    return torch.cat([D["base"], torch.log1p(deg).unsqueeze(1)], 1)


@torch.no_grad()
def predict(model, D, split):
    model.eval()
    logit = model(node_x(D["ei_full"], D), D["ei_full"], D[split]["pairs"], D[split]["S"])
    return torch.sigmoid(logit).cpu().numpy()


def train_eval(mode, D, seed, args, dev, cfg):
    torch.manual_seed(seed)
    in_dim = D["base"].shape[1] + 1
    model = Net(mode, cfg["hid"], in_dim, cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    tp, tn, Sp, Sn = D["tp"], D["tn"], D["Sp"], D["Sn"]
    best_auc, best_res, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tp), device=dev)
        ns = int(args.sup_frac * len(tp))
        sup, mp = perm[:ns], perm[ns:]
        ei_mp = to_undirected(tp[mp].t())
        x = node_x(ei_mp, D)
        nidx = torch.randint(len(tn), (ns,), device=dev)
        pairs = torch.cat([tp[sup], tn[nidx]])
        s = torch.cat([Sp[sup], Sn[nidx]])
        y = torch.cat([torch.ones(ns), torch.zeros(ns)]).to(dev)
        loss = F.binary_cross_entropy_with_logits(model(x, ei_mp, pairs, s), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        pv = predict(model, D, "val")
        auc = roc_auc_score(D["val"]["y"], pv)
        if auc > best_auc:
            best_auc, bad = auc, 0
            thr = best_f1_threshold(D["val"]["y"], pv)
            best_res = metrics(D["test"]["y"], predict(model, D, "test"), thr)
            best_res["alpha"] = model.alpha if model.alpha is not None else float("nan")
            best_res["val_auc"] = auc
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best_res




# ------------------------------------------------------------------ main ----
def run_setting(edges, n, base, neg, args, dev):
    tag = f"{args.dataset}_{neg}"
    print(f"\n===== {tag} =====", flush=True)
    allres = {m: [] for m in HEURISTICS + NEURAL}
    cfgs = {m: {"hid": args.hid, "lr": args.lr, "dropout": 0.3} for m in NEURAL}
    for seed in range(args.seeds):
        print(f"--- seed {seed}: building split + features ...", flush=True)
        D = make_data(edges, n, seed, neg, dev, base)
        if args.tune and seed == 0:
            cfgs = tune(D, args, dev)
        for m, r in eval_heuristics(D).items():
            allres[m].append(r)
        for m in NEURAL:
            allres[m].append(train_eval(MODE[m], D, seed, args, dev, cfgs[m]))
            print(f"    {m}: AUC={allres[m][-1]['AUC']:.4f} F1={allres[m][-1]['F1']:.4f}",
                  flush=True)

    with open(f"seeds_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("method,seed," + ",".join(COLS) + "\n")
        for m, rs in allres.items():
            for s, r in enumerate(rs):
                f.write(f'"{m}",{s},' + ",".join(f"{r[c]:.6f}" for c in COLS) + "\n")

    print(f"\n[{tag}]\n{'Method':24s}" + "".join(f"{c:>18s}" for c in COLS))
    lines = ["Method," + ",".join(COLS)]
    for m, rs in allres.items():
        cells = [f"{np.mean([r[c] for r in rs]):.4f} ± {np.std([r[c] for r in rs]):.4f}"
                 for c in COLS]
        print(f"{m:24s}" + "".join(f"{c:>18s}" for c in cells))
        lines.append(f'"{m}",' + ",".join(f'"{c}"' for c in cells))
    with open(f"results_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"HGSF mean gate alpha: {np.mean([r['alpha'] for r in allres['HGSF (gated fusion)']]):.3f}")

    if args.seeds >= 3:
        from scipy import stats
        print("\nPaired t-test, HGSF vs other (p-value; small = real difference)")
        print(f"{'Other method':24s}{'AUC p':>10s}{'F1 p':>10s}")
        for m in HEURISTICS + NEURAL[:-1]:
            ps = []
            for c in ("AUC", "F1"):
                a = np.array([r[c] for r in allres["HGSF (gated fusion)"]])
                b = np.array([r[c] for r in allres[m]])
                ps.append(stats.ttest_rel(a, b).pvalue)
            print(f"{m:24s}{ps[0]:10.4f}{ps[1]:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["facebook", "cora", "citeseer", "pubmed"], default="facebook")
    ap.add_argument("--edges", default="facebook_combined.txt.gz")
    ap.add_argument("--root", default="data")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--neg", choices=["random", "hard", "both"], default="both")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--sup_frac", type=float, default=0.3)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges, n, base = load_graph(args)
    print(f"dataset={args.dataset} nodes={n} edges={len(edges)} "
          f"node_features={'yes' if base is not None else 'no'} device={dev}")
    for neg in (["random", "hard"] if args.neg == "both" else [args.neg]):
        run_setting(edges, n, base, neg, args, dev)


def dist_class(u, v, adj):
    Nu, Nv = adj[u], adj[v]
    if Nu & Nv:
        return 2
    for w in Nu:
        if adj[w] & Nv:
            return 3
    return 4


def walk(u, steps, rng, adj_list):
    for _ in range(steps):
        nb = adj_list[u]
        if not nb:
            return None
        u = nb[rng.integers(len(nb))]
    return u


def class_counts(pairs, adj):
    cnt = {2: 0, 3: 0, 4: 0}
    for u, v in pairs:
        cnt[dist_class(int(u), int(v), adj)] += 1
    return cnt


def sample_matched(n, counts, taken, rng, adj, adj_list):
    out, seen = [], set()
    for c, k in counts.items():
        got, tries = 0, 0
        while got < k:
            tries += 1
            c_eff = c if tries <= 2000 * k else None
            u = int(rng.integers(n))
            if c_eff == 2:
                v = walk(u, 2, rng, adj_list)
            elif c_eff == 3:
                v = walk(u, 3, rng, adj_list)
            else:
                v = int(rng.integers(n))
            if v is None or v == u:
                continue
            p = (min(u, v), max(u, v))
            if p in taken or p in seen:
                continue
            if c_eff is not None and dist_class(u, v, adj) != c_eff:
                continue
            seen.add(p)
            out.append(p)
            got += 1
    return np.array(sorted(out), dtype=np.int64)


NF = 10


def struct_feats10(pairs, adj, remove_target_edge=False):
    deg = [len(s) for s in adj]
    out = np.zeros((len(pairs), NF), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        Nu, Nv, du, dv = adj[u], adj[v], deg[u], deg[v]
        if remove_target_edge and v in Nu:
            Nu, Nv, du, dv = Nu - {v}, Nv - {u}, du - 1, dv - 1
        common = Nu & Nv
        cn = len(common)
        aa = sum(1.0 / np.log(deg[w]) for w in common if deg[w] > 1)
        ra = sum(1.0 / deg[w] for w in common)
        union = len(Nu | Nv)
        jc = cn / union if union else 0.0
        pa = du * dv
        salton = cn / np.sqrt(du * dv) if du > 0 and dv > 0 else 0.0
        hpi = cn / min(du, dv) if min(du, dv) > 0 else 0.0
        small, other = (Nu, Nv) if len(Nu) <= len(Nv) else (Nv, Nu)
        l3 = 0.0
        for w in small:
            for x in adj[w] & other:
                l3 += 1.0 / np.sqrt(deg[w] * deg[x])
        out[i] = [cn, aa, jc, ra, pa, salton, hpi, l3, min(du, dv), max(du, dv)]
    return out


class NetF(nn.Module):
    def __init__(self, mode, hid, in_dim, dropout, nfeat=NF):
        super().__init__()
        self.mode, self.p = mode, dropout
        d = 4 * hid
        if mode != "struct":
            self.c1, self.c2 = SAGEConv(in_dim, hid), SAGEConv(hid, hid)
        if mode == "struct":
            mlp_in = nfeat
        elif mode == "concat":
            mlp_in = d + nfeat
        elif mode == "hgsf":
            self.proj = nn.Linear(nfeat, d)
            self.gate = nn.Linear(2 * d, d)
            mlp_in = d
        else:
            mlp_in = d
        self.mlp = nn.Sequential(nn.Linear(mlp_in, hid), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hid, 1))
        self.alpha = None

    def encode(self, x, ei):
        z = F.relu(self.c1(x, ei))
        return self.c2(F.dropout(z, self.p, self.training), ei)

    def forward(self, x, ei, pairs, s):
        if self.mode == "struct":
            return self.mlp(s).squeeze(-1)
        z = self.encode(x, ei)
        zu, zv = z[pairs[:, 0]], z[pairs[:, 1]]
        h = torch.cat([zu, zv, (zu - zv).abs(), zu * zv], -1)
        if self.mode == "sage":
            f = h
        elif self.mode == "concat":
            f = torch.cat([h, s], -1)
        else:
            st = F.relu(self.proj(s))
            a = torch.sigmoid(self.gate(torch.cat([h, st], -1)))
            self.alpha = a.detach().mean().item()
            f = a * h + (1 - a) * st
        return self.mlp(f).squeeze(-1)


def make_data_v4(edges, n, seed, neg_mode, dev, base):
    rng = np.random.default_rng(seed)
    E = np.array(edges, dtype=np.int64)
    E = E[rng.permutation(len(E))]
    nv, nt = int(0.05 * len(E)), int(0.10 * len(E))
    val_pos, test_pos, train_pos = E[:nv], E[nv:nv + nt], E[nv + nt:]

    adj = [set() for _ in range(n)]
    for u, v in train_pos:
        adj[u].add(int(v))
        adj[v].add(int(u))
    adj_list = [sorted(s) for s in adj]
    taken = set(map(tuple, E.tolist()))

    if neg_mode == "hard":
        cnt_test = class_counts(test_pos, adj)
        cnt_val = class_counts(val_pos, adj)
        print(f"    positives by distance class (2 / 3 / 4+ hops): test {cnt_test}", flush=True)
        test_neg = sample_matched(n, cnt_test, taken, rng, adj, adj_list)
        taken |= set(map(tuple, test_neg.tolist()))
        val_neg = sample_matched(n, cnt_val, taken, rng, adj, adj_list)
        taken |= set(map(tuple, val_neg.tolist()))
        cnt_train = {c: int(round(len(train_pos) * cnt_val[c] / max(len(val_pos), 1))) for c in (2, 3, 4)}
        cnt_train[4] += len(train_pos) - sum(cnt_train.values())
        cnt_train = {c: max(k, 0) for c, k in cnt_train.items()}
        train_neg = sample_matched(n, cnt_train, taken, rng, adj, adj_list)
    else:
        test_neg = sample_neg(n, len(test_pos), taken, rng, adj_list, False)
        taken |= set(map(tuple, test_neg.tolist()))
        val_neg = sample_neg(n, len(val_pos), taken, rng, adj_list, False)
        taken |= set(map(tuple, val_neg.tolist()))
        train_neg = sample_neg(n, len(train_pos), taken, rng, adj_list, False)

    raw = {
        "train_pos": struct_feats10(train_pos, adj, remove_target_edge=True),
        "train_neg": struct_feats10(train_neg, adj),
        "val_pos": struct_feats10(val_pos, adj), "val_neg": struct_feats10(val_neg, adj),
        "test_pos": struct_feats10(test_pos, adj), "test_neg": struct_feats10(test_neg, adj),
    }

    def transform(a):
        a = a.copy()
        for c in (0, 1, 3, 4, 7, 8, 9):
            a[:, c] = np.log1p(a[:, c])
        return a

    tr_all = transform(np.vstack([raw["train_pos"], raw["train_neg"]]))
    mu, sd = tr_all.mean(0), tr_all.std(0) + 1e-8

    def norm(a):
        return torch.tensor((transform(a) - mu) / sd, dtype=torch.float32, device=dev)

    T = lambda a: torch.tensor(a, dtype=torch.long, device=dev)
    D = {
        "n": n, "ei_full": to_undirected(T(train_pos).t()),
        "tp": T(train_pos), "tn": T(train_neg),
        "Sp": norm(raw["train_pos"]), "Sn": norm(raw["train_neg"]),
        "raw": raw,
        "base": (base.to(dev) if base is not None else torch.ones(n, 1, device=dev)),
    }
    for name, pos, neg in (("val", val_pos, val_neg), ("test", test_pos, test_neg)):
        D[name] = {
            "pairs": T(np.vstack([pos, neg])),
            "S": norm(np.vstack([raw[name + "_pos"], raw[name + "_neg"]])),
            "y": np.r_[np.ones(len(pos)), np.zeros(len(neg))],
        }
    return D


_orig_train_eval = train_eval


def train_eval_w(mode, D, seed, args, dev, cfg):
    a = copy.copy(args)
    a.sup_frac = cfg.get("sup_frac", args.sup_frac)
    return _orig_train_eval(mode, D, seed, a, dev, cfg)


CONFIGS = [
    {"hid": 64, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.3},
    {"hid": 128, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.3},
    {"hid": 64, "lr": 0.001, "dropout": 0.3, "sup_frac": 0.3},
    {"hid": 128, "lr": 0.001, "dropout": 0.3, "sup_frac": 0.3},
    {"hid": 64, "lr": 0.005, "dropout": 0.5, "sup_frac": 0.3},
    {"hid": 128, "lr": 0.005, "dropout": 0.5, "sup_frac": 0.3},
    {"hid": 64, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.1},
    {"hid": 128, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.1},
]


def tune_v4(D, args, dev):
    chosen = {}
    for name in NEURAL:
        best = max(CONFIGS, key=lambda c: train_eval_w(MODE[name], D, 0, args, dev, c)["val_auc"])
        chosen[name] = dict(best)
        print(f"    tuned {name}: {best}", flush=True)
    return chosen


_orig_run_setting = run_setting


def run_setting_v4(edges, n, base, neg, args, dev):
    a = copy.copy(args)
    a.dataset = args.dataset + "_v4"
    return _orig_run_setting(edges, n, base, neg, a, dev)


# ---- replace the earlier (5-feature) versions by the HGSF+ versions ----
Net = NetF
make_data = make_data_v4
train_eval = train_eval_w
tune = tune_v4
run_setting = run_setting_v4

if __name__ == "__main__":
    main()
