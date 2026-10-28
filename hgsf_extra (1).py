"""
Extra experiments for the paper (needs hgsf_all.py in the same folder).
  --task ablation : HGSF with 5 classic features / 9 features (no L3) / all 10 features,
                    same splits, same negatives, same default hyperparameters for every variant.
  --task timing   : parameters, training time, inference time and feature-computation time.
Usage:
  python hgsf_extra.py --task ablation --dataset cora --seeds 5 --neg both
  python hgsf_extra.py --task timing   --dataset cora --seeds 3
"""
import argparse
import time

import numpy as np
import torch

import hgsf_all as H

VARIANTS = {
    "5 classic features": [0, 1, 2, 3, 4],
    "9 features (no L3)": [0, 1, 2, 3, 4, 5, 6, 8, 9],
    "10 features (full)": list(range(10)),
}
CFG = {"hid": 64, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.3}
CUR_N = [10]
# the model must know how many structural features it receives
H.Net = lambda mode, hid, in_dim, dropout: H.NetF(mode, hid, in_dim, dropout, nfeat=CUR_N[0])


def slice_D(D, idx):
    E = dict(D)
    E["Sp"] = D["Sp"][:, idx]
    E["Sn"] = D["Sn"][:, idx]
    E["val"] = dict(D["val"])
    E["val"]["S"] = D["val"]["S"][:, idx]
    E["test"] = dict(D["test"])
    E["test"]["S"] = D["test"]["S"][:, idx]
    return E


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_ablation(args, dev):
    from scipy import stats
    edges, n, base = H.load_graph(args)
    for neg in (["random", "hard"] if args.neg == "both" else [args.neg]):
        tag = f"{args.dataset}_{neg}"
        print(f"\n===== ablation {tag} =====", flush=True)
        res = {v: [] for v in VARIANTS}
        for seed in range(args.seeds):
            print(f"--- seed {seed}", flush=True)
            D = H.make_data(edges, n, seed, neg, dev, base)
            for v, idx in VARIANTS.items():
                CUR_N[0] = len(idx)
                r = H.train_eval("hgsf", slice_D(D, idx), seed, args, dev, CFG)
                res[v].append(r)
                print(f"    {v}: AUC={r['AUC']:.4f} AP={r['AP']:.4f} F1={r['F1']:.4f}", flush=True)
        cols = ["AUC", "AP", "F1"]
        with open(f"ablation_seeds_{tag}.csv", "w", encoding="utf-8") as f:
            f.write("variant,seed," + ",".join(cols) + "\n")
            for v, rs in res.items():
                for s, r in enumerate(rs):
                    f.write(f'"{v}",{s},' + ",".join(f"{r[c]:.6f}" for c in cols) + "\n")
        print(f"\n[ablation {tag}]\n{'Variant':24s}" + "".join(f"{c:>18s}" for c in cols))
        for v, rs in res.items():
            cells = [f"{np.mean([r[c] for r in rs]):.4f} ± {np.std([r[c] for r in rs]):.4f}" for c in cols]
            print(f"{v:24s}" + "".join(f"{c:>18s}" for c in cells))
        if args.seeds >= 3:
            full = np.array([r["AUC"] for r in res["10 features (full)"]])
            print("\nPaired t-test on AUC: 10 features (full) vs ...")
            for v in ("5 classic features", "9 features (no L3)"):
                other = np.array([r["AUC"] for r in res[v]])
                print(f"   {v:24s} p = {stats.ttest_rel(full, other).pvalue:.4f}")


def run_timing(args, dev):
    edges, n, base = H.load_graph(args)
    train_t = {m: [] for m in H.NEURAL}
    feat_ms = []
    CUR_N[0] = 10
    D = None
    for seed in range(args.seeds):
        D = H.make_data(edges, n, seed, "random", dev, base)
        adj = [set() for _ in range(n)]
        for u, v in D["tp"].cpu().numpy():
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))
        pairs = D["test"]["pairs"][:5000].cpu().numpy()
        t0 = time.perf_counter()
        H.struct_feats10(pairs, adj)
        feat_ms.append((time.perf_counter() - t0) / len(pairs) * 1e6)
        for m in H.NEURAL:
            sync(dev)
            t0 = time.perf_counter()
            H.train_eval(H.MODE[m], D, seed, args, dev, CFG)
            sync(dev)
            train_t[m].append(time.perf_counter() - t0)
            print(f"seed {seed} {m}: {train_t[m][-1]:.1f} s", flush=True)
    in_dim = D["base"].shape[1] + 1
    rows = []
    for m in H.NEURAL:
        model = H.Net(H.MODE[m], CFG["hid"], in_dim, CFG["dropout"]).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        H.predict(model, D, "test")  # warm-up
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            H.predict(model, D, "test")
        sync(dev)
        inf = (time.perf_counter() - t0) / 10 / len(D["test"]["pairs"]) * 1e6
        rows.append((m, n_par, np.mean(train_t[m]), np.std(train_t[m]), inf))
    print(f"\n[timing {args.dataset}] device={dev}")
    print(f"{'Model':24s}{'#params':>12s}{'train time (s)':>20s}{'inference (ms/1000 pairs)':>28s}")
    for m, p, tm, ts, inf in rows:
        print(f"{m:24s}{p:12d}{tm:14.1f} ± {ts:4.1f}{inf:28.2f}")
    print(f"\nComputing the 10 structural features: {np.mean(feat_ms):.1f} ms per 1000 pairs (CPU, Python)")
    with open(f"timing_{args.dataset}.csv", "w", encoding="utf-8") as f:
        f.write("model,params,train_s_mean,train_s_std,inference_ms_per_1000_pairs\n")
        for m, p, tm, ts, inf in rows:
            f.write(f'"{m}",{p},{tm:.2f},{ts:.2f},{inf:.3f}\n')
        f.write(f"features_10_ms_per_1000_pairs,,,,{np.mean(feat_ms):.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ablation", "timing"], required=True)
    ap.add_argument("--dataset", choices=["facebook", "cora", "citeseer", "pubmed"], default="cora")
    ap.add_argument("--edges", default="facebook_combined.txt.gz")
    ap.add_argument("--root", default="data")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--neg", choices=["random", "hard", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--sup_frac", type=float, default=0.3)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"task={args.task} dataset={args.dataset} device={dev}")
    if args.task == "ablation":
        run_ablation(args, dev)
    else:
        run_timing(args, dev)


if __name__ == "__main__":
    main()
