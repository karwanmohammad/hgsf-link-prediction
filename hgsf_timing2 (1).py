"""
Cleaner timing study (needs hgsf_all.py in the same folder).
Training cost is measured as seconds per epoch over a fixed number of epochs (no early stopping),
after a warm-up, so that it does not depend on how early a model stops.
Usage:  python hgsf_timing2.py --dataset cora --seeds 3
"""
import argparse
import copy
import time

import numpy as np
import torch

import hgsf_all as H

CFG = {"hid": 64, "lr": 0.005, "dropout": 0.3, "sup_frac": 0.3}
H.Net = lambda mode, hid, in_dim, dropout: H.NetF(mode, hid, in_dim, dropout, nfeat=10)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["facebook", "cora", "citeseer", "pubmed"], default="cora")
    ap.add_argument("--edges", default="facebook_combined.txt.gz")
    ap.add_argument("--root", default="data")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--timed_epochs", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.epochs, args.patience, args.sup_frac, args.hid, args.lr = args.timed_epochs, 10 ** 9, 0.3, 64, 0.005
    print(f"timing dataset={args.dataset} device={dev} timed_epochs={args.epochs}", flush=True)

    edges, n, base = H.load_graph(args)
    per_epoch = {m: [] for m in H.NEURAL}
    feat_ms = []
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
        for m in H.NEURAL:                      # warm-up (not timed)
            w = copy.copy(args)
            w.epochs = 2
            H.train_eval(H.MODE[m], D, seed, w, dev, CFG)
        for m in H.NEURAL:
            sync(dev)
            t0 = time.perf_counter()
            H.train_eval(H.MODE[m], D, seed, args, dev, CFG)
            sync(dev)
            per_epoch[m].append((time.perf_counter() - t0) / args.epochs)
            print(f"seed {seed} {m}: {per_epoch[m][-1]:.3f} s/epoch", flush=True)

    in_dim = D["base"].shape[1] + 1
    n_test = len(D["test"]["pairs"])
    rows = []
    for m in H.NEURAL:
        model = H.Net(H.MODE[m], CFG["hid"], in_dim, CFG["dropout"]).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        H.predict(model, D, "test")
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            H.predict(model, D, "test")
        sync(dev)
        total_ms = (time.perf_counter() - t0) / 10 * 1000
        rows.append((m, n_par, np.mean(per_epoch[m]), np.std(per_epoch[m]), total_ms))

    print(f"\n[timing {args.dataset}] device={dev}, test set = {n_test} pairs")
    print(f"{'Model':24s}{'#params':>12s}{'s / epoch':>18s}{'inference, whole test set (ms)':>34s}")
    for m, p, tm, ts, tot in rows:
        print(f"{m:24s}{p:12d}{tm:11.3f} ± {ts:5.3f}{tot:34.1f}")
    print(f"\nComputing the 10 structural features: {np.mean(feat_ms):.1f} ms per 1000 pairs (CPU, Python)")
    with open(f"timing2_{args.dataset}.csv", "w", encoding="utf-8") as f:
        f.write("model,params,s_per_epoch_mean,s_per_epoch_std,inference_ms_whole_test_set,test_pairs\n")
        for m, p, tm, ts, tot in rows:
            f.write(f'"{m}",{p},{tm:.4f},{ts:.4f},{tot:.2f},{n_test}\n')
        f.write(f"features_10_ms_per_1000_pairs,,,,{np.mean(feat_ms):.3f},\n")


if __name__ == "__main__":
    main()
