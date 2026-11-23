"""No-training baseline for CalMS21 Task 1.

The 23 generic two-mouse features of features.py, then per feature
the value itself plus mean, standard deviation and an 8-bin histogram over centred windows of 0.5, 2, 8 and 30 s,
then z-score and PCA to 128 numbers per frame. Bin edges, z-score moments and PCA are fitted on the unlabeled
videos and the training videos only. No label is used and nothing is trained by gradient descent.

Commands
  fit    fit and freeze the statistics for all candidate layouts
  embed  write per-video embeddings (T, 128) for one layout and one split
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local modules, also under python -I
import features
from t2_common import (EMB_DIM, NBINS, WINDOWS, block_slices, load_video, stack_blocks, video_files, window_stats)

FIT_STRIDE = 3
_MS_SHORT = ["inst", "mean15", "std15", "mean61", "std61"]
_MS_LONG = ["mean241", "std241", "mean901", "std901"]
CONFIGS = {
    "A_joint_meanstd": [(_MS_SHORT + _MS_LONG, 128)],
    "B_joint_meanstd_hist": [(_MS_SHORT + _MS_LONG + [f"hist{w}" for w in WINDOWS], 128)],
    "C_split_meanstd": [(_MS_SHORT, 64), (_MS_LONG, 64)],
    "D_split_meanstd_hist": [(_MS_SHORT + ["hist15", "hist61"], 64), (_MS_LONG + ["hist241", "hist901"], 64)],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_features(path):
    kp, sc, _ = load_video(path)
    X, _ = features.build(kp, sc)
    return X


def cmd_fit(a):
    t0 = time.time()
    files = video_files(a.npz, "unlabeled") + video_files(a.npz, "train")
    if a.limit:
        files = files[:: max(1, len(files) // a.limit)]
    print(f"fit on {len(files)} unlabeled + training videos", flush=True)
    rng = np.random.default_rng(0)
    base, sample = [], []
    for i, (_, f) in enumerate(files):
        X = base_features(f)
        base.append(X)
        sample.append(X[rng.choice(len(X), size=min(2000, len(X)), replace=False)])
    sample = np.concatenate(sample, 0)
    F = sample.shape[1]
    qs = np.linspace(0, 1, NBINS + 1)[1:-1]
    edges = np.quantile(sample, qs, axis=0).T.astype(np.float32)
    print(f"F={F}, frames={sum(len(b) for b in base)}, features in {time.time() - t0:.0f}s", flush=True)
    sl, D = block_slices(F)
    n_acc, s1, s2 = 0, np.zeros(D), np.zeros((D, D))
    for i, b in enumerate(base):
        Z = stack_blocks(window_stats(b, edges))[::FIT_STRIDE].astype(np.float64)
        n_acc += len(Z)
        s1 += Z.sum(0)
        s2 += Z.T @ Z
        if i % 50 == 0:
            print(f"  moments {i}/{len(base)} {time.time() - t0:.0f}s", flush=True)
    mu = s1 / n_acc
    cov = s2 / n_acc - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    out = {"edges": edges, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "F": np.int64(F)}
    for cname, groups in CONFIGS.items():
        for gi, (blocks, k) in enumerate(groups):
            idx = np.concatenate([sl[b] for b in blocks])
            keep = idx[sd[idx] > 1e-6]
            corr = cov[np.ix_(keep, keep)] / np.outer(sd[keep], sd[keep])
            evals, evecs = np.linalg.eigh(corr)
            order = np.argsort(evals)[::-1][:k]
            W = evecs[:, order] / np.sqrt(np.maximum(evals[order], 1e-8))
            out[f"{cname}/g{gi}/idx"] = keep.astype(np.int64)
            out[f"{cname}/g{gi}/W"] = W.astype(np.float32)
            print(f"  {cname} group {gi}: {len(keep)} dims -> {W.shape[1]}, variance kept "
                  f"{evals[order].sum() / evals.sum():.3f}", flush=True)
    np.savez(a.out, **out)
    print(f"saved {a.out} sha256 {sha256(a.out)} in {time.time() - t0:.0f}s", flush=True)


def project(Z, fit, cname):
    parts = []
    for gi in range(len(CONFIGS[cname])):
        idx = fit[f"{cname}/g{gi}/idx"]
        z = (Z[:, idx] - fit["mu"][idx]) / fit["sd"][idx]
        parts.append(z @ fit[f"{cname}/g{gi}/W"])
    return np.concatenate(parts, 1).astype(np.float32)


def cmd_embed(a):
    fit = dict(np.load(a.fit))
    cnames = a.config.split(",")
    for split in a.splits.split(","):
        files = video_files(a.npz, split)
        for c in cnames:
            os.makedirs(os.path.join(a.out, c, split), exist_ok=True)
        for name, f in files:
            Z = stack_blocks(window_stats(base_features(f), fit["edges"]))
            for c in cnames:
                e = project(Z, fit, c)
                assert e.shape[1] <= EMB_DIM and np.isfinite(e).all()
                np.save(os.path.join(a.out, c, split, name + ".npy"), e)
        print(f"embedded {len(files)} {split} videos for {cnames}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--npz", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("embed")
    s.add_argument("--npz", required=True)
    s.add_argument("--fit", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--splits", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args()
    {"fit": cmd_fit, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
