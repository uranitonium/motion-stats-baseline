"""The generic recipe for hBABEL (real human motion from AMASS with BABEL's labels): the 25 joints and their velocities
in the body's heading frame (relative to the pelvis horizontally, the hip line along x), then the same window statistics,
z-score and PCA to at most 64 numbers per frame (the evaluator's limit) as the hand-written baseline. Run with
T12_VARIANT=ego (the generic recipe); T12_VARIANT=world keeps the input's coordinates. Nothing is trained with gradient
descent and no label is used to build the representation. Bin edges, moments and PCA are fitted on BABEL's training
sequences only and frozen.

Data: the poses hBehaveMAE itself receives, BehaveMAE's `joint_positions_processed` (25 NTU-order joints in 3D, metres,
30 frames per second): each frame is moved so its joints' centroid is fixed and the whole sequence is turned by one
rotation fitted on its first frame. Lengths are divided by the body's size (median over the sequence of pelvis-to-head
distance plus mean leg length).

Commands
  fit    bin edges, moments and PCA projections for every candidate layout (training poses, no labels)
  dev    cross-validate the candidate layouts on the training sequences and their own labels; pick one
  embed  write a submission for every validation sequence, in the pose file's order (then hbabel/align_hbabel.py relayout)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np

FPS = 30.0
# BehaveMAE's NTU-order joints (datasets/hbabel.py, NTU_KPTS); 21 to 24 (finger tips and thumbs) are not used
(PELVIS, SPINE_MID, NECK, HEAD, L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, R_SHOULDER, R_ELBOW, R_WRIST, R_HAND,
 L_HIP, L_KNEE, L_ANKLE, L_FOOT, R_HIP, R_KNEE, R_ANKLE, R_FOOT, SPINE_SHOULDER) = range(21)
UP = 2                       # the vertical axis
STILL = 0.05                 # m/s: mean joint speed below which the body counts as still
WINDOWS = (15, 61, 241, 901)  # 0.5 s, 2 s, 8 s, 30 s at 30 Hz (odd, centred); the whole sequence is added below
HIST_WINDOWS = (61, 241, 901)
NBINS = 8
FIT_STRIDE = 3
EMB_DIM = 64

VARIANT = os.environ.get("T12_VARIANT", "")
assert VARIANT in ("world", "ego"), "set T12_VARIANT to world or ego"

FEATURE_NAMES = [f"{kind}_{k}_{ax}" for kind in ("pos", "vel") for k in range(25) for ax in "xyz"]


def _ddt(x):
    """Central difference over +-2 frames, per second, edges padded. x (T, ...)."""
    xp = np.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]], 0)
    return (xp[4:] - xp[:-4]) * (FPS / 4.0)


def _norm(v):
    return np.sqrt((v ** 2).sum(-1))


def _unit(v):
    return v / (_norm(v)[..., None] + 1e-9)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _angle(a, b, c):
    """Angle at b between b->a and b->c, radians. (T, 3) each."""
    return np.arccos(np.clip((_unit(a - b) * _unit(c - b)).sum(-1), -1.0, 1.0))


def frame_features(kp):
    """Raw joints in place of the 40 hand-written measurements. kp (T, 25, 3) -> (T, 150).

    world: the 25 joints as BehaveMAE gives them to hBehaveMAE (centroid fixed per frame, one rotation per sequence)
           and their velocities.
    ego:   the same, made relative to the pelvis horizontally and turned about the vertical axis so that the hip line
           points along x (the body's heading frame); velocities are the world velocities turned the same way.
    Lengths are divided by the body's size and velocities are per second, exactly as for the 40 measurements."""
    p = np.nan_to_num(np.asarray(kp, np.float64))
    j = lambda i: p[:, i]  # noqa: E731
    leg = 0.5 * (_norm(j(L_HIP) - j(L_KNEE)) + _norm(j(L_KNEE) - j(L_ANKLE))
                 + _norm(j(R_HIP) - j(R_KNEE)) + _norm(j(R_KNEE) - j(R_ANKLE)))
    size = float(np.median(_norm(j(HEAD) - j(PELVIS)) + leg)) + 1e-6
    vel = _ddt(p)
    if VARIANT == "ego":
        right = j(R_HIP) - j(L_HIP)
        yaw = np.arctan2(right[:, 1], right[:, 0])
        cs, sn = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
        rel = p.copy()
        rel[..., 0] -= j(PELVIS)[:, None, 0]
        rel[..., 1] -= j(PELVIS)[:, None, 1]

        def turn(x):
            out = x.copy()
            out[..., 0] = cs * x[..., 0] - sn * x[..., 1]
            out[..., 1] = sn * x[..., 0] + cs * x[..., 1]
            return out

        p, vel = turn(rel), turn(vel)
    X = np.concatenate([(p / size).reshape(len(p), -1), (vel / size).reshape(len(p), -1)], 1).astype(np.float32)
    assert X.shape[1] == len(FEATURE_NAMES)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------
# window statistics and layouts (as for the MABe22 mice, which also have 60 s clips at 30 Hz)
# ----------------------------------------------------------------------------
def _roll_mean(X, w):
    T = X.shape[0]
    h = w // 2
    cs = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, 0, dtype=np.float64)])
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    return ((cs[hi] - cs[lo]) / (hi - lo)[:, None]).astype(np.float32)


def window_stats(X, edges):
    T, F = X.shape
    blocks = {"inst": X}
    X2 = X.astype(np.float64) ** 2
    for w in WINDOWS:
        m = _roll_mean(X, w)
        s = np.sqrt(np.maximum(_roll_mean(X2, w) - m.astype(np.float64) ** 2, 0.0)).astype(np.float32)
        blocks[f"mean{w}"] = m
        blocks[f"std{w}"] = s
    blocks["meanclip"] = np.repeat(X.mean(0, keepdims=True), T, 0)
    blocks["stdclip"] = np.repeat(X.std(0, keepdims=True), T, 0)
    idx = (X[:, :, None] > edges[None, :, :]).sum(-1)
    onehot = np.zeros((T, F, NBINS), np.float32)
    np.put_along_axis(onehot, idx[:, :, None], 1.0, axis=2)
    onehot = onehot.reshape(T, F * NBINS)
    for w in HIST_WINDOWS:
        blocks[f"hist{w}"] = _roll_mean(onehot, w)
    blocks["histclip"] = np.repeat(onehot.mean(0, keepdims=True), T, 0)
    return blocks


BLOCK_ORDER = (["inst"] + [f"{s}{w}" for w in WINDOWS for s in ("mean", "std")] + ["meanclip", "stdclip"]
               + [f"hist{w}" for w in HIST_WINDOWS] + ["histclip"])
_LOCAL_MS = ["inst", "mean15", "std15", "mean61", "std61", "mean241", "std241"]
_LONG_MS = ["mean901", "std901", "meanclip", "stdclip"]
CONFIGS = {  # the four layouts of the MABe22 mouse baseline, at the benchmark's 64 numbers
    "A_joint_meanstd": [(_LOCAL_MS + _LONG_MS, 64)],
    "B_joint_meanstd_hist": [(_LOCAL_MS + _LONG_MS + ["hist61", "hist241", "hist901", "histclip"], 64)],
    "C_split_meanstd": [(_LOCAL_MS, 32), (_LONG_MS + ["histclip"], 32)],
    "D_split_meanstd_hist": [(_LOCAL_MS + ["hist61", "hist241"], 32), (_LONG_MS + ["hist901", "histclip"], 32)],
}


def stack_blocks(blocks):
    return np.concatenate([blocks[b] for b in BLOCK_ORDER], 1)


def block_slices(F):
    sl, start = {}, 0
    for b in BLOCK_ORDER:
        d = F * NBINS if b.startswith("hist") else F
        sl[b] = np.arange(start, start + d)
        start += d
    return sl, start


def load_poses(path):
    """BehaveMAE's processed pose file (a list of sequences) -> {babel_id: (T, 25, 3)} in the file's order."""
    return {str(d["babel_id"]): d["joint_positions_processed"] for d in joblib.load(path)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def project(Z, fit, cname):
    parts = []
    for gi in range(len(CONFIGS[cname])):
        idx = fit[f"{cname}/g{gi}/idx"]
        z = (Z[:, idx] - fit["mu"][idx]) / fit["sd"][idx]
        parts.append(z @ fit[f"{cname}/g{gi}/W"])
    return np.concatenate(parts, 1).astype(np.float32)


def embed_clip(kp, fit, cnames):
    Z = stack_blocks(window_stats(frame_features(kp), fit["edges"]))
    return {c: project(Z, fit, c) for c in cnames}


def cmd_fit(a):
    t0 = time.time()
    seqs = load_poses(a.train)
    names = sorted(seqs)
    if a.limit:
        names = names[:: max(1, len(names) // a.limit)]
    base = [frame_features(seqs[n]) for n in names]
    F = base[0].shape[1]
    rng = np.random.default_rng(0)
    sample = np.concatenate([b[rng.choice(len(b), size=min(200, len(b)), replace=False)] for b in base], 0)
    qs = np.linspace(0, 1, NBINS + 1)[1:-1]
    edges = np.quantile(sample, qs, axis=0).T.astype(np.float32)
    sl, D = block_slices(F)
    n_acc, s1, s2 = 0, np.zeros(D), np.zeros((D, D))
    for b in base:
        Z = stack_blocks(window_stats(b, edges))[::FIT_STRIDE].astype(np.float64)
        n_acc += len(Z); s1 += Z.sum(0); s2 += Z.T @ Z
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
            out[f"{cname}/g{gi}/idx"] = keep.astype(np.int64)
            out[f"{cname}/g{gi}/W"] = (evecs[:, order] / np.sqrt(np.maximum(evals[order], 1e-8))).astype(np.float32)
            print(f"  {cname} group {gi}: {len(keep)} dims -> {k}, variance kept {evals[order].sum() / evals.sum():.3f}", flush=True)
    np.savez(a.out, **out)
    print(f"fit on {len(names)} training sequences, F={F}; saved {a.out} sha256 {sha256(a.out)} in {time.time() - t0:.0f}s", flush=True)


# Layout choice on the training sequences and their own labels (BehaveMAE's hbabel_..._train_top_120_60_filtered.npy,
# built by the same script as the test labels). The criterion mirrors the evaluator's summary: F1 per task, then for
# the frame-level tasks the means over the 10, 30, 60 and 90 most frequent, for the sequence-level tasks over the 10,
# 30 and all 60 (the label file lists each group from least to most frequent), and the mean of the two group means.
TOP_FRAME = (10, 30, 60, 90)
TOP_SEQ = (10, 30, 60)


def hbabel_score(per, vocab):
    """per: {task: F1}. Returns (score, frame columns, sequence columns), as the evaluator's summary groups them."""
    frame = [t for t in vocab if t.endswith("_frame")]
    seq = [t for t in vocab if t.endswith("_seg")]
    fc = [float(np.mean([per[t] for t in frame[-k:]])) for k in TOP_FRAME]
    sc = [float(np.mean([per[t] for t in seq[-k:]])) for k in TOP_SEQ]
    return float(np.mean([np.mean(fc), np.mean(sc)])), fc, sc


def cmd_dev(a):
    """5-fold CV over the labelled training sequences with the benchmark's classifier, on every third frame."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold

    t0 = time.time()
    fit = dict(np.load(a.fit))
    seqs = load_poses(a.train)
    lab = np.load(a.labels, allow_pickle=True).item()
    vocab = list(lab["vocabulary"])
    names = list(lab["frame_number_map"])
    if a.limit:
        names = names[:: max(1, len(names) // a.limit)]
    cnames = list(CONFIGS)
    X = {c: [] for c in cnames}
    Y, seq_of = [], []
    for i, n in enumerate(names):
        s, en = lab["frame_number_map"][n]
        assert len(seqs[n]) == en - s, n
        e = embed_clip(seqs[n], fit, cnames)
        for c in cnames:
            X[c].append(e[c][::FIT_STRIDE])
        Y.append(lab["label_array"][:, s:en][:, ::FIT_STRIDE].astype(np.int8))
        seq_of.append(np.full(len(range(0, en - s, FIT_STRIDE)), i))
        if i % 500 == 0:
            print(f"  embedded {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    del lab, seqs
    X = {c: np.concatenate(v, 0) for c, v in X.items()}
    Y = np.concatenate(Y, 1)
    seq_of = np.concatenate(seq_of)
    assert (Y >= 0).all()
    folds = [(np.isin(seq_of, tr), np.isin(seq_of, te))
             for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(names)))]
    summary = {"n_sequences": len(names), "n_frames_used": int(len(seq_of)), "vocabulary": vocab}
    for c in cnames:
        per = {}
        for ti, t in enumerate(vocab):
            y = Y[ti]
            pred = np.zeros(len(y), np.int8)
            for trm, tem in folds:
                if y[trm].min() == y[trm].max():
                    continue
                pred[tem] = RidgeClassifier(class_weight="balanced", alpha=1.0).fit(X[c][trm], y[trm]).predict(X[c][tem])
            per[t] = float(f1_score(y, pred, average="binary", zero_division=0))
        score, fc, sc = hbabel_score(per, vocab)
        summary[c] = {"per_task_f1": per, "frame_top": fc, "seq_top": sc, "criterion": score}
        print(c, "frame top 10/30/60/90", [round(x, 4) for x in fc], "seq top 10/30/60", [round(x, 4) for x in sc],
              "criterion", round(score, 4), f"{time.time() - t0:.0f}s", flush=True)
    summary["selected"] = max(cnames, key=lambda c: summary[c]["criterion"])  # ties: first in A, B, C, D order
    json.dump(summary, open(a.out, "w"), indent=1)
    print("selected", summary["selected"], flush=True)


def cmd_embed(a):
    """Every sequence of the pose file, back to back in the file's order (as BehaveMAE's own test script lays them)."""
    t0 = time.time()
    fit = dict(np.load(a.fit))
    seqs = load_poses(a.test)
    total = sum(len(v) for v in seqs.values())
    emb = np.zeros((total, EMB_DIM), np.float32)
    fmap, start = {}, 0
    for i, (n, kp) in enumerate(seqs.items()):
        x = embed_clip(kp, fit, [a.config])[a.config]
        assert x.shape == (len(kp), EMB_DIM)
        emb[start:start + len(kp)] = x
        fmap[n] = (start, start + len(kp))
        start += len(kp)
        if i % 500 == 0:
            print(f"  embedded {i}/{len(seqs)} {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(emb).all()
    np.save(a.out, {"frame_number_map": fmap, "embeddings": emb})
    print(f"saved {a.out} ({total}, {EMB_DIM}) in {time.time() - t0:.0f}s sha256 {sha256(a.out)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--train", required=True); s.add_argument("--out", required=True); s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("dev")
    s.add_argument("--train", required=True); s.add_argument("--fit", required=True); s.add_argument("--labels", required=True)
    s.add_argument("--out", required=True); s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("embed")
    s.add_argument("--test", required=True); s.add_argument("--fit", required=True)
    s.add_argument("--config", required=True); s.add_argument("--out", required=True)
    a = p.parse_args()
    {"fit": cmd_fit, "dev": cmd_dev, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
