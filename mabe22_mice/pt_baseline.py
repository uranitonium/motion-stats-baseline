"""No-training baseline for MABe22 mouse triplets.

Pipeline: generic per-frame pose and relational features, pooled over the three
mice in a permutation-invariant way; statistics of those features over fixed
time windows (mean, spread, histogram); z-score; PCA to at most 128 dimensions.
Nothing is trained with gradient descent and no behaviour label is used.

All statistics (z-score, histogram bin edges, PCA) are fitted on the benchmark's
user-train clips only (mouse_triplet_train.npy) and then frozen.

Commands
  fit    fit bin edges, moments and PCA projections for every candidate config
  dev    cross-validate the candidate configs on the user-train labels
  embed  write a benchmark submission file for the test clips with one config
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

NOSE, EAR_L, EAR_R, NECK, FPAW_L, FPAW_R, CENTER, HPAW_L, HPAW_R, TAIL_BASE, TAIL_MID, TAIL_TIP = range(12)
ARENA_C = 425.0  # centre of the 850 px grid used by the benchmark
WINDOWS = (15, 61, 241, 901)  # 0.5 s, 2 s, 8 s, 30 s at 30 Hz (odd, centred)
HIST_WINDOWS = (61, 241, 901)
NBINS = 8
FIT_STRIDE = 3  # frames used for fitting moments and PCA: every 3rd frame
EMB_DIM = 128


# ----------------------------------------------------------------------------
# per-frame features
# ----------------------------------------------------------------------------
def fill_missing(kp):
    """kp (T, M, K, 2) raw pixels, (0, 0) = missing. Linear interpolation in time.

    Returns filled float32 keypoints and a (T, M) flag for 'mouse not tracked'.
    """
    kp = kp.astype(np.float32).copy()
    T, M, K, _ = kp.shape
    miss = (kp == 0).all(-1)  # (T, M, K)
    t = np.arange(T)
    for m in range(M):
        for k in range(K):
            bad = miss[:, m, k]
            if bad.any() and not bad.all():
                good = ~bad
                for c in range(2):
                    kp[bad, m, k, c] = np.interp(t[bad], t[good], kp[good, m, k, c])
    for m in range(M):
        never = miss[:, m, :].all(0)
        if never.any():
            if (~never).any():
                fill = kp[:, m, ~never, :].mean(1, keepdims=True)
            else:
                fill = np.full((T, 1, 2), ARENA_C, np.float32)
            kp[:, m, never, :] = fill
    return kp, miss.any(-1)


def _ddt(x):
    """Central difference over +-2 frames, per frame, edges padded. x (T, ...)."""
    xp = np.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]], 0)
    return (xp[4:] - xp[:-4]) / 4.0


def _norm(v):
    return np.sqrt((v ** 2).sum(-1))


def _unit(v):
    return v / (_norm(v)[..., None] + 1e-6)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def frame_features(kp_raw):
    """kp_raw (T, 3, 12, 2) -> (T, F) float32 generic features, mouse-order invariant."""
    kp, miss = fill_missing(kp_raw)
    T, M, K, _ = kp.shape
    nose, neck, cen, tb = kp[:, :, NOSE], kp[:, :, NECK], kp[:, :, CENTER], kp[:, :, TAIL_BASE]

    v_c = _ddt(cen)
    v_n = _ddt(nose)
    v_t = _ddt(tb)
    head_vec = nose - neck
    body_vec = cen - tb
    axis_vec = nose - tb
    head_ang = np.arctan2(head_vec[..., 1], head_vec[..., 0])
    body_ang = np.arctan2(body_vec[..., 1], body_vec[..., 0])

    def ang_speed(a):
        ap = np.concatenate([a[:1], a[:1], a, a[-1:], a[-1:]], 0)
        return np.abs(_wrap(ap[4:] - ap[:-4])) / 4.0

    per_mouse = [
        np.log1p(_norm(v_c)),  # centre speed
        np.log1p(_norm(v_n)),  # nose speed
        np.log1p(_norm(v_t)),  # tail-base speed
        np.log1p(_norm(_ddt(v_c))),  # acceleration
        _norm(axis_vec) / 100.0,  # body length
        _norm(kp[:, :, EAR_L] - kp[:, :, EAR_R]) / 100.0,  # head width
        _norm(neck - cen) / 100.0,  # front length
        _norm(cen - tb) / 100.0,  # back length
        (_norm(tb - kp[:, :, TAIL_MID]) + _norm(kp[:, :, TAIL_MID] - kp[:, :, TAIL_TIP])) / 100.0,  # tail length
        _norm(kp[:, :, FPAW_L] - kp[:, :, FPAW_R]) / 100.0,  # forepaw spread
        _norm(kp[:, :, HPAW_L] - kp[:, :, HPAW_R]) / 100.0,  # hindpaw spread
        ang_speed(head_ang),  # head turning speed
        ang_speed(body_ang),  # body turning speed
        (_unit(head_vec) * _unit(body_vec)).sum(-1),  # head-body alignment
        _norm(cen - ARENA_C) / ARENA_C,  # distance from arena centre
        (cen[..., 0] - ARENA_C) / ARENA_C,  # x
        (cen[..., 1] - ARENA_C) / ARENA_C,  # y
        _slog((v_c * _unit(axis_vec)).sum(-1)),  # forward speed
        miss.astype(np.float32),  # not tracked
    ]
    pm = np.stack(per_mouse, -1)  # (T, M, Fm)
    pm = np.sort(pm, axis=1)  # min, mid, max over mice, per feature
    feats = [pm.reshape(T, -1)]

    sym, dire = [], []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        d_cc = _norm(cen[:, i] - cen[:, j])
        diff = kp[:, i][:, :, None, :] - kp[:, j][:, None, :, :]  # (T, K, K, 2)
        d_min = np.sqrt((diff ** 2).sum(-1)).reshape(T, -1).min(1)
        sym.append(np.stack([
            d_cc / 100.0,
            _norm(nose[:, i] - nose[:, j]) / 100.0,
            d_min / 100.0,
            _slog(_ddt(d_cc)),  # closing speed (negative = approaching)
            (_unit(axis_vec[:, i]) * _unit(axis_vec[:, j])).sum(-1),  # relative heading
        ], -1))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            to_j = cen[:, j] - nose[:, i]
            dire.append(np.stack([
                _norm(nose[:, i] - tb[:, j]) / 100.0,
                np.minimum(_norm(nose[:, i] - kp[:, j, EAR_L]), _norm(nose[:, i] - kp[:, j, EAR_R])) / 100.0,
                _norm(to_j) / 100.0,
                (_unit(head_vec[:, i]) * _unit(to_j)).sum(-1),  # i facing j
                _slog((v_c[:, i] * _unit(cen[:, j] - cen[:, i])).sum(-1)),  # i moving toward j
            ], -1))
    sym = np.sort(np.stack(sym, 1), axis=1)  # (T, 3, 5) min, mid, max over pairs
    feats.append(sym.reshape(T, -1))
    dire = np.stack(dire, 1)  # (T, 6, 5)
    feats.append(dire.min(1))
    feats.append(dire.max(1))
    feats.append(dire.mean(1))

    c0, c1, c2 = cen[:, 0], cen[:, 1], cen[:, 2]
    area = 0.5 * np.abs((c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) - (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1]))
    feats.append((area / 1e4)[:, None])
    feats.append(miss.sum(1, keepdims=True).astype(np.float32))
    out = np.concatenate(feats, 1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------
# window statistics
# ----------------------------------------------------------------------------
def _roll_mean(X, w):
    """Centred rolling mean of width w, truncated at the clip edges. X (T, D)."""
    T = X.shape[0]
    h = w // 2
    cs = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, 0, dtype=np.float64)])
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    return ((cs[hi] - cs[lo]) / (hi - lo)[:, None]).astype(np.float32)


def window_stats(X, edges):
    """X (T, F) per-frame features -> dict of named blocks, each (T, d) float32."""
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
    idx = (X[:, :, None] > edges[None, :, :]).sum(-1)  # (T, F) bin index 0..NBINS-1
    onehot = np.zeros((T, F, NBINS), np.float32)
    np.put_along_axis(onehot, idx[:, :, None], 1.0, axis=2)
    onehot = onehot.reshape(T, F * NBINS)
    for w in HIST_WINDOWS:
        blocks[f"hist{w}"] = _roll_mean(onehot, w)
    blocks["histclip"] = np.repeat(onehot.mean(0, keepdims=True), T, 0)
    return blocks


BLOCK_ORDER = (["inst"] + [f"{s}{w}" for w in WINDOWS for s in ("mean", "std")] + ["meanclip", "stdclip"]
               + [f"hist{w}" for w in HIST_WINDOWS] + ["histclip"])

# Candidate configurations. Each is a list of (block names, output dims) groups;
# every group gets its own z-score + PCA, and the groups are concatenated.
_LOCAL_MS = ["inst", "mean15", "std15", "mean61", "std61", "mean241", "std241"]
_LONG_MS = ["mean901", "std901", "meanclip", "stdclip"]
CONFIGS = {
    "A_joint_meanstd": [(_LOCAL_MS + _LONG_MS, 128)],
    "B_joint_meanstd_hist": [(_LOCAL_MS + _LONG_MS + ["hist61", "hist241", "hist901", "histclip"], 128)],
    "C_split_meanstd": [(_LOCAL_MS, 64), (_LONG_MS + ["histclip"], 64)],
    "D_split_meanstd_hist": [(_LOCAL_MS + ["hist61", "hist241"], 64), (_LONG_MS + ["hist901", "histclip"], 64)],
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


# ----------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------
def load_sequences(path):
    d = np.load(path, allow_pickle=True).item()
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def cmd_fit(args):
    t0 = time.time()
    d = load_sequences(args.train)
    names = sorted(d["sequences"].keys())
    if args.limit:
        names = names[: args.limit]
    print(f"fit on {len(names)} user-train clips", flush=True)

    # pass 1: per-frame features (cached) and histogram bin edges
    base = []
    for n in names:
        base.append(frame_features(d["sequences"][n]["keypoints"]))
    F = base[0].shape[1]
    rng = np.random.default_rng(0)
    sample = np.concatenate([b[rng.choice(len(b), size=min(200, len(b)), replace=False)] for b in base], 0)
    qs = np.linspace(0, 1, NBINS + 1)[1:-1]
    edges = np.quantile(sample, qs, axis=0).T.astype(np.float32)  # (F, NBINS-1)
    print(f"F={F}, features cached in {time.time() - t0:.0f}s", flush=True)

    # pass 2: moments of the full statistics vector
    sl, D = block_slices(F)
    n_acc, s1, s2 = 0, np.zeros(D), np.zeros((D, D))
    for i, b in enumerate(base):
        Z = stack_blocks(window_stats(b, edges))[::FIT_STRIDE].astype(np.float64)
        n_acc += len(Z)
        s1 += Z.sum(0)
        s2 += Z.T @ Z
        if i % 100 == 0:
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
            W = evecs[:, order] / np.sqrt(np.maximum(evals[order], 1e-8))  # whitened scores
            out[f"{cname}/g{gi}/idx"] = keep.astype(np.int64)
            out[f"{cname}/g{gi}/W"] = W.astype(np.float32)
            ev = evals[order].sum() / evals.sum()
            print(f"  {cname} group {gi}: {len(keep)} dims -> {k}, variance kept {ev:.3f}", flush=True)
    np.savez(args.out, **out)
    print(f"saved {args.out} sha256 {sha256(args.out)} in {time.time() - t0:.0f}s", flush=True)


def project(Z, fit, cname):
    parts = []
    for gi in range(len(CONFIGS[cname])):
        idx = fit[f"{cname}/g{gi}/idx"]
        W = fit[f"{cname}/g{gi}/W"]
        z = (Z[:, idx] - fit["mu"][idx]) / fit["sd"][idx]
        parts.append(z @ W)
    return np.concatenate(parts, 1).astype(np.float32)


def embed_clip(kp, fit, cnames):
    Z = stack_blocks(window_stats(frame_features(kp), fit["edges"]))
    return {c: project(Z, fit, c) for c in cnames}


# ----------------------------------------------------------------------------
# dev: cross-validation on the user-train labels (chases, lights)
# ----------------------------------------------------------------------------
def cmd_dev(args):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold

    t0 = time.time()
    fit = dict(np.load(args.fit))
    d = load_sequences(args.train)
    vocab = d["vocabulary"]
    names = sorted(d["sequences"].keys())
    if args.limit:
        names = names[: args.limit]
    cnames = list(CONFIGS)
    T = d["sequences"][names[0]]["keypoints"].shape[0]
    N = len(names)
    os.makedirs(args.tmp, exist_ok=True)
    mm = {c: np.lib.format.open_memmap(os.path.join(args.tmp, f"dev_{c}.npy"), mode="w+", dtype=np.float32,
                                       shape=(N * T, EMB_DIM)) for c in cnames}
    Y = np.zeros((len(vocab), N * T), np.float32)
    for i, n in enumerate(names):
        s = d["sequences"][n]
        e = embed_clip(s["keypoints"], fit, cnames)
        for c in cnames:
            mm[c][i * T:(i + 1) * T] = e[c]
        Y[:, i * T:(i + 1) * T] = s["annotations"]
        if i % 200 == 0:
            print(f"  embedded {i}/{N} {time.time() - t0:.0f}s", flush=True)
    clip_of = np.repeat(np.arange(N), T)
    res = {c: {v: [] for v in vocab} for c in cnames}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(N))):
        trm, tem = np.isin(clip_of, tr), np.isin(clip_of, te)
        for c in cnames:
            Xtr, Xte = np.asarray(mm[c][trm]), np.asarray(mm[c][tem])
            for vi, v in enumerate(vocab):
                clf = RidgeClassifier(class_weight="balanced", alpha=1.0).fit(Xtr, Y[vi, trm])
                res[c][v].append(float(f1_score(Y[vi, tem], clf.predict(Xte), average="binary")))
        print(f"  fold {fold} done {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for c in cnames:
        per = {v: float(np.mean(res[c][v])) for v in vocab}
        summary[c] = {"per_task_f1": per, "folds": res[c], "criterion": float(np.mean(list(per.values())))}
        print(c, {k: round(x, 4) for k, x in per.items()}, "criterion", round(summary[c]["criterion"], 4), flush=True)
    best = max(cnames, key=lambda c: summary[c]["criterion"])
    summary["selected"] = best
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("selected", best, flush=True)
    for c in cnames:
        os.remove(os.path.join(args.tmp, f"dev_{c}.npy"))


# ----------------------------------------------------------------------------
# embed: benchmark submission for the test clips
# ----------------------------------------------------------------------------
def cmd_embed(args):
    t0 = time.time()
    fit = dict(np.load(args.fit))
    fmap = np.load(args.frame_map, allow_pickle=True).item()
    d = load_sequences(args.test)
    seqs = d["sequences"]
    total = max(e for _, e in fmap.values())
    emb = np.zeros((total, EMB_DIM), np.float32)
    done = 0
    for n, (s, e) in fmap.items():
        x = embed_clip(seqs[n]["keypoints"], fit, [args.config])[args.config]
        assert x.shape[0] == e - s, (n, x.shape, s, e)
        emb[s:e] = x
        done += 1
        if done % 500 == 0:
            print(f"  embedded {done}/{len(fmap)} {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(emb).all()
    np.save(args.out, {"frame_number_map": fmap, "embeddings": emb})
    print(f"saved {args.out} shape {emb.shape} in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit")
    a.add_argument("--train", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("dev")
    a.add_argument("--train", required=True)
    a.add_argument("--fit", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tmp", required=True)
    a.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("embed")
    a.add_argument("--test", required=True)
    a.add_argument("--fit", required=True)
    a.add_argument("--frame_map", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args()
    {"fit": cmd_fit, "dev": cmd_dev, "embed": cmd_embed}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
