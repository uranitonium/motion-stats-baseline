"""The generic recipe for the MABe22 fly groups. Each fly's 20 landmarks (entries 0 to 19: the pose landmarks and the
tracked centre) and their velocities in its own body frame (origin at its centre, x axis from abdomen tip to head;
velocities, over +-5 frames, are the dish velocities turned into that frame; millimetres), after gap interpolation and a
5-frame median filter, plus one block of relations to the other animals, the same for every species: the distances
from its centre to the two nearest other flies' centres (sorted), the nearest one's centre position and velocity in its
own frame, and the closing speed to it. Pooled over the flies by mean, spread, minimum and maximum, so the order of the
flies does not matter. No measurement written for the dataset.

Then, as for the hand-written baseline: statistics over fixed time windows (mean, spread, histogram); z-score; PCA to
at most 256 numbers per frame (the benchmark's limit for flies). Nothing is trained with gradient descent and no
behaviour label is used to build the representation. Fitted on the benchmark's 426 user-train clips only
(fly_group_train.npy); the submission clips are only ever transformed with the frozen fit.

Commands:  fit | dev | embed
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import median_filter

FPS = 150.0
# keypoint entries (fly_group_train.npy "keypoint_vocabulary")
WING_L, WING_R, HEAD, EYE_R, EYE_L, SHOULDER_L, SHOULDER_R, NOTUM, ABDOMEN = range(9)
MID_L_B, MID_L_E, MID_R_B, MID_R_E = 9, 10, 11, 12
TIP_FR, TIP_MR, TIP_BR, TIP_BL, TIP_ML, TIP_FL = 13, 14, 15, 16, 17, 18
CENTRE = 19
N_POINTS = 20                  # entries 0..19 are positions in mm; 20..23 are tracker measurements, not used
LEG_TIPS = [TIP_FR, TIP_MR, TIP_BR, TIP_BL, TIP_ML, TIP_FL]
BODY_POINTS = list(range(2, 20))  # every body point except the wing tips
ARENA_R = 24.0                 # mm; coordinates are relative to the dish centre
DT_HALF = 5                    # velocities: central difference over +-5 frames (1/15 s span)
SMOOTH = 5                     # median filter over 5 frames (1/30 s) before anything else
STILL_MM_S = 1.0               # a fly slower than 1 mm/s counts as still

WINDOWS = (75, 301, 1201)      # 0.5 s, 2 s, 8 s at 150 Hz (odd, centred); the whole 30 s clip is added below
HIST_WINDOWS = (301, 1201)
NBINS = 8
FIT_STRIDE = 10                # frames used for fitting moments and PCA: every 10th frame
EMB_DIM = 256

VARIANT = "ego"  # body frame only
POOLS = ["mean", "std", "min", "max"]
RAW = [f"{kind}_{k}_{ax}" for kind in ("pos", "vel") for k in range(N_POINTS) for ax in "xy"] + \
    ["rel_d1", "rel_d2", "rel_px", "rel_py", "rel_vx", "rel_vy", "rel_close"]
FEATURE_NAMES = [f"{p}_{f}" for f in RAW for p in POOLS]


# ----------------------------------------------------------------------------
# per-frame features
# ----------------------------------------------------------------------------
def fill_missing(kp):
    """kp (T, M, K, 2) in mm with NaN = missing. Linear interpolation in time; points never seen in the clip take
    the fly's centre."""
    kp = kp.astype(np.float32).copy()
    T, M, K, _ = kp.shape
    t = np.arange(T)
    for m in range(M):
        for k in range(K):
            bad = np.isnan(kp[:, m, k, 0]) | np.isnan(kp[:, m, k, 1])
            if bad.any() and not bad.all():
                good = ~bad
                for c in range(2):
                    kp[bad, m, k, c] = np.interp(t[bad], t[good], kp[good, m, k, c])
    for m in range(M):
        for k in range(K):
            if np.isnan(kp[:, m, k, 0]).all():
                kp[:, m, k, :] = kp[:, m, CENTRE, :]
    return kp


def _ddt(x):
    """Central difference over +-DT_HALF frames, per second, edges padded. x (T, ...)."""
    pad = [x[:1]] * DT_HALF + [x] + [x[-1:]] * DT_HALF
    xp = np.concatenate(pad, 0)
    return (xp[2 * DT_HALF:] - xp[: -2 * DT_HALF]) * (FPS / (2 * DT_HALF))


def _norm(v):
    return np.sqrt((v ** 2).sum(-1))


def _unit(v):
    return v / (_norm(v)[..., None] + 1e-6)


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def generic_relations(c, cvel, u):
    """The generic relations block. c, cvel (T, M, 2) centres and their velocities; u (T, M, 2) unit heading.
    Returns (T, M, 7): d1, d2, nearest centre (fwd, side), nearest relative velocity (fwd, side), closing speed."""
    T, M = c.shape[:2]
    if M < 2:
        return np.zeros((T, M, 7))
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    diff = c[:, None, :, :] - c[:, :, None, :]                       # (T, i, j, 2): j's centre minus i's
    D = np.sqrt((diff ** 2).sum(-1))
    D[:, np.arange(M), np.arange(M)] = np.inf
    order = np.argsort(D, axis=2)                                      # nearest first
    ds = np.take_along_axis(D, order, axis=2)
    d1 = ds[:, :, 0]
    d2 = ds[:, :, 1] if M > 2 else d1
    j = order[:, :, 0]                                                 # nearest other, (T, M)
    rel = np.take_along_axis(diff, j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]          # (T, M, 2)
    vj = np.take_along_axis(np.broadcast_to(cvel[:, None], (T, M, M, 2)), j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]
    rv = vj - cvel
    proj = lambda v: np.stack([(v * u).sum(-1), (v * perp).sum(-1)], -1)
    close = -_ddt(d1[..., None])[..., 0]
    return np.concatenate([d1[..., None], d2[..., None], proj(rel), proj(rv), close[..., None]], -1)


def frame_features(kp_raw):
    """Raw landmarks in place of the 100 hand-written measurements. kp_raw (T, 11, 24, 2) -> (T, 320).

    Per fly, the 20 landmarks (entries 0 to 19) and their velocities, 80 numbers, after the same cleaning as the
    measurements (empty slots dropped, gaps interpolated, 5-frame median filter); pooled over the flies by mean,
    spread, minimum and maximum, as the measurements were, so the order of the flies does not matter.
    world: dish coordinates in mm. ego: each fly's own frame (origin at its centre, x axis from abdomen tip to head);
    velocities are the dish velocities turned into that frame."""
    present = ~np.isnan(kp_raw[:, :, CENTRE, 0]).all(0)
    kp = fill_missing(kp_raw[:, present, :N_POINTS])
    kp = median_filter(kp, size=(SMOOTH, 1, 1, 1), mode="nearest").astype(np.float64)
    T, M = kp.shape[:2]
    pos = kp
    vel = _ddt(kp)
    if VARIANT == "ego":
        c = kp[:, :, CENTRE]
        u = _unit(kp[:, :, HEAD] - kp[:, :, ABDOMEN])
        perp = np.stack([-u[..., 1], u[..., 0]], -1)
        rel = kp - c[:, :, None, :]
        pos = np.stack([(rel * u[:, :, None, :]).sum(-1), (rel * perp[:, :, None, :]).sum(-1)], -1)
        vel = np.stack([(vel * u[:, :, None, :]).sum(-1), (vel * perp[:, :, None, :]).sum(-1)], -1)
    cvel = _ddt(kp[:, :, CENTRE])
    rel_block = generic_relations(kp[:, :, CENTRE], cvel, _unit(kp[:, :, HEAD] - kp[:, :, ABDOMEN]))
    x = np.concatenate([pos.reshape(T, M, -1), vel.reshape(T, M, -1), rel_block], -1)      # (T, M, 87)
    out = np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], 1)       # (T, 4 * 80), pool-major
    # reorder to the FEATURE_NAMES order (per raw value: mean, std, min, max)
    out = out.reshape(T, 4, -1).transpose(0, 2, 1).reshape(T, -1).astype(np.float32)
    assert out.shape[1] == len(FEATURE_NAMES)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------
# window statistics (as for the MABe22 mice, with windows for 150 Hz and a 30 s clip)
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

# Candidate layouts. Each is a list of (block names, output dims) groups; every group gets its own z-score + PCA,
# and the groups are concatenated. Short half: the frame value, 0.5 s and 2 s; long half: 8 s and the whole clip.
_SHORT = ["inst", "mean75", "std75", "mean301", "std301"]
_LONG = ["mean1201", "std1201", "meanclip", "stdclip"]
CONFIGS = {
    "A_joint_meanstd": [(_SHORT + _LONG, 256)],
    "B_joint_meanstd_hist": [(_SHORT + _LONG + ["hist301", "hist1201", "histclip"], 256)],
    "C_split_meanstd": [(_SHORT, 128), (_LONG + ["histclip"], 128)],
    "D_split_meanstd_hist": [(_SHORT + ["hist301"], 128), (_LONG + ["hist1201", "histclip"], 128)],
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
    return np.load(path, allow_pickle=True).item()


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
    base = []
    for i, n in enumerate(names):
        base.append(frame_features(d["sequences"][n]["keypoints"]))
        if i % 50 == 0:
            print(f"  features {i}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    F = base[0].shape[1]
    rng = np.random.default_rng(0)
    sample = np.concatenate([b[rng.choice(len(b), size=min(200, len(b)), replace=False)] for b in base], 0)
    qs = np.linspace(0, 1, NBINS + 1)[1:-1]
    edges = np.quantile(sample, qs, axis=0).T.astype(np.float32)  # (F, NBINS-1)
    print(f"F={F}, features done in {time.time() - t0:.0f}s", flush=True)

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
# dev: cross-validation on the user-train labels
# ----------------------------------------------------------------------------
def cmd_dev(args):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold

    t0 = time.time()
    fit = dict(np.load(args.fit))
    d = load_sequences(args.train)
    vocab = list(d["vocabulary"])
    names = sorted(d["sequences"].keys())
    if args.limit:
        names = names[: args.limit]
    cnames = list(CONFIGS)
    T = d["sequences"][names[0]]["keypoints"].shape[0]
    N = len(names)
    os.makedirs(args.tmp, exist_ok=True)
    mm = {c: np.lib.format.open_memmap(os.path.join(args.tmp, f"dev_{c}.npy"), mode="w+", dtype=np.float32,
                                       shape=(N * T, EMB_DIM)) for c in cnames}
    Y = np.full((len(vocab), N * T), np.nan, np.float32)
    for i, n in enumerate(names):
        s = d["sequences"][n]
        e = embed_clip(s["keypoints"], fit, cnames)
        for c in cnames:
            mm[c][i * T:(i + 1) * T] = e[c]
        Y[:, i * T:(i + 1) * T] = s["annotations"]
        if i % 50 == 0:
            print(f"  embedded {i}/{N} {time.time() - t0:.0f}s", flush=True)
    clip_of = np.repeat(np.arange(N), T)
    folds = list(KFold(5, shuffle=True, random_state=0).split(np.arange(N)))
    summary = {}
    for c in cnames:
        X = mm[c]
        per = {}
        for vi, v in enumerate(vocab):
            lab = ~np.isnan(Y[vi])
            pred = np.full(N * T, np.nan, np.float32)
            for tr, te in folds:
                trm = np.isin(clip_of, tr) & lab
                tem = np.isin(clip_of, te) & lab
                if trm.sum() == 0 or tem.sum() == 0 or len(np.unique(Y[vi, trm])) < 2:
                    continue
                clf = RidgeClassifier(class_weight="balanced", alpha=1.0).fit(np.asarray(X[trm]), Y[vi, trm])
                pred[tem] = clf.predict(np.asarray(X[tem]))
            ok = lab & ~np.isnan(pred)
            per[v] = {"f1": float(f1_score(Y[vi, ok], pred[ok], average="binary", zero_division=0)) if ok.any() else 0.0,
                      "frames": int(ok.sum()), "positive_frames": int(Y[vi, ok].sum()) if ok.any() else 0}
        crit = float(np.mean([per[v]["f1"] for v in vocab]))
        summary[c] = {"per_task": per, "criterion": crit}
        print(c, {k: round(x["f1"], 4) for k, x in per.items()}, "criterion", round(crit, 4),
              f"{time.time() - t0:.0f}s", flush=True)
    best = max(cnames, key=lambda c: summary[c]["criterion"])  # ties: first in the fixed order
    summary["selected"] = best
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("selected", best, flush=True)
    for c in cnames:
        del mm[c]
        os.remove(os.path.join(args.tmp, f"dev_{c}.npy"))


# ----------------------------------------------------------------------------
# embed: benchmark submission for the submission clips
# ----------------------------------------------------------------------------
def cmd_embed(args):
    t0 = time.time()
    fit = dict(np.load(args.fit))
    fmap = np.load(args.frame_map, allow_pickle=True).item()
    d = load_sequences(args.test)
    seqs = d["sequences"]
    total = max(e for _, e in fmap.values())
    emb = np.lib.format.open_memmap(args.out + ".emb.npy", mode="w+", dtype=np.float32, shape=(total, EMB_DIM))
    done = 0
    for n, (s, e) in fmap.items():
        x = embed_clip(seqs[n]["keypoints"], fit, [args.config])[args.config]
        assert x.shape == (e - s, EMB_DIM), (n, x.shape, s, e)
        emb[s:e] = x
        done += 1
        if done % 50 == 0:
            print(f"  embedded {done}/{len(fmap)} {time.time() - t0:.0f}s", flush=True)
    emb.flush()
    arr = np.asarray(emb)
    assert np.isfinite(arr).all()
    np.save(args.out, {"frame_number_map": fmap, "embeddings": arr})
    del emb, arr
    os.remove(args.out + ".emb.npy")
    print(f"saved {args.out} shape ({total}, {EMB_DIM}) in {time.time() - t0:.0f}s sha256 {sha256(args.out)}", flush=True)


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
