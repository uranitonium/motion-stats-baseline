"""No-training baseline for MABe22 fly groups.

The baseline recipe applied to groups of flies: generic per-frame measurements of each fly and of its
relation to the nearest other fly, pooled over the flies in a permutation-invariant way (mean, spread, minimum,
maximum), plus a few group-level measurements; statistics of those over fixed time windows (mean, spread,
histogram); z-score; PCA to at most 256 numbers per frame (the benchmark's limit for flies).
Nothing is trained with gradient descent and no behaviour label is used to build the representation.

Fitted on the benchmark's 426 user-train clips only (fly_group_train.npy): histogram bin edges, z-score moments
and PCA projections. The submission clips are only ever transformed with the frozen fit.

Only the pose landmarks (entries 0 to 18) and the tracked centre (entry 19) of each fly are used; the tracker's
orientation, ellipse, area and contrast entries (20 to 23) are not.

Commands
  fit    bin edges, moments and PCA projections for every candidate layout
  dev    cross-validate the candidate layouts on the user-train labels, pick one by the fixed rule
  embed  write a benchmark submission file for the submission clips with one layout
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

PER_FLY = ["speed", "head_speed", "tail_speed", "accel", "forward", "sideways", "turning", "body_len", "abdomen_len",
           "wing_angle_max", "wing_angle_min", "wing_reach", "front_leg_reach", "leg_spread", "radial", "still",
           "nn_dist", "nn_closing", "facing_nn", "nn_facing_me", "nn_alignment", "head_to_other", "n_close",
           "toward_nn"]
GROUP = ["mean_pair_dist", "close_pairs", "still_frac", "n_flies"]
POOLS = ["mean", "std", "min", "max"]
FEATURE_NAMES = [f"{p}_{f}" for f in PER_FLY for p in POOLS] + GROUP


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


def frame_features(kp_raw):
    """kp_raw (T, 11, 24, 2) -> (T, F) float32 generic features, invariant to the order of the flies."""
    T = kp_raw.shape[0]
    present = ~np.isnan(kp_raw[:, :, CENTRE, 0]).all(0)
    kp = fill_missing(kp_raw[:, present, :N_POINTS])
    kp = median_filter(kp, size=(SMOOTH, 1, 1, 1), mode="nearest")
    M = kp.shape[1]
    c, head, tail, notum = kp[:, :, CENTRE], kp[:, :, HEAD], kp[:, :, ABDOMEN], kp[:, :, NOTUM]
    axis = head - tail
    u = _unit(axis)
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    v = _ddt(c)
    speed = _norm(v)
    ang = np.arctan2(axis[..., 1], axis[..., 0])
    pad = [ang[:1]] * DT_HALF + [ang] + [ang[-1:]] * DT_HALF
    angp = np.concatenate(pad, 0)
    turning = np.abs(_wrap(angp[2 * DT_HALF:] - angp[: -2 * DT_HALF])) * (FPS / (2 * DT_HALF))

    def wing_angle(w):
        return np.arccos(np.clip((_unit(w - c) * -u).sum(-1), -1.0, 1.0))

    wl, wr = wing_angle(kp[:, :, WING_L]), wing_angle(kp[:, :, WING_R])
    body_len = _norm(axis)
    blen = float(np.median(body_len))

    per = {
        "speed": np.log1p(speed),
        "head_speed": np.log1p(_norm(_ddt(head))),
        "tail_speed": np.log1p(_norm(_ddt(tail))),
        "accel": np.log1p(_norm(_ddt(v))),
        "forward": _slog((v * u).sum(-1)),
        "sideways": np.log1p(np.abs((v * perp).sum(-1))),
        "turning": np.log1p(turning),
        "body_len": body_len,
        "abdomen_len": _norm(tail - notum),
        "wing_angle_max": np.maximum(wl, wr),
        "wing_angle_min": np.minimum(wl, wr),
        "wing_reach": np.maximum(_norm(kp[:, :, WING_L] - c), _norm(kp[:, :, WING_R] - c)),
        "front_leg_reach": np.maximum(_norm(kp[:, :, TIP_FL] - c), _norm(kp[:, :, TIP_FR] - c)),
        "leg_spread": _norm(kp[:, :, LEG_TIPS] - c[:, :, None]).mean(-1),
        "radial": _norm(c) / ARENA_R,
        "still": (speed < STILL_MM_S).astype(np.float32),
    }

    # relations to the nearest other fly
    rel = c[:, None, :, :] - c[:, :, None, :]                        # (T, i, j, 2): from fly i to fly j
    D = _norm(rel)
    eye = np.eye(M, dtype=bool)[None]
    Dm = np.where(eye, np.inf, D)
    nn = Dm.argmin(-1)                                               # (T, M)
    ti = np.arange(T)[:, None]
    mi = np.arange(M)[None, :]
    d_nn = Dm[ti, mi, nn]
    closing_all = -_ddt(D)                                           # (T, i, j)
    to_nn = _unit(rel[ti, mi, nn])
    u_nn = u[ti, nn]
    head_d = _norm(head[:, :, None, None, :] - kp[:, None, :, BODY_POINTS, :]).min(-1)   # (T, i, j)
    head_d = np.where(eye, np.inf, head_d).min(-1)
    per.update({
        "nn_dist": np.log1p(d_nn),
        "nn_closing": _slog(closing_all[ti, mi, nn]),
        "facing_nn": (u * to_nn).sum(-1),
        "nn_facing_me": (u_nn * -to_nn).sum(-1),
        "nn_alignment": (u * u_nn).sum(-1),
        "head_to_other": np.log1p(head_d),
        "n_close": (Dm < 2.0 * blen).sum(-1).astype(np.float32),
        "toward_nn": _slog((v * to_nn).sum(-1)),
    })
    cols = []
    for f in PER_FLY:
        x = per[f].astype(np.float64)
        cols += [x.mean(1), x.std(1), x.min(1), x.max(1)]
    iu = np.triu_indices(M, 1)
    pair = D[:, iu[0], iu[1]]
    cols += [np.log1p(pair.mean(1)), (pair < blen).mean(1), per["still"].mean(1), np.full(T, float(M))]
    out = np.stack(cols, 1).astype(np.float32)
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
