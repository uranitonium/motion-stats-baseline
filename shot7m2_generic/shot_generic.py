"""The generic recipe for Shot7M2 (synthetic basketball, one player): the 26 joints and their velocities in the body's
heading frame, then the same window statistics, z-score and PCA to at most 64 numbers per frame (the evaluator's limit)
as the hand-written baseline. Run with T8_VARIANT=ego (the generic recipe); T8_VARIANT=world keeps the dataset's own
coordinates. Nothing is trained with gradient descent and no label is used to build the representation. Bin edges,
moments and PCA are fitted on the 1,280 user-train episodes only (train/train_dictionary_poses.npy) and frozen.

Data: 26 joints in 3D (metres, y vertical), 30 frames per second, 60 s episodes. The poses are centred on the pelvis
horizontally, so there is no court position or travel. Lengths are divided by the player's size (median over the
episode of pelvis-to-head distance plus mean leg length).

Commands
  fit    bin edges, moments and PCA projections for every candidate layout (user-train poses, no labels)
  dev    cross-validate the candidate layouts on the user-train episodes and their own labels; pick one
  embed  write a submission for the evaluator's 2,720 episodes with one layout
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

FPS = 30.0
(CENTER, L_HIP, L_KNEE, L_ANKLE, L_FOOT, L_TOES, R_HIP, R_KNEE, R_ANKLE, R_FOOT, R_TOES, LUMBARS, LOW_THORAX,
 HIGH_THORAX, CERVICALS, L_SBLADE, L_SHOULDER, L_ELBOW, L_WRIST, NECK, HEAD, HEAD_TOP, R_SBLADE, R_SHOULDER,
 R_ELBOW, R_WRIST) = range(26)
UP = 1                       # the vertical axis
STILL = 0.05                 # m/s: mean joint speed below which the body counts as still
WINDOWS = (15, 61, 241, 901)  # 0.5 s, 2 s, 8 s, 30 s at 30 Hz (odd, centred); the whole 60 s clip is added below
HIST_WINDOWS = (61, 241, 901)
NBINS = 8
FIT_STRIDE = 3
EMB_DIM = 64

VARIANT = os.environ.get("T8_VARIANT", "")
assert VARIANT in ("world", "ego"), "set T8_VARIANT to world or ego"
FEATURE_NAMES = [f"{kind}_{j}_{ax}" for kind in ("pos", "vel") for j in range(26) for ax in "xyz"]


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
    """Raw joints in place of the 40 hand-written measurements. kp (T, 1, 26, 3) -> (T, 156).

    world: the 26 joints as the dataset gives them (already centred on the pelvis horizontally) and their velocities.
    ego:   the same, turned about the vertical axis so that the hip line points along x (the body's heading frame);
           velocities are the world velocities turned the same way.
    Lengths are divided by the player's size and velocities are per second, exactly as for the 40 measurements."""
    p = np.nan_to_num(np.asarray(kp, np.float64)[:, 0])
    j = lambda i: p[:, i]  # noqa: E731
    leg = 0.5 * (_norm(j(L_HIP) - j(L_KNEE)) + _norm(j(L_KNEE) - j(L_ANKLE))
                 + _norm(j(R_HIP) - j(R_KNEE)) + _norm(j(R_KNEE) - j(R_ANKLE)))
    size = float(np.median(_norm(j(HEAD) - j(CENTER)) + leg)) + 1e-6
    vel = _ddt(p)
    if VARIANT == "ego":
        right = j(R_HIP) - j(L_HIP)
        yaw = np.arctan2(right[:, 2], right[:, 0])
        cs, sn = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
        rel = p.copy()
        rel[..., 0] -= j(CENTER)[:, None, 0]
        rel[..., 2] -= j(CENTER)[:, None, 2]

        def turn(x):
            out = x.copy()
            out[..., 0] = cs * x[..., 0] - sn * x[..., 2]
            out[..., 2] = sn * x[..., 0] + cs * x[..., 2]
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
    return np.load(path, allow_pickle=True).item()["sequences"]["keypoints"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    print(f"fit on {len(names)} user-train episodes, F={F}; saved {a.out} sha256 {sha256(a.out)} in {time.time() - t0:.0f}s", flush=True)


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


# Layout choice on the user-train episodes and their own labels (train/train_dictionary_actions.npy). Only the rows
# whose values are integer codes are used; the other rows hold continuous weights whose binary form is not
# documented. Each code becomes one binary task, grouped as the benchmark groups its scales: activity (the four
# episode types), action (sprint, force, and the two spin and two turn directions averaged as pairs), moveme (switch,
# and the two codes of each elbow and knee averaged as pairs). Criterion: mean of the three scale means.
DEV_TASKS = [  # (name, label row, code, scale, pair key)
    ("episode_0", "activity_Episode", 0, "activity", None), ("episode_1", "activity_Episode", 1, "activity", None),
    ("episode_2", "activity_Episode", 2, "activity", None), ("episode_3", "activity_Episode", 3, "activity", None),
    ("sprint", "action_Sprint", 1, "action", None), ("force", "action_Force", 1, "action", None),
    ("spin_neg", "action_Spin", -1, "action", "spin"), ("spin_pos", "action_Spin", 1, "action", "spin"),
    ("turn_neg", "action_Turn", -1, "action", "turn"), ("turn_pos", "action_Turn", 1, "action", "turn"),
    ("switch", "moveme_Switch", 1, "moveme", None),
    ("lelbow_neg", "moveme_Lelbow", -1, "moveme", "lelbow"), ("lelbow_pos", "moveme_Lelbow", 1, "moveme", "lelbow"),
    ("relbow_neg", "moveme_Relbow", -1, "moveme", "relbow"), ("relbow_pos", "moveme_Relbow", 1, "moveme", "relbow"),
    ("lknee_neg", "moveme_Lknee", -1, "moveme", "lknee"), ("lknee_pos", "moveme_Lknee", 1, "moveme", "lknee"),
    ("rknee_neg", "moveme_Rknee", -1, "moveme", "rknee"), ("rknee_pos", "moveme_Rknee", 1, "moveme", "rknee"),
]


def dev_criterion(per):
    scales = {}
    for s in ("activity", "action", "moveme"):
        items, seen = [], set()
        for name, _, _, sc, pair in DEV_TASKS:
            if sc != s:
                continue
            if pair is None:
                items.append(per[name])
            elif pair not in seen:
                seen.add(pair)
                items.append(np.mean([per[n] for n, _, _, _, pk in DEV_TASKS if pk == pair]))
        scales[s] = float(np.mean(items))
    return float(np.mean(list(scales.values()))), scales


def cmd_dev(a):
    """5-fold CV over the user-train episodes with the benchmark's classifier, on their own integer-coded labels."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold

    t0 = time.time()
    fit = dict(np.load(a.fit))
    seqs = load_poses(a.train)
    lab = np.load(a.labels, allow_pickle=True).item()
    vocab = list(lab["vocabulary"])
    names = sorted(seqs)
    if a.limit:
        names = names[:: max(1, len(names) // a.limit)]
    T = 1800
    N = len(names)
    cnames = list(CONFIGS)
    X = {c: np.zeros((N * T, EMB_DIM), np.float32) for c in cnames}
    Y = np.zeros((len(DEV_TASKS), N * T), np.int8)
    for i, n in enumerate(names):
        e = embed_clip(seqs[n], fit, cnames)
        for c in cnames:
            X[c][i * T:(i + 1) * T] = e[c]
        s, en = lab["frame_number_map"][n]
        assert en - s == T
        for ti, (_, row, code, _, _) in enumerate(DEV_TASKS):
            Y[ti, i * T:(i + 1) * T] = lab["label_array"][vocab.index(row), s:en] == code
        if i % 200 == 0:
            print(f"  embedded {i}/{N} {time.time() - t0:.0f}s", flush=True)
    del lab, seqs
    clip_of = np.repeat(np.arange(N), T)
    folds = [(np.isin(clip_of, tr), np.isin(clip_of, te))
             for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(N))]
    summary = {"tasks": [t[:3] for t in DEV_TASKS], "n_episodes": N,
               "positives": {t[0]: int(Y[k].sum()) for k, t in enumerate(DEV_TASKS)}}
    for c in cnames:
        per = {}
        for ti, t in enumerate(DEV_TASKS):
            y = Y[ti]
            pred = np.zeros(N * T, np.int8)
            for trm, tem in folds:
                if y[trm].min() == y[trm].max():
                    continue
                pred[tem] = RidgeClassifier(class_weight="balanced", alpha=1.0).fit(X[c][trm], y[trm]).predict(X[c][tem])
            per[t[0]] = float(f1_score(y, pred, average="binary", zero_division=0))
        crit, scales = dev_criterion(per)
        summary[c] = {"per_task_f1": per, "scales": scales, "criterion": crit}
        print(c, "scales", {k: round(v, 4) for k, v in scales.items()}, "criterion", round(crit, 4),
              f"{time.time() - t0:.0f}s", flush=True)
    summary["selected"] = max(cnames, key=lambda c: summary[c]["criterion"])  # ties: first in A, B, C, D order
    json.dump(summary, open(a.out, "w"), indent=1)
    print("selected", summary["selected"], flush=True)


def cmd_embed(a):
    t0 = time.time()
    fit = dict(np.load(a.fit))
    fmap = np.load(a.frame_map, allow_pickle=True).item()
    seqs = load_poses(a.test)
    total = max(e for _, e in fmap.values())
    emb = np.zeros((total, EMB_DIM), np.float32)
    for i, (n, (s, e)) in enumerate(fmap.items()):
        x = embed_clip(seqs[n], fit, [a.config])[a.config]
        assert x.shape == (e - s, EMB_DIM)
        emb[s:e] = x
        if i % 500 == 0:
            print(f"  embedded {i}/{len(fmap)} {time.time() - t0:.0f}s", flush=True)
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
    s.add_argument("--test", required=True); s.add_argument("--fit", required=True); s.add_argument("--frame_map", required=True)
    s.add_argument("--config", required=True); s.add_argument("--out", required=True)
    a = p.parse_args()
    {"fit": cmd_fit, "dev": cmd_dev, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
