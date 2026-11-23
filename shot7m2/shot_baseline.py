"""No-training baseline for Shot7M2 (synthetic basketball, one player).

The baseline recipe applied to one 3D skeleton: generic per-frame measurements of posture and motion,
statistics over fixed time windows (mean, spread, histogram), z-score, PCA to at most 64 numbers per frame (the
benchmark evaluator's limit for Shot7M2). Nothing is trained with gradient descent and no label is used to build
the representation. Bin edges, moments and PCA are fitted on the 1,280 user-train episodes only
(train/train_dictionary_poses.npy) and frozen.

Data: 26 joints in 3D (metres, y vertical), 30 frames per second, 60 s episodes. The poses are centred on the
pelvis horizontally, so there is no court position or travel; only posture, limb motion, height and turning.
One body, so left and right limbs are kept apart. Positions and speeds are divided by the player's size (median
over the episode of pelvis-to-head distance plus mean leg length), which is also kept as a measurement.

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

FEATURE_NAMES = [
    "size", "centre_height", "head_height", "centre_vvel", "body_speed", "head_speed",
    "l_wrist_fwd", "l_wrist_lat", "l_wrist_up", "l_wrist_speed", "l_wrist_vvel",
    "r_wrist_fwd", "r_wrist_lat", "r_wrist_up", "r_wrist_speed", "r_wrist_vvel",
    "l_ankle_fwd", "l_ankle_lat", "l_ankle_height", "l_ankle_speed", "l_ankle_vvel",
    "r_ankle_fwd", "r_ankle_lat", "r_ankle_height", "r_ankle_speed", "r_ankle_vvel",
    "l_elbow", "r_elbow", "l_knee", "r_knee", "l_elbow_rate", "r_elbow_rate", "l_knee_rate", "r_knee_rate",
    "trunk_lean", "trunk_twist", "yaw_rate", "hand_gap", "foot_gap", "still",
]


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
    """kp (T, 1, 26, 3) -> (T, 40) float32 generic posture and motion measurements of one 3D skeleton."""
    p = np.nan_to_num(np.asarray(kp, np.float64)[:, 0])
    T = p.shape[0]
    j = lambda i: p[:, i]  # noqa: E731
    leg = 0.5 * (_norm(j(L_HIP) - j(L_KNEE)) + _norm(j(L_KNEE) - j(L_ANKLE))
                 + _norm(j(R_HIP) - j(R_KNEE)) + _norm(j(R_KNEE) - j(R_ANKLE)))
    size = float(np.median(_norm(j(HEAD) - j(CENTER)) + leg)) + 1e-6
    vel = _ddt(p) / size                            # (T, 26, 3), sizes per second
    spd = _norm(vel)                                # (T, 26)
    floor = np.median(np.minimum(j(L_ANKLE)[:, UP], j(R_ANKLE)[:, UP]))
    # body frame: right = horizontal hip line (left hip to right hip), forward = right x up (checked: toes point along it)
    right = j(R_HIP) - j(L_HIP)
    right[:, UP] = 0
    right = _unit(right)
    up = np.zeros(3)
    up[UP] = 1.0
    fwd = np.cross(right, up)
    yaw = np.arctan2(right[:, 2], right[:, 0])
    yp = np.concatenate([yaw[:1], yaw[:1], yaw, yaw[-1:], yaw[-1:]])
    yaw_rate = _wrap(yp[4:] - yp[:-4]) * (FPS / 4.0)            # rad/s, signed
    sh = j(R_SHOULDER) - j(L_SHOULDER)
    twist = _wrap(np.arctan2(sh[:, 2], sh[:, 0]) - yaw)         # signed shoulder line against hip line
    trunk = j(CERVICALS) - j(CENTER)
    lean = np.arccos(np.clip((_unit(trunk) * up).sum(-1), -1.0, 1.0))
    rel = (p - j(CENTER)[:, None]) / size                       # (T, 26, 3) relative to the pelvis, in sizes
    f_ = lambda i: (rel[:, i] * fwd).sum(-1)  # noqa: E731
    l_ = lambda i: (rel[:, i] * right).sum(-1)  # noqa: E731
    angles = [_angle(j(L_SHOULDER), j(L_ELBOW), j(L_WRIST)), _angle(j(R_SHOULDER), j(R_ELBOW), j(R_WRIST)),
              _angle(j(L_HIP), j(L_KNEE), j(L_ANKLE)), _angle(j(R_HIP), j(R_KNEE), j(R_ANKLE))]
    body_speed_ms = _norm(_ddt(p)).mean(1)
    cols = [np.full(T, size), (j(CENTER)[:, UP] - floor) / size, (j(HEAD)[:, UP] - floor) / size,
            _slog(vel[:, CENTER, UP]), np.log1p(spd.mean(1)), np.log1p(spd[:, HEAD])]
    for w in (L_WRIST, R_WRIST):
        cols += [f_(w), l_(w), rel[:, w, UP], np.log1p(spd[:, w]), _slog(vel[:, w, UP])]
    for a_ in (L_ANKLE, R_ANKLE):
        cols += [f_(a_), l_(a_), (p[:, a_, UP] - floor) / size, np.log1p(spd[:, a_]), _slog(vel[:, a_, UP])]
    cols += angles + [_slog(_ddt(x)) for x in angles]
    cols += [lean, twist, _slog(yaw_rate), _norm(rel[:, L_WRIST] - rel[:, R_WRIST]),
             _norm(rel[:, L_ANKLE] - rel[:, R_ANKLE]), (body_speed_ms < STILL).astype(np.float64)]
    out = np.stack(cols, 1).astype(np.float32)
    assert out.shape[1] == len(FEATURE_NAMES)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


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
