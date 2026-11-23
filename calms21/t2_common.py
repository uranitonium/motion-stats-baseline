"""Shared helpers for the CalMS21 Task 1 baseline: data access and window statistics."""
import glob
import os

import numpy as np

FPS = 30
WINDOWS = (15, 61, 241, 901)  # 0.5 s, 2 s, 8 s, 30 s (odd, centred)
NBINS = 8
EMB_DIM = 128
CLASSES = ("attack", "investigation", "mount", "other")  # CalMS21 Task 1 label ids 0..3


def video_files(npz_root, split):
    """Sorted list of (name, path) for split in {train, test, unlabeled}."""
    fs = sorted(glob.glob(os.path.join(npz_root, split, "*.npz")))
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in fs]


def load_video(path):
    d = np.load(path, allow_pickle=False)
    kp = d["kp"]  # (T, 2 mice, 2 xy, 7 parts)
    sc = d["sc"]
    ann = d["ann"] if "ann" in d.files else np.zeros(0, np.int8)
    return kp, sc, ann


def roll_mean(X, w):
    """Centred rolling mean of width w, truncated at the sequence edges. X (T, D)."""
    T = X.shape[0]
    h = w // 2
    cs = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(X, 0, dtype=np.float64)])
    lo = np.clip(np.arange(T) - h, 0, T)
    hi = np.clip(np.arange(T) + h + 1, 0, T)
    return ((cs[hi] - cs[lo]) / (hi - lo)[:, None]).astype(np.float32)


def window_stats(X, edges):
    """X (T, F) -> dict of blocks (T, d): inst, mean<w>, std<w>, hist<w> for every window."""
    T, F = X.shape
    blocks = {"inst": X.astype(np.float32)}
    X2 = X.astype(np.float64) ** 2
    idx = (X[:, :, None] > edges[None, :, :]).sum(-1)
    onehot = np.zeros((T, F, NBINS), np.float32)
    np.put_along_axis(onehot, idx[:, :, None], 1.0, axis=2)
    onehot = onehot.reshape(T, F * NBINS)
    for w in WINDOWS:
        m = roll_mean(X, w)
        blocks[f"mean{w}"] = m
        blocks[f"std{w}"] = np.sqrt(np.maximum(roll_mean(X2, w) - m.astype(np.float64) ** 2, 0.0)).astype(np.float32)
        blocks[f"hist{w}"] = roll_mean(onehot, w)
    return blocks


BLOCK_ORDER = ["inst"] + [f"{s}{w}" for w in WINDOWS for s in ("mean", "std", "hist")]


def stack_blocks(blocks):
    return np.concatenate([blocks[b] for b in BLOCK_ORDER], 1)


def block_slices(F):
    sl, start = {}, 0
    for b in BLOCK_ORDER:
        d = F * NBINS if b.startswith("hist") else F
        sl[b] = np.arange(start, start + d)
        start += d
    return sl, start
