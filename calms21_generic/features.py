"""The generic recipe for CalMS21 Task 1. It has the signature build(kp, sc) -> (X, mask) of calms21/features.py, so the
same pipeline (window statistics, standardization, PCA, layouts in t2_baseline.py) runs on top of it.

For each mouse: its seven keypoints in its own body frame (origin at its centre, the mean of neck and hips; x axis from
that centre to its nose) and their velocities (the image velocities turned into that frame), plus the relations block
used for every animal benchmark (the distances to the two nearest other animals, here the one other mouse, repeated;
the nearest one's centre position and velocity in the focal mouse's frame; the closing speed). The two mice keep their
roles (resident first), as the benchmark defines them. A 3-frame median filter, lengths in body lengths (the video's
median nose-to-tail distance), velocities per second. 70 numbers per frame. Nothing is written for the species.
"""
import numpy as np
from scipy.ndimage import median_filter

FPS = 30.0
NOSE, NECK, TAIL = 0, 3, 6
HIPS = (4, 5)


def _ddt(p):
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / 2.0
    v[0] = p[1] - p[0]
    v[-1] = p[-1] - p[-2]
    return v * FPS


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def generic_relations(c, cvel, u):
    """The relations block (as in the MABe22 generic scripts). c, cvel (T, M, 2); u (T, M, 2) unit heading -> (T, M, 7)."""
    T, M = c.shape[:2]
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    diff = c[:, None, :, :] - c[:, :, None, :]
    D = np.sqrt((diff ** 2).sum(-1))
    D[:, np.arange(M), np.arange(M)] = np.inf
    order = np.argsort(D, axis=2)
    ds = np.take_along_axis(D, order, axis=2)
    d1 = ds[:, :, 0]
    d2 = ds[:, :, 1] if M > 2 else d1
    j = order[:, :, 0]
    rel = np.take_along_axis(diff, j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]
    vj = np.take_along_axis(np.broadcast_to(cvel[:, None], (T, M, M, 2)), j[:, :, None, None].repeat(2, -1), axis=2)[:, :, 0]
    rv = vj - cvel
    proj = lambda v: np.stack([(v * u).sum(-1), (v * perp).sum(-1)], -1)
    close = -_ddt(d1)
    return np.concatenate([d1[..., None], d2[..., None], proj(rel), proj(rv), close[..., None]], -1)


def build(kp, sc, score_min=0.2):
    kp = np.asarray(kp, dtype=np.float64)
    T = kp.shape[0]
    pts = np.transpose(kp, (0, 1, 3, 2))  # T, agent, part, xy
    pts = median_filter(pts, size=(3, 1, 1, 1), mode="nearest")
    blen = np.linalg.norm(pts[:, :, NOSE] - pts[:, :, TAIL], axis=-1)
    L = max(np.median(blen), 1.0)
    pts = pts / L
    c = pts[:, :, [NECK, *HIPS], :].mean(axis=2)                       # (T, 2, 2)
    u = _unit(pts[:, :, NOSE] - c)
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    rel = pts - c[:, :, None, :]
    pos = np.stack([(rel * u[:, :, None, :]).sum(-1), (rel * perp[:, :, None, :]).sum(-1)], -1)        # (T, 2, 7, 2)
    v = _ddt(pts)
    vel = np.stack([(v * u[:, :, None, :]).sum(-1), (v * perp[:, :, None, :]).sum(-1)], -1)
    relb = generic_relations(c, _ddt(c), u)                                                            # (T, 2, 7)
    X = np.concatenate([pos.reshape(T, 2, -1), vel.reshape(T, 2, -1), relb], -1).reshape(T, -1)       # (T, 70)
    mask = np.asarray(sc).mean(axis=(1, 2)) >= score_min
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), mask
