"""Generic dyadic relational features from two-agent keypoints (CalMS21 layout).

kp: (T, 2 agents, 2 xy, 7 parts) ordered nose, l ear, r ear, neck, l hip, r hip, tail base.
Every feature is defined from two agents' keypoints only, with no arena axis.
Distances are in body lengths, speeds in body lengths per second.
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FPS = 30.0
NOSE, NECK, TAIL = 0, 3, 6
FEATURE_NAMES = [
    "dist", "closing_speed", "speed_0", "speed_1", "toward_0", "toward_1",
    "elong_0", "elong_1", "headlead_0", "headlead_1",
    "still_occ_0", "still_occ_1", "still_trans_0", "still_trans_1", "speed_corr",
    "nose_to_partner_0", "nose_to_partner_1", "contact_margin", "contact",
    "facing_0", "facing_1", "facing_rate_0", "facing_rate_1",
]
CRITICAL_INDEX = 0   # distance in body lengths; the contact threshold is 1 body length
STILL_THR = 0.25     # body lengths per second, fixed before any label inspection
WIN = 15             # centered window, as in the fencing features


def _vel(p):
    v = np.zeros_like(p)
    v[1:-1] = (p[2:] - p[:-2]) / 2.0
    v[0] = p[1] - p[0]; v[-1] = p[-1] - p[-2]
    return v * FPS


def _centered_corr(a, b, w):
    ma, mb = uniform_filter1d(a, w, mode="nearest"), uniform_filter1d(b, w, mode="nearest")
    cov = uniform_filter1d(a * b, w, mode="nearest") - ma * mb
    va = uniform_filter1d(a * a, w, mode="nearest") - ma ** 2
    vb = uniform_filter1d(b * b, w, mode="nearest") - mb ** 2
    return cov / np.sqrt(np.clip(va, 1e-8, None) * np.clip(vb, 1e-8, None))


def build(kp, sc, score_min=0.2):
    kp = np.asarray(kp, dtype=np.float64)
    T = kp.shape[0]
    pts = np.transpose(kp, (0, 1, 3, 2))  # T, agent, part, xy
    pts = median_filter(pts, size=(3, 1, 1, 1), mode="nearest")
    cen = pts[:, :, [NECK, 4, 5], :].mean(axis=2)  # neck + hips
    blen = np.linalg.norm(pts[:, :, NOSE] - pts[:, :, TAIL], axis=-1)  # T,2
    L = max(np.median(blen), 1.0)  # sequence scale: median body length over both agents
    Lm = np.maximum(np.median(blen, axis=0), 1.0)
    rel = cen[:, 1] - cen[:, 0]
    d = np.linalg.norm(rel, axis=-1)
    u01 = rel / np.maximum(d, 1e-6)[:, None]
    v = _vel(cen)  # T,2,xy
    spd = np.linalg.norm(v, axis=-1) / L
    toward0 = (v[:, 0] * u01).sum(-1) / L
    toward1 = (v[:, 1] * -u01).sum(-1) / L
    dn = d / L
    ddot = _vel(dn[:, None])[:, 0]
    closing = -ddot
    elong = blen / Lm[None]
    nose_spd = np.linalg.norm(_vel(pts[:, :, NOSE]), axis=-1) / L
    headlead = nose_spd - spd
    still = (spd < STILL_THR).astype(np.float64)
    occ = uniform_filter1d(still, WIN, axis=0, mode="nearest")
    trans = np.zeros_like(still); trans[1:] = np.abs(np.diff(still, axis=0))
    trans = uniform_filter1d(trans, WIN, axis=0, mode="nearest")
    corr = _centered_corr(spd[:, 0], spd[:, 1], WIN)
    n2p = np.stack([
        np.linalg.norm(pts[:, 0, NOSE][:, None] - pts[:, 1], axis=-1).min(-1),
        np.linalg.norm(pts[:, 1, NOSE][:, None] - pts[:, 0], axis=-1).min(-1)], axis=1) / L
    margin = 1.0 - dn
    contact = (dn < 1.0).astype(np.float64)
    head = pts[:, :, NOSE] - pts[:, :, NECK]
    head = head / np.maximum(np.linalg.norm(head, axis=-1, keepdims=True), 1e-6)
    to_other = np.stack([u01, -u01], axis=1)
    facing = (head * to_other).sum(-1)
    ang = np.arctan2(head[..., 1], head[..., 0])
    dang = np.zeros_like(ang); dang[1:] = np.angle(np.exp(1j * np.diff(ang, axis=0)))
    frate = uniform_filter1d(np.abs(dang) * FPS, 3, axis=0, mode="nearest")
    X = np.column_stack([
        dn, closing, spd[:, 0], spd[:, 1], toward0, toward1, elong[:, 0], elong[:, 1],
        headlead[:, 0], headlead[:, 1], occ[:, 0], occ[:, 1], trans[:, 0], trans[:, 1], corr,
        n2p[:, 0], n2p[:, 1], margin, contact, facing[:, 0], facing[:, 1], frate[:, 0], frate[:, 1]])
    mask = np.asarray(sc).mean(axis=(1, 2)) >= score_min
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), mask


def swap_partner(kp_a, kp_b):
    """Resident (agent 0) from sequence a, intruder (agent 1) from sequence b, truncated to equal length."""
    T = min(len(kp_a), len(kp_b))
    out = np.array(kp_a[:T], copy=True)
    out[:, 1] = kp_b[:T, 1]
    return out
