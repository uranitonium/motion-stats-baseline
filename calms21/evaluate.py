"""CalMS21 Task 1 with the shared ridge classifier used for every method in the paper.

The class-balanced ridge classifier at alpha 1 (sklearn RidgeClassifier) that the MABe22, Shot7M2 and hBABEL evaluators
train on frozen embeddings, fitted on every frame of the 70 training videos and scored on the 19 test videos as macro F1
over attack, investigation and mount (frames pooled over videos). 95% intervals come from 10,000 resamples of the test
videos; with --emb_b the interval is for the paired gap (A minus B).

    python evaluate.py --npz CALMS21_NPZ --emb EMB_DIR [--emb_b OTHER_EMB_DIR]

CALMS21_NPZ holds train/ and test/ folders of per-video .npz files (convert.py); EMB_DIR holds train/ and test/
folders of per-video (T, D) .npy embeddings named like the .npz files (t2_baseline.py embed writes them).
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_common import load_video, video_files  # noqa: E402

BEHAVIOURS = (0, 1, 2)  # attack, investigation, mount; 3 = other


def load(emb, npz, split):
    out = []
    for name, f in video_files(npz, split):
        _, _, ann = load_video(f)
        X = np.load(os.path.join(emb, split, name + ".npy")).astype(np.float32)
        assert len(X) == len(ann), (name, X.shape, ann.shape)
        out.append((name, X, ann.astype(np.int64)))
    return out


def confusions(train, test):
    clf = RidgeClassifier(class_weight="balanced", alpha=1.0).fit(np.concatenate([v[1] for v in train]),
                                                                    np.concatenate([v[2] for v in train]))
    cms = []
    for _, X, y in test:
        cm = np.zeros((4, 4), np.int64)
        np.add.at(cm, (y, clf.predict(X)), 1)
        cms.append(cm)
    return np.stack(cms)  # (videos, 4, 4)


def macro_f1(cm):
    f = []
    for c in BEHAVIOURS:
        tp = cm[..., c, c]
        d = 2 * tp + (cm[..., :, c].sum(-1) - tp) + (cm[..., c, :].sum(-1) - tp)
        f.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    return np.mean(f, 0), f


def boot(cms_list, n_boot=10000, seed=1):
    V = cms_list[0].shape[0]
    idx = np.random.default_rng(seed).integers(0, V, (n_boot, V))
    cnt = np.zeros((n_boot, V))
    np.add.at(cnt, (np.repeat(np.arange(n_boot), V), idx.ravel()), 1)
    return [macro_f1(np.einsum("bv,vij->bij", cnt, c.astype(np.float64)))[0] for c in cms_list]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--emb_b")
    a = p.parse_args()
    runs = []
    for emb in [a.emb] + ([a.emb_b] if a.emb_b else []):
        cms = confusions(load(emb, a.npz, "train"), load(emb, a.npz, "test"))
        m, per = macro_f1(cms.sum(0))
        print(f"{emb}: macro F1 {100 * m:.1f} (attack {100 * per[0]:.1f}, investigation {100 * per[1]:.1f}, "
              f"mount {100 * per[2]:.1f})")
        runs.append(cms)
    b = boot(runs)
    if len(runs) == 1:
        print(f"95% interval {100 * np.percentile(b[0], 2.5):.1f} to {100 * np.percentile(b[0], 97.5):.1f}")
    else:
        g = b[0] - b[1]
        gap = macro_f1(runs[0].sum(0))[0] - macro_f1(runs[1].sum(0))[0]
        print(f"A minus B {100 * gap:+.1f}, paired 95% interval {100 * np.percentile(g, 2.5):+.1f} to "
              f"{100 * np.percentile(g, 97.5):+.1f}")


if __name__ == "__main__":
    main()
