"""Turn a benchmark evaluator's result table (CSV) into the score the paper reports.

    python summarize.py mice   RESULT.csv   # clip-level F1 (strain, lights) and frame-level F1 (the 9 behaviors)
    python summarize.py flies  RESULT.csv   # mean F1 over the evaluator's 50 rows
    python summarize.py shot   RESULT.csv   # All F1: mean of the activity, action and moveme scale means
    python summarize.py hbabel RESULT.csv   # mean of the frame-level (top 10/30/60/90) and sequence-level (10/30/60) means
Each follows the benchmark paper's own definition, using the evaluator's "Private Score" column.
"""
import sys

import numpy as np
import pandas as pd


def mice(res):
    f1 = res[res["Metric"] == "f1_score"]
    seq = f1[f1["Sequence Level Task"] == True]  # noqa: E712
    frm = f1[f1["Sequence Level Task"] == False]  # noqa: E712
    return {"clip-level F1": 100 * float(seq["Private Score"].mean()), "frame-level F1": 100 * float(frm["Private Score"].mean())}


def flies(res):
    return {"mean F1": 100 * float(res["Private Score"].astype(float).mean())}


def shot(res):
    tid = res["Task ID"].astype(str)
    act = [i for i in range(len(res.index)) if tid[i].startswith("activity_Episode")]
    activity = round(res.loc[act, "Private Score"].mean(), 3)
    a = res[tid.str.startswith("action")]["Private Score"].to_numpy()
    action = np.concatenate([a[:6], a[8:10], a[6:8].reshape(-1, 2).mean(1), a[10:].reshape(-1, 2).mean(1)]).mean()
    m = res[tid.str.startswith("moveme")]["Private Score"].to_numpy()
    moveme = np.concatenate([m[:6], m[6:].reshape(-1, 2).mean(1)]).mean()
    assert len(res) == 30 and len(act) == 4 and len(a) == 12 and len(m) == 14
    return {"All F1": 100 * float(np.mean([activity, action, moveme])), "activity": 100 * float(activity),
            "action": 100 * float(action), "moveme": 100 * float(moveme)}


def hbabel(res):
    f1 = res.loc[res["Metric"] == "f1_score", "Private Score"].to_numpy()
    out = {}
    for group, tops in (("frame", (10, 30, 60, 90)), ("seg", (10, 30, 90))):  # 90 asks for all 60 sequence-level tasks
        idx = [k for k in res.index if res["Task ID"][k].endswith("_" + group)]
        out[group] = [round(100 * round(float(f1[idx[-top:]].mean()), 3), 1) for top in tops]
    return {"score": float(np.mean([np.mean(out["frame"]), np.mean(out["seg"])])), "frame": out["frame"], "sequence": out["seg"]}


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    r = {"mice": mice, "flies": flies, "shot": shot, "hbabel": hbabel}[kind](pd.read_csv(path))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
