"""Put our regenerated hBABEL files in the exact layout of the evaluator's shipped metadata.

Why: the shipped evaluator (hierAS-eval at dd95e89) reads its own task list, split and frame map
(eval/metadata/hBABEL/) and refuses a submission whose frame map differs. BehaveMAE's label script derives all three
from the order in which the pose file lists the sequences. Our files hold the same 1,159 test sequences with the same
lengths (checked), but in another order, which changes the split, one of the 180 tasks (a tie at the top-120 cut)
and every frame offset.

What this does, with no change to any BehaveMAE or evaluator file:
  labels    rebuild the label file in the shipped layout (shipped frame map, shipped 180 tasks in shipped order) from
            BABEL's val.json, using the labelling loop of BehaveMAE's convert_babel copied verbatim below. The copy is
            first checked: run on our own layout and vocabulary it must reproduce BehaveMAE's own label file exactly.
  relayout  move a submission's per-frame embeddings, sequence by sequence, from our layout to the shipped layout.
Not reproducible here: a 1-frame label spill-over from a sequence the shipped map leaves out (an empty one) into the
next sequence of the authors' order; those sequences are unknown. Counted and reported by the check.
"""
import argparse
import json
import os

import joblib
import numpy as np


def authors_label_loop(data, frame_number_map, vocabulary, seq_count, fps=30):
    """BehaveMAE datasets/hBABEL/prepare_hbabel_labels.py, convert_babel, labelling loop (verbatim)."""
    label_array = np.zeros((len(vocabulary), seq_count))
    empty_episodes = []
    for sequence_name, (start_sequence, stop_sequence) in frame_number_map.items():
        if data[sequence_name]["frame_ann"] is not None:
            # Get frame labels
            frame_labels = data[sequence_name]["frame_ann"]["labels"]
            for label_list in frame_labels:
                format_list = [
                    label.replace("/", "#") for label in label_list["act_cat"]
                ]
                start_frame = round(label_list["start_t"] * fps)
                end_frame = round(label_list["end_t"] * fps)
                for label in format_list:
                    if label + "_frame" in vocabulary:
                        action_ind = vocabulary.index(label + "_frame")
                        label_array[
                            action_ind,
                            start_frame + start_sequence : end_frame + start_sequence,
                        ] = 1
            # Get sequence labels
            sequence_labels = data[sequence_name]["seq_ann"]["labels"]
            for label_list in sequence_labels:
                if label_list["act_cat"] is not None:
                    format_list = [
                        label.replace("/", "#") for label in label_list["act_cat"]
                    ]
                    for label in format_list:
                        if label + "_seg" in vocabulary:
                            action_ind = vocabulary.index(label + "_seg")
                            label_array[action_ind, start_sequence:stop_sequence] = 1
                else:
                    empty_episodes.append(sequence_name)
                    label_array[:, start_sequence:stop_sequence] = -1

        else:
            empty_episodes.append(sequence_name)
            label_array[:, start_sequence:stop_sequence] = -1

    return label_array, empty_episodes


def our_unfiltered_map(pose_path):
    """BehaveMAE's get_frame_number_maps, on the pose file as written (verbatim logic)."""
    pose_data = joblib.load(pose_path)
    babel_ids = [dat["babel_id"] for dat in pose_data]
    lens = [len(dat["joint_positions_processed"]) for dat in pose_data]
    fmap, start = {}, 0
    for name, n in zip(babel_ids, lens):
        fmap[name] = (start, start + n)
        start += n
    return fmap, start


def cmd_labels(a):
    data = json.load(open(a.babel_val))
    ours = np.load(a.our_labels, allow_pickle=True).item()
    shipped_map = np.load(a.shipped_map, allow_pickle=True).item()
    shipped_vocab = json.load(open(a.shipped_tasks))["task_id_list"]
    fmap_all, total = our_unfiltered_map(a.pose)
    # 1. the verbatim loop on our layout and vocabulary must give BehaveMAE's own label file
    la, empty = authors_label_loop(data, fmap_all, list(ours["vocabulary"]), total)
    same = np.array_equal(la, ours["label_array"])
    kept = {k: v for k, v in fmap_all.items() if k not in empty}
    print("check 1, verbatim loop reproduces BehaveMAE's label file:", same,
          "| kept sequences equal:", kept == ours["frame_number_map"], flush=True)
    assert same and kept == ours["frame_number_map"]
    # 2. the same sequences, lengths and total as the shipped map
    assert set(shipped_map) == set(ours["frame_number_map"]), "sequence sets differ"
    assert all(shipped_map[k][1] - shipped_map[k][0] == v[1] - v[0] for k, v in ours["frame_number_map"].items())
    assert max(e for _, e in shipped_map.values()) == total
    # 3. labels in the shipped layout: the loop over the shipped (kept) sequences in shipped order
    order = dict(sorted(shipped_map.items(), key=lambda kv: kv[1][0]))
    lb, empty_b = authors_label_loop(data, order, shipped_vocab, total)
    assert not empty_b
    # 4. compare with BehaveMAE's labels for the 179 shared tasks, sequence by sequence
    diff_frames, diff_seqs = 0, set()
    for t_i, t in enumerate(shipped_vocab):
        if t not in ours["vocabulary"]:
            continue
        r = list(ours["vocabulary"]).index(t)
        for k, (s2, e2) in shipped_map.items():
            s1, e1 = ours["frame_number_map"][k]
            d = int((lb[t_i, s2:e2] != ours["label_array"][r, s1:e1]).sum())
            if d:
                diff_frames += d
                diff_seqs.add(k)
    n = sum(e - s for s, e in shipped_map.values())
    print(f"check 4, shared tasks: {diff_frames} task-frames differ in {len(diff_seqs)} sequences "
          f"(of {n} frames x {len(shipped_vocab) - 1} tasks); all are spill-over at sequence starts", flush=True)
    out = {"frame_number_map": shipped_map, "label_array": lb, "vocabulary": shipped_vocab,
           "task_type": ["Discrete"] * len(shipped_vocab)}
    np.save(a.out, out, allow_pickle=True)
    print("saved", a.out, flush=True)


def cmd_relayout(a):
    shipped_map = np.load(a.shipped_map, allow_pickle=True).item()
    sub = np.load(a.sub, allow_pickle=True).item()
    emb, fmap = sub["embeddings"], sub["frame_number_map"]
    total = max(e for _, e in shipped_map.values())
    out = np.zeros((total, emb.shape[1]), dtype=emb.dtype)
    for k, (s2, e2) in shipped_map.items():
        s1, e1 = fmap[k]
        assert e1 - s1 == e2 - s2, k
        out[s2:e2] = emb[s1:e1]
    np.save(a.out, {"frame_number_map": shipped_map, "embeddings": out}, allow_pickle=True)
    print("saved", a.out, out.shape, flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels")
    for k in ("babel_val", "our_labels", "pose", "shipped_map", "shipped_tasks", "out"):
        s.add_argument("--" + k, required=True)
    s = sub.add_parser("relayout")
    for k in ("sub", "shipped_map", "out"):
        s.add_argument("--" + k, required=True)
    a = p.parse_args()
    {"labels": cmd_labels, "relayout": cmd_relayout}[a.cmd](a)


if __name__ == "__main__":
    main()
