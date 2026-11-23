"""Merge the per-subset outputs of TEACH's process_amass.py (run in parallel groups) into one amass.pth.tar,
in TEACH's AMASS_DIRS order, exactly as a single run of the script would have built all_data: a subset that is
listed twice (TCDhandMocap) is taken once, subsets not on disk are skipped."""
import argparse
import ast
import glob
import os

import joblib


def amass_dirs(script):
    tree = ast.parse(open(script).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "AMASS_DIRS":
            return ast.literal_eval(node.value)
    raise SystemExit("AMASS_DIRS not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", required=True)
    p.add_argument("--teach", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    all_data, seen = [], set()
    for x in amass_dirs(a.teach):
        name = x.strip().split("/")[-1]
        if name in seen:
            continue
        hits = glob.glob(f"{a.groups}/g*/{name}/{name}.pth.tar")
        if not hits:
            print("not on disk:", x, flush=True)
            continue
        assert len(hits) == 1, hits
        seen.add(name)
        part = joblib.load(hits[0])
        print(f"{name}: {len(part)} sequences, {sum(s['poses'].shape[0] for s in part)} frames", flush=True)
        all_data.extend(part)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print("total", len(all_data), "sequences", flush=True)
    joblib.dump(all_data, a.out)


if __name__ == "__main__":
    main()
