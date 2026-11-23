# Self-supervised behavior encoders rarely beat a training-free baseline

Code and frozen fits for *Self-supervised behavior encoders rarely beat a training-free baseline* (Ziwen Chen,
[arXiv:2611.24792](https://arxiv.org/abs/2611.24792)).

Self-supervised encoders of animal and human pose are rarely compared with simple summaries of the poses they encode.
This repository holds such a summary, which needs no training and no knowledge of the species, and the commands that
score it on five benchmarks (CalMS21 Task 1, MABe22 mouse triplets and fly groups, Shot7M2 and hBABEL) with each
benchmark's own evaluator.

The baseline is deliberately plain, and comes in two variants.

- **The generic recipe** (the paper's main baseline). Each body's keypoints and their velocities in its own frame
  (centered on the body, turned to its heading); for animal benchmarks, one relations block, the same for every
  species: the distances to the two nearest other animals, the nearest one's position and velocity in the animal's
  own frame, and the closing speed. Nothing is written for a particular species.
- **The hand-written variant.** A few dozen ordinary kinematic measurements per benchmark (distances, speeds, closing
  speed, facing, joint angles, heights) in place of the keypoints.

Both summarize each number by its value, mean, spread and an 8-bin histogram over centered windows from 0.5 s to the
whole clip, and reduce the result with PCA to the benchmark's budget of numbers per frame. Nothing is trained by
gradient descent and no label is used to fit them; the layout of the windows (one of four) is chosen by
cross-validation on training labels. Fitting takes minutes on a CPU.

## Results (all labels, each benchmark's own score)

| Benchmark | Score | Generic recipe | Hand-written variant | Encoder pretrained on the benchmark's data |
|---|---|---|---|---|
| CalMS21 Task 1 | macro F1 (shared ridge) | 79.7 | 80.1 | hBehaveMAE, retrained: 75.9 |
| MABe22 mouse triplets | clip-level F1 | 83.1 | 83.7 | hBehaveMAE, released: 80.3 |
| MABe22 fly groups | mean F1, 50 tasks | 52.4 | 51.6 | BAMS, published: 48.2 |
| Shot7M2 | All F1 | 66.0 | 68.8 | hBehaveMAE, released: 64.7 |
| hBABEL | benchmark score | 20.9 | 22.1 | hBehaveMAE, released: 16.5 |

Every method is scored with the class-balanced ridge classifier (alpha 1) that the MABe22, Shot7M2 and hBABEL
evaluators train on frozen embeddings; CalMS21 ships no evaluator, so `calms21/evaluate.py` applies the same classifier.
Of the seven encoders in the paper, only HiDRA, a forecaster built for pairs of mice, beats the baseline with all labels,
on CalMS21 (83.2); it trails on the mouse triplets (75.8) and cannot run on the flies or the human benchmarks. The paper
reports intervals, the other encoders and the checks.

## What is here

| Folder | Benchmark | Variant | Code | Frozen fit | Layout |
|---|---|---|---|---|---|
| `calms21_generic/` | CalMS21 Task 1 (two mice) | generic | `features.py`, `t2_common.py`, `t2_baseline.py` | `generic_fit_calms21.npz` | `B_joint_meanstd_hist` |
| `mabe22_mice_generic/` | MABe22 mouse triplets | generic | `mice_generic.py` | `generic_fit_mabe22.npz` | `C_split_meanstd` |
| `mabe22_flies_generic/` | MABe22 fly groups | generic | `fly_generic.py` | `generic_fit_flies.npz` | `D_split_meanstd_hist` |
| `shot7m2_generic/` | Shot7M2 (synthetic basketball) | generic | `shot_generic.py` (`T8_VARIANT=ego`) | `generic_fit_shot7m2.npz` | `C_split_meanstd` |
| `hbabel_generic/` | hBABEL (human motion capture) | generic | `hbabel_generic.py` (`T12_VARIANT=ego`) | `generic_fit_hbabel.npz` | `B_joint_meanstd_hist` |
| `calms21/` | CalMS21 Task 1 | hand-written | `features.py`, `t2_common.py`, `t2_baseline.py` | `baseline_fit_calms21.npz` | `D_split_meanstd_hist` |
| `mabe22_mice/` | MABe22 mouse triplets | hand-written | `pt_baseline.py` | `baseline_fit_mabe22.npz` | `C_split_meanstd` |
| `mabe22_flies/` | MABe22 fly groups | hand-written | `fly_baseline.py` | `baseline_fit_flies.npz` | `A_joint_meanstd` |
| `shot7m2/` | Shot7M2 | hand-written | `shot_baseline.py` | `baseline_fit_shot7m2.npz` | `C_split_meanstd` |
| `hbabel/` | hBABEL | hand-written | `hbabel_baseline.py` | `baseline_fit_hbabel.npz` | `C_split_meanstd` |

Each fit holds the histogram bin edges, standardization moments and PCA projections, fitted on the benchmark's training
data without labels; `SHA256SUMS` lists their hashes. Every script has a `fit` command that recomputes its fit from the
training file and an `embed` command that writes a submission; the MABe22, Shot7M2 and hBABEL scripts also have a `dev`
command that picks the layout by cross-validation on training labels. The scripts here are the ones that produced the
paper's numbers, with only their comments edited. `summarize.py` turns each evaluator's result table into the paper's
score.

## Setup

Python 3.11 with `pip install -r requirements.txt`. The evaluators are used unmodified from their repositories:

- `hierAS-eval` from [BehaveMAE](https://github.com/amathislab/BehaveMAE) at commit `dd95e89` (MABe22 mice, Shot7M2, hBABEL);
- the [MABe 2022 public evaluator](https://github.com/damaggu/MABe2022) at commit `e97ddee` (flies), in its own
  environment with numpy 1.26.4, pandas 1.5.3 and scikit-learn 1.5.2.

Data are not redistributed: CalMS21 (data.caltech.edu record `s0vdx-0k302`), the MABe22 mouse triplets and fly groups
(records `8kdn3-95j37` and `rdsa8-rde65`), Shot7M2 (huggingface.co/datasets/amathislab/SHOT7M2), and hBABEL, which is
rebuilt from AMASS and BABEL as described in `hbabel/README.md`.

## Reproducing each number

`HIER` is a checkout of BehaveMAE at `dd95e89` (the evaluator runs from `$HIER/hierAS-eval`); paths to the data files are
placeholders.

**Generic recipe.** The same data and evaluators as below, with the generic scripts, fits and layouts:

    python calms21_generic/t2_baseline.py embed --npz npz --fit calms21_generic/generic_fit_calms21.npz \
        --config B_joint_meanstd_hist --splits train,test --out emb_generic
    python calms21/evaluate.py --npz npz --emb emb_generic/B_joint_meanstd_hist                       # 79.7
    python mabe22_mice_generic/mice_generic.py embed --test mouse_triplet_test.npy \
        --fit mabe22_mice_generic/generic_fit_mabe22.npz \
        --frame_map $HIER/hierAS-eval/eval/metadata/mouse_frame_number_map.npy --config C_split_meanstd --out mice_generic.npy
    python mabe22_flies_generic/fly_generic.py embed --test fly_group_test.npy \
        --fit mabe22_flies_generic/generic_fit_flies.npz \
        --frame_map MABe2022/round1/metadata/flies_frame_number_map.npy --config D_split_meanstd_hist --out flies_generic.npy
    T8_VARIANT=ego python shot7m2_generic/shot_generic.py embed --test test/test_dictionary_poses.npy \
        --fit shot7m2_generic/generic_fit_shot7m2.npz \
        --frame_map $HIER/hierAS-eval/eval/metadata/Shot7M2/frame_number_map_Shot7M2.npy --config C_split_meanstd --out shot_generic.npy
    T12_VARIANT=ego python hbabel_generic/hbabel_generic.py embed --test val_proc_realigned_procrustes.pth.tar \
        --fit hbabel_generic/generic_fit_hbabel.npz --config B_joint_meanstd_hist --out hbabel_generic_own.npy
    python hbabel/align_hbabel.py relayout --sub hbabel_generic_own.npy \
        --shipped_map $HIER/hierAS-eval/eval/metadata/hBABEL/frame_number_map_val_filtered.npy --out hbabel_generic.npy

Then score each submission exactly as the hand-written one below (the triplets give 83.1, the flies 52.4, Shot7M2 66.0
and hBABEL 20.9).

**Hand-written variant.**

**CalMS21 Task 1 (80.1).** From `task1_classic_classification.zip` (and, only to refit, `unlabeled_videos.zip`),
convert to one `.npz` per video, then embed and score:

    python calms21/convert.py task1_classic_classification/calms21_task1_train.json npz/train
    python calms21/convert.py task1_classic_classification/calms21_task1_test.json npz/test
    for p in 1 2 3 4; do unzip -p unlabeled_videos.zip unlabeled_videos/calms21_unlabeled_videos_part$p.json \
        | python calms21/convert_unlabeled.py npz/unlabeled; done   # only needed for a new fit
    python calms21/t2_baseline.py embed --npz npz --fit calms21/baseline_fit_calms21.npz \
        --config D_split_meanstd_hist --splits train,test --out emb
    python calms21/evaluate.py --npz npz --emb emb/D_split_meanstd_hist

**MABe22 mouse triplets (83.7).**

    python mabe22_mice/pt_baseline.py embed --test mouse_triplet_test.npy --fit mabe22_mice/baseline_fit_mabe22.npz \
        --frame_map $HIER/hierAS-eval/eval/metadata/mouse_frame_number_map.npy --config C_split_meanstd --out mice.npy
    (cd $HIER/hierAS-eval && python evaluator.py --task mabe_mice --output-dir out_mice \
        --labels mouse_triplets_test_labels.npy --submission mice.npy)
    python summarize.py mice out_mice/<result>.csv

**MABe22 fly groups (51.6).**

    python mabe22_flies/fly_baseline.py embed --test fly_group_test.npy --fit mabe22_flies/baseline_fit_flies.npz \
        --frame_map MABe2022/round1/metadata/flies_frame_number_map.npy --config A_joint_meanstd --out flies.npy
    MABE2022_REPO=MABe2022 EVAL_PY=python bash mabe22_flies/run_eval.sh flies.npy fly_groups_test_labels.npy out_flies
    python summarize.py flies out_flies/flies-round-1--flies.npy.csv

**Shot7M2 (68.8).**

    python shot7m2/shot_baseline.py embed --test test/test_dictionary_poses.npy --fit shot7m2/baseline_fit_shot7m2.npz \
        --frame_map $HIER/hierAS-eval/eval/metadata/Shot7M2/frame_number_map_Shot7M2.npy --config C_split_meanstd --out shot.npy
    (cd $HIER/hierAS-eval && python evaluator.py --task Shot7M2 --output-dir out_shot \
        --labels test/benchmark_labels.npy --submission shot.npy)
    python summarize.py shot out_shot/<result>.csv

**hBABEL (22.1).** After rebuilding the data (`hbabel/README.md`):

    python hbabel/hbabel_baseline.py embed --test val_proc_realigned_procrustes.pth.tar \
        --fit hbabel/baseline_fit_hbabel.npz --config C_split_meanstd --out hbabel_own.npy
    python hbabel/align_hbabel.py relayout --sub hbabel_own.npy \
        --shipped_map $HIER/hierAS-eval/eval/metadata/hBABEL/frame_number_map_val_filtered.npy --out hbabel.npy
    (cd $HIER/hierAS-eval && python evaluator.py --task hBABEL --output-dir out_hbabel \
        --labels hbabel_val_test_actions_val_top_120_60_filtered.npy --submission hbabel.npy)
    python summarize.py hbabel out_hbabel/<result>.csv

## Citation

    @article{chen2026encoders,
      title   = {Self-supervised behavior encoders rarely beat a training-free baseline},
      author  = {Chen, Ziwen},
      journal = {arXiv preprint arXiv:2611.24792},
      year    = {2026}
    }

## License

Code: MIT (`LICENSE`); third-party code: `THIRD_PARTY_NOTICES.md`. The frozen fits are statistics computed from the
datasets above and are shared under those datasets' terms.
