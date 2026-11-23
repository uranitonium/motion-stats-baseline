#!/usr/bin/env bash
# Runs the MABe 2022 public evaluator (github.com/damaggu/MABe2022 at e97ddee, unmodified) on one fly submission.
#   bash run_eval.sh SUBMISSION.npy LABELS.npy OUT_DIR
# The evaluator reads its metadata by relative path and writes its logs next to itself, so each run works in a
# fresh copy of the repository inside OUT_DIR; the cloned repository itself is never written to.
# Output: OUT_DIR/flies-round-1--<submission name>.csv (the evaluator's own result table) and OUT_DIR/eval.log.
set -euo pipefail
SUB="$(realpath "$1")"
LAB="$(realpath "$2")"
OUT="$(realpath -m "$3")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MABE2022_REPO:?set MABE2022_REPO to a clone of github.com/damaggu/MABe2022}"
PY="${EVAL_PY:-python}"  # an environment with numpy 1.26.4, pandas 1.5.3, scikit-learn 1.5.2
COMMIT=e97ddeedb873c503aab86a1b9bb2f8b47360aece
[ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT" ] || { echo "evaluator repository is not at $COMMIT" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "evaluator repository has local changes" >&2; exit 1; }
mkdir -p "$OUT"
rm -rf "$OUT/evaluator"
git -C "$REPO" archive --format=tar "$COMMIT" | (mkdir -p "$OUT/evaluator" && tar -x -C "$OUT/evaluator")
cd "$OUT/evaluator"
"$PY" -c "import numpy, pandas, sklearn, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)" > "$OUT/eval.log" 2>&1
OMP_NUM_THREADS="${EVAL_THREADS:-6}" OPENBLAS_NUM_THREADS="${EVAL_THREADS:-6}" MKL_NUM_THREADS="${EVAL_THREADS:-6}" \
    nice -n "${EVAL_NICE:-10}" "$PY" round1_evaluator.py --task flies --submission "$SUB" --labels "$LAB" --output-dir "$OUT" >> "$OUT/eval.log" 2>&1
ls "$OUT"/flies-round-1--*.csv
