#!/usr/bin/env bash
#
# End-to-end runner for the landmark-only ablation study, intended for a remote
# server. It (1) installs the training dependencies, (2) downloads + MediaPipe-
# preprocesses the WHOLE HaGRIDv2 dataset into the layout the trainer expects
# (all non-target one-hand classes -> N/A), and (3) trains the landmark-only
# ablation and saves its outputs.
#
# The heavy step is preprocessing ~600k images; it is resumable, so the script
# is safe to re-run after an interruption.
#
# Usage (from the repo root):
#   bash run_ablation.sh [--skip-install] [--skip-preprocess]
#
# Configuration via environment variables (all optional):
#   GESTURE_DATA_ROOT  where preprocessed crops/landmarks live  (default: ./data)
#   HF_CACHE_DIR       HuggingFace download cache               (default: ./.hf_cache)
#   RUN_NAME           run folder under runs/                   (default: ablation_lm)
#   EPOCHS             training epochs                          (default: 30)
#   WORKERS            training dataloader workers              (default: nproc-1)
#                      (the preprocessor auto-sizes its own worker pool)
#   PYTHON             python interpreter                       (default: python)
#
# Tip for long remote runs, so it survives logout:
#   nohup bash run_ablation.sh > run_ablation.out 2>&1 &

set -euo pipefail

# ── locate repo root (this script's directory) ───────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── configuration ─────────────────────────────────────────────────────────────
export GESTURE_DATA_ROOT="${GESTURE_DATA_ROOT:-$REPO_ROOT/data}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/.hf_cache}"
RUN_NAME="${RUN_NAME:-ablation_lm}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-$(( $(nproc 2>/dev/null || echo 4) - 1 ))}"
[ "$WORKERS" -lt 1 ] && WORKERS=1
PYTHON="${PYTHON:-python}"

SKIP_INSTALL=0
SKIP_PREPROCESS=0
for arg in "$@"; do
  case "$arg" in
    --skip-install)    SKIP_INSTALL=1 ;;
    --skip-preprocess) SKIP_PREPROCESS=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
RUNNER_LOG="$RUN_DIR/runner.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNNER_LOG"; }

log "Repo root        : $REPO_ROOT"
log "Data root        : $GESTURE_DATA_ROOT"
log "HF cache         : $HF_CACHE_DIR"
log "Run name         : $RUN_NAME  (-> $RUN_DIR)"
log "Epochs / workers : $EPOCHS / $WORKERS"
log "Python           : $PYTHON ($($PYTHON --version 2>&1))"

# ── step 1: dependencies ──────────────────────────────────────────────────────
if [ "$SKIP_INSTALL" -eq 0 ]; then
  log "=== Step 1/3: installing training dependencies ==="
  "$PYTHON" -m pip install --quiet --upgrade pip
  "$PYTHON" -m pip install --quiet -r "$REPO_ROOT/requirements-train.txt"
  log "Dependencies installed."
else
  log "=== Step 1/3: skipped (--skip-install) ==="
fi

# ── step 2: download + preprocess the whole dataset ──────────────────────────
if [ "$SKIP_PREPROCESS" -eq 0 ]; then
  # Fail fast: confirm the HuggingFace dataset is reachable before committing to
  # hours of preprocessing. Uses HF_TOKEN from the environment if it is set.
  log "Preflight: checking HuggingFace access to the dataset ..."
  if ! "$PYTHON" - <<'PY'
import sys
from huggingface_hub import list_repo_files
try:
    files = list(list_repo_files("testdummyvt/hagRIDv2_512px", repo_type="dataset"))
    print(f"OK: dataset reachable, {len(files)} files listed.")
except Exception as e:
    print("HF ACCESS FAILED:", repr(e))
    print("If this is a 401/403 auth error, create a free read token at")
    print("https://huggingface.co/settings/tokens and run:  export HF_TOKEN=hf_xxx")
    sys.exit(1)
PY
  then
    log "ERROR: cannot reach the dataset on HuggingFace (see message above)."
    exit 1
  fi

  log "=== Step 2/3: download + preprocess whole HaGRIDv2 (resumable) ==="
  "$PYTHON" "$REPO_ROOT/download_and_preprocess.py" 2>&1 | tee -a "$RUN_DIR/preprocess.log"
  log "Preprocessing finished."
else
  log "=== Step 2/3: skipped (--skip-preprocess) ==="
fi

# sanity check: the trainer needs at least the train split present
if [ ! -d "$GESTURE_DATA_ROOT/crops/train" ]; then
  log "ERROR: $GESTURE_DATA_ROOT/crops/train not found. Preprocessing did not"
  log "       produce data. Re-run without --skip-preprocess."
  exit 1
fi

# ── step 3: landmark-only ablation training ──────────────────────────────────
log "=== Step 3/3: training landmark-only ablation ==="
"$PYTHON" "$REPO_ROOT/train/ablation_landmark_only.py" \
  --run-name "$RUN_NAME" --epochs "$EPOCHS" --workers "$WORKERS" \
  2>&1 | tee -a "$RUN_DIR/train.log"

# ── collect results ───────────────────────────────────────────────────────────
RESULTS="$RUN_DIR/RESULTS.txt"
{
  echo "Landmark-only ablation — results"
  echo "run name : $RUN_NAME"
  echo "data root: $GESTURE_DATA_ROOT"
  echo "finished : $(date '+%F %T')"
  echo
  grep -E "best val accuracy|best val contest score|params," \
    "$RUN_DIR/ablation_lm.log" 2>/dev/null || true
} > "$RESULTS"

log "=== DONE ==="
log "Outputs in $RUN_DIR :"
log "  - landmark_only_best.pt / landmark_only_last.pt  (checkpoints)"
log "  - ablation_lm.log        (full training log)"
log "  - RESULTS.txt            (best val accuracy + contest score)"
echo
echo "================ RESULTS ================"
cat "$RESULTS"
echo "========================================="
echo "Report the 'best val accuracy' as the landmark-only ablation row."
