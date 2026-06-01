#!/usr/bin/env bash
# Batch-evaluate trained DL3DV probe checkpoints on val.json.
#
# For each run under logs/dl3dv/runs/ matching the requested views count:
#   1. Reuses the original training overrides from .hydra/overrides.yaml
#      (experiment=..., job_name=..., feat_postfix=..., ~trainer.plugins, ...)
#      but drops trainer.devices=* (we force single-GPU for eval).
#   2. Appends gt_num_frames/batch_size/target_spatial_size read from the
#      run's .hydra/config.yaml, since committed yaml defaults may drift.
#   3. Switches task_name to dl3dv-eval so outputs land in logs/dl3dv-eval/.
#   4. Runs train.py with train=false test=true autoresume=false ckpt_path=<last.ckpt>.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_dl3dv.sh --views 4 --video-tss "[15, 26]" --video-tss "[15, 22]" --project 3d-eval2
#                              [--project <wandb-project>] [--run-name <exact-run-dir>] [--vfm wan-t2v-1.3b] [--vfm qwen3-vl-8b] [--dry-run] [--skip-done]
#
# Notes:
#   * Run directories can use short release names such as
#     `vlm_dl3dv_qwen3-vl-8b`. Evaluation metadata is read from
#     `.hydra/config.yaml`, not parsed from the folder name.
#   * --video-tss restricts wan-t2v-1.3b/opensora to only the matching
#     target_spatial_size variants (e.g. "[15, 26]"); InternVL3/Qwen3VL
#     runs are unaffected and run as-is. Pass multiple times to allow
#     several variants, e.g. --video-tss "[15, 26]" --video-tss "[15, 22]".
#   * --vfm filters by VFM substring in the run dir name (e.g. wan-t2v-1.3b, opensora,
#     qwen3-vl-8b, internvl3). Matches both old `dl3dv_<vfm>_...` and new
#     tag-prefixed `vg_dl3dv_<vfm>_...` names. Pass multiple times to allow
#     several. SenseNova runs are skipped by default because their DL3DV
#     features are not part of the regular FEAT tree.
#   * --bd filters by the backbone-depth suffix in the run dir name, e.g.
#     --bd 1 --bd 2 --bd 6 keeps only runs containing `_bd1_`, `_bd2_`, or
#     `_bd6_`. This is useful for the 3D probe-depth ablation.
#   * --project overrides logger.wandb.project (default: VideoProbe3D from
#     configs/logger/wandb.yaml). Useful for routing finegrained eval runs
#     into a separate wandb project.
#   * --task-name changes the Hydra task/output folder for eval results.
#     Default is dl3dv-eval, which writes to logs/dl3dv-eval/. Use a new
#     value such as dl3dv-eval-full-main to keep full main-table reruns
#     separate from previous quick/viz evals.
#   * --run-name filters by exact run directory basename. Pass multiple times
#     to evaluate an explicit whitelist, which is the safest way to rerun the
#     main table without accidentally including layer/depth ablations.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/logs/dl3dv/runs"

VIEWS=""
DRY_RUN=0
SKIP_DONE=0
DONE_DIR=""
VIZ_SAMPLES=0
SAVE_VIZ=0
LIMIT_TEST_SCENES=""
LIMIT_TEST_BATCHES=""
VIDEO_TSS_LIST=()
RUN_NAME_LIST=()  # exact run directory basenames to evaluate. Empty = no filter.
VFM_LIST=()  # only run ckpts whose name matches one of these VFM substrings. Empty = no filter.
BD_LIST=()   # only run ckpts whose name contains _bdN_. Empty = no filter.
WANDB_PROJECT=""
EVAL_TASK_NAME="dl3dv-eval"
CASE_OUTPUT_ROOT="" # Optional extra case-study root organized as scene/model files.
CASE_OUTPUT_KEY=""
VIZ_BATCH_SAMPLES=""
VIZ_SCENE_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --done-dir) DONE_DIR="$2"; shift 2 ;;
    --viz-samples) VIZ_SAMPLES="$2"; shift 2 ;;
    --save-viz) SAVE_VIZ=1; shift ;;
    --limit-test-scenes) LIMIT_TEST_SCENES="$2"; shift 2 ;;
    --limit-test-batches) LIMIT_TEST_BATCHES="$2"; shift 2 ;;
    --video-tss) VIDEO_TSS_LIST+=("$2"); shift 2 ;;
    --run-name) RUN_NAME_LIST+=("$2"); shift 2 ;;
    --vfm) VFM_LIST+=("$2"); shift 2 ;;
    --bd) BD_LIST+=("$2"); shift 2 ;;
    --task-name) EVAL_TASK_NAME="$2"; shift 2 ;;
    --project) WANDB_PROJECT="$2"; shift 2 ;;
    --case-output-root) CASE_OUTPUT_ROOT="$2"; shift 2 ;;
    --case-output-key) CASE_OUTPUT_KEY="$2"; shift 2 ;;
    --viz-batch-sample) VIZ_BATCH_SAMPLES="${VIZ_BATCH_SAMPLES:+${VIZ_BATCH_SAMPLES},}$2"; shift 2 ;;
    --viz-scene-id) VIZ_SCENE_IDS+=("$2"); shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$VIEWS" ]]; then
  echo "error: --views is required (e.g. --views 4)" >&2
  exit 2
fi

if [[ -z "$DONE_DIR" ]]; then
  DONE_DIR="$PROJECT_ROOT/logs/${EVAL_TASK_NAME}/done"
fi

cd "$PROJECT_ROOT"

cfg_value() {
  local cfg_path="$1"
  local key="$2"
  python - "$cfg_path" "$key" <<'PY'
import sys
from omegaconf import ListConfig, OmegaConf

cfg = OmegaConf.load(sys.argv[1])
value = OmegaConf.select(cfg, sys.argv[2])
if value is None:
    print("")
elif isinstance(value, (list, tuple, ListConfig)):
    print("[" + ",".join(str(int(x)) for x in value) + "]")
else:
    print(value)
PY
}

is_video_model_name() {
  local n="$1"
  [[ "$n" == wan-* || "$n" == opensora* || "$n" == cogvideox-* \
     || "$n" == aether* || "$n" == vjepa* ]]
}

# Collect all Hydra runs and filter by config instead of folder suffix. This
# keeps short release names and older sweep-style names both evaluable.
mapfile -t CANDIDATE_DIRS < <(find "$RUNS_DIR" -maxdepth 1 -type d | sort)

if [[ ${#CANDIDATE_DIRS[@]} -eq 0 ]]; then
  echo "No runs found under $RUNS_DIR" >&2
  exit 1
fi

# Pre-filter pass: apply --video-tss filter to wan-t2v-1.3b/opensora runs so the "Found"
# listing reflects what we'll actually evaluate. ckpt/overrides existence and
# bs-parse failures are still handled inside the eval loop (they emit [skip]
# lines there with full context).
RUN_DIRS=()
SKIPPED_TSS=()
SKIPPED_RUN_NAME=()
SKIPPED_VFM=()
SKIPPED_BD=()
SKIPPED_DONE=()
SKIPPED_SENSENOVA=()
MATCHED_RUN_NAMES=()

done_path_for() {
  local run_name="$1"
  printf '%s/%s.done' "$DONE_DIR" "$run_name"
}

for d in "${CANDIDATE_DIRS[@]}"; do
  RUN_NAME="$(basename "$d")"
  CFG="$d/.hydra/config.yaml"
  if [[ ! -f "$CFG" ]]; then
    continue
  fi

  CFG_VIEWS="$(cfg_value "$CFG" "gt_num_frames")"
  [[ "$CFG_VIEWS" == "$VIEWS" ]] || continue
  CFG_TSS="$(cfg_value "$CFG" "target_spatial_size")"
  CFG_VFM="$(cfg_value "$CFG" "vfm_name")"

  if [[ ${#RUN_NAME_LIST[@]} -gt 0 ]]; then
    run_name_matched=0
    for wanted in "${RUN_NAME_LIST[@]}"; do
      if [[ "$RUN_NAME" == "$wanted" ]]; then
        run_name_matched=1; break
      fi
    done
    if [[ $run_name_matched -eq 0 ]]; then
      SKIPPED_RUN_NAME+=("$RUN_NAME")
      continue
    fi
    MATCHED_RUN_NAMES+=("$RUN_NAME")
  fi

  # --vfm filter: prefer the stored config vfm_name, with run-name matching as
  # a fallback for legacy runs.
  if [[ ${#VFM_LIST[@]} -gt 0 ]]; then
    vfm_matched=0
    for v in "${VFM_LIST[@]}"; do
      if [[ "$CFG_VFM" == "$v" || "$RUN_NAME" == "$v" || "$RUN_NAME" == ${v}_* || "$RUN_NAME" == *_${v} || "$RUN_NAME" == *_${v}_* ]]; then
        vfm_matched=1; break
      fi
    done
    if [[ $vfm_matched -eq 0 ]]; then
      SKIPPED_VFM+=("$RUN_NAME")
      continue
    fi
  fi

  if [[ "$RUN_NAME" == *sensenova* ]]; then
    SKIPPED_SENSENOVA+=("$RUN_NAME")
    continue
  fi

  if [[ ${#BD_LIST[@]} -gt 0 ]]; then
    bd_matched=0
    for bd in "${BD_LIST[@]}"; do
      if [[ "$RUN_NAME" == *_bd${bd}_* ]]; then
        bd_matched=1; break
      fi
    done
    if [[ $bd_matched -eq 0 ]]; then
      SKIPPED_BD+=("$RUN_NAME")
      continue
    fi
  fi

  if [[ ${#VIDEO_TSS_LIST[@]} -gt 0 ]] && is_video_model_name "$CFG_VFM"; then
    GOT="${CFG_TSS#[}"
    GOT="${GOT%]}"
    GOT="${GOT// /}"
    matched=0
    for v in "${VIDEO_TSS_LIST[@]}"; do
      w="${v#[}"; w="${w%]}"; w="${w// /}"
      [[ "$w" == "$GOT" ]] && matched=1
    done
    if [[ $matched -eq 0 ]]; then
      SKIPPED_TSS+=("$RUN_NAME")
      continue
    fi
  fi
  if [[ $SKIP_DONE -eq 1 && -f "$(done_path_for "$RUN_NAME")" ]]; then
    SKIPPED_DONE+=("$RUN_NAME")
    continue
  fi
  RUN_DIRS+=("$d")
done

if [[ ${#RUN_NAME_LIST[@]} -gt 0 ]]; then
  MISSING_RUN_NAMES=()
  for wanted in "${RUN_NAME_LIST[@]}"; do
    found=0
    for matched in "${MATCHED_RUN_NAMES[@]}"; do
      if [[ "$wanted" == "$matched" ]]; then
        found=1; break
      fi
    done
    if [[ $found -eq 0 ]]; then
      MISSING_RUN_NAMES+=("$wanted")
    fi
  done
  if [[ ${#MISSING_RUN_NAMES[@]} -gt 0 ]]; then
    echo "Missing ${#MISSING_RUN_NAMES[@]} requested --run-name value(s) under $RUNS_DIR for views=${VIEWS}:" >&2
    for n in "${MISSING_RUN_NAMES[@]}"; do echo "  - $n" >&2; done
    exit 1
  fi
fi

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  if [[ $SKIP_DONE -eq 1 && ${#SKIPPED_DONE[@]} -gt 0 ]]; then
    echo "No runs left: ${#SKIPPED_DONE[@]} run(s) already have done markers in $DONE_DIR"
    exit 0
  fi
  if [[ ${#SKIPPED_SENSENOVA[@]} -gt 0 ]]; then
    echo "No runs left: ${#SKIPPED_SENSENOVA[@]} SenseNova run(s) skipped by default"
    exit 0
  fi
    echo "No runs left after filters (--views: $VIEWS, --run-name: ${RUN_NAME_LIST[*]:-none}, --vfm: ${VFM_LIST[*]:-none}, --bd: ${BD_LIST[*]:-none}, --video-tss: ${VIDEO_TSS_LIST[*]:-none}, --skip-done: $SKIP_DONE)" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) for views=${VIEWS}:"
for d in "${RUN_DIRS[@]}"; do echo "  - $(basename "$d")"; done
if [[ $SKIP_DONE -eq 1 ]]; then
  echo
  echo "Done markers: $DONE_DIR"
fi
if [[ ${#RUN_NAME_LIST[@]} -gt 0 ]]; then
  echo
  echo "Exact run-name filter enabled (${#RUN_NAME_LIST[@]} wanted)."
fi
if [[ ${#SKIPPED_VFM[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_VFM[@]} run(s) by --vfm filter (wanted: ${VFM_LIST[*]}):"
  for n in "${SKIPPED_VFM[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_TSS[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_TSS[@]} wan-t2v-1.3b/opensora run(s) by --video-tss filter (wanted: ${VIDEO_TSS_LIST[*]}):"
  for n in "${SKIPPED_TSS[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_BD[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_BD[@]} run(s) by --bd filter (wanted: ${BD_LIST[*]}):"
  for n in "${SKIPPED_BD[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_DONE[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_DONE[@]} run(s) with existing done markers:"
  for n in "${SKIPPED_DONE[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_SENSENOVA[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_SENSENOVA[@]} SenseNova run(s) by default:"
  for n in "${SKIPPED_SENSENOVA[@]}"; do echo "  - $n"; done
fi
echo

mark_done() {
  local run_name="$1"
  shift
  local done_path
  done_path="$(done_path_for "$run_name")"
  mkdir -p "$DONE_DIR"
  {
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'run_name=%s\n' "$run_name"
    printf 'command='
    printf '%q ' python -m probing_vlm_vgm.train "$@"
    printf '\n'
  } > "$done_path"
}

for RUN_DIR in "${RUN_DIRS[@]}"; do
  RUN_NAME="$(basename "$RUN_DIR")"
  CKPT="$RUN_DIR/checkpoints/last.ckpt"
  OVERRIDES="$RUN_DIR/.hydra/overrides.yaml"
  CFG="$RUN_DIR/.hydra/config.yaml"

  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] $RUN_NAME — no last.ckpt" >&2
    continue
  fi
  if [[ ! -f "$OVERRIDES" ]]; then
    echo "[skip] $RUN_NAME — no .hydra/overrides.yaml" >&2
    continue
  fi
  if [[ ! -f "$CFG" ]]; then
    echo "[skip] $RUN_NAME — no .hydra/config.yaml" >&2
    continue
  fi

  BS="$(cfg_value "$CFG" "batch_size")"
  TSS_VALUE="$(cfg_value "$CFG" "target_spatial_size")"

  # Extract original training overrides, dropping trainer.devices=* (we force =1).
  ARGS=()
  while IFS= read -r line; do
    # strip leading "- " and surrounding whitespace
    arg="${line#- }"
    arg="${arg#"${arg%%[![:space:]]*}"}"
    [[ -z "$arg" ]] && continue
    [[ "$arg" == trainer.devices=* ]] && continue
    ARGS+=("$arg")
  done < "$OVERRIDES"

  # Append our eval-mode overrides. Order matters: later overrides win in Hydra.
  # skip_test_viz=true skips per-batch PCA+grid+viser (metrics still computed).
  EVAL_BS="$BS"
  SKIP_TEST_VIZ=true
  if [[ "$VIZ_SAMPLES" -gt 0 ]]; then
    EVAL_BS="$VIZ_SAMPLES"
    SKIP_TEST_VIZ=false
  elif [[ "$SAVE_VIZ" -eq 1 ]]; then
    SKIP_TEST_VIZ=false
  fi
  LIMIT_BATCHES_FOR_RUN="$LIMIT_TEST_BATCHES"
  if [[ -n "$LIMIT_TEST_SCENES" ]]; then
    LIMIT_BATCHES_FOR_RUN=$(( (LIMIT_TEST_SCENES + EVAL_BS - 1) / EVAL_BS ))
  fi

  ARGS+=(
    "gt_num_frames=${VIEWS}"
    "batch_size=${EVAL_BS}"
    "trainer.devices=1"
    "task_name=${EVAL_TASK_NAME}"
    "train=false"
    "test=true"
    "autoresume=false"
    "+model.skip_test_viz=${SKIP_TEST_VIZ}"
    # Single-quote the path so Hydra doesn't try to parse "[15, 26]" in the
    # folder name as a list literal. The outer double quotes are bash; the
    # inner single quotes are part of Hydra's value grammar.
    "ckpt_path='${CKPT}'"
  )

  if [[ "$VIZ_SAMPLES" -gt 0 ]]; then
    ARGS+=("+trainer.limit_test_batches=1")
  elif [[ -n "$LIMIT_BATCHES_FOR_RUN" ]]; then
    ARGS+=("+trainer.limit_test_batches=${LIMIT_BATCHES_FOR_RUN}")
  fi

  # Re-pass target_spatial_size from the stored run config when present.
  if [[ -n "$TSS_VALUE" ]]; then
    ARGS+=("target_spatial_size=${TSS_VALUE}")
  fi

  # Override wandb project if --project was passed.
  if [[ -n "$WANDB_PROJECT" ]]; then
    ARGS+=("logger.wandb.project=${WANDB_PROJECT}")
  fi

  if [[ -n "$CASE_OUTPUT_ROOT" ]]; then
    ARGS+=(
      "+model.case_output_root='${CASE_OUTPUT_ROOT}'"
      "+model.case_model_name='${RUN_NAME}'"
    )
    if [[ -n "$CASE_OUTPUT_KEY" ]]; then
      ARGS+=("+model.case_output_key='${CASE_OUTPUT_KEY}'")
    fi
    if [[ -n "$VIZ_BATCH_SAMPLES" ]]; then
      ARGS+=("+model.case_viz_batch_samples='${VIZ_BATCH_SAMPLES}'")
    fi
    if [[ ${#VIZ_SCENE_IDS[@]} -gt 0 ]]; then
      VIZ_SCENE_JOINED="$(IFS=,; echo "${VIZ_SCENE_IDS[*]}")"
      ARGS+=("+model.case_viz_scene_ids='${VIZ_SCENE_JOINED}'")
    fi
  fi

  echo "=== $RUN_NAME ==="
  printf '  %s\n' "${ARGS[@]}"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] skipping execution"
    echo
    continue
  fi

  python -m probing_vlm_vgm.train "${ARGS[@]}"
  mark_done "$RUN_NAME" "${ARGS[@]}"
  echo
done
