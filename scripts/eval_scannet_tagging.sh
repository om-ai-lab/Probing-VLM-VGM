#!/usr/bin/env bash
# Batch-evaluate trained ScanNet semantic-tagging (Exp-B) checkpoints on val.json.
#
# Mirrors scripts/eval_scannet.sh, with the instance-probe specifics removed:
#   - No HDBSCAN / num_eval_workers   (tagging eval is just logit accumulation
#                                      + compute_tagging_metrics; cheap, GPU-bound)
#   - No viz_random_n / viz_scene_ids (tagging has no per-scene cluster viz)
#
# For each run under logs/scannet-tagging/runs/ matching the requested views:
#   - Reuses the original training overrides from .hydra/overrides.yaml
#     (experiment=..., job_name=..., feat_postfix=..., +trainer.* adds, ...)
#     minus trainer.devices=* (we force single-GPU for eval).
#   - Re-injects gt_num_frames / batch_size / target_spatial_size read from
#     the run's .hydra/config.yaml in case yaml defaults drifted.
#   - Forces trainer=local so eval runs interactively regardless of how the
#     ckpt was trained (a SLURM-trained run's overrides.yaml may carry
#     trainer=ddp, which would crash an interactive eval on KeyError SLURM_*).
#   - Switches task_name to scannet-tagging-eval so outputs land in
#     logs/scannet-tagging-eval/.
#   - Loads checkpoints/best.ckpt (highest val/mAP — written by the tagging
#     callbacks preset). Falls back to last.ckpt with a warning for runs that
#     predate the static filename or crashed before the first validation.
#   - Runs train.py with train=false test=true autoresume=false ckpt_path=<ckpt>.
#
# test_step in tagging_probe_module.py delegates to validation_step, and
# on_test_epoch_end delegates to on_validation_epoch_end — so a single eval
# pass produces all val/* metrics (val/mAP, val/AP_head|mid|tail, val/OF1,
# val/CF1). No extra flags required.
#
# IMPORTANT: SemanticTagHead.clip_init is a non-persistent buffer (NOT in the
# ckpt) — it is re-loaded from `clip_init_path` (configs/model/probe_tagging.yaml)
# at model construction. The .npy file produced by build_clip_class_embeds.py
# must still exist at eval time, or model init fails. The eval reuses the
# training config so the path is inherited automatically; just don't delete
# data/ScanNet/clip_class_embeds_*.npy between train and eval.
#
# Usage:
#   # Minimal: single GPU, sequential.
#   bash scripts/eval_scannet_tagging.sh --views 8 --project VLM-VG-3D-TaggingEval
#
#   # Restrict video models to specific token-matched variants.
#   bash scripts/eval_scannet_tagging.sh --views 8 --video-tss "[15, 26]" --video-tss "[15, 22]"
#
#   # Multi-GPU work-pool (recommended for the 11-VFM sweep).
#   bash scripts/eval_scannet_tagging.sh --views 8 --gpus "0,1,2,3,4,5,6,7"
#
#   # Only one VFM family (matches by run-name token, so `--vfm qwen3-vl-8b`
#   # covers both the plain and sensenova variants). Pass --vfm repeatedly.
#   bash scripts/eval_scannet_tagging.sh --views 8 --vfm wan-t2v-1.3b
#
#   # Route all eval runs to a separate wandb project.
#   bash scripts/eval_scannet_tagging.sh --views 8 --gpus "0,1" --project VLM-VG-3D-TaggingEval
#
#   # Resume an interrupted sweep: skip ckpts that already finished eval.
#   bash scripts/eval_scannet_tagging.sh --views 8 --skip-done --project VLM-VG-3D-TaggingEval
#
#   # Re-run everything even if done markers exist.
#   bash scripts/eval_scannet_tagging.sh --views 8 --skip-done --force
#
#   # Store/read done markers somewhere else.
#   bash scripts/eval_scannet_tagging.sh --views 8 --skip-done --done-dir /tmp/tagging-eval-done
#
#   # Anything after `--` is appended verbatim to every per-ckpt train.py call.
#   bash scripts/eval_scannet_tagging.sh --views 8 -- model.probe.semantic_classifier_mode=open_vocab
#
# Notes:
#   * Run directories can use short release names such as
#     `vlm_scannet-tagging_qwen3-vl-8b`. Evaluation metadata is read from
#     `.hydra/config.yaml`, not parsed from the folder name.
#   * --video-tss restricts wan-t2v-1.3b/opensora/cogvideox-i2v-5b/aether/vjepa to only the
#     matching target_spatial_size variants; InternVL3/Qwen3VL/DINO runs are
#     unaffected. Pass multiple times to allow several variants.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/logs/scannet-tagging/runs"
DEFAULT_DONE_DIR="$PROJECT_ROOT/logs/scannet-tagging-eval/done"

VIEWS=""
DRY_RUN=0
SKIP_DONE=0
FORCE=0
DONE_DIR="$DEFAULT_DONE_DIR"
VIDEO_TSS_LIST=()
VFM_LIST=()  # only run ckpts for these VFMs (matched by config vfm_name or run name). Empty = no filter.
RUN_NAME_LIST=() # exact run directory basenames to evaluate. Empty = no filter.
GPUS=""      # comma-separated list, e.g. "0,1,2,3". Empty = single-GPU sequential.
PROJECT=""   # wandb project name. Empty = use logger/wandb.yaml default.
CASE_OUTPUT_ROOT="" # Optional extra case-study root organized as scene/model files.
EXTRA_ARGS=() # everything after `--` is appended verbatim to train.py CLI.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --force) FORCE=1; shift ;;
    --done-dir) DONE_DIR="$2"; shift 2 ;;
    --video-tss) VIDEO_TSS_LIST+=("$2"); shift 2 ;;
    --run-name) RUN_NAME_LIST+=("$2"); shift 2 ;;
    --vfm) VFM_LIST+=("$2"); shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --case-output-root) CASE_OUTPUT_ROOT="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$VIEWS" ]]; then
  echo "error: --views is required (e.g. --views 8)" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "error: no runs dir at $RUNS_DIR — train some scannet_tagging ckpts first" >&2
  exit 1
fi

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

# Pre-filter pass: apply --vfm and --video-tss filters so the "Found" listing
# reflects what we'll actually evaluate. Video models in ScanNet:
# wan-t2v-1.3b / opensora / cogvideox-i2v-5b / aether / vjepa. (DINO and the VLM families are
# excluded from the --video-tss token-matched axis — they don't share it.)
RUN_DIRS=()
SKIPPED_TSS=()
SKIPPED_VFM=()
SKIPPED_RUN_NAME=()
SKIPPED_DONE=()
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

  # --video-tss filter: only constrains video-model runs.
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

  if [[ $SKIP_DONE -eq 1 && $FORCE -eq 0 && -f "$(done_path_for "$RUN_NAME")" ]]; then
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
  echo "No runs left after filters (--views: $VIEWS, --run-name: ${RUN_NAME_LIST[*]:-none}, --vfm: ${VFM_LIST[*]:-none}, --video-tss: ${VIDEO_TSS_LIST[*]:-none}, --skip-done: $SKIP_DONE)" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) for views=${VIEWS}:"
for d in "${RUN_DIRS[@]}"; do echo "  - $(basename "$d")"; done
if [[ ${#RUN_NAME_LIST[@]} -gt 0 ]]; then
  echo
  echo "Exact run-name filter enabled (${#RUN_NAME_LIST[@]} wanted)."
fi
if [[ $SKIP_DONE -eq 1 ]]; then
  echo
  if [[ $FORCE -eq 1 ]]; then
    echo "--force set: ignoring done markers in $DONE_DIR"
  else
    echo "Done markers: $DONE_DIR"
  fi
fi
if [[ ${#SKIPPED_VFM[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_VFM[@]} run(s) by --vfm filter (wanted: ${VFM_LIST[*]}):"
  for n in "${SKIPPED_VFM[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_TSS[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_TSS[@]} video-model run(s) by --video-tss filter (wanted: ${VIDEO_TSS_LIST[*]}):"
  for n in "${SKIPPED_TSS[@]}"; do echo "  - $n"; done
fi
if [[ ${#SKIPPED_DONE[@]} -gt 0 ]]; then
  echo
  echo "Skipped ${#SKIPPED_DONE[@]} run(s) with existing done markers:"
  for n in "${SKIPPED_DONE[@]}"; do echo "  - $n"; done
fi
echo

# If --gpus was given, spin up a bash FIFO semaphore so each ckpt grabs a
# free GPU, dispatches, then releases. True work-pool scheduling (no
# batch-of-N barrier waiting on the slowest job). Pool size = #GPU IDs.
POOL_LOG_DIR=""
if [[ -n "$GPUS" ]]; then
  IFS=',' read -ra GPU_IDS <<< "$GPUS"
  N_GPUS=${#GPU_IDS[@]}
  if [[ $N_GPUS -lt 1 ]]; then
    echo "error: --gpus parsed to 0 GPUs from '$GPUS'" >&2
    exit 2
  fi
  POOL_LOG_DIR="$PROJECT_ROOT/logs/scannet-tagging-eval/pool-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$POOL_LOG_DIR"
  SEM="$(mktemp -u)"
  mkfifo "$SEM"
  exec 9<>"$SEM"
  rm "$SEM"
  for g in "${GPU_IDS[@]}"; do echo "$g" >&9; done
  echo "Pool: ${N_GPUS} GPU(s) [${GPU_IDS[*]}], per-ckpt logs → $POOL_LOG_DIR"
  echo
fi

mark_done() {
  local run_name="$1"
  shift
  local done_path
  done_path="$(done_path_for "$run_name")"
  mkdir -p "$DONE_DIR"
  {
    printf 'run_name=%s\n' "$run_name"
    printf 'finished_at=%s\n' "$(date -Is)"
    printf 'command='
    printf '%q ' python -m probing_vlm_vgm.train "$@"
    printf '\n'
  } > "$done_path"
}

# Function isolates ARGS at fork time (positional args, copy-by-value),
# so the parent loop is free to reassign ARGS for the next iteration.
run_one() {
  local gpu="$1"
  local run_name="$2"
  shift 2
  local args=("$@")
  local log_path="$POOL_LOG_DIR/${run_name}.log"
  echo "[start] gpu=$gpu  $run_name"
  local rc=0
  if CUDA_VISIBLE_DEVICES="$gpu" python -m probing_vlm_vgm.train "${args[@]}" \
      > "$log_path" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL]  gpu=$gpu  $run_name  rc=$rc  (see $log_path)" >&2
  else
    mark_done "$run_name" "${args[@]}"
    echo "[done]  gpu=$gpu  $run_name"
  fi
  # Release GPU back to the semaphore.
  echo "$gpu" >&9
  return "$rc"
}

for RUN_DIR in "${RUN_DIRS[@]}"; do
  RUN_NAME="$(basename "$RUN_DIR")"
  OVERRIDES="$RUN_DIR/.hydra/overrides.yaml"
  CFG="$RUN_DIR/.hydra/config.yaml"

  # Prefer best.ckpt (highest val/mAP, written by configs/callbacks/tagging.yaml's
  # ModelCheckpoint with filename="best"). Fall back to last.ckpt for runs that
  # crashed before the first validation, or were trained before the static
  # filename was introduced.
  if [[ -f "$RUN_DIR/checkpoints/best.ckpt" ]]; then
    CKPT="$RUN_DIR/checkpoints/best.ckpt"
  elif [[ -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
    CKPT="$RUN_DIR/checkpoints/last.ckpt"
    echo "[warn] $RUN_NAME — no best.ckpt, falling back to last.ckpt" >&2
  else
    echo "[skip] $RUN_NAME — no best.ckpt or last.ckpt" >&2
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

  # Extract original training overrides, dropping trainer.devices=* (we force
  # =1) and any trainer= group selection (we force =local for interactive eval).
  ARGS=()
  while IFS= read -r line; do
    arg="${line#- }"
    arg="${arg#"${arg%%[![:space:]]*}"}"
    [[ -z "$arg" ]] && continue
    [[ "$arg" == trainer.devices=* ]] && continue
    [[ "$arg" == trainer=* ]] && continue
    ARGS+=("$arg")
  done < "$OVERRIDES"

  # Append eval-mode overrides. Order matters: later overrides win in Hydra.
  ARGS+=(
    "gt_num_frames=${VIEWS}"
    "batch_size=${BS}"
    "trainer=local"
    "trainer.devices=1"
    "task_name=scannet-tagging-eval"
    "train=false"
    "test=true"
    "autoresume=false"
    "ckpt_path='${CKPT}'"
    "data.data_module.num_workers_val=4"
  )

  # Re-pass target_spatial_size from the stored run config when present.
  if [[ -n "$TSS_VALUE" ]]; then
    ARGS+=("target_spatial_size=${TSS_VALUE}")
  fi

  # --project flips the wandb project (creates it on first push if it
  # doesn't exist). Bare override since logger.wandb.project already exists
  # in configs/logger/wandb.yaml.
  if [[ -n "$PROJECT" ]]; then
    ARGS+=("logger.wandb.project=${PROJECT}")
  fi

  if [[ -n "$CASE_OUTPUT_ROOT" ]]; then
    ARGS+=(
      "+model.case_output_root='${CASE_OUTPUT_ROOT}'"
      "+model.case_model_name='${RUN_NAME}'"
    )
  fi

  # Caller-supplied hydra overrides via `--` pass-through (last → wins).
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA_ARGS[@]}")
  fi

  echo "=== $RUN_NAME ==="
  printf '  %s\n' "${ARGS[@]}"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] skipping execution"
    echo
    continue
  fi

  if [[ -n "$GPUS" ]]; then
    # If the pool is saturated, print a one-time hint so the user knows the
    # next `read -u 9 GPU` will block (not hang).
    if [[ "$(jobs -pr | wc -l)" -ge "$N_GPUS" ]]; then
      echo "[wait]  pool saturated ($N_GPUS in-flight); next slot blocks until a GPU frees"
    fi
    # Block until a GPU is free, then dispatch to background.
    read -u 9 GPU
    run_one "$GPU" "$RUN_NAME" "${ARGS[@]}" &
  else
    # Single-GPU sequential.
    python -m probing_vlm_vgm.train "${ARGS[@]}"
    mark_done "$RUN_NAME" "${ARGS[@]}"
    echo
  fi
done

# Wait for all pool workers to finish, close the semaphore.
if [[ -n "$GPUS" ]]; then
  WAIT_RC=0
  wait || WAIT_RC=$?
  exec 9>&-
  echo
  echo "All pool jobs complete. Per-ckpt logs in $POOL_LOG_DIR"
  if [[ $WAIT_RC -ne 0 ]]; then
    echo "At least one pool job failed; failed runs were not marked done." >&2
    exit "$WAIT_RC"
  fi
fi

echo
echo "Done. Aggregate with:"
echo "  python scripts/parse_results.py --groups scannet-tagging-eval \\"
echo "      --metrics 'val/mAP,val/AP_head,val/AP_mid,val/AP_tail,val/OF1,val/CF1'"
