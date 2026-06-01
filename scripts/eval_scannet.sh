#!/usr/bin/env bash
# Batch-evaluate trained ScanNet instance-probe checkpoints on val.json.
#
# Mirrors scripts/eval_dl3dv.sh:
#   1. For each run under logs/scannet-instance/runs/ matching the requested views:
#       - Reuses the original training overrides from .hydra/overrides.yaml
#         (experiment=..., job_name=..., feat_postfix=..., ~trainer.plugins, ...)
#         minus trainer.devices=* (we force single-GPU for eval).
#       - Re-injects gt_num_frames/batch_size/target_spatial_size read from
#         the run's .hydra/config.yaml in case yaml defaults drifted.
#       - Switches task_name to scannet-instance-eval so outputs land in
#         logs/scannet-instance-eval/.
#       - Loads checkpoints/best.ckpt (best val/loss — written by the instance
#         callbacks preset). Falls back to the single epoch_*.ckpt for runs that
#         predate the static filename (save_top_k=1 → that one IS the best),
#         then to last.ckpt.
#       - Runs train.py with train=false test=true autoresume=false ckpt_path=<ckpt>.
#
# Usage:
#   # Minimal: single GPU, sequential, no viz.
#   bash scripts/eval_scannet.sh --views 8 --video-tss "[15, 26]" --video-tss "[15, 22]"
#
#   # Multi-GPU (work-pool, recommended for batch sweep).
#   bash scripts/eval_scannet.sh --views 8 --video-tss "[15, 26]" --video-tss "[15, 22]" --gpus "1,2,3,4,5,6,7"
#
#   # Only one VFM family (e.g. wan-t2v-1.3b layer sweep). Pass --vfm multiple times for
#   # several. Matches by run-name prefix scannet-instance_<vfm>_, so `--vfm
#   # qwen3-vl-8b` covers both the plain and sensenova variants.
#   bash scripts/eval_scannet.sh --views 8 --video-tss "[15, 26]" --vfm wan-t2v-1.3b
#
#   # Route all 37 runs to a new wandb project (created on first push).
#   bash scripts/eval_scannet.sh --views 8 --video-tss "[15, 26]" --video-tss "[15, 22]" --project VLM-VG-3D-InstanceEval -- "data.data_module.validation_datasets=['ScanNetInstanceDataset(root=\"\${data.data_root}\", root_vfm=\"data/ScanNet/FEAT\", split=\"val\", vfm_name=\"\${vfm_name}\", feat_postfix=\"\${feat_postfix}\", feat_pixalign=True, num_views=\${gt_num_frames}, min_view_interval=5, context_len=76, query_idx_divisor=4, seed=0, target_spatial_size=\${target_spatial_size}, load_images=True)']"

#   # Resume an interrupted sweep: skip ckpts that already finished eval.
#   bash scripts/eval_scannet.sh --views 8 --skip-done --project VLM-VG-3D-InstanceEval
#
#   # Re-run everything even if done markers exist.
#   bash scripts/eval_scannet.sh --views 8 --skip-done --force
#
#   # Store/read done markers somewhere else.
#   bash scripts/eval_scannet.sh --views 8 --skip-done --done-dir /tmp/instance-eval-done
#
#   # Random-3 viz across every ckpt (deterministic via seed). `load_images=True`
#   # has to enter the ScanNet val-dataset string, so we CLI-override the whole
#   # validation_datasets list — anything after `--` is passed verbatim to each
#   # per-ckpt train.py call. The dataset string uses Python double-quotes
#   # internally so they don't collide with hydra's list-element single-quotes;
#   # `\$` keeps `${data.data_root}` etc. literal through bash so hydra (not
#   # bash) resolves them from each per-ckpt config.
#   bash scripts/eval_scannet.sh --views 8 --video-tss "[15, 26]" --video-tss "[15, 22]" --gpus "0,1,2,3,4,5,6,7" -- "data.data_module.validation_datasets=['ScanNetInstanceDataset(root=\"\${data.data_root}\", root_vfm=\"data/ScanNet/FEAT\", split=\"val\", vfm_name=\"\${vfm_name}\", feat_postfix=\"\${feat_postfix}\", feat_pixalign=True, num_views=\${gt_num_frames}, min_view_interval=5, context_len=76, query_idx_divisor=4, seed=0, target_spatial_size=\${target_spatial_size}, load_images=True)']"
       
#
# Notes:
#   * Run directories can use short release names such as
#     `vlm_scannet-instance_qwen3-vl-8b`. Evaluation metadata is read from
#     `.hydra/config.yaml`, not parsed from the folder name.
#   * --video-tss restricts wan-t2v-1.3b/opensora/cogvideox-i2v-5b/aether/vjepa to only the
#     matching target_spatial_size variants (e.g. "[15, 22]"); InternVL3/Qwen3VL
#     and DINO runs are unaffected and run as-is. Pass multiple times to allow
#     several variants.
#   * test_step in instance_probe_module.py runs the heavy path (MVC loss +
#     HDBSCAN + T-mIoU/T-SR aggregation), so a single eval pass produces all
#     val/* metrics — no extra flags required.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/logs/scannet-instance/runs"
DEFAULT_DONE_DIR="$PROJECT_ROOT/logs/scannet-instance-eval/done"

VIEWS=""
DRY_RUN=0
SKIP_DONE=0
FORCE=0
DONE_DIR="$DEFAULT_DONE_DIR"
VIDEO_TSS_LIST=()
VFM_LIST=()  # only run ckpts for these VFMs (matched by config vfm_name or run name). Empty = no filter.
GPUS=""   # comma-separated list, e.g. "1,2,3,4,5,6,7". Empty = single-GPU sequential.
PROJECT=""  # wandb project name. Empty = use logger/wandb.yaml default.
EXTRA_ARGS=()  # everything after `--` is appended verbatim to train.py CLI.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --force) FORCE=1; shift ;;
    --done-dir) DONE_DIR="$2"; shift 2 ;;
    --video-tss) VIDEO_TSS_LIST+=("$2"); shift 2 ;;
    --vfm) VFM_LIST+=("$2"); shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$VIEWS" ]]; then
  echo "error: --views is required (e.g. --views 8)" >&2
  exit 2
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

# Pre-filter pass: apply --video-tss filter to video-model runs so the
# "Found" listing reflects what we'll actually evaluate. Video models in
# ScanNet: wan-t2v-1.3b / opensora / cogvideox-i2v-5b / aether / vjepa. (DINO and the VLM
# families are excluded — they don't share a token-matched comparison axis.)
RUN_DIRS=()
SKIPPED_TSS=()
SKIPPED_VFM=()
SKIPPED_DONE=()
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

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  if [[ $SKIP_DONE -eq 1 && ${#SKIPPED_DONE[@]} -gt 0 ]]; then
    echo "No runs left: ${#SKIPPED_DONE[@]} run(s) already have done markers in $DONE_DIR"
    exit 0
  fi
  echo "No runs left after filters (--views: $VIEWS, --vfm: ${VFM_LIST[*]:-none}, --video-tss: ${VIDEO_TSS_LIST[*]:-none}, --skip-done: $SKIP_DONE)" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) for views=${VIEWS}:"
for d in "${RUN_DIRS[@]}"; do echo "  - $(basename "$d")"; done
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
  POOL_LOG_DIR="$PROJECT_ROOT/logs/scannet-instance-eval/pool-$(date +%Y%m%d-%H%M%S)"
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
  CKPT_DIR="$RUN_DIR/checkpoints"

  # Checkpoint resolution, best → epoch_* → last:
  #   1. best.ckpt          — new runs (configs/callbacks/instance.yaml writes
  #                           the best-val/loss ckpt to this static name).
  #   2. the single epoch_*.ckpt — old runs predate the static filename. With
  #                           save_top_k=1 there's exactly one, and it IS the
  #                           best-val/loss ckpt (just epoch-numbered). If more
  #                           than one is found (unexpected — would mean
  #                           save_top_k>1), we can't tell which is best from
  #                           the name alone, so fall through to last.ckpt.
  #   3. last.ckpt          — degenerate fallback (run crashed before val, or
  #                           a non-standard checkpoint config).
  CKPT=""
  if [[ -f "$CKPT_DIR/best.ckpt" ]]; then
    CKPT="$CKPT_DIR/best.ckpt"
  else
    mapfile -t EPOCH_CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -name 'epoch_*.ckpt' 2>/dev/null | sort)
    if [[ ${#EPOCH_CKPTS[@]} -eq 1 ]]; then
      CKPT="${EPOCH_CKPTS[0]}"
      echo "[info] $RUN_NAME — no best.ckpt, using ${EPOCH_CKPTS[0]##*/} (save_top_k=1 best)" >&2
    elif [[ -f "$CKPT_DIR/last.ckpt" ]]; then
      CKPT="$CKPT_DIR/last.ckpt"
      if [[ ${#EPOCH_CKPTS[@]} -gt 1 ]]; then
        echo "[warn] $RUN_NAME — ${#EPOCH_CKPTS[@]} epoch_*.ckpt found, can't pick best; using last.ckpt" >&2
      else
        echo "[warn] $RUN_NAME — no best.ckpt/epoch_*.ckpt, falling back to last.ckpt" >&2
      fi
    fi
  fi
  if [[ -z "$CKPT" ]]; then
    echo "[skip] $RUN_NAME — no best.ckpt / epoch_*.ckpt / last.ckpt" >&2
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
    arg="${line#- }"
    arg="${arg#"${arg%%[![:space:]]*}"}"
    [[ -z "$arg" ]] && continue
    [[ "$arg" == trainer.devices=* ]] && continue
    ARGS+=("$arg")
  done < "$OVERRIDES"

  # Append eval-mode overrides. Order matters: later overrides win in Hydra.
  ARGS+=(
    "gt_num_frames=${VIEWS}"
    "batch_size=${BS}"
    "trainer.devices=1"
    "task_name=scannet-instance-eval"
    "train=false"
    "test=true"
    "autoresume=false"
    "ckpt_path='${CKPT}'"
    "model.num_eval_workers=8"
    "data.data_module.num_workers_val=4"
    "model.viz_random_n=3"
    "model.viz_random_seed=0"
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

  # Caller-supplied hydra overrides via `--` pass-through (last → wins in hydra).
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
    # next `read -u 9 GPU` is going to block (not hang). `jobs -p` lists
    # background PIDs in this shell — once N_GPUS of them are still alive,
    # the next read waits for one to release its slot.
    if [[ "$(jobs -pr | wc -l)" -ge "$N_GPUS" ]]; then
      echo "[wait]  pool saturated ($N_GPUS in-flight); next slot blocks until a GPU frees"
    fi
    # Block until a GPU is free, then dispatch to background.
    read -u 9 GPU
    run_one "$GPU" "$RUN_NAME" "${ARGS[@]}" &
  else
    # Single-GPU sequential (original behaviour).
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
