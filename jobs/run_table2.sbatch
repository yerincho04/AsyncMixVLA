#!/usr/bin/env bash
#SBATCH --job-name=asyncmixvla_table2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-9

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OFT_REPO="$ROOT/openvla-oft"
ADAPTER_REPO="$ROOT/VLA-Adapter"
WORKDIR=${ASYNC_MIX_VLA_WORKDIR:-$ROOT/runs}
TRIGGER=${TRIGGER:-v1_v2_cascade_continuous}
METHODS=${METHODS:-sync,naive,ours}
TAKEOVER_K=${TAKEOVER_K:-4}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}

: "${TASK_ID:?set TASK_ID outside Slurm, or submit this file as an array}"
: "${ADAPTER_CHECKPOINT:?set ADAPTER_CHECKPOINT to the Adapter policy directory}"
: "${OFT_CHECKPOINT:?set OFT_CHECKPOINT to the OFT policy directory}"
: "${ADAPTER_PYTHON:?set ADAPTER_PYTHON to the Adapter environment python}"
: "${OFT_PYTHON:?set OFT_PYTHON to the OFT environment python}"

case "$TRIGGER" in
  v1_v2_cascade_continuous)
    RUN_NAME=${RUN_NAME:-table2_learned_k4_jointv2}
    MARK=${MARK:-v1v2cont_k4_jointv2}
    ;;
  oracle_onset)
    RUN_NAME=${RUN_NAME:-table2_oracle_k4_jointv2}
    MARK=${MARK:-oracle_k4_jointv2}
    ;;
  *)
    echo "TRIGGER must be v1_v2_cascade_continuous or oracle_onset" >&2
    exit 2
    ;;
esac

OUT="$WORKDIR/$RUN_NAME"
LOGDIR="$WORKDIR/logs"
MANIFEST="$ROOT/results/manifests/final_test/test_episode_manifest_portable.json"
ORACLE_MANIFEST="$ROOT/results/manifests/oracle_trigger/oracle_manifest.json"
RESIDUAL="$OFT_REPO/asyncmixvla/calibration/joint_deploy_residual_v2.pt"
mkdir -p "$OUT/traces" "$LOGDIR" "$WORKDIR/libero-config-adapter" "$WORKDIR/libero-config-oft"
EXTRA=()
[[ -z "${LIMIT:-}" ]] || EXTRA+=(--limit "$LIMIT")
[[ -z "${FORCE:-}" ]] || EXTRA+=(--force)

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

JOB_TOKEN=${SLURM_ARRAY_JOB_ID:-$$}
PORT_BASE=$((20000 + (JOB_TOKEN * 17 + TASK_ID * 181) % 18000))
export ADAPTER_PORT=$PORT_BASE OFT_PORT=$((PORT_BASE + 1))
LOGPRE="$LOGDIR/${RUN_NAME}_${JOB_TOKEN}_${TASK_ID}"
AP='' OP=''

cleanup() {
  [[ -z "$AP" ]] || kill "$AP" 2>/dev/null || true
  [[ -z "$OP" ]] || kill "$OP" 2>/dev/null || true
  [[ -z "$AP" ]] || wait "$AP" 2>/dev/null || true
  [[ -z "$OP" ]] || wait "$OP" 2>/dev/null || true
}
trap cleanup EXIT

ready() {
  local pid=$1 port=$2 logfile=$3
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then tail -100 "$logfile"; return 1; fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 3<&-; return 0
    fi
    sleep 5
  done
  tail -100 "$logfile"; return 1
}

(
  cd "$ADAPTER_REPO"
  export PYTHONPATH="$ADAPTER_REPO/LIBERO:${PYTHONPATH:-}"
  export LIBERO_CONFIG_PATH="$WORKDIR/libero-config-adapter"
  exec "$ADAPTER_PYTHON" serve_adapter_libero10.py \
    --checkpoint "$ADAPTER_CHECKPOINT" --task-suite-name libero_10 \
    --host 127.0.0.1 --port "$ADAPTER_PORT" --chunk-size 8 --seed 7 \
    --no-use-pro-version --model-version v1
) >"${LOGPRE}_adapter.log" 2>&1 & AP=$!
ready "$AP" "$ADAPTER_PORT" "${LOGPRE}_adapter.log"

(
  cd "$OFT_REPO"
  export PYTHONPATH="$OFT_REPO/LIBERO:${PYTHONPATH:-}"
  export LIBERO_CONFIG_PATH="$WORKDIR/libero-config-oft"
  exec "$OFT_PYTHON" serve_oft_libero10.py \
    --checkpoint "$OFT_CHECKPOINT" --task-suite-name libero_10 \
    --host 127.0.0.1 --port "$OFT_PORT" --chunk-size 8 --seed 7 \
    --action_consistency_checkpoint "$RESIDUAL"
) >"${LOGPRE}_oft.log" 2>&1 & OP=$!
ready "$OP" "$OFT_PORT" "${LOGPRE}_oft.log"

cd "$OFT_REPO"
export PYTHONPATH="$OFT_REPO/LIBERO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$WORKDIR/libero-config-oft"
"$OFT_PYTHON" run_final_switching_tables_corrected.py \
  --task_id "$TASK_ID" --trigger "$TRIGGER" --methods "$METHODS" \
  --manifest "$MANIFEST" --oracle_manifest "$ORACLE_MANIFEST" \
  --out_dir "$OUT/traces" --mark "$MARK" --takeover_k "$TAKEOVER_K" \
  --action_consistency_checkpoint "$RESIDUAL" "${EXTRA[@]}"
