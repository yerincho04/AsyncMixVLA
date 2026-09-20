#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKDIR=${ASYNC_MIX_VLA_WORKDIR:-$ROOT/runs}
OFT_PYTHON=${OFT_PYTHON:-python}
TRIGGER=${TRIGGER:-v1_v2_cascade_continuous}

case "$TRIGGER" in
  v1_v2_cascade_continuous)
    RUN_NAME=${RUN_NAME:-table2_learned_k4_jointv2}
    MARK=${MARK:-v1v2cont_k4_jointv2}
    ;;
  oracle_onset)
    RUN_NAME=${RUN_NAME:-table2_oracle_k4_jointv2}
    MARK=${MARK:-oracle_k4_jointv2}
    ;;
  *) echo "unsupported TRIGGER=$TRIGGER" >&2; exit 2 ;;
esac

cd "$ROOT/openvla-oft"
"$OFT_PYTHON" analyze_final_switching_corrected.py \
  --trigger "$TRIGGER" --root "$WORKDIR/$RUN_NAME" \
  --manifest "$ROOT/results/manifests/final_test/test_episode_manifest_portable.json" \
  --mark "$MARK" --split TEST_K4_JOINTV2 --require_complete
