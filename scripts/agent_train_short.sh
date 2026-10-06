#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=/dev/null
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

NUM_ENVS="${NUM_ENVS:-${SHORT_NUM_ENVS:-64}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-${SHORT_MAX_ITERATIONS:-50}}"
LIMIT="${SHORT_TIMEOUT_SECONDS:-900}"
STAMP="$(timestamp)"
LOGDIR="${LOGDIR:-$RUN_ROOT/short_$STAMP}"
CONSOLE_LOG="$LOGDIR/console.log"
mkdir -p "$LOGDIR"

echo "=== Agent skrl PPO CommandFollow short training ==="
print_context
echo "num_envs=$NUM_ENVS"
echo "max_iterations=$MAX_ITERATIONS"
echo "console_log=$CONSOLE_LOG"

check_skrl_layout

if ! check_task_registered "$TASK"; then
  echo "[FAIL] Training task is not registered: $TASK" >&2
  exit 2
fi

cmd=(
  "$ISAACLAB_SH" -p "$TRAIN_SCRIPT"
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
)
append_skrl_selection_args cmd
append_headless_arg cmd "$TRAIN_HEADLESS"

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

set +e
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
run_guarded "$LIMIT" "${cmd[@]}" "$@" 2>&1 | tee "$CONSOLE_LOG"
rc=${PIPESTATUS[0]}
set -e

if ! scan_log_for_fatal_errors "$CONSOLE_LOG"; then
  exit 1
fi

case "$rc" in
  0)
    echo "[PASS] Short PPO training completed normally."
    ;;
  124|130|137)
    echo "[PASS] Short PPO training reached the configured safety time limit without a detected fatal error."
    ;;
  *)
    echo "[FAIL] Short PPO training exit code: $rc" >&2
    exit "$rc"
    ;;
esac

echo "$LOGDIR" > "$AGENT_DIR/status/last_short_run.txt"

echo
echo "Next:"
echo "  1. Inspect the console log: $CONSOLE_LOG"
echo "  2. Inspect the skrl run under /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/..."
echo "  3. Check speed tracking, heading tracking, stability, and action smoothness trends."
echo "  4. Run agent_eval.sh only after a usable skrl checkpoint exists."
