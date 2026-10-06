#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=/dev/null
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

NUM_ENVS="${NUM_ENVS:-${SMOKE_NUM_ENVS:-4}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-${SMOKE_MAX_ITERATIONS:-2}}"
LIMIT="${SMOKE_TIMEOUT_SECONDS:-240}"
STAMP="$(timestamp)"
LOGDIR="${LOGDIR:-$RUN_ROOT/smoke_$STAMP}"
CONSOLE_LOG="$LOGDIR/console.log"
mkdir -p "$LOGDIR"

echo "=== Agent skrl PPO CommandFollow smoke test ==="
print_context
echo "num_envs=$NUM_ENVS"
echo "max_iterations=$MAX_ITERATIONS"
echo "console_log=$CONSOLE_LOG"

echo "[1/3] Checking Isaac Lab / skrl script layout..."
check_skrl_layout

echo "[2/3] Checking training task registration..."
if ! check_task_registered "$TASK"; then
  cat >&2 <<EOF2
[FAIL] Gym cannot find:
  $TASK

Check the new package import/registration, especially:
  isaaclab_tasks.user.traction_protective_control.config.car4wd
and the repository-level isaaclab_tasks package import path.
EOF2
  exit 2
fi

echo "[3/3] Starting a bounded skrl PPO training smoke run..."
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
    echo "[PASS] Smoke training completed normally."
    ;;
  124|130|137)
    echo "[PASS] Smoke run reached the safety time limit without a detected fatal error."
    ;;
  *)
    echo "[FAIL] Smoke run exit code: $rc" >&2
    exit "$rc"
    ;;
esac

echo "$LOGDIR" > "$AGENT_DIR/status/last_smoke_run.txt"
echo "console_log=$CONSOLE_LOG"
echo "Note: skrl outputs are under /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow"
