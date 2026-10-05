#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=/dev/null
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/agent_common.sh"

NUM_ENVS="${NUM_ENVS:-${EVAL_NUM_ENVS:-1}}"
LIMIT="${EVAL_TIMEOUT_SECONDS:-300}"
PLAY_TASK="${PLAY_TASK:-$EVAL_TASK}"
STAMP="$(timestamp)"
LOGDIR="${LOGDIR:-$RUN_ROOT/eval_$STAMP}"
CONSOLE_LOG="$LOGDIR/console.log"
mkdir -p "$LOGDIR"

echo "=== Agent skrl PPO CommandFollow evaluation ==="
print_context
echo "play_task=$PLAY_TASK"
echo "num_envs=$NUM_ENVS"
echo "console_log=$CONSOLE_LOG"

check_skrl_layout

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "CHECKPOINT is required." >&2
  echo "Example:" >&2
  echo "  CHECKPOINT=logs/skrl/.../checkpoints/agent_XXXXX.pt ./scripts/agent_eval.sh" >&2
  exit 2
fi

if [[ "$CHECKPOINT" != /* && -e "$REPO_ROOT/$CHECKPOINT" ]]; then
  CHECKPOINT="$REPO_ROOT/$CHECKPOINT"
fi

if [[ ! -e "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi

if ! check_task_registered "$PLAY_TASK"; then
  echo "[FAIL] Play task is not registered: $PLAY_TASK" >&2
  exit 2
fi

cmd=(
  "$ISAACLAB_SH" -p "$PLAY_SCRIPT"
  --task "$PLAY_TASK"
  --num_envs "$NUM_ENVS"
  --checkpoint "$CHECKPOINT"
)
append_skrl_selection_args cmd
append_headless_arg cmd "$EVAL_HEADLESS"

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
    echo "[PASS] Evaluation/play exited normally."
    ;;
  124|130|137)
    echo "[PASS] Evaluation/play reached the configured safety time limit without a detected fatal error."
    ;;
  *)
    echo "[FAIL] Evaluation/play exit code: $rc" >&2
    exit "$rc"
    ;;
esac

echo "$LOGDIR" > "$AGENT_DIR/status/last_eval_run.txt"
echo "console_log=$CONSOLE_LOG"
