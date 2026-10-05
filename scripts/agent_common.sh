#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd))"
AGENT_DIR="$REPO_ROOT/agent"

if [[ -f "$AGENT_DIR/agent.env" ]]; then
  # shellcheck disable=SC1091
  source "$AGENT_DIR/agent.env"
fi

TASK="${TASK:-Isaac-TractionProtect-CommandFollow-Car4WD-v0}"
EVAL_TASK="${EVAL_TASK:-Isaac-TractionProtect-CommandFollow-Car4WD-Play-v0}"
ALGORITHM="${ALGORITHM:-PPO}"
ML_FRAMEWORK="${ML_FRAMEWORK:-torch}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-}"
SEED="${SEED:-}"

ISAACLAB_SH="${ISAACLAB_SH:-isaaclab.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/reinforcement_learning/skrl/train.py}"
PLAY_SCRIPT="${PLAY_SCRIPT:-scripts/reinforcement_learning/skrl/play.py}"
RUN_ROOT="${RUN_ROOT:-runs/agent}"

TRAIN_HEADLESS="${TRAIN_HEADLESS:-1}"
EVAL_HEADLESS="${EVAL_HEADLESS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resolve_repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}

ISAACLAB_SH="$(resolve_repo_path "$ISAACLAB_SH")"
TRAIN_SCRIPT="$(resolve_repo_path "$TRAIN_SCRIPT")"
PLAY_SCRIPT="$(resolve_repo_path "$PLAY_SCRIPT")"
RUN_ROOT="$(resolve_repo_path "$RUN_ROOT")"

mkdir -p "$RUN_ROOT" "$AGENT_DIR/status"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "[agent] Missing $description: $path" >&2
    return 1
  fi
}

require_executable() {
  local path="$1"
  local description="$2"
  require_file "$path" "$description" || return 1
  if [[ ! -x "$path" ]]; then
    echo "[agent] $description is not executable: $path" >&2
    echo "[agent] Fix with: chmod +x \"$path\"" >&2
    return 1
  fi
}

check_skrl_layout() {
  require_executable "$ISAACLAB_SH" "Isaac Lab launcher" || return 1
  require_file "$TRAIN_SCRIPT" "skrl training script" || return 1
  require_file "$PLAY_SCRIPT" "skrl play script" || return 1
}

check_task_registered() {
  local task="$1"
  echo "[agent] Checking Gym registration: $task"
  TASK_TO_CHECK="$task" "$ISAACLAB_SH" -p -c '
import os
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

task = os.environ["TASK_TO_CHECK"]
try:
    spec = gym.spec(task)
except Exception as exc:
    raise SystemExit(f"Task is not registered: {task}\n{type(exc).__name__}: {exc}")
print(f"Task registered: {spec.id}")
' >/dev/null
}

append_skrl_selection_args() {
  local -n cmd_ref="$1"
  cmd_ref+=(--algorithm "$ALGORITHM" --ml_framework "$ML_FRAMEWORK")
  if [[ -n "$AGENT_ENTRY_POINT" ]]; then
    cmd_ref+=(--agent "$AGENT_ENTRY_POINT")
  fi
  if [[ -n "$SEED" ]]; then
    cmd_ref+=(--seed "$SEED")
  fi
}

append_headless_arg() {
  local -n cmd_ref="$1"
  local enabled="$2"
  if [[ "$enabled" == "1" || "$enabled" == "true" || "$enabled" == "TRUE" ]]; then
    cmd_ref+=(--headless)
  fi
}

print_context() {
  echo "repo_root=$REPO_ROOT"
  echo "isaaclab_sh=$ISAACLAB_SH"
  echo "train_script=$TRAIN_SCRIPT"
  echo "play_script=$PLAY_SCRIPT"
  echo "task=$TASK"
  echo "eval_task=$EVAL_TASK"
  echo "algorithm=$ALGORITHM"
  echo "ml_framework=$ML_FRAMEWORK"
  if [[ -n "$AGENT_ENTRY_POINT" ]]; then
    echo "agent_entry_point=$AGENT_ENTRY_POINT"
  else
    echo "agent_entry_point=<default skrl_cfg_entry_point for PPO>"
  fi
  echo "agent_log_root=$RUN_ROOT"
}

scan_log_for_fatal_errors() {
  local file="$1"
  if grep -Eiq \
    'Traceback \(most recent call last\)|CUDA out of memory|ModuleNotFoundError|ImportError:|Segmentation fault|core dumped|Fatal Python error|Task is not registered' \
    "$file"; then
    echo "[agent] Fatal error signature found in: $file" >&2
    return 1
  fi
  return 0
}

run_guarded() {
  local timeout_seconds="$1"
  shift

  if [[ "$timeout_seconds" =~ ^[0-9]+$ ]] && (( timeout_seconds > 0 )); then
    timeout --signal=INT --kill-after=30s "${timeout_seconds}s" "$@"
  else
    "$@"
  fi
}
