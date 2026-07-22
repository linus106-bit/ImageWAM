#!/usr/bin/env bash
# Shared helpers for ImageWAM release scripts.
# Source this file from bash entrypoints after `set -euo pipefail`.

imagewam_init() {
  local default_root="$1"
  REPO_ROOT="${REPO_ROOT:-$(cd "${default_root}" && pwd)}"
  export REPO_ROOT
  cd "${REPO_ROOT}"

  if [ -f "${REPO_ROOT}/.env.local" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env.local"
    set +a
  fi
}

imagewam_require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    echo "Set it in the shell or in ${REPO_ROOT}/.env.local" >&2
    exit 2
  fi
}

imagewam_print_config() {
  if [ "${IMAGEWAM_QUIET:-false}" = "true" ]; then
    return 0
  fi
  local name
  for name in "$@"; do
    printf '[config] %s=%s\n' "${name}" "${!name:-<unset>}"
  done
}

imagewam_run() {
  if [ "${DRY_RUN:-false}" = "true" ]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

imagewam_activate_env() {
  local env_file="${1:-}"
  if [ -n "${env_file}" ]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

imagewam_python() {
  "${PYTHON_BIN:-python}" "$@"
}

imagewam_ckpt_from_exp() {
  if [ -z "${CKPT_PATH:-}" ]; then
    imagewam_require_env EXP_PATH
    imagewam_require_env EVAL_TRAIN_STEP
    local eval_step="${EVAL_TRAIN_STEP}"
    if [[ "${eval_step}" =~ ^[0-9]+$ ]]; then
      printf -v eval_step '%06d' "$((10#${eval_step}))"
    fi
    CKPT_PATH="${EXP_PATH}/checkpoints/weights/step_${eval_step}.pt"
    export CKPT_PATH
  fi
  if [ -z "${DATASET_STATS_PATH:-}" ] && [ -n "${EXP_PATH:-}" ]; then
    DATASET_STATS_PATH="${EXP_PATH}/dataset_stats.json"
    export DATASET_STATS_PATH
  fi
}

imagewam_prepare_eval_ckpt() {
  if [ -n "${LOCAL_CKPT_ROOT:-}" ]; then
    imagewam_require_env CKPT_PATH
    local task_name="${TASK:-eval}"
    local run_name="$(basename "$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")")"
    local local_path="${LOCAL_CKPT_ROOT}/runs/${task_name}/${run_name}/checkpoints/weights/$(basename "${CKPT_PATH}")"
    mkdir -p "$(dirname "${local_path}")"
    if [ ! -f "${local_path}" ] || [ "${CKPT_PATH}" -nt "${local_path}" ]; then
      echo "Copying checkpoint to local disk: ${local_path}"
      cp "${CKPT_PATH}" "${local_path}.tmp"
      mv "${local_path}.tmp" "${local_path}"
    else
      echo "Using existing local checkpoint: ${local_path}"
    fi
    CKPT_PATH="${local_path}"
    export CKPT_PATH
  fi
}
