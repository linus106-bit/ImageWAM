#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

SUITE="libero"
CONFIG_NAME="sim_libero_omnigen2"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}" # 4b | 9b
ROPE_SCHEME="${ROPE_SCHEME:-current}"
TASK="${TASK:-libero_flux2_klein_${FLUX2_VARIANT}_base_imagewam}"
if [ "false" = "true" ]; then
  TASK="${TASK/_imagewam/_clean_imagewam}"
fi

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_MODEL_PATH
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export WORKER_PYTHONPATH="${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
LIBERO_WORKER_ENV_SOURCE="${LIBERO_WORKER_ENV_SOURCE:-}"
export LIBERO_WORKER_ENV_SOURCE
imagewam_prepare_eval_ckpt

COMMON=(
  --config-name "${CONFIG_NAME}"
  task="${TASK}"
  ckpt="${CKPT_PATH}"
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}"
  model.flux2_src_path="${FLUX2_SRC}"
  model.flux2_model_path="${FLUX2_MODEL_PATH}"
  model.ae_model_path="${FLUX2_AE_MODEL_PATH}"
  model.variant="klein-base-${FLUX2_VARIANT}"
  model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
  model.load_text_encoder=true
  model.pack_proprio_after_text=true
  model.chunkwise_causal.rope_scheme="${ROPE_SCHEME}"
  MULTIRUN.num_gpus="${NUM_GPUS:-8}"
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-4}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-12}"
)

if [ -n "${QWEN_CACHE_DIR}" ]; then
  COMMON+=(data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}")
fi

COMMON+=(
  model.proprio_dim="${PROPRIO_DIM:-8}"
  data.train.qwen_context_len="${QWEN_CONTEXT_LEN:-512}"
  data.train.qwen_text_cache_format=qwen3_flux2
  MULTIRUN.task_suite_names="${TASK_SUITE_NAMES:-[libero_10,libero_goal,libero_spatial,libero_object]}"
  EVALUATION.num_trials="${NUM_TRIALS:-25}"
  MULTIRUN.chunk_size="${CHUNK_SIZE:-1}"
)

imagewam_print_config SUITE TASK ROPE_SCHEME CKPT_PATH DATASET_STATS_PATH FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
imagewam_run imagewam_python experiments/libero/run_libero_manager.py "${COMMON[@]}" "$@"
