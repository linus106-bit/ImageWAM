#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

SUITE="robotwin"
CONFIG_NAME="sim_robotwin"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}" # 4b | 9b
ROPE_SCHEME="${ROPE_SCHEME:-current}"
TASK="${TASK:-robotwin_flux2_klein_${FLUX2_VARIANT}_base_imagewam}"
if [ "true" = "true" ]; then
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
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-3}"
  EVALUATION.action_horizon="${ACTION_HORIZON:-16}"
  EVALUATION.replan_steps="${REPLAN_STEPS:-16}"
)

if [ -n "${QWEN_CACHE_DIR}" ]; then
  COMMON+=(data.train.qwen_text_cache_dir="${QWEN_CACHE_DIR}")
fi

COMMON+=(
  model.proprio_dim="${PROPRIO_DIM:-14}"
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES:-50}"
  MULTIRUN.phases="${PHASES:-[clean,random]}"
  EVALUATION.skip_get_obs_within_replan="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
  EVALUATION.robotwin_camera_layout="${ROBOTWIN_CAMERA_LAYOUT:-compact_288x256}"
)

imagewam_print_config SUITE TASK ROPE_SCHEME CKPT_PATH DATASET_STATS_PATH FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH
imagewam_run imagewam_python experiments/robotwin/run_robotwin_manager.py "${COMMON[@]}" "$@"
