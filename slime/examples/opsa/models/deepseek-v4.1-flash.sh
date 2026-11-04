# UkisAI preset for the swifter DeepSeek-V4.1-Flash checkpoint.
#
# The swifter checkpoint carries its own architecture (vision tower + native
# MTP preserved for export; only the language model trains). Its converted
# Megatron model definition is NOT bundled here. Point DEEPSEEK_V41_MODEL_CONFIG
# at a scripts/models-style definition file (absolute path allowed):
#
#   export DEEPSEEK_V41_MODEL_CONFIG=/path/to/your/model/definition.sh
#
# If unset, the bundled DeepSeek-V3 definition is used as a placeholder and the
# launcher will fail at conversion/load time until the real definition exists.
MODEL_DISPLAY_NAME="DeepSeek-V4.1-Flash (UkisAI swifter)"
MODEL_CONFIG_RELATIVE="${DEEPSEEK_V41_MODEL_CONFIG:-scripts/models/deepseek-v3.sh}"

# 150-step pilot (extend to 300-600 with --steps if the eval slope is positive).
NUM_ROLLOUT=150

# Parallelism: override per cluster. A 552B-class MoE typically needs
# large TP and optimizer CPU offload; adjust to the training node layout.
ACTOR_GPUS="${DEEPSEEK_V41_ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${DEEPSEEK_V41_ROLLOUT_GPUS:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${DEEPSEEK_V41_TP:-8}"
ROLLOUT_GPUS_PER_ENGINE="${DEEPSEEK_V41_ROLLOUT_GPUS_PER_ENGINE:-1}"

# Canonical OPSA decoding: train at temperature 1, top-p 1, top-k -1 with
# 12,000-token responses; eval at temperature 0.7, top-k 20, top-p 0.8 with
# 32,768-token responses.
ROLLOUT_MAX_RESPONSE_LEN="${DEEPSEEK_V41_ROLLOUT_MAX_RESPONSE_LEN:-12000}"
EVAL_MAX_RESPONSE_LEN="${DEEPSEEK_V41_EVAL_MAX_RESPONSE_LEN:-32768}"
MAX_TOKENS_PER_GPU="${DEEPSEEK_V41_MAX_TOKENS_PER_GPU:-16384}"
SGLANG_MEM_FRACTION_STATIC="${DEEPSEEK_V41_SGLANG_MEM_FRACTION:-0.85}"
OPTIMIZER_CPU_OFFLOAD="${DEEPSEEK_V41_OPTIMIZER_CPU_OFFLOAD:-true}"
