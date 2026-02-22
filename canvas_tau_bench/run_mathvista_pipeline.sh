#!/usr/bin/env bash
set -euo pipefail

POLICY_API_KEY="${POLICY_API_KEY:-sk-WlQjMouxWTzJKjNZGntDA09VFqX4F3daQZPhHMKAJx2Sq1Bi}"
POLICY_API_BASE_URL="${POLICY_API_BASE_URL:-http://172.96.160.199:3000/v1}"
POLICY_MODEL="${POLICY_MODEL:-gpt-4.1-mini}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-32768}"

CRITIC_API_KEY="${CRITIC_API_KEY:-sk-WlQjMouxWTzJKjNZGntDA09VFqX4F3daQZPhHMKAJx2Sq1Bi}"
CRITIC_API_BASE_URL="${CRITIC_API_BASE_URL:-http://172.96.160.199:3000/v1}"
CRITIC_MODEL="${CRITIC_MODEL:-gpt-4.1-mini}"
CRITIC_MAX_TOKENS="${CRITIC_MAX_TOKENS:-1024}"

MODEL_PROVIDER="openai"
TEMPERATURE="0.0"
MAX_STEPS="10"
USER_STRATEGY="llm"
USER_MODEL_PROVIDER="openai"
USER_MODEL="${CRITIC_MODEL}"
USER_API_BASE_URL="${CRITIC_API_BASE_URL}"

BASE_DATA_JSON="/m2/slz/lrt/data/testmini.json"
BASE_DATA_ROOT="/m2/slz/lrt/data"
BASE_MAX_SAMPLES="3"
BASE_OFFSET="0"

NUM_TRIALS="1"
SEED="10"
SHUFFLE="1"
SKIP_ANSWER_TYPE_LIST="1"

LOG_DIR="/m2/slz/lrt/results/canvas_sft"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
SFT_JSONL="${LOG_DIR}/mathvista-${RUN_TAG}.sft.jsonl"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"

if [[ -z "${POLICY_API_KEY}" ]]; then
  echo "POLICY_API_KEY is empty. Please set it." >&2
  exit 1
fi
if [[ -z "${POLICY_API_BASE_URL}" ]]; then
  echo "POLICY_API_BASE_URL is empty. Please set it." >&2
  exit 1
fi
if [[ -z "${POLICY_MODEL}" ]]; then
  echo "POLICY_MODEL is empty. Please set it." >&2
  exit 1
fi
if [[ -z "${CRITIC_API_KEY}" ]]; then
  echo "CRITIC_API_KEY is empty. Please set it." >&2
  exit 1
fi
if [[ -z "${CRITIC_API_BASE_URL}" ]]; then
  echo "CRITIC_API_BASE_URL is empty. Please set it." >&2
  exit 1
fi
if [[ -z "${CRITIC_MODEL}" ]]; then
  echo "CRITIC_MODEL is empty. Please set it." >&2
  exit 1
fi

if [[ ! -f "${BASE_DATA_JSON}" ]]; then
  echo "Base data json not found: ${BASE_DATA_JSON}" >&2
  exit 1
fi

echo "[Config]"
echo "POLICY_MODEL=${POLICY_MODEL}"
echo "POLICY_API_BASE_URL=${POLICY_API_BASE_URL}"
echo "POLICY_MAX_TOKENS=${POLICY_MAX_TOKENS}"
echo "CRITIC_MODEL=${CRITIC_MODEL}"
echo "CRITIC_API_BASE_URL=${CRITIC_API_BASE_URL}"
echo "CRITIC_MAX_TOKENS=${CRITIC_MAX_TOKENS}"
echo "BASE_DATA_JSON=${BASE_DATA_JSON}"
echo "LOG_DIR=${LOG_DIR}"
echo "SFT_JSONL=${SFT_JSONL}"

# =========================
# Run
# =========================
CMD=(
  python3 -m canvas_tau_bench.run
  --model "${POLICY_MODEL}"
  --model-provider "${MODEL_PROVIDER}"
  --api-key "${POLICY_API_KEY}"
  --api-base-url "${POLICY_API_BASE_URL}"
  --policy-max-tokens "${POLICY_MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --max-steps "${MAX_STEPS}"
  --user-strategy "${USER_STRATEGY}"
  --user-model "${USER_MODEL}"
  --user-model-provider "${USER_MODEL_PROVIDER}"
  --user-api-key "${CRITIC_API_KEY}"
  --user-api-base-url "${USER_API_BASE_URL}"
  --critic-max-tokens "${CRITIC_MAX_TOKENS}"
  --num-trials "${NUM_TRIALS}"
  --seed "${SEED}"
  --log-dir "${LOG_DIR}"
  --sft-jsonl "${SFT_JSONL}"
  --base-data-json "${BASE_DATA_JSON}"
  --base-data-root "${BASE_DATA_ROOT}"
  --base-max-samples "${BASE_MAX_SAMPLES}"
  --base-offset "${BASE_OFFSET}"
)

if [[ "${SHUFFLE}" == "1" ]]; then
  CMD+=(--shuffle)
fi

if [[ "${SKIP_ANSWER_TYPE_LIST}" == "1" ]]; then
  CMD+=(--skip-answer-type-list)
fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}"

echo "[Done] SFT jsonl: ${SFT_JSONL}"
