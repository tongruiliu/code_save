#!/usr/bin/env bash
set -euo pipefail

POLICY_API_KEY="${POLICY_API_KEY:-}"
POLICY_API_BASE_URL="${POLICY_API_BASE_URL:-}"
POLICY_MODEL="${POLICY_MODEL:-mgg-2}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-32768}"
POLICY_TIMEOUT_SEC="${POLICY_TIMEOUT_SEC:-120}"

CRITIC_API_KEY="${CRITIC_API_KEY:-${POLICY_API_KEY}}"
CRITIC_API_BASE_URL="${CRITIC_API_BASE_URL:-${POLICY_API_BASE_URL}}"
# Critic model is intentionally configured independently from policy model.
CRITIC_MODEL="${CRITIC_MODEL:-mog-2}"
CRITIC_MAX_TOKENS="${CRITIC_MAX_TOKENS:-4096}"
CRITIC_TIMEOUT_SEC="${CRITIC_TIMEOUT_SEC:-120}"

MODEL_PROVIDER="openai"
CRITIC_PROVIDER="openai"
TEMPERATURE="0.0"
MAX_ROUNDS="6"
MAX_FINAL_RETRIES="2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_DATA_JSON="${BASE_DATA_JSON:-${PROJECT_ROOT}/data/mathvista.json}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-${PROJECT_ROOT}/data}"
START="${START:-0}"
END="${END:-3}"

SKIP_ANSWER_TYPE_LIST="${SKIP_ANSWER_TYPE_LIST:-1}"
SHUFFLE="${SHUFFLE:-1}"
SEED="${SEED:-10}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUT_DIR}"

if [[ -z "${POLICY_API_KEY}" ]]; then
  echo "POLICY_API_KEY is empty." >&2
  exit 1
fi
if [[ -z "${POLICY_API_BASE_URL}" ]]; then
  echo "POLICY_API_BASE_URL is empty." >&2
  exit 1
fi
if [[ -z "${CRITIC_MODEL}" ]]; then
  echo "CRITIC_MODEL is empty. Please set it explicitly (can differ from POLICY_MODEL)." >&2
  exit 1
fi
if [[ ! -f "${BASE_DATA_JSON}" ]]; then
  echo "BASE_DATA_JSON not found: ${BASE_DATA_JSON}" >&2
  exit 1
fi
if ! [[ "${START}" =~ ^[0-9]+$ ]]; then
  echo "START must be non-negative integer: ${START}" >&2
  exit 1
fi
if ! [[ "${END}" =~ ^[0-9]+$ ]]; then
  echo "END must be non-negative integer: ${END}" >&2
  exit 1
fi
if (( END <= START )); then
  echo "Invalid range [START, END): [${START}, ${END})" >&2
  exit 1
fi

CMD=(
  python3 -m canvas_sft_pipeline.run
  --model "${POLICY_MODEL}"
  --model-provider "${MODEL_PROVIDER}"
  --api-key "${POLICY_API_KEY}"
  --api-base-url "${POLICY_API_BASE_URL}"
  --policy-max-tokens "${POLICY_MAX_TOKENS}"
  --policy-timeout-sec "${POLICY_TIMEOUT_SEC}"
  --temperature "${TEMPERATURE}"
  --critic-model "${CRITIC_MODEL}"
  --critic-model-provider "${CRITIC_PROVIDER}"
  --critic-api-key "${CRITIC_API_KEY}"
  --critic-api-base-url "${CRITIC_API_BASE_URL}"
  --critic-max-tokens "${CRITIC_MAX_TOKENS}"
  --critic-timeout-sec "${CRITIC_TIMEOUT_SEC}"
  --base-data-json "${BASE_DATA_JSON}"
  --base-data-root "${BASE_DATA_ROOT}"
  --start "${START}"
  --end "${END}"
  --max-rounds "${MAX_ROUNDS}"
  --max-final-retries "${MAX_FINAL_RETRIES}"
  --seed "${SEED}"
  --output-dir "${OUT_DIR}"
  --run-tag "${RUN_TAG}"
)

if [[ "${SKIP_ANSWER_TYPE_LIST}" == "1" ]]; then
  CMD+=(--skip-answer-type-list)
fi
if [[ "${SHUFFLE}" == "1" ]]; then
  CMD+=(--shuffle)
fi

echo "[Run] ${CMD[*]}"
"${CMD[@]}"
