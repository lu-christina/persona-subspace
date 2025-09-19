#!/usr/bin/env bash
# run_all.sh — sequential UV jobs, always run, per-job logs, summary at the end
set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-./logs}"
SUMMARY="${SUMMARY:-${LOG_DIR}/summary.txt}"
mkdir -p "$LOG_DIR"

declare -a JOBS=(
'uv run traits/3_response_activations.py --model-name google/gemma-3-27b-it --responses-dir /workspace/gemma-3-27b/traits_240/responses --output-dir /workspace/gemma-3-27b/traits_240/response_activations --multi-gpu --num-gpus 8 --batch-size 32'
'uv run roles/3_response_activations.py --model-name google/gemma-3-27b-it --responses-dir /workspace/gemma-3-27b/roles_240/responses --output-dir /workspace/gemma-3-27b/roles_240/response_activations --multi-gpu --num-gpus 8 --batch-size 32'
)

printf "" > "$SUMMARY"

run_job () {
  local cmd="$1"
  local stamp; stamp="$(date +'%Y%m%d-%H%M%S')"
  local tag="job"
  if [[ "$cmd" =~ --output_jsonl[[:space:]]+([^[:space:]]+) ]]; then
    tag="$(basename "${BASH_REMATCH[1]}" .jsonl)"
  fi
  local log="${LOG_DIR}/${stamp}_${tag}.log"

  echo "=== $(date -Is) === $cmd" | tee -a "$log"
  if bash -lc "$cmd" 2>&1 | tee -a "$log"; then
    echo "[OK] $(date -Is)  ${tag}" | tee -a "$SUMMARY"
  else
    echo "[FAIL] $(date -Is) ${tag}  (see $log)" | tee -a "$SUMMARY" >&2
    return 1
  fi
}

overall_rc=0
for j in "${JOBS[@]}"; do
  if ! run_job "$j"; then
    overall_rc=1   # record failure but keep going
  fi
done

echo -e "\n------------- SUMMARY -------------"
cat "$SUMMARY" || true
echo "-----------------------------------"
exit "$overall_rc"
