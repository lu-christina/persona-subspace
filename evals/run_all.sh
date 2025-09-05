#!/usr/bin/env bash
# run_all.sh — sequential UV jobs, always run, per-job logs, summary at the end
set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-./logs}"
SUMMARY="${SUMMARY:-${LOG_DIR}/summary.txt}"
mkdir -p "$LOG_DIR"

declare -a JOBS=(
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/steered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/steered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/steered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/steered/default_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/steered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/steered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/steered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/steered/default_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/steered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/steered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/steered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/steered/default_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/unsteered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/unsteered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/unsteered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/qwen-3-32b/unsteered/default_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/unsteered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/unsteered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/unsteered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/llama-3.3-70b/unsteered/default_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/unsteered/susceptibility_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/unsteered/susceptibility_50_scores.jsonl'
'uv run 2_susceptibility_judge.py /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/unsteered/default_50.jsonl --output /root/git/persona-subspace/evals/susceptibility/gemma-2-27b/unsteered/default_50_scores.jsonl'
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
