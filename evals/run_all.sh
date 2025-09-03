#!/usr/bin/env bash
# run_all.sh — sequential UV jobs, always run, per-job logs, summary at the end
set -Eeuo pipefail

LOG_DIR="${LOG_DIR:-./logs}"
SUMMARY="${SUMMARY:-${LOG_DIR}/summary.txt}"
mkdir -p "$LOG_DIR"

declare -a JOBS=(
'uv run 1_steering.py --pca_filepath /workspace/qwen-3-32b/roles_traits/pca/layer32_roles_pos23_traits_pos40-100.pt --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl --magnitudes 25.0 50.0 --output_jsonl /root/git/persona-subspace/evals/jailbreak/qwen-3-32b/steered/default_1100.jsonl --batch_size 32 --company Alibaba --no_system_prompt --layer 32 --thinking=False --model Qwen/Qwen3-32B'
'uv run 1_steering.py --pca_filepath /workspace/llama-3.3-70b/roles_traits/pca/layer40_roles_pos23_traits_pos40-100.pt --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl --magnitudes -3.0 -6.0 --output_jsonl /root/git/persona-subspace/evals/jailbreak/llama-3.3-70b/steered/jailbreak_1100.jsonl --batch_size 16 --company Meta --layer 40 --model meta-llama/Llama-3.3-70B-Instruct'
'uv run 1_steering.py --pca_filepath /workspace/llama-3.3-70b/roles_traits/pca/layer40_roles_pos23_traits_pos40-100.pt --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl --magnitudes -3.0 -6.0 --output_jsonl /root/git/persona-subspace/evals/jailbreak/llama-3.3-70b/steered/default_1100.jsonl --batch_size 16 --company Meta --no_system_prompt --layer 40 --model meta-llama/Llama-3.3-70B-Instruct'
'uv run 1_steering.py --pca_filepath /workspace/gemma-2-27b/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl --magnitudes 500.0 1000.0 --output_jsonl /root/git/persona-subspace/evals/jailbreak/gemma-2-27b/steered/jailbreak_1100.jsonl --batch_size 16 --company Google --layer 22 --model google/gemma-2-27b-it'
'uv run 1_steering.py --pca_filepath /workspace/gemma-2-27b/roles_traits/pca/layer22_roles_pos23_traits_pos40-100.pt --prompts_file /root/git/persona-subspace/evals/jailbreak/jailbreak_1100.jsonl --magnitudes 500.0 1000.0 --output_jsonl /root/git/persona-subspace/evals/jailbreak/gemma-2-27b/steered/default_1100.jsonl --batch_size 16 --company Google --no_system_prompt --layer 22 --model google/gemma-2-27b-it'
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
