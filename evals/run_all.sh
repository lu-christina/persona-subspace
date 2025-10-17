#!/usr/bin/env bash
set -euo pipefail

# Judge all completed results into *_scores.jsonl
LOG_DIR="/workspace/judge_logs"
mkdir -p "$LOG_DIR"

judge() {
  local in_jsonl="$1"
  local out_jsonl="${in_jsonl%.jsonl}_scores.jsonl"
  local log_file="$LOG_DIR/$(basename "${in_jsonl%.jsonl}")_judge.log"

  if [[ ! -f "$in_jsonl" ]]; then
    echo "[skip] missing: $in_jsonl"
    return 0
  fi

  echo "[run] judge: $in_jsonl -> $out_jsonl (log: $log_file)"
  uv run 2_jailbreak_judge.py \
    "$in_jsonl" \
    --output "$out_jsonl" \
    >"$log_file" 2>&1 &
}

# === Add your result files here ===
RESULTS=(
  "/workspace/llama-3.3-70b/capped/results/role_trait_jailbreak_1100.jsonl"
  "/workspace/llama-3.3-70b/capped/results/lmsys_10000_jailbreak_1100.jsonl"
)

for f in "${RESULTS[@]}"; do
  judge "$f"
done

# Wait for all background judges to finish
wait
echo "All scoring jobs finished. Logs in $LOG_DIR."
