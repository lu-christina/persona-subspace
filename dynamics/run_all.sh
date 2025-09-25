#!/usr/bin/env bash
# run_roles.sh — sequential UV jobs over roles, per-job logs, summary at the end
set -Eeuo pipefail

# ---- Config (override via env) ----
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TRANSCRIPTS_ROOT="${TRANSCRIPTS_ROOT:-/root/git/persona-subspace/dynamics/results/qwen-3-32b/transcripts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/git/persona-subspace/dynamics/results/qwen-3-32b/activations}"
LOG_DIR="${LOG_DIR:-./logs}"
SUMMARY="${SUMMARY:-${LOG_DIR}/summary.txt}"

# Roles to process
declare -a ROLES=("default" "validator" "podcaster" "visionary" "crystalline" "whale" "leviathan")

mkdir -p "$LOG_DIR"
: > "$SUMMARY"

# ---- Build the JOB list from roles ----
declare -a JOBS=()
for role in "${ROLES[@]}"; do
  JOBS+=(
    "uv run 1_activations.py --model-name ${MODEL} --target-dir ${TRANSCRIPTS_ROOT}/${role} --output-dir ${OUTPUT_ROOT}/${role}"
  )
done

# ---- Helpers ----
fmt_duration() {
  local s=$1 h m
  h=$(( s/3600 ))
  m=$(( (s%3600)/60 ))
  s=$(( s%60 ))
  printf "%d:%02d:%02d" "$h" "$m" "$s"
}

run_job () {
  local cmd="$1"
  local stamp tag log start end dur

  # Tag = last path segment of --output-dir (usually the role); fallback to last of --target-dir; else "job"
  tag="job"
  if [[ "$cmd" =~ --output-dir[[:space:]]+([^[:space:]]+) ]]; then
    tag="$(basename "${BASH_REMATCH[1]}")"
  elif [[ "$cmd" =~ --target-dir[[:space:]]+([^[:space:]]+) ]]; then
    tag="$(basename "${BASH_REMATCH[1]}")"
  fi

  stamp="$(date +'%Y%m%d-%H%M%S')"
  log="${LOG_DIR}/${stamp}_${tag}.log"

  echo "=== $(date -Is) === $cmd" | tee -a "$log"
  start=$(date +%s)
  if bash -lc "$cmd" 2>&1 | tee -a "$log"; then
    end=$(date +%s); dur=$(( end - start ))
    echo "[OK]   $(date -Is)  ${tag}  (took $(fmt_duration "$dur"))" | tee -a "$SUMMARY"
    return 0
  else
    end=$(date +%s); dur=$(( end - start ))
    echo "[FAIL] $(date -Is)  ${tag}  (took $(fmt_duration "$dur"))  see: $log" | tee -a "$SUMMARY" >&2
    return 1
  fi
}

# ---- Run all ----
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
