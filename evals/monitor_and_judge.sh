#!/bin/bash

# Configuration
MODELS=("llama-3.1-70b" "gemma-2-27b")
TARGET_LINES=67200
MAX_ATTEMPTS=2
CHECK_INTERVAL=60

# State tracking
declare -A attempts
declare -A done
for model in "${MODELS[@]}"; do
    attempts[$model]=0
    done[$model]=0
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_line_count() {
    local file=$1
    if [[ -f "$file" ]]; then
        wc -l < "$file"
    else
        echo 0
    fi
}

process_file() {
    local model=$1
    local input="/workspace/$model/evals/results/base/prefill_8.jsonl"
    local output="/workspace/$model/evals/results/base/prefill_8_scores.jsonl"

    log "[$model] Starting judge script (attempt $((${attempts[$model]} + 1))/$MAX_ATTEMPTS)..."

    # Run the judge script
    if uv run 2_introspective_judge.py "$input" --output "$output"; then
        log "[$model] Judge script completed successfully"
    else
        log "[$model] ERROR: Judge script failed with exit code $?"
        return 1
    fi

    # Check if output has correct number of lines
    local output_lines=$(get_line_count "$output")
    log "[$model] Output has $output_lines lines (expected $TARGET_LINES)"

    if [[ $output_lines -eq $TARGET_LINES ]]; then
        log "[$model] SUCCESS: Processing complete!"
        done[$model]=1
        return 0
    else
        log "[$model] WARNING: Output has incorrect line count"
        return 1
    fi
}


# Main monitoring loop
log "Starting file monitor..."
log "Monitoring models: ${MODELS[*]}"
log "Target line count: $TARGET_LINES"
log "Max attempts per file: $MAX_ATTEMPTS"
log "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
    all_done=true

    for model in "${MODELS[@]}"; do
        # Check if this model is already done
        if [[ ${done[$model]} -eq 1 ]]; then
            continue
        fi

        all_done=false

        # Check if max attempts reached
        if [[ ${attempts[$model]} -ge $MAX_ATTEMPTS ]]; then
            log "[$model] Max attempts exhausted, marking as done"
            done[$model]=1
            continue
        fi

        # Check input file line count
        input_file="/workspace/$model/evals/results/base/prefill_8.jsonl"
        output_file="/workspace/$model/evals/results/base/prefill_8_scores.jsonl"
        input_lines=$(get_line_count "$input_file")
        output_lines=$(get_line_count "$output_file")

        if [[ $input_lines -lt $TARGET_LINES ]]; then
            log "[$model] Waiting for input... ($input_lines/$TARGET_LINES lines)"
        elif [[ $output_lines -eq $TARGET_LINES ]]; then
            log "[$model] Output already complete with $output_lines lines"
            done[$model]=1
        else
            log "[$model] Input ready ($input_lines lines), output has $output_lines lines"

            # Increment attempt counter
            attempts[$model]=$((${attempts[$model]} + 1))

            # Process the file
            process_file "$model"

            # Add separator after processing
            echo ""
        fi
    done

    # Check if all files are done
    all_done=true
    for model in "${MODELS[@]}"; do
        if [[ ${done[$model]} -ne 1 ]]; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        log "All files processed. Exiting."
        break
    fi

    # Wait before next check
    log "Next check in ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done

log "Monitoring complete!"

# Print final summary
echo ""
echo "=== Final Summary ==="
for model in "${MODELS[@]}"; do
    echo "  $model: Attempts=${attempts[$model]}, Done=${done[$model]}"
done
