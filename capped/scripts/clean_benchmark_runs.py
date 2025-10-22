#!/usr/bin/env python3
"""
Clean up benchmark run directories by:
  a) Renaming verbose directories (with _seed*_limit*_shots*) to timestamp-only
  b) Deleting mmlu_pro runs with max_gen_toks=512
  c) Deleting runs with zero metrics

Usage:
    python clean_benchmark_runs.py [base_dir]

Default base_dir: /workspace/qwen-3-32b/capped/benchmarks
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Metric definitions for each benchmark task
EVAL_METRICS = {
    'ifeval': {
        'metric': 'inst_level_strict_acc,none',
        'display_name': 'IFEval Instruction-level Accuracy',
        'higher_is_better': True
    },
    'mmlu_pro': {
        'metric': 'exact_match,custom-extract',
        'display_name': 'MMLU Pro Exact Match Accuracy',
        'higher_is_better': True
    },
    'eq_bench': {
        'metric': 'eqbench,none',
        'display_name': 'EQ-Bench Score',
        'higher_is_better': True
    },
    'gsm8k': {
        'metric': 'exact_match,flexible-extract',
        'display_name': 'GSM8K Exact Match Accuracy',
        'higher_is_better': True
    }
}

# Pattern for verbose directory names: YYYY-MM-DD_HH-MM-SS_seed*_limit*_shots*
VERBOSE_DIR_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_seed\d+_limit\d+_shots\d+$')
TIMESTAMP_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')


class CleanupOperation:
    """Represents a cleanup operation to be performed."""
    def __init__(self, op_type: str, path: str, reason: str, target: Optional[str] = None):
        self.op_type = op_type  # 'delete', 'rename'
        self.path = path
        self.reason = reason
        self.target = target  # For rename operations


def find_verbose_dirs(base_dir: str) -> List[CleanupOperation]:
    """Find directories with verbose names that should be renamed."""
    operations = []

    for root, dirs, files in os.walk(base_dir):
        for dirname in dirs:
            match = VERBOSE_DIR_PATTERN.match(dirname)
            if match:
                timestamp = match.group(1)
                verbose_path = os.path.join(root, dirname)
                target_path = os.path.join(root, timestamp)

                # Check if target already exists
                if os.path.exists(target_path):
                    operations.append(CleanupOperation(
                        'delete',
                        verbose_path,
                        f'Conflict: {timestamp} already exists',
                        target_path
                    ))
                else:
                    operations.append(CleanupOperation(
                        'rename',
                        verbose_path,
                        f'Rename to {timestamp}',
                        target_path
                    ))

    return operations


def find_mmlu_max_gen_toks_512(base_dir: str) -> List[CleanupOperation]:
    """Find mmlu_pro runs with max_gen_toks=512."""
    operations = []

    # Walk through benchmarks/{task}/{config}/{experiment_id}/{date_dir}
    for task_name in os.listdir(base_dir):
        if task_name != 'mmlu_pro':
            continue

        task_path = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        for config_name in os.listdir(task_path):
            config_path = os.path.join(task_path, config_name)
            if not os.path.isdir(config_path):
                continue

            for experiment_id in os.listdir(config_path):
                experiment_path = os.path.join(config_path, experiment_id)
                if not os.path.isdir(experiment_path):
                    continue

                for date_dir in os.listdir(experiment_path):
                    date_path = os.path.join(experiment_path, date_dir)
                    if not os.path.isdir(date_path):
                        continue

                    manifest_path = os.path.join(date_path, 'manifest.json')
                    if os.path.exists(manifest_path):
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)

                            if manifest.get('max_gen_toks') == 512:
                                operations.append(CleanupOperation(
                                    'delete',
                                    date_path,
                                    'mmlu_pro run with max_gen_toks=512'
                                ))
                        except Exception as e:
                            print(f"Warning: Error reading {manifest_path}: {e}")

    return operations


def find_zero_metric_runs(base_dir: str) -> List[CleanupOperation]:
    """Find runs with zero metrics based on task type."""
    operations = []

    for task_name in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        # Skip if task not in our metric definitions
        if task_name not in EVAL_METRICS:
            continue

        metric_name = EVAL_METRICS[task_name]['metric']

        for config_name in os.listdir(task_path):
            config_path = os.path.join(task_path, config_name)
            if not os.path.isdir(config_path):
                continue

            for experiment_id in os.listdir(config_path):
                experiment_path = os.path.join(config_path, experiment_id)
                if not os.path.isdir(experiment_path):
                    continue

                for date_dir in os.listdir(experiment_path):
                    date_path = os.path.join(experiment_path, date_dir)
                    if not os.path.isdir(date_path):
                        continue

                    results_path = os.path.join(date_path, 'results.json')
                    if os.path.exists(results_path):
                        try:
                            with open(results_path, 'r') as f:
                                results = json.load(f)

                            # Get the task results (task_name should be the key)
                            task_results = results.get('results', {}).get(task_name, {})
                            metric_value = task_results.get(metric_name)

                            if metric_value == 0.0:
                                operations.append(CleanupOperation(
                                    'delete',
                                    date_path,
                                    f'{task_name} run with {metric_name}=0.0'
                                ))
                        except Exception as e:
                            print(f"Warning: Error reading {results_path}: {e}")

    return operations


def execute_operations(operations: List[CleanupOperation], dry_run: bool = True) -> None:
    """Execute the cleanup operations."""
    if dry_run:
        return

    for op in operations:
        try:
            if op.op_type == 'delete':
                shutil.rmtree(op.path)
                print(f"✓ Deleted: {op.path}")
            elif op.op_type == 'rename':
                os.rename(op.path, op.target)
                print(f"✓ Renamed: {op.path} -> {op.target}")
        except Exception as e:
            print(f"✗ Error processing {op.path}: {e}")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/qwen-3-32b/capped/benchmarks"

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)

    print(f"Scanning benchmark directories in: {base_dir}")
    print()

    # Collect all operations
    print("Collecting cleanup operations...")
    verbose_dir_ops = find_verbose_dirs(base_dir)
    mmlu_512_ops = find_mmlu_max_gen_toks_512(base_dir)
    zero_metric_ops = find_zero_metric_runs(base_dir)

    all_operations = verbose_dir_ops + mmlu_512_ops + zero_metric_ops

    if not all_operations:
        print("No cleanup operations needed.")
        return

    # Display operations by type
    print(f"\n{'='*80}")
    print(f"TASK A: Rename/Delete Verbose Directory Names")
    print(f"{'='*80}")
    if verbose_dir_ops:
        for op in verbose_dir_ops:
            action = "DELETE" if op.op_type == 'delete' else "RENAME"
            print(f"  [{action}] {op.path}")
            print(f"           Reason: {op.reason}")
            if op.target and op.op_type == 'rename':
                print(f"           Target: {op.target}")
            print()
    else:
        print("  None found.")

    print(f"\n{'='*80}")
    print(f"TASK B: Delete MMLU Pro runs with max_gen_toks=512")
    print(f"{'='*80}")
    if mmlu_512_ops:
        for op in mmlu_512_ops:
            print(f"  [DELETE] {op.path}")
            print(f"           Reason: {op.reason}")
            print()
    else:
        print("  None found.")

    print(f"\n{'='*80}")
    print(f"TASK C: Delete runs with zero metrics")
    print(f"{'='*80}")
    if zero_metric_ops:
        for op in zero_metric_ops:
            print(f"  [DELETE] {op.path}")
            print(f"           Reason: {op.reason}")
            print()
    else:
        print("  None found.")

    # Summary
    delete_count = sum(1 for op in all_operations if op.op_type == 'delete')
    rename_count = sum(1 for op in all_operations if op.op_type == 'rename')

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Total operations: {len(all_operations)}")
    print(f"    - Deletes: {delete_count}")
    print(f"    - Renames: {rename_count}")
    print()

    # Confirmation
    response = input("Proceed with these operations? (y/N) ").strip().lower()
    if response == 'y':
        print("\nExecuting operations...")
        execute_operations(all_operations, dry_run=False)
        print(f"\nCompleted {len(all_operations)} operations.")
    else:
        print("Aborted. No changes made.")


if __name__ == "__main__":
    main()
