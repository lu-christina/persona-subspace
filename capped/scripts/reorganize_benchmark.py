#!/usr/bin/env python3
"""
Reorganize benchmark directories from:
  {config_name}/{experiment_id}/{task_name}/*
to:
  {task_name}/{config_name}/{experiment_id}/*

Special case: baseline has no experiment_id, so:
  baseline/{task_name}/* -> {task_name}/baseline/*

Usage:
  python reorganize_benchmark.py <benchmarks_dir> <task_name> [config_names...]

Example:
  python reorganize_benchmark.py /workspace/qwen-3-32b/capped/benchmarks eq_bench jailbreak lmsys_10000 role_trait baseline
  python reorganize_benchmark.py /workspace/qwen-3-32b/capped/benchmarks ifeval jailbreak lmsys_10000 role_trait baseline
"""

import sys
import shutil
from pathlib import Path


def reorganize_benchmark(base_dir: Path, task_name: str, config_names: list[str]):
    """
    Reorganize benchmark directories for a specific task.

    Args:
        base_dir: Base directory containing config directories
        task_name: Name of the task (e.g., 'eq_bench', 'ifeval')
        config_names: List of config names to process (e.g., ['jailbreak', 'lmsys_10000', 'role_trait', 'baseline'])
    """
    # Create the new task root directory if it doesn't exist
    new_task_root = base_dir / task_name
    new_task_root.mkdir(exist_ok=True)
    print(f"Created/verified: {new_task_root}")

    for config_name in config_names:
        config_dir = base_dir / config_name

        if not config_dir.exists():
            print(f"Skipping {config_name} - directory doesn't exist")
            continue

        # Special case for baseline
        if config_name == "baseline":
            old_task_dir = config_dir / task_name
            if old_task_dir.exists():
                new_config_dir = new_task_root / config_name
                print(f"Moving: {old_task_dir} -> {new_config_dir}")
                shutil.move(str(old_task_dir), str(new_config_dir))
            else:
                print(f"No {task_name} directory found in {config_dir}")
        else:
            # For other configs, iterate through experiment_id directories
            for experiment_dir in config_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue

                experiment_id = experiment_dir.name
                old_task_dir = experiment_dir / task_name

                if not old_task_dir.exists():
                    print(f"No {task_name} in {experiment_dir}, skipping")
                    continue

                # Create the new structure: {task_name}/{config_name}/{experiment_id}
                new_config_dir = new_task_root / config_name
                new_config_dir.mkdir(exist_ok=True)

                new_experiment_dir = new_config_dir / experiment_id

                print(f"Moving: {old_task_dir} -> {new_experiment_dir}")
                shutil.move(str(old_task_dir), str(new_experiment_dir))

    print("\nReorganization complete!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python reorganize_benchmark.py <benchmarks_dir> <task_name> <config_names...>")
        print("\nExample:")
        print("  python reorganize_benchmark.py /workspace/qwen-3-32b/capped/benchmarks eq_bench jailbreak lmsys_10000 role_trait baseline")
        sys.exit(1)

    benchmarks_dir = Path(sys.argv[1])
    task_name = sys.argv[2]
    config_names = sys.argv[3:]

    if not benchmarks_dir.exists():
        print(f"Error: Directory {benchmarks_dir} does not exist")
        sys.exit(1)

    # Ask for confirmation before proceeding
    print(f"This script will reorganize {task_name} directories.")
    print(f"Base directory: {benchmarks_dir}")
    print(f"Current structure: {{config_name}}/{{experiment_id}}/{task_name}/*")
    print(f"New structure: {task_name}/{{config_name}}/{{experiment_id}}/*")
    print(f"Config names: {', '.join(config_names)}")
    print()
    response = input("Proceed? (yes/no): ")

    if response.lower() in ["yes", "y"]:
        reorganize_benchmark(benchmarks_dir, task_name, config_names)
    else:
        print("Aborted.")
