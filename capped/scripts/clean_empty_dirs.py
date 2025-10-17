#!/usr/bin/env python3
"""
Clean up empty directories in the benchmarks folder.

Usage:
    python clean_empty_dirs.py [base_dir]

Default base_dir: /workspace/qwen-3-32b/capped/benchmarks
"""

import os
import sys
from pathlib import Path


def find_empty_dirs(base_dir):
    """Find all empty directories under base_dir."""
    empty_dirs = []
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            if not os.listdir(dirpath):
                empty_dirs.append(dirpath)
    return sorted(empty_dirs)


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/qwen-3-32b/capped/benchmarks"

    print(f"Searching for empty directories in: {base_dir}")
    print()

    empty_dirs = find_empty_dirs(base_dir)

    if not empty_dirs:
        print("No empty directories found.")
        return

    print(f"Found {len(empty_dirs)} empty directories:")
    for d in empty_dirs:
        print(f"  {d}")
    print()

    response = input("Delete these empty directories? (y/N) ").strip().lower()
    if response == 'y':
        for d in empty_dirs:
            try:
                os.rmdir(d)
                print(f"Deleted: {d}")
            except Exception as e:
                print(f"Error deleting {d}: {e}")
        print(f"\nDeleted {len(empty_dirs)} empty directories.")
    else:
        print("Aborted. No directories deleted.")


if __name__ == "__main__":
    main()
