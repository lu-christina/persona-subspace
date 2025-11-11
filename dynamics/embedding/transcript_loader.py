"""
Custom dataset loader for transcript JSON files.

Loads multi-turn conversations from JSON transcript files and yields individual
user turns with metadata for embedding.
"""

import json
import re
from pathlib import Path
from typing import Any, Iterator


def load_transcript_dataset(
    input_root: Path,
    auditor_models: list[str],
    text_field: str = "text",
    max_rows: int | None = None,  # Kept for API compatibility but not used
) -> Iterator[dict[str, Any]]:
    """
    Load transcript JSON files and yield user turns for embedding.

    Note: max_rows limiting is handled by chatspace's _rows_from_dataset, not here.

    Args:
        input_root: Base path like /workspace/{model_short}/dynamics
        auditor_models: List of auditor model short names (e.g., ["gpt-5", "sonnet-4.5"])
        text_field: Name of field to put user message text in (default: "text")
        max_rows: Ignored (for API compatibility only)

    Yields:
        Dictionary with fields:
            - text: User message content
            - model: Full HF model name from JSON
            - auditor_model: Full HF auditor model name from JSON
            - short_model: Shortened model name extracted from path
            - short_auditor_model: Shortened auditor model name from path
            - domain: Domain (coding, therapy, writing)
            - persona_id: Persona ID (integer)
            - topic_id: Topic ID (integer)
            - response_id: Turn index (0, 2, 4, ... only even indices)
            - source_file: Absolute path to source JSON file
    """
    # Extract short model name from input_root path
    # E.g., /workspace/qwen-3-32b/dynamics -> qwen-3-32b
    short_model = input_root.parent.name

    for auditor_short in auditor_models:
        # Construct path: {input_root}/{auditor}/default/transcripts/
        transcript_dir = input_root / auditor_short / "default" / "transcripts"

        if not transcript_dir.exists():
            print(f"Warning: Transcript directory not found: {transcript_dir}")
            continue

        # Find all JSON files matching pattern: {domain}_persona{id}_topic{id}.json
        json_files = sorted(transcript_dir.glob("*.json"))

        for json_path in json_files:
            # Parse filename to extract metadata
            # Expected pattern: {domain}_persona{p_id}_topic{t_id}.json
            filename = json_path.stem
            match = re.match(r"^(\w+)_persona(\d+)_topic(\d+)$", filename)

            if not match:
                print(f"Warning: Skipping file with unexpected name: {json_path.name}")
                continue

            domain = match.group(1)
            persona_id = int(match.group(2))
            topic_id = int(match.group(3))

            # Load JSON file
            try:
                with json_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue

            # Validate structure
            if "conversation" not in data:
                print(f"Warning: No 'conversation' field in {json_path}")
                continue

            conversation = data["conversation"]
            if not isinstance(conversation, list):
                print(f"Warning: 'conversation' is not a list in {json_path}")
                continue

            # Extract full model names from JSON
            full_model_name = data.get("model", "unknown")
            full_auditor_name = data.get("auditor_model", "unknown")

            # Yield each user turn (even indices: 0, 2, 4, ...)
            for idx, turn in enumerate(conversation):
                # Only process user turns
                if idx % 2 != 0:
                    continue

                # Validate turn structure
                if not isinstance(turn, dict):
                    print(f"Warning: Turn {idx} in {json_path} is not a dict")
                    continue

                if "content" not in turn:
                    print(f"Warning: Turn {idx} in {json_path} has no 'content' field")
                    continue

                # Check role (should be "user")
                role = turn.get("role", "")
                if role != "user":
                    # Even indices should be user turns, but verify
                    print(f"Warning: Turn {idx} in {json_path} has role '{role}', expected 'user'")
                    continue

                # Yield row with all metadata
                row = {
                    text_field: turn["content"],
                    "model": full_model_name,
                    "auditor_model": full_auditor_name,
                    "short_model": short_model,
                    "short_auditor_model": auditor_short,
                    "domain": domain,
                    "persona_id": persona_id,
                    "topic_id": topic_id,
                    "response_id": idx,
                    "source_file": str(json_path.absolute()),
                }

                yield row
