#!/usr/bin/env python3
"""
Fix transcripts where <END_CONVERSATION> appears in messages
other than the final user message by truncating at the first occurrence
and replacing with a proper end message.
"""

import json
import sys
from pathlib import Path


def fix_transcript(transcript_path: Path) -> bool:
    """
    Fix transcript if <END_CONVERSATION> appears in inappropriate locations.

    Truncates the conversation at the first instance of <END_CONVERSATION>
    (if not in the final user message), replaces it with a proper end message,
    sets "ended_early": true, and updates the turns count.

    Returns True if the transcript was modified.
    """
    with open(transcript_path) as f:
        data = json.load(f)

    conversation = data.get("conversation", [])
    if not conversation:
        return False

    # Find first occurrence of <END_CONVERSATION>
    end_index = None
    for i, msg in enumerate(conversation):
        if "<END_CONVERSATION>" in msg.get("content", ""):
            end_index = i
            break

    # If no <END_CONVERSATION> found, nothing to fix
    if end_index is None:
        return False

    # Check if it's already properly formatted:
    # - Must be the last message (index == len - 1)
    # - Must be a user message
    # - Content must be exactly "<END_CONVERSATION>" with no other text
    if (end_index == len(conversation) - 1 and
        conversation[end_index].get("role") == "user" and
        conversation[end_index].get("content") == "<END_CONVERSATION>"):
        # Already properly formatted
        return False

    # Truncate conversation at the first <END_CONVERSATION> and replace
    data["conversation"] = conversation[:end_index] + [
        {
            "role": "user",
            "content": "<END_CONVERSATION>"
        }
    ]

    # Update metadata
    data["ended_early"] = True
    data["turns"] = len(data["conversation"])

    # Write back to file
    with open(transcript_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_early_end.py <transcript_directory>")
        sys.exit(1)

    transcript_dir = Path(sys.argv[1])

    if not transcript_dir.is_dir():
        print(f"Error: {transcript_dir} is not a directory")
        sys.exit(1)

    # Find all JSON files in the directory
    transcript_files = list(transcript_dir.glob("*.json"))

    if not transcript_files:
        print(f"No JSON files found in {transcript_dir}")
        return

    # Fix each transcript
    fixed_transcripts = []
    for transcript_file in sorted(transcript_files):
        if fix_transcript(transcript_file):
            fixed_transcripts.append(transcript_file.name)

    # Print results
    if fixed_transcripts:
        print(f"Fixed {len(fixed_transcripts)} transcript(s):")
        for name in fixed_transcripts:
            print(f"  {name}")
    else:
        print("No transcripts needed fixing.")


if __name__ == "__main__":
    main()
