#!/usr/bin/env python3
"""
Analyze transcript lengths across domains and validate turn counts.

Usage:
    python average_length.py <transcript_folder>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_transcript(file_path: Path) -> Dict:
    """Load a transcript JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def count_actual_turns(conversation: List[Dict]) -> int:
    """Count the actual number of turns (user-assistant exchanges) in a conversation."""
    # Count the number of user messages (each represents one turn)
    return sum(1 for msg in conversation if msg.get('role') == 'user')


def analyze_transcripts(folder_path: Path) -> None:
    """Analyze transcripts in the given folder."""
    domains = ["coding", "writing", "therapy", "philosophy"]

    # Store stats per domain
    domain_stats = {domain: [] for domain in domains}

    # Track longest turns with at least 10 transcripts
    turn_counts = defaultdict(list)  # turn_count -> list of (domain, file_path)

    # Iterate through all JSON files
    for json_file in folder_path.glob("*.json"):
        try:
            transcript = load_transcript(json_file)

            # Extract domain and turn from filename
            # Expected format: domain_turn_X.json or similar
            filename = json_file.stem
            parts = filename.split('_')

            # Find domain
            domain = None
            for d in domains:
                if d in filename:
                    domain = d
                    break

            if domain is None:
                continue

            # Get turn from transcript metadata
            declared_turn = transcript.get('turns', None)
            conversation = transcript.get('conversation', [])

            # Calculate actual turn count
            actual_turn = count_actual_turns(conversation)

            # Store stats
            domain_stats[domain].append({
                'file': json_file.name,
                'declared_turn': declared_turn,
                'actual_turn': actual_turn,
                'message_count': len(conversation)
            })

            # Track for longest message count analysis
            turn_counts[len(conversation)].append((domain, json_file.name))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)

    # Print domain statistics
    print("=" * 80)
    print("TRANSCRIPT LENGTH STATISTICS BY DOMAIN")
    print("=" * 80)
    print()

    for domain in domains:
        stats = domain_stats[domain]
        if not stats:
            print(f"{domain.upper()}: No transcripts found")
            print()
            continue

        message_counts = [s['message_count'] for s in stats]
        user_turns = [s['actual_turn'] for s in stats]
        mismatches = [s for s in stats if s['declared_turn'] != s['message_count'] and s['declared_turn'] is not None]

        print(f"{domain.upper()}:")
        print(f"  Total transcripts: {len(stats)}")
        print(f"  Message count statistics:")
        print(f"    Mean: {sum(message_counts) / len(message_counts):.2f}")
        print(f"    Min: {min(message_counts)}")
        print(f"    Max: {max(message_counts)}")
        print(f"    Median: {sorted(message_counts)[len(message_counts) // 2]}")
        print(f"  User turns (for reference):")
        print(f"    Mean: {sum(user_turns) / len(user_turns):.2f}")
        print(f"    Min: {min(user_turns)}")
        print(f"    Max: {max(user_turns)}")

        if mismatches:
            print(f"  ⚠ Mismatches found: {len(mismatches)} transcripts have declared_turns != message_count")
            for m in mismatches[:3]:  # Show first 3
                print(f"    - {m['file']}: declared={m['declared_turn']}, actual_messages={m['message_count']}")

        print()

    # Find longest message count with at least 10 transcripts (overall)
    print("=" * 80)
    print("LONGEST MESSAGE COUNT WITH AT LEAST 10 TRANSCRIPTS (OVERALL)")
    print("=" * 80)
    print()

    # Sort by message count descending
    sorted_counts = sorted(turn_counts.items(), key=lambda x: x[0], reverse=True)

    found = False
    for msg_count, transcript_list in sorted_counts:
        if len(transcript_list) >= 10:
            print(f"Message count: {msg_count}")
            print(f"Number of transcripts: {len(transcript_list)}")
            print()

            # Group by domain
            by_domain = defaultdict(list)
            for domain, filename in transcript_list:
                by_domain[domain].append(filename)

            print("By domain:")
            for domain in domains:
                if domain in by_domain:
                    print(f"  {domain}: {len(by_domain[domain])} transcripts")

            print()
            print("Sample transcripts:")
            for domain, filename in transcript_list[:10]:
                print(f"  - {domain}/{filename}")

            found = True
            break

    if not found:
        print("No message count has at least 10 transcripts.")
        print()
        print("Message count distribution:")
        for msg_count, transcript_list in sorted_counts[:10]:
            print(f"  {msg_count} messages: {len(transcript_list)} transcripts")

    print()

    # Find longest message count per domain with at least 10 transcripts
    print("=" * 80)
    print("LONGEST MESSAGE COUNT PER DOMAIN (WITH AT LEAST 10 TRANSCRIPTS)")
    print("=" * 80)
    print()

    # Group transcripts by domain and message count
    domain_msg_counts = {domain: defaultdict(list) for domain in domains}

    for msg_count, transcript_list in turn_counts.items():
        for domain, filename in transcript_list:
            domain_msg_counts[domain][msg_count].append(filename)

    # For each domain, find the longest message count with at least 10 transcripts
    for domain in domains:
        msg_counts = domain_msg_counts[domain]
        sorted_domain_counts = sorted(msg_counts.items(), key=lambda x: x[0], reverse=True)

        found_for_domain = False
        for msg_count, file_list in sorted_domain_counts:
            if len(file_list) >= 10:
                print(f"{domain.upper()}:")
                print(f"  Longest message count with ≥10 transcripts: {msg_count}")
                print(f"  Number of transcripts: {len(file_list)}")
                print(f"  Sample files:")
                for filename in file_list[:5]:
                    print(f"    - {filename}")
                print()
                found_for_domain = True
                break

        if not found_for_domain:
            print(f"{domain.upper()}: No message count has at least 10 transcripts")
            print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python average_length.py <transcript_folder>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)

    analyze_transcripts(folder_path)


if __name__ == "__main__":
    main()
