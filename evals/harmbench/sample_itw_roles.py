"""Sample in-the-wild jailbreak personas for the HarmBench persona×question eval.

Loads TrustAIRLab/in-the-wild-jailbreak-prompts (jailbreak_2023_12_25, 1405 prompts),
takes a seeded stratified sample over the `community` attack-family clusters so the
5 personas span distinct strategies rather than 5 DAN variants, and writes a
roles_itw.jsonl in the same {role, id, type} schema as roles.jsonl.

A single empty 'default' role is appended as id=n so baseline (no system prompt)
is computed in the same run.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_chars",
        type=int,
        default=4000,
        help="drop prompts longer than this to keep context manageable",
    )
    p.add_argument("--config", default="jailbreak_2023_12_25")
    p.add_argument("--out", default=str(Path(__file__).parent / "roles_itw.jsonl"))
    args = p.parse_args()

    ds = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts", args.config, split="train"
    )
    print(f"loaded {len(ds)} prompts from {args.config}")

    by_community: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        if len(row["prompt"]) > args.max_chars:
            continue
        by_community[str(row.get("community", ""))].append(i)

    communities = sorted(c for c in by_community if by_community[c])
    print(f"{len(communities)} non-empty communities after length filter")

    rng = random.Random(args.seed)
    rng.shuffle(communities)
    picks: list[int] = []
    for c in communities:
        if len(picks) >= args.n:
            break
        picks.append(rng.choice(by_community[c]))
    assert len(picks) == args.n, f"only found {len(picks)}/{args.n}"

    roles = []
    for rid, idx in enumerate(picks):
        row = ds[idx]
        # 1_steering_{vllm,hf}.py call .format(company=..., name=...) on role text;
        # escape braces so in-the-wild prompts containing {...} don't raise.
        role_text = row["prompt"].replace("{", "{{").replace("}", "}}")
        roles.append(
            {
                "role": role_text,
                "id": rid,
                "type": "itw",
                "community": row.get("community", ""),
                "platform": row.get("platform", ""),
                "source": row.get("source", ""),
                "ds_index": int(idx),
            }
        )
    roles.append({"role": "", "id": args.n, "type": "default"})

    with open(args.out, "w") as f:
        for r in roles:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(roles)} roles → {args.out}")
    for r in roles:
        preview = r["role"][:70].replace("\n", " ")
        print(
            f"  id={r['id']} type={r['type']} community={r.get('community', '-')}: {preview}..."
        )


if __name__ == "__main__":
    main()
