"""Decode base64 responses from sorrybench_encode runs and rewrite for judging.

The Wei et al. base64 attack asks the model to respond in base64. To score with
2_harmbench_judge.py (or any plain-text judge), we need:
  - response: best-effort base64-decoded model output
  - prompt:   the ORIGINAL plain-text behavior (not the encoded wrapper)

Input is the --output_jsonl produced by 1_steering_{vllm,hf}.py /
1_capping_vllm.py. Those scripts copy all question fields through, so each row
already carries `behavior` (plain text) alongside the encoded `prompt`/`question`.

Output is a new JSONL with the same rows but:
  - `response`       overwritten with the decoded text
  - `response_raw`   preserves the original model output
  - `prompt`         overwritten with `behavior` (plain text, for the judge)
  - `decode_status`  one of: full | chunk | none

Usage:
  python3 evals/sorrybench_encode/decode_responses.py \
      /workspace/qwen-3-32b/evals/sorrybench_encode/asst_pc1_contrast.jsonl \
      --out /workspace/qwen-3-32b/evals/sorrybench_encode/asst_pc1_contrast_decoded.jsonl

  # then judge as usual
  uv run evals/2_harmbench_judge.py \
      /workspace/qwen-3-32b/evals/sorrybench_encode/asst_pc1_contrast_decoded.jsonl \
      --output /workspace/qwen-3-32b/evals/sorrybench_encode/asst_pc1_contrast_scores.jsonl
"""

import argparse
import base64
import binascii
import json
import re
from pathlib import Path

B64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def try_b64(s: str) -> str | None:
    s = s.strip()
    if not s:
        return None
    pad = (-len(s)) % 4
    try:
        raw = base64.b64decode(s + "=" * pad, validate=True)
        return raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def decode_response(resp: str) -> tuple[str, str]:
    """Return (decoded_text, status). status ∈ {lines, full, chunk, none}.

    lines — every non-empty line decodes individually (model emitted base64 per-line)
    full  — whole response (whitespace-stripped) decodes as one base64 blob
    chunk — one or more base64-looking substrings decode; concatenated
    none  — nothing decodes (likely a plain-text refusal); response unchanged
    """
    compact = "".join(resp.split())
    full = try_b64(compact)
    if full is not None:
        return full, "full"

    lines = [ln for ln in resp.splitlines() if ln.strip()]
    if lines:
        per_line = [try_b64(ln) for ln in lines]
        if all(d is not None for d in per_line):
            return "\n".join(per_line), "lines"  # type: ignore[arg-type]

    parts: list[str] = []
    for m in B64_RUN.finditer(resp):
        d = try_b64(m.group(0))
        if d is not None:
            parts.append(d)
    if parts:
        return "\n".join(parts), "chunk"

    return resp, "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_jsonl")
    p.add_argument("--out", default=None, help="defaults to <input>_decoded.jsonl")
    args = p.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = (
        Path(args.out)
        if args.out
        else in_path.with_name(in_path.stem + "_decoded.jsonl")
    )

    n = 0
    status_counts: dict[str, int] = {"lines": 0, "full": 0, "chunk": 0, "none": 0}
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("response", "")
            decoded, status = decode_response(raw)
            row["response_raw"] = raw
            row["response"] = decoded
            row["decode_status"] = status
            if "behavior" in row:
                row["prompt"] = row["behavior"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            status_counts[status] += 1
            n += 1

    print(f"decoded {n} rows → {out_path}")
    print("  " + " ".join(f"{k}={v}" for k, v in status_counts.items()))


if __name__ == "__main__":
    main()
