#!/usr/bin/env python3
"""Convert open-thoughts/OpenThoughts-Agent-SFT-100K into a slime OPSA prompt file.

OPSA consumes only prompts: the policy generates its own fresh responses at
temperature 1 and the dataset responses are never trained on. This script
therefore extracts the leading system message(s) plus the first user message
of every terminus-2 trajectory as a standalone prompt and writes slime-format
JSONL:

    {"prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}], "label": ""}

Failed GLM traces (AgentTimeoutError etc.) are kept by default: the task prompt
is still valid for on-policy generation. Pass --drop-failed to exclude them.

Usage:
    python tools/prepare_openthoughts_agent_prompts.py \
        --out data/openthoughts_agent_100k_prompts.jsonl [--limit 20000]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="open-thoughts/OpenThoughts-Agent-SFT-100K")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/openthoughts_agent_100k_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0, help="keep first N prompts after shuffle (0 = all)")
    parser.add_argument("--max-prompt-chars", type=int, default=32000)
    parser.add_argument("--drop-failed", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    seen: set[str] = set()
    rows: list[dict] = []
    dropped = 0
    for record in ds:
        result = record.get("result")
        if args.drop_failed and isinstance(result, str) and result.strip():
            dropped += 1
            continue
        conversations = record["conversations"]
        system_parts = [c["content"] for c in conversations if c.get("role") == "system" and (c.get("content") or "").strip()]
        first_user = next(
            (c["content"] for c in conversations if c.get("role") == "user" and (c.get("content") or "").strip()),
            None,
        )
        if first_user is None:
            dropped += 1
            continue
        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": first_user.strip()})
        text = json.dumps(messages, ensure_ascii=False)
        if len(text) > args.max_prompt_chars:
            dropped += 1
            continue
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        rows.append({"prompt": messages, "label": ""})

    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} prompts to {out} (dropped {dropped} rows: empty/oversize/duplicate"
          f"{'/failed' if args.drop_failed else ''})")


if __name__ == "__main__":
    main()
