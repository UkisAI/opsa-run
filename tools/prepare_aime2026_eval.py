#!/usr/bin/env python3
"""Build the AIME 2026 in-training eval file for slime.

30 problems from MathArena/aime_2026. The prompt matches the MathArena/UkisAI
protocol used by UkisAI/Benchmark_configs (final answer in \\boxed{}, integer
0-999), which is exactly what slime's --eval-rm-type math grader extracts.

Eval decodes 4 samples per prompt (pass@4) at temperature 0.7, top-p 0.8,
top-k 20, 32,768 max tokens - the canonical OPSA evaluation settings.

Usage:
    python tools/prepare_aime2026_eval.py --out slime/examples/opsa/data/aime2026_eval.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

PROMPT_PREFIX = "Put your final answer within \\boxed{}."
AIME_SUFFIX = "\nThe answer is an integer between 0 and 999 inclusive."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MathArena/aime_2026")
    parser.add_argument("--out", default="data/aime2026_eval.jsonl")
    args = parser.parse_args()

    rows = list(load_dataset(args.dataset, split="train"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in rows:
            problem = record["problem"].strip()
            prompt = f"{PROMPT_PREFIX}{AIME_SUFFIX}\n\n{problem}"
            payload = {
                "prompt": [{"role": "user", "content": prompt}],
                "label": str(record["answer"]),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} AIME 2026 problems to {out}")


if __name__ == "__main__":
    main()
