#!/usr/bin/env python3
"""Build the UkisAI OPSA ~50k multi-domain prompt set from NVIDIA Nemotron
post-training datasets (collection: nvidia/nemotron-post-training-datasets).

OPSA consumes prompts only - the policy generates fresh responses at
temperature 1 and dataset responses are never trained on. Extraction is
prompt-only:

* SFT conversation sets: leading system message(s) + first user turn.
* RL `responses_create_params` sets: the Responses-API `input` messages
  (+ optional `tools`, kept for slime's --tool-key).
* Standalone fields: Math-v4 `problem`, RL-Science `problem`, RL-Safety
  `prompt`, OpenCode `agent_prompt`.

All sources are read as raw lines (JSONL) via HfFileSystem range reads, or as
row-group slices for parquet shards - no full downloads, no `datasets`
streaming (immune to heterogeneous-column parse failures).

Output: slime-format JSONL rows
    {"prompt": [{"role": ..., "content": ...}], "label": "", "source": repo,
     "tools": [...]}   # tools only when the source provides them

Usage:
    python tools/prepare_nemotron_prompts.py --out data/opsa_50k_training_v1
"""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from huggingface_hub import HfFileSystem

SEED = 1234
MIN_PROMPT_CHARS = 50
MAX_PROMPT_CHARS = 30000

# (dataset, extractor, quota, path, start_fraction)
SOURCES = [
    # math (long-form reasoning core)
    ("nvidia/Nemotron-SFT-Math-v4", "problem", 12000, "data/train-00000-of-00012.parquet", 0.0),
    # science
    ("nvidia/Nemotron-SFT-Science-v2", "messages", 1500, "rqa.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Science-v2", "messages", 1500, "so.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Science-v2", "messages", 1500, "syn_mcq.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Science-v2", "messages", 1500, "vendor.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Science-v1", "problem", 1000, "so_openq.jsonl", 0.05),
    # ARC-AGI reasoning puzzles (Oct 2026)
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 1037, "data/large_no_reasoning_no_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 737, "data/large_reasoning_and_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/large_reasoning_no_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/large_tools_no_reasoning/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/small_no_reasoning_no_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/small_reasoning_and_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/small_reasoning_no_tools/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-ARC-AGI-v1", "messages", 437, "data/small_tools_no_reasoning/data.jsonl", 0.05),
    # multilingual (4 of the 12 language/domain files)
    ("nvidia/Nemotron-SFT-Multilingual-v2", "messages", 750, "ultra-v3_code_hi_translated_final.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Multilingual-v2", "messages", 750, "ultra-v3_code_ja_translated_final.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Multilingual-v2", "messages", 750, "ultra-v3_math_hi_translated_final.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Multilingual-v2", "messages", 750, "ultra-v3_math_ja_translated_final.jsonl", 0.05),
    # safety / behavior
    ("nvidia/Nemotron-SFT-Safety-v1", "messages", 1500, "data/train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Safety-v1", "prompt", 1000, "data/train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-litmus-bench-v0.1", "responses", 500, "train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Identity-Following-v1", "responses", 500, "train.jsonl", 0.05),
    # software engineering
    ("nvidia/Nemotron-SFT-SWE-v3", "messages", 2000, "data/train-00000-of-00096.parquet", 0.0),
    # instruction following
    ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "messages", 3500, "data/chat.jsonl", 0.30),
    ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "messages", 2500, "data/instruction_following.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", "responses", 500, "direct_generation/train-00000-of-00001.parquet", 0.0),
    ("nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", "responses", 500, "diversified_tasks/train-00000-of-00001.parquet", 0.0),
    ("nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", "responses", 500, "tool_calling_extraction/train-00000-of-00001.parquet", 0.0),
    ("nvidia/Nemotron-RL-Instruction-Following-Calendar-v2", "responses", 500, "train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1", "responses", 500, "ds3_citation_train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1", "responses", 500, "ds2_freeform_train.jsonl", 0.05),
    # agentic (thin slices of the huge SFT shards + RL pivots)
    ("nvidia/Nemotron-SFT-Agentic-v2", "messages", 400, "data/tool_calling.jsonl", 0.10),
    ("nvidia/Nemotron-SFT-Agentic-v2", "messages", 400, "data/interactive_agent.jsonl", 0.10),
    ("nvidia/Nemotron-SFT-Agentic-v2", "messages", 400, "data/search.jsonl", 0.10),
    ("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", "responses", 1500, "train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1", "responses", 1000, "train.jsonl", 0.05),
    ("nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1", "responses", 500, "train.jsonl", 0.05),
    # code
    ("nvidia/Nemotron-SFT-OpenCode-v1", "agent_prompt", 2400, "general/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-OpenCode-v1", "agent_prompt", 1680, "bash_only_tool/data.jsonl", 0.30),
    ("nvidia/Nemotron-SFT-OpenCode-v1", "agent_prompt", 1200, "agent_skills/data.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "messages", 500, "data/competitive_programming_python_00.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "messages", 500, "data/competitive_programming_cpp_00.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "user_only", 500, "data/exercism.jsonl", 0.05),
    ("nvidia/Nemotron-SFT-Competitive-Programming-v2", "messages", 500, "data/text_to_sql.jsonl", 0.05),
]

FS = HfFileSystem()


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return ""


def _normalize_messages(messages):
    """Leading system messages + first user message, as clean {role, content}."""
    system, first_user = [], None
    for msg in messages:
        role, content = msg.get("role"), _text_of(msg.get("content"))
        if role == "system" and content.strip():
            system.append({"role": "system", "content": content.strip()})
        elif role == "user" and content.strip() and first_user is None:
            first_user = content.strip()
    if first_user is None:
        return None
    return system + [{"role": "user", "content": first_user}]


def _row_to_output(messages, source, label="", tools=None):
    text = json.dumps(messages, ensure_ascii=False)
    if not (MIN_PROMPT_CHARS <= len(text) <= MAX_PROMPT_CHARS):
        return None
    row = {"prompt": messages, "label": label, "source": source}
    if tools:
        row["tools"] = tools
    return row


def extract_messages(row, source):
    messages = _normalize_messages(row.get("messages") or [])
    return _row_to_output(messages, source) if messages else None


def extract_problem(row, source):
    problem = _text_of(row.get("problem"))
    if not problem.strip():
        return None
    label = str(row.get("expected_answer", "")) if row.get("expected_answer") is not None else ""
    return _row_to_output([{"role": "user", "content": problem.strip()}], source, label=label)


def extract_prompt(row, source):
    prompt = _text_of(row.get("prompt"))
    if not prompt.strip():
        return None
    return _row_to_output([{"role": "user", "content": prompt.strip()}], source)


def extract_agent_prompt(row, source):
    prompt = _text_of(row.get("agent_prompt"))
    if not prompt.strip():
        return extract_messages(row, source)
    return _row_to_output([{"role": "user", "content": prompt.strip()}], source)


def extract_user_only(row, source):
    """First user message only - for sources whose shared system template is
    oversized boilerplate (e.g. exercism)."""
    for msg in row.get("messages") or []:
        content = _text_of(msg.get("content"))
        if msg.get("role") == "user" and content.strip():
            return _row_to_output([{"role": "user", "content": content.strip()}], source)
    return None


def extract_responses(row, source):
    rcp = row.get("responses_create_params") or {}
    raw_input = rcp.get("input")
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input.strip()}]
    else:
        messages = []
        for item in raw_input or []:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item.strip()})
            elif isinstance(item, dict):
                content = _text_of(item.get("content"))
                if content.strip():
                    messages.append({"role": item.get("role") or "user", "content": content.strip()})
        messages = _normalize_messages(messages) or messages
    tools = rcp.get("tools") or None
    return _row_to_output(messages, source, tools=tools) if messages else None


EXTRACTORS = {
    "messages": extract_messages,
    "problem": extract_problem,
    "prompt": extract_prompt,
    "agent_prompt": extract_agent_prompt,
    "user_only": extract_user_only,
    "responses": extract_responses,
}


def read_jsonl_slice(dataset, path, start_fraction, count, seed):
    """Read `count` parsed lines from a seeded slice of an HF JSONL file."""
    fs_path = f"datasets/{dataset}/{path}"
    info = FS.info(fs_path)
    start = int(info["size"] * start_fraction) if start_fraction > 0 else 0
    rows, f = [], None
    try:
        f = FS.open(fs_path)
        if start:
            f.seek(max(0, start))
            f.readline()  # discard the partial line at the seek boundary
        while len(rows) < count:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    finally:
        if f is not None:
            f.close()
    return rows


def read_parquet_slice(dataset, path, count, seed):
    """Read `count` rows from a seeded slice of an HF parquet shard."""
    import pyarrow.parquet as pq

    fs_path = f"datasets/{dataset}/{path}"
    with FS.open(fs_path) as f:
        pf = pq.ParquetFile(io.BytesIO(f.read()))
    total = pf.metadata.num_rows
    start = int(total * random.Random(seed).uniform(0.0, 0.5))
    end = min(total, start + count)
    rows = []
    row_group = 0
    offset = 0
    while offset <= end and row_group < pf.metadata.num_row_groups:
        table = pf.read_row_group(row_group)
        group_start, group_len = offset, table.num_rows
        lo = max(start - group_start, 0)
        hi = min(end - group_start, group_len)
        if hi > lo:
            rows.extend(table.slice(lo, hi - lo).to_pylist())
        offset += group_len
        row_group += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/opsa_50k_training_v1")
    parser.add_argument("--cap", type=int, default=50000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []

    for dataset, extractor, quota, path, start_fraction in SOURCES:
        seed = SEED + (hash(f"{dataset}|{path}") % 100000)
        if path.endswith(".parquet"):
            raw_rows = read_parquet_slice(dataset, path, quota + 200, seed)
        else:
            raw_rows = read_jsonl_slice(dataset, path, start_fraction, quota * 2, seed)
        got = 0
        for raw in raw_rows:
            if got >= quota:
                break
            out_row = EXTRACTORS[extractor](raw, dataset)
            if out_row is None:
                continue
            digest = hashlib.sha1(json.dumps(out_row["prompt"], ensure_ascii=False).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(out_row)
            got += 1
        print(f"{dataset} {path}: {got}/{quota}", flush=True)

    random.Random(SEED).shuffle(rows)
    rows = rows[: args.cap]

    jsonl_path = out / "opsa_50k_training_v1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import datasets as hfds

    parquet_rows = [
        {"prompt": r["prompt"], "label": r["label"], "source": r["source"], "tools_json": json.dumps(r.get("tools", []), ensure_ascii=False)}
        for r in rows
    ]
    hfds.Dataset.from_list(parquet_rows).to_parquet(out / "train-00000-of-00001.parquet")

    from collections import Counter

    counts = Counter(r["source"] for r in rows)
    print(f"\ntotal {len(rows)} rows -> {jsonl_path}")
    for source, count in counts.most_common():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
