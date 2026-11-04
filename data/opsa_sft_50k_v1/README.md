---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
- multilingual
size_categories:
- 10K<n<100K
tags:
- sft
- nemotron
- multi-domain
- opsa
configs:
- config_name: default
  data_files: train-00000-of-00001.parquet
---

# ukisai/opsa_sft_50k_v1

SFT companion to [`ukisai/opsa_50k_training_v1`](https://huggingface.co/datasets/ukisai/opsa_50k_training_v1):
the **same source rows with full message traces** (assistant turns included),
built in the same pass, linked 1:1 by `row_id` / `row_sha1`.

**45,965 rows** - **38,465 `full` traces** (multi-turn conversations from the
Nemotron SFT datasets, assistant turns included) and **7,500
`prompt_only`** rows from the Nemotron RL sets, which have no assistant turns
by design (prompt + verifier data); filter on `trace_type == "full"` for SFT.
Assistant `reasoning_content` and tool-call payloads were dropped; `content`
only.

## Format

- `opsa_sft_50k_v1.jsonl`:
  `{"messages": [{"role", "content"}...], "trace_type": "full"|"prompt_only",
  "label", "source", "source_path", "source_row", "source_seek", "row_sha1",
  "row_id"}`
- `train-00000-of-00001.parquet`: same rows with `messages` serialized as
  `messages_json` (string).

## Provenance and verification

Identical scheme to the prompt dataset (see its card): `source` + `source_path`
+ `source_row` + `source_seek` + `row_sha1` locate the exact source line/row in
the NVIDIA repo, and `row_sha1` lets any agent verify it byte-for-byte:

```python
from huggingface_hub import HfFileSystem
import hashlib
fs = HfFileSystem()
row = ...  # any row of this dataset
with fs.open(f"datasets/{row['source']}/{row['source_path']}") as f:
    f.seek(row["source_seek"]); f.readline()  # discard partial boundary line
    for _ in range(row["source_row"]):
        f.readline()
    line = f.readline().strip()
assert hashlib.sha1(line).hexdigest() == row["row_sha1"]
```

To turn a prompt row back into its full trace: join
`ukisai/opsa_50k_training_v1` and this dataset on `row_id` (identical values),
or on (`source`, `source_path`, `source_row`).

## Handoff note for agents

Full traces for SFT: download `opsa_sft_50k_v1.jsonl`, filter
`trace_type == "full"`, feed `messages` to the SFT pipeline. Any row can be
audited against its NVIDIA source with the recipe above; `row_sha1` mismatches
mean the source file changed (check the repo revision pin in the build tool
`tools/prepare_nemotron_prompts.py` in [UkisAI/opsa-run](https://github.com/UkisAI/opsa-run)).
