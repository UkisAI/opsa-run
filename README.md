# UkisAI OPSA Run (DeepSeek-V4.1-Flash, swifter checkpoint)

This repo is a copy of [DripNowhy/On-Policy-Self-Adaptation](https://github.com/DripNowhy/On-Policy-Self-Adaptation)
(the canonical OPSA training code, paper arXiv:2608.31046) adapted for the
UkisAI recovery run on the swifter DeepSeek-V4.1-Flash checkpoint.

## What was audited (upstream code, no errors found)

- Core OPSA math (`slime/slime/backends/megatron_utils/opsa.py`): bottom-20%
  lowest-logp selection across the DP-local packed batch, entropy-ranked
  advantages in `[-1.0, -0.5]`, zero loss mask for unselected tokens, zero task
  reward, no reference model / KL. `tests/test_opsa.py` (29 tests) passes.
- Training recipe: Adam LR 1e-6 constant, betas .9/.98, weight decay .1,
  gradient clip 1.0, batch 64 prompts x 1 fresh response, 1 optimizer step per
  rollout batch, temperature 1 / top-p 1 / top-k -1 (slime defaults; the
  launcher pins temperature 1).
- Eval recipe: 4 samples per prompt, temperature 1, top-p 1, top-k -1,
  32,768 max tokens, math grader (boxed answer), passrate logged. Note: this
  deviates from the paper's canonical eval decoding (0.7/0.8/20) by explicit
  user request; expect slightly noisier and typically lower pass@4 values.
- Known canonical nuance (documented upstream): lowest-token selection is per
  data-parallel rank's packed batch, not global.

## What changed for this run

| Item | Upstream | This run |
|---|---|---|
| Base model | Qwen3 / Qwen3.5 / OLMo3 presets | `deepseek-v4.1-flash` preset (swifter checkpoint) |
| Training steps | 700-1000 | **150** (pilot; extend with `--steps` if the eval slope is positive) |
| Save interval | 20 | **10** |
| Eval interval | 20 | **10** |
| Eval benchmark | AIME validation set | **AIME 2026** (MathArena/aime_2026, 30 problems, **pass@4**) |
| Training prompts | DAPO-17k | **ukisai/opsa_50k_training_v1** (48,448 multi-domain prompts built from NVIDIA Nemotron post-training datasets; prompts only) |

## Run

## Model and data for this run

- **Path to the model (starting checkpoint)**: the **step80 OPSA
  DeepSeek-V4.1** checkpoint:

  `s3://ukisai-model-backups-373383136062/radoslav-opsa/dsv41-flash-opsa/dsv41-s160-opsa-agent-lr1e-6/ckpt/update_0080/`

  Provenance: that run was OPSA on `deepseek-ai/DeepSeek-V4.1-Flash-1017`
  starting from the merged Swift s160 LoRA (r32/alpha64, 280 modules); update 80
  is the step80 model. The checkpoint is a **full training checkpoint**
  (`adapter/` + `optimizer.pt` + `rng.pt`), so pass it via `--resume-from`.
  (Weights-only checkpoints would need `tools/convert_hf_to_torch_dist.py`
  conversion and `--actor-checkpoint` instead. Benchmark artifacts for step80
  live alongside at `.../dsv41-s160-opsa-agent-lr1e-6/bench_update_0080/`.)
- **Training prompts**: https://huggingface.co/datasets/ukisai/opsa_50k_training_v1
  (48,811 multi-domain prompts built from NVIDIA's Nemotron post-training
  datasets, Sep-Oct 2026 - fresher and broader than the earlier
  open-thoughts/OpenThoughts-Agent-SFT-100K plan, which is superseded.
  OPSA consumes prompts only - the model generates fresh responses, dataset
  responses are never trained on). Every row carries verifiable provenance
  (`source`, `source_path`, `source_row`, `source_seek`, `row_sha1`, `row_id`).
- **SFT companion (full traces)**: https://huggingface.co/datasets/ukisai/opsa_sft_50k_v1
  - 45,965 rows, 38,465 with full multi-turn traces (`trace_type == "full"`,
    assistant turns included); the 7,500 `prompt_only` rows come from the
    Nemotron RL sets, which have no assistant turns. Linked 1:1 to the prompt
    dataset by `row_id`/`row_sha1`; each row is byte-verifiable against its
    NVIDIA source (recipe in the dataset card). Use for SFT; 2-epoch
    muon+adamw convention applies.
- **Eval**: AIME 2026 (`data/aime2026_eval.jsonl`, from
  `MathArena/aime_2026`, 30 problems), pass@4, every 10 steps.

### Commands

```bash
cd slime

# 1. Training prompts: ukisai/opsa_50k_training_v1 (48,448 multi-domain prompts
#    built from NVIDIA Nemotron post-training datasets - fresher and broader
#    than the earlier OpenThoughts-Agent-100K plan). slime-ready JSONL:
huggingface-cli download ukisai/opsa_50k_training_v1 \
    data/opsa_50k_training_v1.jsonl --repo-type dataset --local-dir data
# (regenerate from source with: python tools/prepare_nemotron_prompts.py)

# 2. AIME 2026 eval file (30 problems, committed at data/aime2026_eval.jsonl):
python tools/prepare_aime2026_eval.py --out data/aime2026_eval.jsonl

# 3. Convert the swifter checkpoint to a Megatron actor checkpoint, then:
export DEEPSEEK_V41_MODEL_CONFIG=/path/to/swifter/model/definition.sh

# starting model: the step80 OPSA DeepSeek-V4.1 checkpoint - this is the path
# to the model for this run (full training checkpoint: adapter + optimizer + RNG)
aws s3 cp --recursive s3://ukisai-model-backups-373383136062/radoslav-opsa/dsv41-flash-opsa/dsv41-s160-opsa-agent-lr1e-6/ckpt/update_0080/ /path/to/step80/

bash examples/opsa/run_opsa_ukis.sh \
    --hf-checkpoint /path/to/swifter/hf \
    --actor-checkpoint /path/to/swifter/megatron \
    --resume-from /path/to/step80 \
    --save-dir /path/to/opsa_run_out \
    --prompt-data data/opsa_50k_training_v1.jsonl \
    --eval-data data/aime2026_eval.jsonl \
    --megatron-path /path/to/Megatron-LM \
    --steps 150
```

Ship the best checkpoint by the logged AIME 2026 pass@4 (eval every 10 steps),
never a fixed step. To continue past 150 steps: `--steps 300` (the prompt pool
supports ~1,400 steps without repeating a prompt).

Standalone AIME 2026 scoring of a served model (same protocol as
UkisAI/Benchmark_configs):

```bash
python benchmarks/aime_2026.py --api-base http://127.0.0.1:8000/v1
```

---

<div align="center">

# Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement

**Teacher-free on-policy training from the policy's own token probabilities and entropies.**

[![Paper](https://img.shields.io/badge/PAPER-arXiv%3A2608.31046-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2608.31046)
[![Code](https://img.shields.io/badge/CODE-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DripNowhy/On-Policy-Self-Adaptation)
[![Blog](https://img.shields.io/badge/BLOG-2F6BFF?style=for-the-badge&logo=githubpages&logoColor=white)](https://dripnowhy.github.io/On-Policy-Self-Adaptation/)
[![W&B Logs](https://img.shields.io/badge/W%26B%20TRAINING%20LOGS-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/whywhyyy0731-purdue-university/opsa-public/workspace?nw=nwuserwhywhyyy0731)
[![Checkpoints](https://img.shields.io/badge/CHECKPOINTS-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/collections/Tuwhy/on-policy-self-adaptation)
[![License](https://img.shields.io/badge/APACHE--2.0-A42C25?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

**English** · [中文](README_zh.md)

<p>
  <a href="#news">📰 News</a> •
  <a href="#introduction">📖 Introduction</a> •
  <a href="#results">📊 Results</a> •
  <a href="#reproduce-opsa">🚀 Reproduce OPSA</a> •
  <a href="#dataset-preparation">🗂️ Dataset Preparation</a>
</p>
<p>
  <a href="#training-logs">📈 Training Logs</a> •
  <a href="#citation">📝 Citation</a> •
  <a href="#license">📄 License</a>
</p>

</div>

<a id="news"></a>

## News

- **[2026-09-14]** Released [Olmo-3-7B-Think-OPSA](https://huggingface.co/Tuwhy/Olmo-3-7B-Think-OPSA), our OPSA-trained OLMo 3 Think model.
- **[2026-09-03]** Released [Olmo-3-7B-Instruct-OPSA](https://huggingface.co/Tuwhy/Olmo-3-7B-Instruct-OPSA), our OPSA-trained OLMo 3 Instruct model.
- **[2026-09-01]** Our paper ranked **#1 on [Hugging Face Daily Papers](https://huggingface.co/papers/date/2026-09-01)**! Thank you for your interest and support.
- **[2026-08-31]** Released our [paper](https://arxiv.org/abs/2608.31046), [code](https://github.com/DripNowhy/On-Policy-Self-Adaptation), and OPSA-trained Qwen-series checkpoints: [Qwen3-1.7B](https://huggingface.co/Tuwhy/Qwen3-1.7B-OPSA), [Qwen3-4B](https://huggingface.co/Tuwhy/Qwen3-4B-OPSA), and [Qwen3.5-9B](https://huggingface.co/Tuwhy/Qwen3.5-9B-OPSA).

<a id="introduction"></a>

## Introduction

OPSA improves a policy without a teacher model, reward model, reference-model
forward pass, or task reward. The canonical configuration trains only the 20%
of valid response tokens with the lowest actor log probabilities and assigns
entropy-adaptive negative advantages between `-0.5` and `-1.0`. All other
tokens are excluded from the policy loss.

<p align="center">
  <img src="assets/opsa-overview.png" alt="Overview of On-Policy Self-Adaptation" width="96%">
</p>

<a id="results"></a>

## Results

The three math columns report **Avg@32 / Pass@32**; the two O.O.D. columns
report **Avg@32** only. All values are from the OPSA manuscript.

| Model | Variant | AIME24 | AIME25 | HMMT25 | MBPP+ Avg@32 | GPQA_D Avg@32 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 13.44 / 40.00 | 9.69 / 30.00 | 5.73 / 23.33 | 58.24 | 27.92 |
| Qwen3-1.7B | **+ OPSA** | **48.85 / 80.00** | **35.31 / 66.67** | **23.33 / 50.00** | **59.44** | **32.40** |
| Qwen3-4B | Base | 23.33 / 56.67 | 20.52 / 56.67 | 13.13 / 33.33 | 66.93 | 38.46 |
| Qwen3-4B | **+ OPSA** | **62.08 / 83.33** | **58.44 / 83.33** | **37.40 / 60.00** | **68.35** | **41.29** |
| Qwen3.5-9B | Base | 76.35 / 93.33 | 56.04 / 93.33 | 44.48 / 86.67 | 77.33 | 70.53 |
| Qwen3.5-9B | **+ OPSA** | **87.81 / 96.67** | **76.98 / 96.67** | **67.40 / 93.33** | **79.27** | **73.70** |

The models are trained and evaluated in non-thinking mode. Training uses only
the questions from DAPO-17k. Evaluation samples 32 responses per prompt with
temperature `0.7`, top-k `20`, top-p `0.8`, and maximum response length
`32,768`.

<a id="reproduce-opsa"></a>

## Reproduce OPSA

### Installation

```bash
git clone https://github.com/DripNowhy/On-Policy-Self-Adaptation.git
cd On-Policy-Self-Adaptation/slime
pip install -e .
```

<a id="dataset-preparation"></a>

### Dataset preparation

Install the Hugging Face CLI and choose an external data directory:

```bash
python3 -m pip install --upgrade huggingface_hub
export OPSA_DATA_DIR=/path/to/opsa-data
mkdir -p "${OPSA_DATA_DIR}"
```

Download the exact Slime-ready JSONL files used by this project:

```bash
hf download zhuzilin/dapo-math-17k dapo-math-17k.jsonl \
  --repo-type dataset \
  --revision 2e65612930298bde4c5d58fd97b3f23a483aaff9 \
  --local-dir "${OPSA_DATA_DIR}/dapo-math-17k"

hf download zhuzilin/aime-2024 aime-2024.jsonl \
  --repo-type dataset \
  --revision 1c625e328db94ec7ef7ff169016b097c468d60b9 \
  --local-dir "${OPSA_DATA_DIR}/aime-2024"
```

The resulting training file has 17,398 rows and the AIME24 evaluation file has
30 rows. Both already contain the required `prompt` and `label` fields, so no
conversion is needed. OPSA training loads only `prompt` and always uses zero
task reward; `label` is read only during evaluation.

The training JSONL is the pinned
[Slime-ready mirror](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
of the
[original DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
Use the pinned mirror for exact reproduction rather than the currently
duplicated 1.79M-row upstream parquet. The AIME24 file comes from the
[Slime-ready AIME24 mirror](https://huggingface.co/datasets/zhuzilin/aime-2024).
These external datasets remain subject to their own terms.

### CPU dry run

This resolves and prints the complete command without checking paths, querying
GPUs, starting Ray, or training:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

### GPU training

Prepare a Hugging Face checkpoint, its converted Megatron actor checkpoint, and
a Megatron-LM checkout. See the
[checkpoint conversion instructions](slime/examples/opsa/README.md#prepare-checkpoints).

After replacing the model and output paths below, this command starts the
canonical Qwen3-1.7B OPSA run on a single machine with 8 GPUs:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --hf-checkpoint /path/to/Qwen3-1.7B \
  --actor-checkpoint /path/to/qwen3-1.7b-megatron \
  --save-dir /path/to/outputs/opsa-qwen3-1.7b-lowest20 \
  --prompt-data "${OPSA_DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --eval-data "${OPSA_DATA_DIR}/aime-2024/aime-2024.jsonl" \
  --megatron-path /path/to/Megatron-LM
```

This preset uses 4 actor GPUs and 4 rollout GPUs, runs for 700 steps, and saves
and evaluates every 20 steps. The launcher starts a local Ray cluster and stops
only the cluster it started. See the
[launcher guide](slime/examples/opsa/README.md) for existing Ray clusters and the
other supported models.

OLMo 3 training is available through `--model olmo3-7b-instruct`,
`olmo3-7b-instruct-sft`, and `olmo3-7b-think`; see the
[model presets and checkpoint conversion guide](slime/examples/opsa/README.md#model-presets).

<a id="training-logs"></a>

## Training logs

Reference training curves and metrics are available in the
[public W&B workspace](https://wandb.ai/whywhyyy0731-purdue-university/opsa-public/workspace?nw=nwuserwhywhyyy0731).

<a id="citation"></a>

## Citation

If you find this work useful, please cite:

```bibtex
@article{ding2026does,
  title={Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement},
  author={Ding, Yi and Zhang, Ruqi},
  journal={arXiv preprint arXiv:2608.31046},
  year={2026}
}
```

<a id="license"></a>

## License

This project is built on [slime](https://github.com/THUDM/slime) and released
under the [Apache License 2.0](LICENSE). See [UPSTREAM.md](UPSTREAM.md) for
snapshot provenance.
