"""AIME 2026: 30 problems x 4 seeds, 250k tokens, avg@4 + pass@4.

Standalone, protocol-compatible with UkisAI/Benchmark_configs (MathArena data,
MathArena prompt, boxed-answer grading). Point it at a served model:

    python benchmarks/aime_2026.py --api-base http://127.0.0.1:8000/v1
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
THINKING = {"enable_thinking": True, "reasoning_effort": "xhigh"}
PROMPT = "Put your final answer within \\boxed{}.\nThe answer is an integer between 0 and 999 inclusive.\n\n{problem}"
KEY = os.environ.get("API_KEY", "none")


def boxed(text):
    """Contents of the last \\boxed{...}, braces balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + 7
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth < 0:
            return text[j:k].strip()
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="AIME 2026 avg@4 / pass@4.")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="default: the one model served at --api-base")
    p.add_argument("--conc", type=int, default=30)
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--max-tokens", type=int, default=250_000)
    p.add_argument("--timeout", type=int, default=10800)
    p.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    if not a.model:
        served = [m["id"] for m in json.load(urllib.request.urlopen(a.api_base.rstrip("/") + "/models", timeout=10))["data"]]
        if len(served) != 1:
            sys.exit(f"{a.api_base} serves {served}; pick one with --model")
        a.model = served[0]
    out = Path(a.out or Path(__file__).parent / f"runs/aime_2026-{a.model.replace('/', '_')}").resolve()

    rows = list(load_dataset("MathArena/aime_2026", split="train"))[: a.limit or None]

    def ask(job):
        seed, i, r = job
        body = dict(
            model=a.model,
            messages=[{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
            max_tokens=a.max_tokens,
            seed=seed,
            chat_template_kwargs=THINKING,
            **SAMPLING,
        )
        t = time.time()
        try:
            req = urllib.request.Request(
                a.api_base.rstrip("/") + "/chat/completions",
                json.dumps(body).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
            )
            with urllib.request.urlopen(req, timeout=a.timeout) as f:
                resp = json.load(f)
            c = resp["choices"][0]
            x = dict(
                content=c["message"].get("content") or "",
                finish_reason=c.get("finish_reason"),
                completion_tokens=(resp.get("usage") or {}).get("completion_tokens"),
            )
        except Exception as e:  # kept and scored wrong; a rerun asks again
            x = dict(content="", finish_reason="error", error=f"{type(e).__name__}: {e}")
        return dict(x, seed=seed, problem=f"{i:02}", gold=str(r["answer"]), pred=boxed(x["content"]), seconds=time.time() - t)

    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = {f"s{s}_{i:02}": (s, i, r) for s in a.seeds for i, r in enumerate(rows)}
    todo = [k for k in jobs if not (raw / f"{k}.json").exists() or "error" in json.loads((raw / f"{k}.json").read_text())]
    print(f"aime_2026: {len(jobs)} total, {len(jobs) - len(todo)} cached, {len(todo)} to run, conc={a.conc}", flush=True)
    start = time.time()

    def one(key):
        x = dict(key=key, **ask(jobs[key]))
        tmp = raw / f"{key}.tmp"
        tmp.write_text(json.dumps(x, ensure_ascii=False))
        tmp.replace(raw / f"{key}.json")
        return x

    with ThreadPoolExecutor(a.conc) as pool:
        for n, f in enumerate(as_completed(pool.submit(one, k) for k in todo), 1):
            rate = n / max(time.time() - start, 1) * 60
            print(json.dumps(dict(done=n, total=len(todo), per_min=round(rate, 1),
                                  eta_min=round((len(todo) - n) / max(rate, 0.01)))), flush=True)

    results = [json.loads((raw / f"{k}.json").read_text()) for k in jobs]
    for x in results:
        x["correct"] = (
            not x.get("error")
            and x.get("finish_reason") != "length"
            and x.get("content", "").strip() != ""
            and x.get("pred") is not None
            and x["pred"].strip().lstrip("0") == x["gold"].lstrip("0")
        )
    per_seed = {s: 100 * sum(x["correct"] for x in results if x["seed"] == s) / max(1, sum(x["seed"] == s for x in results)) for s in a.seeds}
    solved_problems = {x["problem"] for x in results if x["correct"]}
    n_problems = len(rows)
    score = dict(
        benchmark="aime_2026",
        model=a.model,
        avg_at_4=round(sum(per_seed.values()) / len(per_seed), 2),
        pass_at_4=round(100 * len(solved_problems) / max(1, n_problems), 2),
        problems_solved=len(solved_problems),
        problems_total=n_problems,
        per_seed=per_seed,
        n=len(results),
        errors=sum(bool(x.get("error")) for x in results),
        truncated=sum(x.get("finish_reason") == "length" for x in results),
    )
    (out / "score_aime_2026.json").write_text(json.dumps(score, indent=2))
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
