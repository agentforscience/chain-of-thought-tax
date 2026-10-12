"""Run one local chat model over the evaluation set under the given prompt conditions.

Greedy decoding, batched, resumable (appends to results/model_outputs/<model>.jsonl and skips
(id, condition) pairs that are already present).

Usage: CUDA_VISIBLE_DEVICES=0 python src/run_inference.py --model Qwen/Qwen2.5-7B-Instruct \
           --conditions direct cot cot2 --batch-size 48 [--limit 5]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from scoring import CONDITIONS, build_user_prompt, extract_answer, score  # noqa: E402

SEED = 42
SYSTEM_PROMPT = "You are a helpful assistant."


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_batch(model, tok, batch_prompts, max_new):
    """Greedy-generate for a list of prompts; on CUDA OOM (the GPUs are shared) split the batch and retry.

    Returns a list of 1-D tensors of new token ids (padding included)."""
    try:
        enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        return list(gen[:, enc["input_ids"].shape[1]:].cpu())
    except torch.OutOfMemoryError:
        if len(batch_prompts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(batch_prompts) // 2
        print(f"  OOM at batch {len(batch_prompts)} -> splitting", flush=True)
        return (generate_batch(model, tok, batch_prompts[:half], max_new)
                + generate_batch(model, tok, batch_prompts[half:], max_new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="+", default=["direct", "cot"])
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="items per task (pilot runs)")
    ap.add_argument("--tasks", default="datasets/processed/tasks.jsonl")
    ap.add_argument("--out-dir", default="results/model_outputs")
    ap.add_argument("--max-memory", nargs="+", default=None,
                    help="per-visible-GPU caps, e.g. 20GiB 12GiB: split the model over several shared GPUs")
    args = ap.parse_args()
    set_seed()

    items = [json.loads(l) for l in open(args.tasks)]
    if args.limit:
        seen, kept = {}, []
        for it in items:
            seen[it["task"]] = seen.get(it["task"], 0) + 1
            if seen[it["task"]] <= args.limit:
                kept.append(it)
        items = kept

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.model.split("/")[-1] + ".jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["condition"]) for r in map(json.loads, open(out_path))}
    print(f"{args.model}: {len(items)} items, {len(done)} (id, condition) pairs already done", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"  # required for batched generation with a decoder-only model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.max_memory:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto",
                                                     max_memory=dict(enumerate(args.max_memory)))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    for cond in args.conditions:
        _, prefill, max_new = CONDITIONS[cond]
        todo = [it for it in items if (it["id"], cond) not in done]
        prompts = []
        for it in todo:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(it, cond)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill)
        # Sort by prompt length so each batch needs little padding.
        order = sorted(range(len(todo)), key=lambda i: len(prompts[i]))
        t0 = time.time()
        with open(out_path, "a") as f:
            for b in range(0, len(order), args.batch_size):
                idx = order[b:b + args.batch_size]
                new = generate_batch(model, tok, [prompts[i] for i in idx], max_new)
                for row, i in zip(new, idx):
                    n_tok = int((row != tok.pad_token_id).sum())
                    text = tok.decode(row, skip_special_tokens=True)
                    ans, has_marker = extract_answer(text, cond)
                    it = todo[i]
                    f.write(json.dumps(dict(
                        id=it["id"], task=it["task"], category=it["category"], model=args.model,
                        condition=cond, output=text, answer=ans, has_marker=has_marker,
                        # hit the token budget without finishing (no EOS inside the budget)
                        truncated=bool(n_tok >= max_new and cond != "direct"),
                        n_out_tokens=n_tok, correct=score(it, ans))) + "\n")
                f.flush()
                print(f"  [{cond}] {min(b + args.batch_size, len(order))}/{len(order)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("DONE", args.model, flush=True)


if __name__ == "__main__":
    main()
