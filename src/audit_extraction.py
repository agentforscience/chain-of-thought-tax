"""Audit answer extraction: sample CoT responses scored wrong and print gold / extracted answer / tail
of the raw output, so scoring artefacts (correct reasoning, mis-parsed answer) can be counted by hand.

Usage: python src/audit_extraction.py [n_per_task] [condition]
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scoring import extract_answer, score  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cond = sys.argv[2] if len(sys.argv) > 2 else "cot"
items = {r["id"]: r for r in map(json.loads, open("datasets/processed/tasks.jsonl"))}
rows = [r for p in sorted(glob.glob("results/model_outputs/*.jsonl")) for r in map(json.loads, open(p))]
df = pd.DataFrame([r for r in rows if r["condition"] == cond])
df[["answer", "has_marker"]] = [extract_answer(o, cond) for o in df.output]
df["correct"] = [score(items[i], a) for i, a in zip(df.id, df.answer)]
wrong = df[(df.correct == 0) & (~df.truncated)]
for task, g in wrong.groupby("task"):
    print(f"\n################ {task}: {len(g)} wrong (non-truncated) of {len(df[df.task == task])}")
    for _, r in g.sample(min(n, len(g)), random_state=0).iterrows():
        gold = items[r.id]["gold"]
        gold = gold[:4] if isinstance(gold, list) else gold
        print(f"--- {r.model.split('/')[-1]} {r.id} | gold={gold} | extracted={r.answer!r} | marker={r.has_marker}")
        print("    tail:", r.output[-160:].replace("\n", " ⏎ "))
