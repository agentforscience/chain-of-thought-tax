"""Build the fixed evaluation set: 8 tasks in 4 categories, written to datasets/processed/tasks.jsonl.

Each record: id, task, category, text (task-specific question incl. answer-format hint),
answer_type (label | mcq | number | freeform), gold (str or list of aliases), labels (for label tasks).
"""
import glob
import json
import os
import random

import numpy as np
import pandas as pd

SEED = 42
N = 300
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
RAW = "datasets/raw"
OUT = "datasets/processed/tasks.jsonl"
LETTERS = "ABCDE"

# MMLU subjects that are dominated by fact retrieval rather than multi-step reasoning.
MMLU_RECALL_SUBJECTS = [
    "miscellaneous", "world_religions", "high_school_geography", "prehistory",
    "anatomy", "nutrition", "astronomy",
]


def cached(pattern):
    """Resolve a parquet file inside the shared HF hub cache."""
    hits = sorted(glob.glob(os.path.join(HF_HUB, pattern)))
    assert hits, f"not found in HF cache: {pattern}"
    return hits[0]


def sample(df, n, seed=SEED):
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def ag_news():
    df = sample(pd.read_parquet(cached("datasets--fancyzhx--ag_news/snapshots/*/data/test-*.parquet")), N)
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    for i, r in df.iterrows():
        yield dict(id=f"agnews-{i}", task="ag_news", category="pattern_matching", answer_type="label",
                   labels=labels, gold=labels[int(r["label"])],
                   text=f"Classify the topic of this news article.\n\nArticle: {r['text']}\n\n"
                        f"The answer must be exactly one of: World, Sports, Business, Sci/Tech.")


def sst2():
    df = sample(pd.read_parquet(f"{RAW}/stanfordnlp__sst2/data/validation-00000-of-00001.parquet"), N)
    labels = ["negative", "positive"]
    for i, r in df.iterrows():
        yield dict(id=f"sst2-{i}", task="sst2", category="pattern_matching", answer_type="label",
                   labels=labels, gold=labels[int(r["label"])],
                   text=f"Classify the sentiment of this movie review sentence.\n\nSentence: {r['sentence'].strip()}\n\n"
                        f"The answer must be exactly one of: positive, negative.")


def triviaqa():
    df = pd.read_parquet(f"{RAW}/mandarjoshi__trivia_qa/rc.nocontext/validation-00000-of-00001.parquet")
    df = sample(df.drop_duplicates("question"), N)
    for i, r in df.iterrows():
        aliases = sorted(set(list(r["answer"]["normalized_aliases"]) + [r["answer"]["normalized_value"]]))
        yield dict(id=f"triviaqa-{i}", task="triviaqa", category="factual_recall", answer_type="freeform",
                   gold=aliases,
                   text=f"Answer this trivia question.\n\nQuestion: {r['question']}\n\n"
                        f"The answer should be a short name or phrase.")


def mmlu():
    df = pd.read_parquet(cached("datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))
    df = sample(df[df["subject"].isin(MMLU_RECALL_SUBJECTS)], N)
    for i, r in df.iterrows():
        opts = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(r["choices"]))
        yield dict(id=f"mmlu-{i}", task="mmlu_recall", category="factual_recall", answer_type="mcq",
                   gold=LETTERS[int(r["answer"])], n_options=4, subject=r["subject"],
                   text=f"Answer this multiple-choice question.\n\nQuestion: {r['question']}\n{opts}\n\n"
                        f"The answer must be a single letter: A, B, C, or D.")


def bbh_logic():
    df = pd.read_parquet(f"{RAW}/lukaemon__bbh/logical_deduction_five_objects/test-00000-of-00001.parquet")
    df = sample(df, N)  # only 250 items exist; all are used
    for i, r in df.iterrows():
        yield dict(id=f"bbhlogic-{i}", task="bbh_logical_deduction", category="logical_deduction",
                   answer_type="mcq", gold=r["target"].strip("()"), n_options=5,
                   text=f"{r['input']}\n\nThe answer must be a single letter: A, B, C, D, or E.")


def proofwriter():
    df = pd.read_parquet(glob.glob(f"{RAW}/tasksource__proofwriter/data/validation-*.parquet")[0])
    # Open-world, depth-5 theories; keep questions that need >=1 inference step (or are Unknown).
    df = df[(df["config"] == "depth-5") & df["id"].str.contains("OWA")]
    df = df[(df["answer"] == "Unknown") | (df["QDep"].astype(str).isin(["1", "2", "3", "4", "5"]))]
    parts = [sample(df[df["answer"] == a], N // 3, seed=SEED + k) for k, a in enumerate(["True", "False", "Unknown"])]
    df = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    labels = ["True", "False", "Unknown"]
    for i, r in df.iterrows():
        yield dict(id=f"proofwriter-{i}", task="proofwriter", category="logical_deduction", answer_type="label",
                   labels=labels, gold=r["answer"], qdep=str(r["QDep"]),
                   text=f"Here are some facts and rules:\n{r['theory']}\n\n"
                        f"Based only on the facts and rules above, is the following statement True, False, or "
                        f"Unknown (cannot be determined)?\nStatement: {r['question']}\n\n"
                        f"The answer must be exactly one of: True, False, Unknown.")


def gsm8k():
    df = sample(pd.read_parquet(cached("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")), N)
    for i, r in df.iterrows():
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        yield dict(id=f"gsm8k-{i}", task="gsm8k", category="arithmetic", answer_type="number", gold=gold,
                   text=f"Solve this math problem.\n\nProblem: {r['question']}\n\nThe answer must be a single number.")


def synth_arith():
    """Freshly generated 3-operation integer expressions, e.g. (47 * 23) + 618 - 95."""
    rng = random.Random(SEED)
    for i in range(N):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        c, d = rng.randint(100, 999), rng.randint(100, 999)
        form = rng.choice(["mul_add_sub", "add_mul", "mul_sub_mul"])
        if form == "mul_add_sub":
            expr, val = f"({a} * {b}) + {c} - {d}", a * b + c - d
        elif form == "add_mul":
            e = rng.randint(3, 9)
            expr, val = f"({c} + {d}) * {e} - {a}", (c + d) * e - a
        else:
            e, f = rng.randint(3, 9), rng.randint(3, 9)
            expr, val = f"({a} * {b}) - ({e} * {f}) + {c}", a * b - e * f + c
        yield dict(id=f"syntharith-{i}", task="synthetic_arithmetic", category="arithmetic",
                   answer_type="number", gold=str(val),
                   text=f"Compute the value of this expression.\n\n{expr}\n\nThe answer must be a single integer.")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for gen in [ag_news, sst2, triviaqa, mmlu, bbh_logic, proofwriter, gsm8k, synth_arith]:
        part = list(gen())
        rows += part
        golds = pd.Series([str(r["gold"]) if not isinstance(r["gold"], list) else "<aliases>" for r in part])
        print(f"{part[0]['task']:28s} n={len(part):4d}  gold dist: {golds.value_counts().head(6).to_dict()}")
    assert len({r["id"] for r in rows}) == len(rows)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items -> {OUT}")


if __name__ == "__main__":
    main()
