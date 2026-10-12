# The Chain-of-Thought Tax

A paired comparison of direct answering versus zero-shot chain-of-thought (CoT) on 8 tasks in 4 categories
(pattern matching, factual recall, logical deduction, arithmetic), across five sizes of Qwen2.5-Instruct
(0.5B to 14B), run locally on GPU. Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **No accuracy tax on pattern matching or factual recall.** CoT changed accuracy by -2.2 to +3.8 points
  across all model sizes; nothing was significantly negative, and nothing was in the hypothesised -5 to
  -15 range.
- **Arithmetic gains were far larger than hypothesised** (+48 to +85 points versus the predicted +10 to
  +25), because answering with no visible working is near 0% on these tasks.
- **Logical deduction gains grew with model size**, from -7 points at 0.5B to +29 points at 14B.
- **The scale trend was the opposite of the hypothesis**: 2 of 8 tasks benefited significantly at 0.5B, 4
  of 8 at 14B, and the share of correct answers that CoT broke fell with size in every category.
- **The cost of CoT here is tokens**: about 75x more output tokens, for no measurable gain on
  classification.

Caveats: one model family, 14B maximum, zero-shot only; item-level outputs are not bit-reproducible across
batch compositions; the prompt-robustness check is incomplete for the 14B model. No API key was available,
so frontier API models were not tested.

## Reproduce

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync                                   # installs from pyproject.toml
python src/prepare_data.py                # builds datasets/processed/tasks.jsonl (see datasets/README.md)
bash src/supervise.sh Qwen/Qwen2.5-7B-Instruct 24000 64    # one model: direct + cot, then cot2 on a subset
python src/analyze.py > results/analysis_output.txt        # tables, statistics, figures
python src/audit_extraction.py 7          # optional: sample wrong-scored CoT answers for manual review
```

`supervise.sh <model> <free MiB needed> <batch size>` waits for a GPU with enough free memory and retries
on failure; `run_inference.py` is resumable. Analysis alone (from the saved outputs) takes about a minute.

## Files

| Path | Contents |
|---|---|
| `planning.md` | Motivation, hypotheses, pre-specified analysis plan |
| `REPORT.md` | Full report with results and limitations |
| `src/prepare_data.py` | Builds the 2,350-item evaluation set |
| `src/scoring.py` | Prompt templates, answer extraction, scoring |
| `src/run_inference.py` | Batched greedy generation, resumable |
| `src/supervise.sh`, `src/run_all.sh` | Launch scripts for shared GPUs |
| `src/analyze.py` | Statistics, tables, figures |
| `src/audit_extraction.py` | Manual audit helper for answer extraction |
| `results/model_outputs/` | Raw model outputs, one JSONL per model |
| `results/*.csv`, `results/analysis_output.txt` | Result tables |
| `results/config.json` | Models, snapshot hashes, prompts, library versions |
| `figures/` | Plots used in the report |
| `datasets/` | Processed evaluation set and download notes |
