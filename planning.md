# Planning: The Chain-of-Thought Tax

## Motivation & Novelty Assessment

### Why This Research Matters
Chain-of-thought (CoT) prompting is applied almost by default, yet it costs 10-50x more output tokens than
a direct answer. If CoT also *lowers* accuracy on some task types, practitioners are paying twice. Knowing
which task categories benefit, which are taxed, and how that changes with model size lets people route
prompts sensibly instead of using CoT everywhere.

### Gap in Existing Work
No pre-gathered literature review was present in the workspace (the resource-finder stage was skipped), so
this is from my own knowledge of the literature rather than a systematic review:
- Wei et al. 2022 and Kojima et al. 2022 established CoT gains, mostly on arithmetic/symbolic tasks.
- Sprague et al. 2024 ("To CoT or not to CoT?") meta-analysed 100+ papers and found gains concentrated on
  math/symbolic tasks, with near-zero benefit elsewhere - but mostly reports "no gain", not harm.
- Liu et al. 2024 ("Mind Your Step (by Step)") show CoT *hurts* on tasks where verbal deliberation hurts
  humans (implicit pattern learning, etc.).
- Most comparisons mix model families, so "scale" is confounded with training data and recipe.

Missing: a controlled *within-family* scale ladder, with paired item-level statistics, that separates
"CoT gives no gain" from "CoT actively breaks answers the model otherwise gets right".

### Our Novel Contribution
A paired direct-vs-CoT comparison over 4 task categories x 2 datasets each, on five sizes of one model
family (Qwen2.5-Instruct 0.5B-14B, a 30x parameter range), reporting not only the accuracy delta but the
*harm rate* P(CoT wrong | direct right) and *fix rate* P(CoT right | direct wrong), which are not
distorted by ceiling effects the way raw deltas are.

### Experiment Justification
- Exp 1 (main grid: 5 models x 8 tasks x {direct, CoT}): directly tests the sign and size of the CoT
  effect per category (H1, H2).
- Exp 2 (scale trend): slope of the CoT delta against log2(parameters) per category tests the claimed
  shift of the crossover with scale (H3).
- Exp 3 (prompt-paraphrase robustness: a second CoT wording): greedy decoding has no seed variance, so
  prompt wording is the relevant nuisance variable; checks that conclusions are not one prompt's artefact.
- Exp 4 (error analysis: harm/fix rates, parse failures, truncation, reasoning length): distinguishes a
  real reasoning tax from formatting/extraction artefacts.

## Research Question
Does zero-shot CoT lower accuracy on pattern-matching and factual-recall tasks (by 5-15 points) while
raising it on logical deduction and arithmetic (by 10-25 points), and does the set of categories that
benefit shrink as model scale grows?

## Hypothesis Decomposition
- H1: For pattern matching and factual recall, delta = acc(CoT) - acc(direct) lies in [-15, -5] pp.
- H2: For logical deduction and arithmetic, delta lies in [+10, +25] pp.
- H3: As parameters increase, fewer categories show a positive delta (deltas trend downward with scale).
- (Mechanistic clause "larger models internalise reasoning that traces disrupt" is NOT directly testable
  with behavioural data; at most the results can be consistent or inconsistent with it.)
"5-15%" is read as percentage points of accuracy; relative changes are reported alongside.

Independent variables: prompt condition (direct / CoT), task category, model size.
Dependent variable: item-level correctness (0/1).

## Proposed Methodology

### Approach
The environment has no API key (OPENROUTER_KEY and all provider keys are unset), so API models are not
available. Instead I run real open-weight models locally on the 4x RTX A6000. This is a deviation from the
default instruction, but it gives a cleaner test of the scale claim: one family, one training recipe, five
sizes. Models: Qwen2.5-{0.5B, 1.5B, 3B, 7B, 14B}-Instruct (already in the local HF cache; larger sizes are
not cached and the data disk has ~6 GB free, so 14B is the ceiling).

### Tasks (n = 300 items each, fixed seed 42)
| Category | Task | Source | Format |
|---|---|---|---|
| Pattern matching | AG News topic | fancyzhx/ag_news test | 4 labels |
| Pattern matching | SST-2 sentiment | stanfordnlp/sst2 validation | 2 labels |
| Factual recall | TriviaQA | rc.nocontext validation | free-form, alias match |
| Factual recall | MMLU recall-heavy subjects | cais/mmlu test | 4-way MCQ |
| Logical deduction | BBH logical_deduction_five_objects | lukaemon/bbh (250 items, all used) | 5-way MCQ |
| Logical deduction | ProofWriter OWA depth-5 | tasksource/proofwriter validation | True/False/Unknown |
| Arithmetic | GSM8K | openai/gsm8k test | number |
| Arithmetic | Synthetic multi-step arithmetic | generated here (contamination-free) | number |

### Conditions
- Direct: instruction to give only the answer; the assistant turn is prefilled with "Answer:" and capped
  at 24 new tokens, so the model cannot reason in the output.
- CoT: "think step by step", then a final "Answer: ..." line; 640 new tokens.
- CoT-v2 (robustness): a differently worded reasoning instruction.
Zero-shot, greedy decoding (temperature 0), default system prompt, identical items across conditions.

### Baselines
Direct answering is the baseline for CoT. Chance levels per task are reported (25%, 50%, 20%, 33%, ~0%).

### Evaluation Metrics
Accuracy; paired delta (CoT - direct); harm rate; fix rate; parse-failure and truncation rates; output
tokens (the literal cost side of the "tax").

### Statistical Analysis Plan (fixed before seeing results)
- Per model x task and per model x category: paired bootstrap 95% CI on delta (10,000 resamples of items),
  exact McNemar test, Holm correction across the family of tests at alpha = 0.05.
- H1/H2 are judged per model x category: supported if the point estimate is inside the stated band and the
  CI excludes 0 in the predicted direction.
- H3: OLS slope of category delta on log2(params), CI by item bootstrap; plus the count of categories/tasks
  with a significantly positive delta at each size.
- CoT answers with no parsable final answer count as wrong (reported separately, with a sensitivity
  analysis that excludes them).

## Expected Outcomes
Support: negative deltas on classification/recall, positive on logic/arithmetic, positive deltas shrinking
with size. Refutation: deltas near zero or positive on recall/pattern tasks; no monotone scale trend.

## Timeline
Setup and data 30 min; harness + pilot 40 min; full runs ~60 min on 4 GPUs; analysis 40 min; report 30 min.

## Potential Challenges
- Small models ignoring the output format -> track parse failures, sensitivity analysis.
- Ceiling effects on SST-2 -> harm/fix rates.
- Truncated CoT -> 640-token budget, report truncation rate.
- Disk nearly full -> venv symlinked to the uv cache, only small parquet files downloaded.

## Success Criteria
All 5 models x 8 tasks x 2 conditions completed with item-level outputs saved; each sub-hypothesis gets an
explicit supported / not supported verdict with CIs.
