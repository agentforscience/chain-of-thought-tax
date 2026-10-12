# Resources

The resource-finder stage was skipped for this run (`.neurico/pipeline_state.json`: `"skipped": true`), so no
literature review, papers or code were pre-gathered. Everything below was assembled during the experiment run.

## Models (local Hugging Face cache, run on GPU)
Qwen/Qwen2.5-{0.5B, 1.5B, 3B, 7B, 14B}-Instruct. Snapshot hashes are in `results/config.json`.

## Datasets
See `datasets/README.md` for sources, files and revisions.

## Prior work referred to (from memory, not re-read for this run)
- Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models".
- Kojima et al. 2022, "Large Language Models are Zero-Shot Reasoners".
- Sprague et al. 2024, "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning".
- Liu et al. 2024, "Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse".
- Suzgun et al. 2022 (BIG-Bench Hard); Tafjord et al. 2021 (ProofWriter); Cobbe et al. 2021 (GSM8K);
  Joshi et al. 2017 (TriviaQA); Hendrycks et al. 2021 (MMLU); Zhang et al. 2015 (AG News); Socher et al. 2013 (SST-2).

## Research process notes
- No API key was present in the environment, so the study uses local open-weight models instead of API models.
- GPUs were shared with other jobs; runs were made resumable and split batches on out-of-memory errors.
