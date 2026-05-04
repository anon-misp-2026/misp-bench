# MISP-Bench

**Decomposing user-provided false priors into answer, rationale, and guard effects.**

This repository contains the corpus, prompts, evaluation scripts, and item-level
audit list for MISP-Bench. The benchmark is designed to decompose which
structural component of a wrong user prior — the answer, the rationale, or
their combination — drives downstream model error, and to measure whether
common safety prompts ("verify the reasoning first") actually mitigate the
damage. See the paper for full motivation and findings.

License: **CC-BY-4.0** for the corpus and audit list. Code is released under
**MIT**.

---

## Repository contents

| File | Description |
|---|---|
| `Benchmark.json` | 2,494 items × 14 prompt levels (the full pre-audit pool). The audited 1,724-item subset is materialized at runtime by `03_quality_audit.ipynb`. |
| `01_question_generation.ipynb` | Builds `Benchmark.json` from MedMCQA validation + GSM8K test. |
| `02_run_experiment.ipynb` | Runs all 11 models against `Benchmark.json` via vLLM and writes `results_<model>_<timestamp>.csv` per model. |
| `03_quality_audit.ipynb` | Reproduces the six-category audit (paper Table 1) and writes `tables/t0_question_flags.csv`. |
| `04_analysis.ipynb` | Loads result CSVs + flags table; produces all paper figures and tables. |
| `EXCLUSIONS.md` | Per-category audit documentation with detection criteria and case writeups. |
| `croissant.json` | Croissant 1.0 metadata (core + Responsible AI fields) per NeurIPS E&D Track requirements. |

---

## Reproduction pipeline

```
                      MedMCQA (validation)
                      GSM8K   (test)
                            │
                            ▼
            01_question_generation.ipynb
                            │
                            ▼
                   Benchmark.json (2,494 items)
                     │                    │
                     ▼                    ▼
        03_quality_audit.ipynb     02_run_experiment.ipynb
                     │                    │
                     ▼                    ▼
       tables/t0_question_flags.csv   results_<model>_*.csv
                     │                    │
                     └────────┬───────────┘
                              ▼
                    04_analysis.ipynb
                              │
                              ▼
                paper figures, tables
```

### Quick start

```bash
# 1. Environment
conda create -n misp python=3.11 -y && conda activate misp
pip install vllm transformers huggingface_hub openai datasets pandas tqdm \
            numpy scipy matplotlib seaborn statsmodels

# 2. Authenticate (no plaintext tokens; either:)
export OPENAI_API_KEY=sk-...     # for 01_question_generation
export HF_TOKEN=hf_...           # for gated MedGemma/Gemma3
# ── OR ──
huggingface-cli login            # interactive, sets up cache
# (use whichever your team standard prefers)

# 3. Audit (produces tables/t0_question_flags.csv)
jupyter nbconvert --execute --to notebook --inplace 03_quality_audit.ipynb

# 4. Inference (runs 11 models)
jupyter nbconvert --execute --to notebook --inplace 02_run_experiment.ipynb

# 5. Analysis
jupyter nbconvert --execute --to notebook --inplace 04_analysis.ipynb
```

`02_run_experiment.ipynb` loads each model sequentially with `gc.collect()` +
`torch.cuda.empty_cache()` between models. The notebook can run on either a
single GPU with enough VRAM or a multi-GPU node; tune `gpu_memory_utilization`
and `max_model_len` per model in the `MODELS` dict to fit your setup.

### Compute budget

Inference for the released response set was performed on 2× NVIDIA H100 GPUs.
Per-model wall time scales with model size; see the run logs in
`tripod_<timestamp>.json` (written by `02_run_experiment.ipynb`) for the exact
timings on your own hardware.

---

## Output schema (`results_<model>_*.csv`)

`04_analysis.ipynb` consumes these columns:

| Column | Type | Description |
|---|---|---|
| `model`, `model_id`, `category` | str | Identification; `category ∈ {open_general, domain_specific}`. |
| `question_id`, `global_idx`, `domain`, `subject`, `difficulty` | mixed | Item identification. |
| `level`, `base_level` | str | Prompt level (`L1`–`L7b`); `base_level` collapses `L4*`/`L6*`/`L7*` into the family. |
| `cot` | str | `cot` or `direct`. |
| `repeat` | int | 0, 1, 2 (three samples per cell). |
| `correct_answer`, `distractor_answer` | str/num | Gold and (if applicable) seeded distractor. |
| `extracted` | str/num | The parsed final answer (or `UNK` / `ERROR`). |
| `is_correct`, `is_sycophantic`, `has_correction` | bool | Outcome flags. |
| `response`, `reasoning` | str | Cleaned response and stripped thinking trace. |
| `in_tok`, `out_tok` | int | vLLM-reported token counts. |
| `finish_reason` | str | `stop` / `length` (used for the truncation audit). |
| `timestamp`, `model_returned`, `error` | str | Run metadata. |

The schema is asserted by the last cell of `02_run_experiment.ipynb`.

---

## Citing

```bibtex
@inproceedings{mispbench2026,
  title  = {MISP-Bench: Decomposing User-Provided False Priors into Answer, Rationale, and Guard Effects},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Evaluations \& Datasets Track},
  year   = {2026}
}
```

---

## Notes for reviewers

- **Anonymity.** The repository is anonymized (no author / institution
  metadata) per the E&D Track double-blind default.
- **Audit transparency.** Every item in the 770-exclusion list is
  reproducible from `Benchmark.json` alone via `03_quality_audit.ipynb`.
  Hand-curated lists (`EXACT_DUP`, `IMAGE_REFERENCING`, `LABEL_ERROR`)
  are reproduced verbatim with provenance comments; the audit notebook
  asserts that the `EXACT_DUP` autodetector matches the hand list.
- **Phi-4-14B-reasoning.** The model is configured in
  `02_run_experiment.ipynb` but is excluded from the paper's main analysis
  due to 86–98% truncation (paper §3.3, §6.4). Re-inference with
  extended budget is in progress.
