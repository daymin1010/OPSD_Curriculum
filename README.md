# OPSD Curriculum — Representation-Guided Curriculum for On-Policy Self-Distillation

Research code backup. Two-axis (difficulty × representation-geometry) curriculum
for OPSD on Qwen3-8B math reasoning.

## Layout
- `OPSD_Curriculum/reasoning_pivot/` — activation extraction, representation-axis construction
- `OPSD_Curriculum/analysis_qwen3_8b/` — pass-rate / activation-shift analysis
- `OPSD_Curriculum/training/` — curriculum builder, stage manifests, SLURM jobs
  - `training/curriculum/` — stage construction (`stagebuild.py`), OPSD trainer wrappers
  - `training/opsd_src/` — vendored OPSD source (fork of `siyan-zhao/OPSD`, with local env changes)
  - `training/stages_*/` — generated stage manifests (`*.json`)
- `OPSD_Curriculum/labeling/` — dataset subject/difficulty labeling
- `OPSD_Curriculum/REPORT_*.md`, `RESEARCH_NOTE_*.md` — progress reports

## Not in this repo (excluded by `.gitignore`)
Large regenerable artifacts: model checkpoints / activation tensors (`*.pt`, `*.npy`,
`*.npz`), parquet/jsonl/csv data dumps, caches, wandb runs, SLURM logs.
Unmodified upstream clones (`OPSD_original/`, `dataset_decision/OPSD_repo/`) are
omitted — recover via `git clone https://github.com/siyan-zhao/OPSD`.
