# Cross-Select

Implementation of the Cross-Select pre-trained model recommender from the thesis in [thesis.pdf](thesis.pdf). Cross-Select uses cross-attention over a parameter-based model encoding and a CLIP-based dataset encoding to predict fine-tune compatibility without running fine-tuning.

This repo consumes pre-computed model tokens (`(512,)` per model) and dataset tokens (`(16, 102, 512)` CLIP feature shards). Feature extraction lives outside this repo.

## Layout

- `src/cross_select/` — package source (data, models, losses, baselines, training, eval, cli)
- `configs/` — Hydra configs (model, data, trainer, experiment)
- `tests/` — pytest suite
- `thesis/` — LaTeX source for reference
- `third_party/model-spider/` — Model Spider reference repo (gitignored)

## Quickstart (after implementation)

```bash
pip install -e ".[dev]"
python -m cross_select.cli.train experiment=quadrant1
```
