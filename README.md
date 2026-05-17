# Gemma 4 Good ReXVQA

This repository contains the code for a Gemma 4 Good project on chest X-ray visual question answering with ReXVQA. It was developed for the [ReXrank Challenge V2.0](https://rexrank.ai/), which evaluates VQA systems on radiology studies from ReXGradient and ReXVQA.

The project includes data preparation utilities, inference runners, evaluation code, a Gradio demo, and an experimental self-distillation fine-tuning pipeline for Gemma-style vision-language models.

This project is for research and competition use. It is not a clinical diagnostic tool.

## Repository Layout

- `inference.py`: Hydra entry point for validation or test inference.
- `train_sdft.py`: self-distillation fine-tuning entry point.
- `build_quota_dataset.py`: creates balanced training subsets from ReXVQA metadata.
- `demo.py`: Gradio demo for the trained adapter.
- `conf/`: Hydra configuration groups for data, models, runtime backends, training, and dataset building.
- `rexvqa_models/`: local Python package with prompt builders, model backends, evaluators, image utilities, and training helpers.
- `notebooks/data.ipynb`: notebook version of the data download and preparation flow.
- `notebooks/error_analysis.ipynb`: exploratory notebook for inspecting validation failures.
- `demo_assets/`: small public demo images used by the Gradio app.
- `results/`: local output directory for runs, checkpoints, merged models, and predictions.

Large data files, checkpoints, merged model weights, and generated predictions are produced locally.

## Requirements

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency management. The lockfile is committed, so new users should be able to reproduce the Python environment with `uv sync`.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the base environment:

```bash
uv sync
```

Useful dependency groups:

- `notebook`: Jupyter, pandas, Hugging Face download helpers, and plotting.
- `vllm`: vLLM runtime for faster batched inference.
- `fast_inference`: optional fast attention packages. These can require a working CUDA build toolchain.
- `training`: PyTorch, Transformers, PEFT, Unsloth, TRL, bitsandbytes, xFormers, TensorBoard, and Trackio for SDFT training.
- `demo`: Gradio plus the PyTorch, Transformers, PEFT, safetensors, and image stack needed to load the demo model and adapter.
- `dev`: Ruff and ty.

Examples:

```bash
uv sync --group notebook
uv sync --group vllm
uv sync --group training --group vllm
uv sync --all-groups
```

If you are unsure which workflow you need, `uv sync --all-groups` installs the full environment.

The lockfile is configured for CUDA 12.8 PyTorch wheels. On a different CUDA stack or CPU-only machine, expect to adjust `pyproject.toml` or use a compatible environment.

## Data Access

The code expects the following local layout:

```text
data/
  ReXVQA/
    metadata/
      train_vqa_data.json
      valid_vqa_data.json
      valid_mini_vqa_data.json
      test_vqa_data.json
    deid_png/
      ...
```

The metadata comes from `rajpurkarlab/ReXVQA` on Hugging Face. The de-identified PNG images come from `rajpurkarlab/ReXGradient-160K`. Both datasets may require accepting the dataset terms and authenticating with Hugging Face.

Log in once before running the data preparation notebook:

```bash
uv run --group notebook huggingface-cli login
```

Then run `notebooks/data.ipynb`. The notebook downloads the ReXVQA metadata, downloads and extracts the ReXGradient PNG archives, moves the PNGs into `data/ReXVQA/deid_png`, and creates `valid_mini_vqa_data.json`.

```bash
uv run --group notebook jupyter notebook notebooks/
```

## Configuration

The scripts use Hydra. You select config groups with arguments like `inference/data=valid_mini` and override individual values with dotted keys like `run.batch_size=16`.

Common inference groups:

- Data: `valid_mini`, `valid`, `test`
- Models: `medgemma`, `chexone`, `gemma4`, `gemma4_unsloth`, `gemma4_thinking`, `gemma4_trained`
- Runtime: `transformers`, `vllm`

Common training groups:

- Data: `smoke`, `quota_fastlift_v1`, `quota_fastlift_v2`
- Dataset format: `sdft_explanation`, `sdft_explanation_rich`
- Training: `smoke`, `sdft_v1`, `sdft_v2`
- Eval: `disabled`, `valid_mini`
- Logging: `tensorboard`, `trackio_tensorboard`

Hydra writes resolved configs and summaries under `results/`.

## Inference

Run inference on the test split:

```bash
uv run --group vllm inference.py \
  inference/data=test \
  inference/model=gemma4_trained \
  inference/runtime=vllm \
  run.batch_size=64 \
  run.max_new_tokens=2048 \
  runtime.max_model_len=4096
```

Outputs are written to:

```text
results/inference_results/<split>/<model-or-adapter>-<timestamp>.json
results/inference_runs/<timestamp>_<model>_<backend>/summary.json
```

If an output file already exists, `inference.py` resumes incomplete predictions by skipping IDs already present in the JSON.

## Evaluation

For multiple-choice VQA, the evaluator extracts `A`, `B`, `C`, or `D` from model responses and reports overall and per-category accuracy. For free-text mode, it reports exact match, token F1, and ROUGE-L style scores against the correct option text.

Validation metrics are printed at the end of `inference.py` and saved in each run summary under `results/`.

## Building a Training Subset

The training configs expect a quota-balanced subset file. Build the default `fastlift_v2` subset:

```bash
uv run build_quota_dataset.py quota_dataset=fastlift_v2
```

For a quick end-to-end check:

```bash
uv run build_quota_dataset.py quota_dataset=smoke
```

The default subset is written to:

```text
data/ReXVQA/metadata/train_quota_fastlift_v2.json
data/ReXVQA/metadata/train_quota_fastlift_v2_summary.json
```

## Training

Install the training group:

```bash
uv sync --group training
```

Run a short smoke test:

```bash
uv run --group training train_sdft.py \
  train_sdft/data=smoke \
  train_sdft/training=smoke \
  train_sdft/eval=disabled \
  train_sdft/logging=tensorboard
```

Run the default SDFT experiment:

```bash
uv run --group training train_sdft.py \
  train_sdft/experiment=sdft_explanation_gemma4_fastlift_v2
```

Enable mini-validation during training:

```bash
uv run --group training train_sdft.py \
  train_sdft/experiment=sdft_explanation_gemma4_fastlift_v2 \
  train_sdft/eval=valid_mini
```

Training outputs are written to:

```text
results/sdft_runs/<timestamp>_<experiment>/
  adapter/
  checkpoints/
  resolved_config.json
  summary.json
```

Resume from the latest checkpoint in the current run config:

```bash
uv run --group training train_sdft.py \
  train_sdft/experiment=sdft_explanation_gemma4_fastlift_v2 \
  training.resume_from_checkpoint=latest
```

Or point at a previous run or checkpoint directory:

```bash
uv run --group training train_sdft.py \
  train_sdft/experiment=sdft_explanation_gemma4_fastlift_v2 \
  training.resume_from_checkpoint=results/sdft_runs/<run>/checkpoints/checkpoint-500
```

## Running a Trained Adapter

`conf/inference/model/gemma4_trained.yaml` points at a local adapter path under `results/sdft_runs/.../adapter`. Update that path to your trained adapter or override it on the command line:

```bash
uv run --group vllm inference.py \
  inference/data=valid \
  inference/model=gemma4_trained \
  inference/runtime=vllm \
  model.adapter_path=results/sdft_runs/<run>/adapter
```

For vLLM, LoRA adapters are merged into a temporary local model by default and stored under `results/merged_models/`. Use `model.merge_adapter_for_vllm=false` only if your runtime can load the adapter directly.

## Demo

A hosted Gradio demo is available on Hugging Face Spaces: [carlesonielfa/rexvqa-SDFT-gemma-4-E2B](https://huggingface.co/spaces/carlesonielfa/rexvqa-SDFT-gemma-4-E2B).

Install demo dependencies:

```bash
uv sync --group demo
```

Run the Gradio app:

```bash
uv run --group demo demo.py
```

The demo currently loads:

- Base model: `unsloth/gemma-4-E2B-it`
- Adapter: `carlesonielfa/ReXVQA-SDFT-gemma-4-E2B`

Update `BASE_MODEL_ID` and `ADAPTER_ID` in `demo.py` if you publish a different adapter.

## Development

Run formatting and lint fixes:

```bash
uv run --group dev ruff check --fix
uv run --group dev ruff format
```

Run type checking:

```bash
uv run --group dev ty check
```
