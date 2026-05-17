This is a project in which we attempt to beat the #1 spot in the ReXrank Challenge V2.0 (https://rexrank.ai/)

ReXrank Challenge V2.0 is a competition in VQA task utilizing VQA dataset constructed from ReXGradient, including 41,007 VQA pairs with 10,000 radiological studies. We benchmarked 8 state-of-the-art models.

ReXGradient-160K is the largest publicly available multi-site chest X-ray dataset, containing 273,004 unique chest X-ray images from 160,000 radiological studies, collected from 109,487 unique patients across 3 U.S. health systems (79 medical sites). In ReXrank, we use additional private test set ReXGradient, 10,000 studies for benchmarking.

ReXVQA is the largest and most comprehensive benchmark for VQA in chest radiology, comprising 653834 questions paired with 160,000 radiological studies. The dataset is constructed from ReXGradient-160K.
## Files

`data/` contains the downloaded and processed data.
`results/` default save path for run results.
`rexvqa_models/` python library with the code for running the various models.
`inference.py` code for running inference on the test set and generating the submission file.


## Description

We will be developing new models and testing them on the ReXrank Challenge V2.0 dataset. We will be using the `rexvqa_models` library to implement our models and the `inference.py` script to run inference and generate submission files.

Example of how to run inference on the mini validation set:

```bash
uv run inference.py inference/data=valid_mini inference/model=medgemma inference/runtime=vllm run.batch_size=60 run.max_new_tokens=4098 runtime.max_model_len=8096
```

The current evaluation is extracted from the example given by the challenge authors but its very basic and will likely need to be improved as we develop new models and test them on the dataset.

`results/` is local scratch space and should not be committed. Copy any release-worthy score summaries into tracked docs outside `results/`.

## Other
Dependency management is handled using `uv` + `pyproject.toml` + `uv.lock`.

A virtual environment (.venv) is present at the root. Any `uv run` command will automatically use the virtual environment.

ruff is used for linting and formatting. You can run `uv run ruff --fix` to automatically fix any linting issues.
ty is used for type checking. You can run `uv run ty check` to check for type errors.
