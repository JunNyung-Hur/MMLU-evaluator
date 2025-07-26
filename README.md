# LLM Evaluator
This code assists in evaluating your LLM with evaluation datasets.

Currently supported evaluation datasets:
- MMLU
- MMLU Pro


## 1. Configuration
Edit the `.env` file to configure your settings.

- `MODEL_NAME`: The name of the model served by OpenAI-compatible API endpoint
- `OPENAI_API_URL`: The URL of OpenAI compatible API endpoint
- `OPENAI_API_KEY`: The API key for OpenAI-compatible API


## 2. Usage

To run the evaluator, use the following commands:

```
uv sync
uv run main.py <dataset> [flags]
```

- dataset
  - `mmlu`: Evaluate using the MMLU dataset.
  - `mmlu_pro`: Evaluate using the MMLU-PRO dataset.

- flags
  - `--no-think`: Disables the reasoning feature of the model.
  
    This flag only takes effect if the model has been trained to support disabling reasoning.
  
  - `--only-print`: Prints only the accuracy, skipping further evaluation steps.