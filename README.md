# MMLU Pro Evaluator
This code assists in evaluating your LLM using the MMLU Pro dataset.

## 1. Configuration
Edit the `.env` file to configure your settings.

- `MODEL_NAME`: The name of the model served by OpenAI-compatible API endpoint
- `OPENAI_API_URL`: The URL of OpenAI compatible API endpoint
- `OPENAI_API_KEY`: The API key for OpenAI-compatible API
- `DATASET`: MMLU Pro dataset name in HuggingFace. Default is `"TIGER-Lab/MMLU-Pro"`
- `THINKING`: If you want to turn off the reasoning feature of your model, change the `THINKING` variable from true to false (This setting only applies if your model supports toggling reasoning.)


## 2. Run
```sh
uv sync
uv run main.py
```