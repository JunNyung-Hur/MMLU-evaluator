import os
import json
from typing import List

from langchain_openai import ChatOpenAI

from settings import Settings


class BaseEvaluator:
    def __init__(self, dataset_name: str, settings: Settings):
        self.model = ChatOpenAI(model=settings.model_name, base_url=settings.openai_api_url, api_key=settings.openai_api_key)
        self.no_think: bool = settings.no_think
        self.save_dir = os.path.join(settings.save_dir, dataset_name.split("/")[-1])
        self.save_path = os.path.join(
            self.save_dir,
            f"{settings.model_name.split("/")[-1]}{'-no_think' if settings.no_think else ''}.jsonl"
        )
        self.save_step = settings.save_step

    def _write_result(self, result: dict):
        jsonl_result = json.dumps(result, ensure_ascii=False)+"\n"
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.isfile(self.save_path):
            with open(self.save_path, "w", encoding="utf-8") as f:
                f.write(jsonl_result)
        else:
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(jsonl_result)

    def _get_len_results(self) -> int:
        if not os.path.isfile(self.save_path):
            return 0
        with open(self.save_path, "r", encoding="utf-8") as f:
            num_lines = len(f.readlines())
        return num_lines
    
    def _read_results(self) -> List[dict]:
        if not os.path.isfile(self.save_path):
            return []
        results = []
        with open(self.save_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    result = json.loads(line)
                except json.decoder.JSONDecodeError:
                    continue
                results.append(result)
        return results
