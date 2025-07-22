import json
import os
from typing import Optional

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from settings import Settings


class Evaulator:

    def __init__(self):
        settings = Settings()
        self.model = ChatOpenAI(model=settings.model_name, base_url=settings.openai_api_url, api_key=settings.openai_api_key)
        self.dataset = load_dataset(settings.dataset, split="test")
        self.thinking = settings.thinking
        self.answer_sheet = None
        self.save_dir = settings.save_dir
        self.answer_sheet_path = f"{self.save_dir}/{settings.model_name.split("/")[-1]}{"" if self.thinking else "-no_think"}.json"
        self.save_step = settings.save_step
        self._load_answer_sheet()

    def _load_answer_sheet(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        if os.path.isfile(self.answer_sheet_path):
            with open(self.answer_sheet_path, "r", encoding="utf-8") as f:
                self.answer_sheet = json.load(f)
        else:
            self.answer_sheet = {}
            for sample in self.dataset:
                self.answer_sheet[str(sample["question_id"])] = {
                    "category": sample["category"],
                    "answer": sample["answer"],
                    "answer_pred": None
                }        
            self._save_answer_sheet()

    def _save_answer_sheet(self):
        with open(self.answer_sheet_path, "w", encoding="utf-8") as f:
            json.dump(self.answer_sheet, f, indent=2, ensure_ascii=False)

    def evaluate(self):
        for i, sample in enumerate(tqdm(self.dataset)):
            if i and not (i % self.save_step):
                self._save_answer_sheet()
            question_id = str(sample["question_id"])
            if self.answer_sheet[question_id]["answer_pred"] is not None:
                continue
            question = sample["question"]
            options = sample["options"]
            system_prompt = (
                "Solve and choose the symbol of answer in [Options] using following format at the end of response.\n"
                "```json\n"
                '{"answer_symbol": "A"}\n'
                "```"
            )
            if not self.thinking:
                system_prompt += "\n\n/no_think"
            messages = [
                ["system", system_prompt],
                [
                    "user",
                    (
                        f"[Question]\n{question}\n"
                        f"[Options]\n{"\n".join([f"{chr(ord("A")+idx)}) {option}" for idx, option in enumerate(options)])}"
                    )
                ]
            ]
            response = self.model.invoke(messages).content
            answer_pred = self._parse_answer(response)
            self.answer_sheet[question_id]["answer_pred"] = answer_pred
        self._save_answer_sheet() 

    @staticmethod
    def _parse_answer(response: str) -> Optional[str]:
        if "```json" not in response:
            return
        front_split = response.split("```json")[1]
        if "```" not in front_split:
            return
        end_split = front_split.split("```")[0]
        return json.loads(end_split)["answer_symbol"]

    def log_accuracy(self):
        stat = {"overall": {"correct": 0, "total": len(self.answer_sheet)}}
        for item in self.answer_sheet.values():
            category = item["category"]
            answer = item["answer"]
            answer_pred = item["answer_pred"]
            if answer_pred is None:
                continue
            if category not in stat:
                stat[category] = {"correct": 0, "total": 0}
            stat[category]["total"] += 1
            if answer == answer_pred:
                stat[category]["correct"] += 1
                stat["overall"]["correct"] += 1

        for category, item in stat.items():
            print(f"{category}: {item["correct"]/item["total"]}")
