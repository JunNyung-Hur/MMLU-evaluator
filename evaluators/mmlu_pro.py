import json
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from evaluators.base import BaseEvaluator
from settings import Settings


DATASET_NAME = "TIGER-Lab/MMLU-Pro"


class MMLUProEvaulator(BaseEvaluator):

    def __init__(self, settings: Settings):
        super().__init__(DATASET_NAME, settings)
        self.dataset = load_dataset(DATASET_NAME, split="test")

    def evaluate(self):
        num_done = self._get_len_results()
        for i, sample in enumerate(tqdm(self.dataset)):
            if i < num_done:
                continue
            question = sample["question"]
            category = sample["category"]
            options = sample["options"]
            answer = sample["answer"]
            user_prompt = (
                "Solve the [Question] and provide the symbol of answer in [Options] using following format, at the end of response\n"
                "```json\n"
                '{"answer_symbol": "A"}\n'
                "```\n\n"
            )
            messages = [
                [
                    "user",
                    (
                        f"{user_prompt}"
                        f"[Question]\n{question}\n"
                        f"[Options]\n{"\n".join([f"{chr(ord("A")+idx)}) {option}" for idx, option in enumerate(options)])}"
                    )
                ]
            ]
            if self.no_think:
                messages = [["system", "/no_think"]] + messages
            response = self.model.invoke(messages).content
            answer_pred = self._parse_answer(response)
            result = {"category": category, "answer": answer, "answer_pred": answer_pred}
            self._write_result(result)

    @staticmethod
    def _parse_answer(response: str) -> Optional[str]:
        if "```json" in response:
            response = response.split("```json")[1]
        if "```" in response:
            response = response.split("```")[0]
        try:
            json_response = json.loads(response)
            answer_symbol = json_response["answer_symbol"]
        except json.JSONDecodeError:
            if '{"answer_symbol": "' in response and '"}' in response:
                return response.split('{"answer_symbol": "')[-1].split('"}')[0]
            else:
                return "Error"
        except KeyError:
            return "Error"
        except TypeError:
            return "Error"
        return answer_symbol
    
    def print_accuracy(self):
        results = self._read_results()
        stat = {"overall": {"correct": 0, "total": len(results)}}
        for result in results:
            category = result["category"]
            answer = result["answer"]
            answer_pred = result["answer_pred"]
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
