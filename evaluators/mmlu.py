import json
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from evaluators.base import BaseEvaluator
from settings import Settings


DATASET_NAME = "cais/mmlu"


class MMLUEvaulator(BaseEvaluator):

    def __init__(self, settings: Settings):
        super().__init__(DATASET_NAME, settings)
        self.dataset = load_dataset(DATASET_NAME, name="all", split="test")  

    def evaluate(self):
        num_done = self._get_len_results()
        for i, sample in enumerate(tqdm(self.dataset)):
            if i < num_done:
                continue
            question = sample["question"]
            subject = sample["subject"]
            choices = sample["choices"]
            answer = chr(ord("A")+sample["answer"])
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
                        f"[Options]\n{"\n".join([f"{chr(ord("A")+idx)}) {choice}" for idx, choice in enumerate(choices)])}"
                    )
                ]
            ]
            if self.no_think:
                messages = [["system", "/no_think"]] + messages
            response = self.model.invoke(messages).content
            answer_pred = self._parse_answer(response)
            result = {"subject": subject, "answer": answer, "answer_pred": answer_pred}
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
            subject = result["subject"]
            answer = result["answer"]
            answer_pred = result["answer_pred"]
            if answer_pred is None:
                continue
            if subject not in stat:
                stat[subject] = {"correct": 0, "total": 0}
            stat[subject]["total"] += 1
            if answer == answer_pred:
                stat[subject]["correct"] += 1
                stat["overall"]["correct"] += 1

        for subject, item in stat.items():
            print(f"{subject}: {item["correct"]/item["total"]}")
