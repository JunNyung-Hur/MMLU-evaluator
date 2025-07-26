import argparse

from evaluators import *
from settings import Settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        choices=['mmlu', 'mmlu_pro'],
        help="target evaluation dataset (mmlu, mmlu_pro)"
    )
    parser.add_argument(
        '--no-think',
        action='store_true',
        help="options for disabling model's reasoning feature"
    )
    parser.add_argument(
        '--only-print',
        action='store_true',
        help="print only accuracy without more evaulation."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    settings = Settings()
    if args.no_think:
        settings.no_think = True

    if args.dataset == "mmlu":
        evaluator = MMLUEvaulator(settings)
    elif args.dataset == "mmlu_pro":
        evaluator = MMLUProEvaulator(settings)
    else:
        raise RuntimeError(f"Dataset '{args.dataset}' is not supported.")
    
    if not args.only_print:
        evaluator.evaluate()
    
    evaluator.print_accuracy()


if __name__ == "__main__":
    main()