import os
import json
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TruthfulQA JSON for evaluator_truthfulqa.py"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation"],
        help="Which splits to prepare (e.g. train validation test)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/benchmarks",
        help="Where to write the JSON files"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ds = load_dataset("truthful_qa", "generation")

    for split in args.splits:
        if split not in ds:
            print(f"Warning: split '{split}' not in dataset, available: {list(ds.keys())}")
            continue

        out_list = []
        for ex in ds[split]:
            question = ex.get("question")
            answers  = ex.get("correct_answers") or ex.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            out_list.append({
                "question": question,
                "answers": answers
            })

        out_path = os.path.join(args.output_dir, f"truthfulqa_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(out_list)} examples to {out_path}")

if __name__ == "__main__":
    main()
