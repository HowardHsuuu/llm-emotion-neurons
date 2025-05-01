import argparse
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(layer_str: str, num_layers: int):
    if layer_str.lower() == 'all':
        return list(range(num_layers))
    layers = set()
    for part in layer_str.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            layers.update(range(a, b+1))
        else:
            layers.add(int(part))
    zero_based = [l-1 for l in layers if 1 <= l <= num_layers]
    return sorted(zero_based)


def add_steering_hooks(model, layer_idxs, diff_matrix, device, alpha: float):
    handles = []
    for l in layer_idxs:
        vec = torch.from_numpy(diff_matrix[l]).float().to(device) * alpha
        try:
            block = model.model.layers[l]
        except AttributeError:
            block = model.transformer.h[l]
        def hook(module, inp, output, vec=vec):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] += vec
            return None
        handles.append(block.register_forward_hook(hook))
    return handles


def generate_answer(model, tokenizer, question: str, device: torch.device, max_length: int):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def score_truthfulqa(pred: str, gold_answers: list):
    p = pred.lower()
    for ans in gold_answers:
        if ans.lower() in p:
            return 1.0
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluator for TruthfulQA with steering")
    parser.add_argument("--model_name", required=True,
                        help="HF model id, e.g. Llama-3.2-3B-Instruct")
    parser.add_argument("--vector_path", required=True,
                        help=".npy steering diff matrix [L,D]")
    parser.add_argument("--layers", default="all",
                        help="1-based 层号或 'all'，例如 '1,3,5-7' 或 'all'")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling factor for steering")
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda")
    parser.add_argument("--max_length", type=int, default=60,
                        help="Max gen tokens")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--data_path", required=True,
                        help="TruthfulQA JSON file path, e.g. data/benchmarks/truthfulqa_validation.json")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional JSON results file")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA unavailable, using CPU.")

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    ).to(device).eval()
    diff_matrix = np.load(args.vector_path)
    num_layers = diff_matrix.shape[0]
    layer_idxs = parse_layers(args.layers, num_layers)
    if not layer_idxs:
        raise ValueError(f"No valid layers parsed from '{args.layers}'.")

    results = []
    base_scores, steer_scores = [], []
    for item in tqdm(data, desc="Evaluating TruthfulQA"):
        q = item['question']
        gold = item.get('answers', [])

        # Baseline
        ans_b = generate_answer(model, tokenizer, q, device, args.max_length)
        score_b = score_truthfulqa(ans_b, gold)
        base_scores.append(score_b)

        # Steered
        handles = add_steering_hooks(model, layer_idxs, diff_matrix, device, args.alpha)
        ans_s = generate_answer(model, tokenizer, q, device, args.max_length)
        for h in handles: h.remove()
        score_s = score_truthfulqa(ans_s, gold)
        steer_scores.append(score_s)

        results.append({
            'question': q,
            'baseline_answer': ans_b,
            'baseline_score': score_b,
            'steered_answer': ans_s,
            'steered_score': score_s
        })

    avg_b = np.mean(base_scores)
    avg_s = np.mean(steer_scores)
    print(f"Baseline Acc: {avg_b:.4f}")
    print(f"Steered  Acc: {avg_s:.4f}")
    print(f"Delta Acc: {avg_s-avg_b:.4f}")

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
