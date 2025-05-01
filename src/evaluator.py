import argparse
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layers(layer_str: str, max_layer: int):
    if layer_str.lower() == 'all':
        return list(range(1, max_layer + 1))
    layers = set()
    for part in layer_str.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            layers.update(range(a, b + 1))
        else:
            layers.add(int(part))
    return sorted([l for l in layers if 1 <= l <= max_layer])


def add_steering_hook(model, layer_idx, vec, alpha, device):
    idx0 = layer_idx - 1
    try:
        block = model.model.layers[idx0]
    except AttributeError:
        block = model.transformer.h[idx0]

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += alpha * vec.to(device)
        return None

    return block.register_forward_hook(hook)


def generate_text(model, tokenizer, prompt: str, device: torch.device, max_length: int, do_sample: bool, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    gen_kwargs = dict(
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluator with multi-layer steering using diff matrix"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True,
                        help="Path to diffmatrix .npy of shape [L, D]")
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated 1-based layer indices, ranges with '-', or 'all'")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling factor for steering vector")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling (True) or greedy decoding (False)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional JSON file to save results summary")
    parser.add_argument("prompts", nargs='+', help="Prompts to generate from")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, fallback to CPU.")

    diff_matrix = np.load(args.vector_path)
    num_layers, hidden_dim = diff_matrix.shape
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=False
    ).to(device).eval()

    try:
        model_layers = len(model.model.layers)
    except AttributeError:
        model_layers = len(model.transformer.h)
    if num_layers != model_layers:
        raise RuntimeError(f"Diff matrix has {num_layers} layers but model has {model_layers}.")

    layer_idxs = parse_layers(args.layers, num_layers)
    results = []
    for prompt in args.prompts:
        entry = {'prompt': prompt}
        # Baseline
        baseline = generate_text(
            model, tokenizer, prompt, device, args.max_length,
            args.do_sample, args.temperature, args.top_p
        )
        print(f"--- Baseline ---\n{baseline}\n")
        entry['baseline'] = baseline

        # Steered
        handles = []
        for l in layer_idxs:
            vec = torch.from_numpy(diff_matrix[l-1]).float()
            handles.append(add_steering_hook(model, l, vec, args.alpha, device))
        steered = generate_text(
            model, tokenizer, prompt, device, args.max_length,
            args.do_sample, args.temperature, args.top_p
        )
        for h in handles:
            h.remove()
        print(f"+++ Steered (α={args.alpha}, layers={layer_idxs}) +++\n{steered}\n")
        entry['steered'] = steered
        results.append(entry)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
