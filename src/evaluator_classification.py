import argparse
import json
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
            layers.update(range(a, b + 1))
        else:
            layers.add(int(part))
    return sorted([l for l in layers if 0 <= l < num_layers])


def add_steering_hooks(model, layer_idxs, diff_matrix, device):
    handles = []
    for l in layer_idxs:
        vec = torch.from_numpy(diff_matrix[l]).to(device)
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


def generate_text(model, tokenizer, prompt: str, device: torch.device, max_length: int):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def emotion_score(text: str, clf, emotion_label: str):
    scores = clf(text)[0]
    for entry in scores:
        if entry['label'].lower() == emotion_label:
            return entry['score']
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Classification evaluator with multi-layer steering")
    parser.add_argument("--model_name", required=True,
                        help="HF model id, e.g. meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--vector_path", required=True,
                        help="Path to steering diff matrix .npy of shape [L, D]")
    parser.add_argument("--layers", default="all",
                        help="Comma-separated 0-based layer indices or ranges, or 'all'")
    parser.add_argument("--emotion", default="anger",
                        help="Emotion label to score, e.g. anger, joy, sadness")
    parser.add_argument("--device", default="cpu",
                        help="Device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--max_length", type=int, default=60,
                        help="Max generation tokens")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional JSON file to save results")
    parser.add_argument("prompts", nargs='+', help="Prompts to evaluate")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA unavailable, using CPU.")

    clf_device = 0 if device.type=='cuda' else -1
    emotion_clf = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True, device=clf_device)
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
    emotion_label = args.emotion.lower()

    for prompt in args.prompts:
        print(f"\n=== Prompt ===\n{prompt}")
        # Baseline generation & score
        base_out = generate_text(model, tokenizer, prompt, device, args.max_length)
        base_prob = emotion_score(base_out, emotion_clf, emotion_label)
        base_scores.append(base_prob)
        print(f"\n--- Baseline:\n{base_out}\n{emotion_label.capitalize()} prob: {base_prob:.4f}")

        # Steered generation & score
        handles = add_steering_hooks(model, layer_idxs, diff_matrix, device)
        steer_out = generate_text(model, tokenizer, prompt, device, args.max_length)
        for h in handles: h.remove()
        steer_prob = emotion_score(steer_out, emotion_clf, emotion_label)
        steer_scores.append(steer_prob)
        print(f"\n+++ Steered (layers={layer_idxs}):\n{steer_out}\n{emotion_label.capitalize()} prob: {steer_prob:.4f}")

        results.append({
            'prompt': prompt,
            'baseline': base_out,
            f'baseline_{emotion_label}': base_prob,
            'steered': steer_out,
            f'steered_{emotion_label}': steer_prob
        })

    avg_base = np.mean(base_scores)
    avg_steer = np.mean(steer_scores)
    print(f"\nAvg {emotion_label} prob: baseline={avg_base:.4f}, steered={avg_steer:.4f}, Δ={avg_steer-avg_base:.4f}")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
