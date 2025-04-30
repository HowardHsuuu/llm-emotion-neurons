# src/evaluator.py

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def add_steering_hook(model, layer_idx, steering_vector, device):
    try:
        block = model.model.layers[layer_idx]
    except AttributeError:
        block = model.transformer.h[layer_idx]

    def hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += steering_vector.to(device)
        return (hidden, output[1]) if isinstance(output, tuple) else hidden
    return block.register_forward_hook(hook)

def generate_text(model, tokenizer, prompt, device, max_length):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        return_dict_in_generate=True
    )
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

def anger_score(text, clf):
    scores = clf(text)[0]
    for entry in scores:
        if entry["label"].lower() == "anger":
            return entry["score"]
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anger steering effect")
    parser.add_argument("--model_name",  required=True,
                        help="HF model id, e.g. gpt2 or meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--vector_path", required=True,
                        help="Path to steering vector numpy file")
    parser.add_argument("--layer",       type=int, default=-1,
                        help="Layer to inject at")
    parser.add_argument("--device",      default="cpu",
                        help="Device, e.g. cpu or cuda")
    parser.add_argument("--max_length",  type=int, default=60,
                        help="Generation max length")
    parser.add_argument("prompts", nargs="+",
                        help="Prompts to evaluate")
    args = parser.parse_args()

    # emotion classifier
    emotion_clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    ).to(device).eval()

    steering_vec = torch.from_numpy(np.load(args.vector_path)).float()

    base_scores = []
    steer_scores = []

    for prompt in args.prompts:
        print(f"\n=== Prompt ===\n{prompt}")

        base_out = generate_text(model, tokenizer, prompt, device, args.max_length)
        base_anger = anger_score(base_out, emotion_clf)
        base_scores.append(base_anger)
        print(f"\n--- Baseline:\n{base_out}\nAnger prob: {base_anger:.4f}")

        hook = add_steering_hook(model, args.layer, steering_vec, device)
        steer_out = generate_text(model, tokenizer, prompt, device, args.max_length)
        hook.remove()
        steer_anger = anger_score(steer_out, emotion_clf)
        steer_scores.append(steer_anger)
        print(f"\n+++ Steered:\n{steer_out}\nAnger prob: {steer_anger:.4f}")

    avg_base = np.mean(base_scores)
    avg_steer = np.mean(steer_scores)
    print(f"\nAvg anger prob: baseline={avg_base:.4f}, steered={avg_steer:.4f}, Δ={avg_steer-avg_base:.4f}")
