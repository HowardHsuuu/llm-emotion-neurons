import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def add_steering_hook(model, layer_idx, vec, alpha, device):
    try:
        blk = model.model.layers[layer_idx-1]
    except AttributeError:
        blk = model.transformer.h[layer_idx-1]
    def hook(_, __, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += alpha * vec.to(device)
        return None
    return blk.register_forward_hook(hook)

def generate_text(model, tokenizer, prompt, device, max_length):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        use_cache=False,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    return tokenizer.decode(out.sequences[0], skip_special_tokens=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluator with multi-layer steering")
    p.add_argument("--model_name", required=True)
    p.add_argument("--emotion_npy", required=True, help="train_emotion_layers.npy")
    p.add_argument("--neutral_npy", required=True, help="train_neutral_layers.npy")
    p.add_argument("--layers", type=str, default="all", help="Comma-separated 1-based indices or 'all'")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_length", type=int, default=60)
    p.add_argument("prompts", nargs="+")
    args = p.parse_args()
    device = torch.device(args.device)
    emo_all = np.load(args.emotion_npy)   # [L_sel, N_em, D]
    neu_all = np.load(args.neutral_npy)   # [L_sel, N_neu, D]
    L_sel = emo_all.shape[0]

    if args.layers.lower() == "all":
        layer_idxs = list(range(1, L_sel+1))
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]

    steering = {}
    for l in layer_idxs:
        mean_e = emo_all[l-1].mean(axis=0)
        mean_n = neu_all[l-1].mean(axis=0)
        steering[l] = torch.from_numpy(mean_e - mean_n).float()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    ).to(device).eval()

    for prompt in args.prompts:
        print(f"\n=== Prompt ===\n{prompt}")

        base = generate_text(model, tokenizer, prompt, device, args.max_length)
        print(f"\n--- Baseline ---\n{base}")
        handles = []
        for l in layer_idxs:
            handles.append(add_steering_hook(
                model, l, steering[l], args.alpha, device
            ))
        steer = generate_text(model, tokenizer, prompt, device, args.max_length)
        for h in handles:
            h.remove()
        print(f"\n+++ Steered (α={args.alpha}, layers={layer_idxs}) +++\n{steer}")
