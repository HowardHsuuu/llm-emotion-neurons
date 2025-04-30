import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def add_steering_hook(model, layer_idx, steering_vector, device):
    try:
        target = model.model.layers[layer_idx]
    except AttributeError:
        target = model.transformer.h[layer_idx]
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        print(f"Hidden state shape: {hidden.shape}, Steering vector shape: {steering_vector.shape}")
        if hidden.shape[-1] != steering_vector.shape[-1]:
            raise ValueError(f"Shape mismatch: hidden {hidden.shape} vs steering {steering_vector.shape}")
        hidden[:, -1, :] += steering_vector.to(device)
        return (hidden, output[1]) if isinstance(output, tuple) else hidden
    return target.register_forward_hook(hook)

def generate_text(model, tokenizer, prompt, device, max_length):
    print(f"Tokenizing prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        return_dict_in_generate=True,
    )
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--vector_path", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("prompts", nargs="+", help="One or more prompts to test")
    args = parser.parse_args()

    # Validate vector
    try:
        vec_np = np.load(args.vector_path)
        print(f"Loaded vector shape: {vec_np.shape}, dtype: {vec_np.dtype}")
        if np.any(np.isnan(vec_np)) or np.any(np.isinf(vec_np)):
            print("Warning: Steering vector contains NaN or Inf values")
        vec = torch.from_numpy(vec_np).float().to(args.device)
    except Exception as e:
        print(f"Error loading vector: {e}")
        exit(1)

    device = torch.device(args.device)
    print(f"Loading model: {args.model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    if args.layer == -1:
        try:
            args.layer = len(model.model.layers) - 1
        except AttributeError:
            args.layer = len(model.transformer.h) - 1
    print(f"Using layer: {args.layer}")

    for prompt in args.prompts:
        print("\n=== Prompt ===")
        print(prompt)
        # Baseline
        try:
            base_out = generate_text(model, tokenizer, prompt, device, args.max_length)
            print("\n--- Baseline ---")
            print(base_out)
        except Exception as e:
            print(f"Baseline generation failed: {e}")
            continue
        # Steered
        try:
            hook = add_steering_hook(model, args.layer, vec, device)
            steer_out = generate_text(model, tokenizer, prompt, device, args.max_length)
            print("\n+++ Steered +++")
            print(steer_out)
        except Exception as e:
            print(f"Steered generation failed: {e}")
        finally:
            hook.remove()