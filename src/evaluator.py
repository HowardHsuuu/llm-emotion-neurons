import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def generate_text(model, tokenizer, prompt, device, max_length):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        use_cache=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluator with multi-layer steering using diff matrix")
    parser.add_argument("--model_name",  type=str,   required=True)
    parser.add_argument("--vector_path", type=str,   required=True,
                        help="Path to diffmatrix .npy of shape [L, D]")
    parser.add_argument("--layers",      type=str,   default="all",
                        help="Comma-separated 1-based layer indices or 'all'")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Scaling factor for steering vector")
    parser.add_argument("--device",      type=str,   default="cuda",
                        help="Device to run on, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--max_length",  type=int,   default=60)
    parser.add_argument("prompts",       nargs='+',  help="Prompts to generate from")
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        if args.device.startswith("cuda"):
            print("CUDA not available, falling back to CPU")

    diff_matrix = np.load(args.vector_path)  # [L, D]
    L, D = diff_matrix.shape
    if args.layers.lower() == "all":
        layer_idxs = list(range(1, L+1))
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]
        for l in layer_idxs:
            if not (1 <= l <= L):
                raise ValueError(f"Layer index {l} out of range 1..{L}")

    steering = {l: torch.from_numpy(diff_matrix[l-1]).float() for l in layer_idxs}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=False
    ).to(device).eval()
    for prompt in args.prompts:
        print(f"\n=== Prompt ===\n{prompt}\n")
        # Baseline generation
        baseline = generate_text(model, tokenizer, prompt, device, args.max_length)
        print(f"--- Baseline ---\n{baseline}\n")
        # Steered generation
        handles = []
        for l in layer_idxs:
            handles.append(add_steering_hook(model, l, steering[l], args.alpha, device))
        steered = generate_text(model, tokenizer, prompt, device, args.max_length)
        for h in handles:
            h.remove()
        print(f"+++ Steered (α={args.alpha}, layers={layer_idxs}) +++\n{steered}\n")
