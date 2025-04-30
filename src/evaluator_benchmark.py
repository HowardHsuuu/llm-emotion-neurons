import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def generate_answer(model, tokenizer, question, device, max_length):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=False,
        return_dict_in_generate=True
    )
    return tokenizer.decode(out.sequences[0], skip_special_tokens=True)

def score_truthfulqa(pred, gold_answers):
    """
    Simple scoring: 1 if any gold answer substring in pred; else 0.
    Replace with GPT-judge for higher fidelity.
    """
    p = pred.lower()
    for a in gold_answers:
        if a.lower() in p:
            return 1.0
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA eval with steering")
    parser.add_argument("--model_name", required=True, help="HF model ID")
    parser.add_argument("--vector_path", required=True, help=".npy steering vector")
    parser.add_argument("--layer", type=int, default=-1, help="Layer idx")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--max_length", type=int, default=60, help="Gen max length")
    parser.add_argument("--data_path", required=True,
                        help="JSON file list of {question, answers(list)}")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    ).to(device).eval()
    steering_vec = torch.from_numpy(np.load(args.vector_path)).float()

    data = json.load(open(args.data_path, encoding="utf-8"))
    base = []
    steer = []
    for item in data:
        q = item["question"]
        gold = item["answers"]
        # baseline
        ans_b = generate_answer(model, tokenizer, q, device, args.max_length)
        score_b = score_truthfulqa(ans_b, gold)
        base.append(score_b)
        # steered
        h = add_steering_hook(model, args.layer, steering_vec, device)
        ans_s = generate_answer(model, tokenizer, q, device, args.max_length)
        h.remove()
        score_s = score_truthfulqa(ans_s, gold)
        steer.append(score_s)

    print(f"Baseline Acc: {np.mean(base):.4f}")
    print(f"Steered  Acc: {np.mean(steer):.4f}")
    print(f"Delta Acc: {(np.mean(steer)-np.mean(base)):.4f}")
