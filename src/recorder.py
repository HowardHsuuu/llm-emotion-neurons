# src/recorder.py

import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_hidden_states(texts, model, tokenizer, layer, batch_size, device):
    all_states = []
    model.to(device).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt",
                            padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer]       # [B, seq_len, hidden_dim]
            mask = enc["attention_mask"]        # [B, seq_len]
            lengths = mask.sum(dim=1)           # true token lengths per sample
            # collect each sample's last-token state
            last_states = []
            for b, length in enumerate(lengths):
                last_states.append(hs[b, length-1, :].cpu().numpy())
            all_states.append(np.stack(last_states, axis=0))
    return np.concatenate(all_states, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states on CPU")
    parser.add_argument("--emotion",    type=str, required=True,
                        help="Emotion name (e.g. anger, joy)")
    parser.add_argument("--split",      type=str, default="train",
                        help="Dataset split: train/validation/test")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model identifier (e.g. gpt2, meta-llama/Llama-3.2-1B)")
    parser.add_argument("--layer",      type=int, default=-1,
                        help="Hidden_states layer index, -1 for last")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for extraction (small for CPU)")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="Device to run on, e.g. 'cpu' or 'cuda'")
    args = parser.parse_args()

    base     = f"data/processed/{args.emotion}"
    emo_path = os.path.join(base, f"{args.split}_emotion.txt")
    neu_path = os.path.join(base, f"{args.split}_neutral.txt")
    out_dir  = os.path.join(base, "hidden_states")
    os.makedirs(out_dir, exist_ok=True)

    emo_texts = read_texts(emo_path)
    neu_texts = read_texts(neu_path)
    print(f"Loaded {len(emo_texts)} emotion texts, {len(neu_texts)} neutral texts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        output_hidden_states=True
    )

    emo_states = extract_hidden_states(
        emo_texts, model, tokenizer,
        layer=args.layer, batch_size=args.batch_size,
        device=args.device
    )
    neu_states = extract_hidden_states(
        neu_texts, model, tokenizer,
        layer=args.layer, batch_size=args.batch_size,
        device=args.device
    )

    np.save(os.path.join(out_dir, f"{args.split}_emotion.npy"), emo_states)
    np.save(os.path.join(out_dir, f"{args.split}_neutral.npy"), neu_states)
    print("Saved hidden states to:", out_dir)
