import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_selected_layers(texts, model, tokenizer, batch_size, device, layer_idxs):
    model.to(device).eval()
    collected = None
    first = True

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt",
                            padding=True, truncation=True).to(device)
            out = model(**enc, output_hidden_states=True)
            hs_tuple = out.hidden_states      # length = num_layers+1 (0=embedding)
            mask = enc["attention_mask"]
            lengths = mask.sum(dim=1)         # [B]
            layer_states = []
            for l in layer_idxs:
                hs = hs_tuple[l]              # [B, T, D]
                idx = (torch.arange(hs.size(0), device=device), lengths - 1)
                last = hs[idx].cpu().numpy()  # [B, D]
                layer_states.append(last)

            # [L_sel, B, D]
            batch_arr = np.stack(layer_states, axis=0)

            if first:
                collected = batch_arr
                first = False
            else:
                collected = np.concatenate([collected, batch_arr], axis=1)

    return collected  # [L_sel, N, D]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recorder: extract selected Transformer layers")
    p.add_argument("--emotion",    type=str, required=True)
    p.add_argument("--split",      type=str, default="train")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--layers",     type=str, default="all", help="Comma-separated 1-based layer indices or 'all'")
    args = p.parse_args()

    base    = f"data/processed/{args.emotion}"
    emo_txt = os.path.join(base, f"{args.split}_emotion.txt")
    neu_txt = os.path.join(base, f"{args.split}_neutral.txt")
    out_dir = os.path.join(base, "hidden_states")
    os.makedirs(out_dir, exist_ok=True)

    emo_texts = read_texts(emo_txt)
    neu_texts = read_texts(neu_txt)
    print(f"Loaded {len(emo_texts)} emotion and {len(neu_texts)} neutral texts")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, output_hidden_states=True
    )

    try:
        num = len(model.model.layers)
    except AttributeError:
        num = len(model.transformer.h)

    if args.layers.lower() == "all":
        layer_idxs = list(range(1, num+1))
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]
        for l in layer_idxs:
            if not (1 <= l <= num):
                raise ValueError(f"Layer index {l} out of range 1..{num}")

    print(f"Extracting Transformer layers: {layer_idxs}")
    emo_arr = extract_selected_layers(
        emo_texts, model, tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        layer_idxs=layer_idxs
    )
    emo_file = os.path.join(out_dir, f"{args.split}_emotion_layers.npy")
    np.save(emo_file, emo_arr)
    print(f"Saved emotion activations to {emo_file}, shape {emo_arr.shape}")
    neu_arr = extract_selected_layers(
        neu_texts, model, tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        layer_idxs=layer_idxs
    )
    neu_file = os.path.join(out_dir, f"{args.split}_neutral_layers.npy")
    np.save(neu_file, neu_arr)
    print(f"Saved neutral activations to {neu_file}, shape {neu_arr.shape}")
