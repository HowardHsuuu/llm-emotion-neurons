# src/analyzer.py

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def load_states_all(path_emotion_layers, path_neutral_layers, layer_idx):
    emo_all = np.load(path_emotion_layers)   # shape [L_sel, N_e, D]
    neu_all = np.load(path_neutral_layers)   # shape [L_sel, N_n, D]
    L_sel = emo_all.shape[0]
    if not (1 <= layer_idx <= L_sel):
        raise ValueError(f"layer {layer_idx} out of range 1..{L_sel}")
    emo = emo_all[layer_idx - 1]
    neu = neu_all[layer_idx - 1]
    return emo, neu

def compute_activation_difference(emo, neu):
    mean_emo = emo.mean(axis=0)
    mean_neu = neu.mean(axis=0)
    return mean_emo - mean_neu

def compute_ttest(emo, neu):
    # Welch’s t-test
    _, p_vals = ttest_ind(emo, neu, axis=0, equal_var=False)
    return p_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze neuron statistics for emotion vs neutral"
    )
    parser.add_argument("--emotion",             type=str, required=True,
                        help="Emotion name, e.g. anger, joy")
    parser.add_argument("--split",               type=str, default="train",
                        help="Dataset split: train/validation/test")
    parser.add_argument("--emotion_npy",         type=str, required=True,
                        help="Path to *_emotion_layers.npy")
    parser.add_argument("--neutral_npy",         type=str, required=True,
                        help="Path to *_neutral_layers.npy")
    parser.add_argument("--layer",               type=int, default=1,
                        help="1-based Transformer layer index to analyze")
    parser.add_argument("--top_k",               type=int, default=50,
                        help="Number of top neurons to output")
    args = parser.parse_args()

    emo_states, neu_states = load_states_all(
        args.emotion_npy, args.neutral_npy, args.layer
    )
    print(f"Layer {args.layer}: emotion states shape {emo_states.shape}, neutral {neu_states.shape}")
    diff   = compute_activation_difference(emo_states, neu_states)
    p_vals = compute_ttest(emo_states, neu_states)
    df = pd.DataFrame({
        "neuron_idx": np.arange(diff.shape[0]),
        "delta":      diff,
        "abs_delta":  np.abs(diff),
        "p_value":    p_vals
    })
    df_sorted = df.sort_values("abs_delta", ascending=False).head(args.top_k)
    out_dir = f"results/neuron_stats/{args.emotion}/layer_{args.layer}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.split}_top{args.top_k}.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"Top {args.top_k} neurons saved to {out_csv}")
    print(df_sorted)
