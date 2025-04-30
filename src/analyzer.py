import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def load_states(path_emotion, path_neutral):
    emo = np.load(path_emotion)   # shape [N_e, D]
    neu = np.load(path_neutral)   # shape [N_n, D]
    return emo, neu

def compute_activation_difference(emo, neu):
    mean_emo = emo.mean(axis=0)
    mean_neu = neu.mean(axis=0)
    diff     = mean_emo - mean_neu
    return diff

def compute_ttest(emo, neu):
    t_vals, p_vals = ttest_ind(emo, neu, axis=0, equal_var=False)
    return p_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze emotion vs neutral neuron activations")
    parser.add_argument("--emotion",   type=str, required=True,
                        help="Emotion name (e.g. anger, joy)")
    parser.add_argument("--split",     type=str, default="train",
                        help="Dataset split: train/validation/test")
    parser.add_argument("--layer",     type=int, default=-1,
                        help="Hidden_states layer index (for record-keeping)")
    parser.add_argument("--top_k",     type=int, default=50,
                        help="输出前 top_k 个 neuron")
    args = parser.parse_args()

    base = f"data/processed/{args.emotion}/hidden_states"
    emo_path = os.path.join(base, f"{args.split}_emotion.npy")
    neu_path = os.path.join(base, f"{args.split}_neutral.npy")
    assert os.path.isfile(emo_path) and os.path.isfile(neu_path), \
        "请先运行 recorder.py 生成 hidden_states .npy 文件"

    emo_states, neu_states = load_states(emo_path, neu_path)
    diff   = compute_activation_difference(emo_states, neu_states)
    p_vals = compute_ttest(emo_states, neu_states)
    df = pd.DataFrame({
        "neuron_idx": np.arange(diff.shape[0]),
        "delta":       diff,
        "abs_delta":   np.abs(diff),
        "p_value":     p_vals
    })
    df_sorted = df.sort_values("abs_delta", ascending=False).head(args.top_k)
    out_dir = f"results/neuron_stats/{args.emotion}/layer_{args.layer}"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.split}_top{args.top_k}.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"Top {args.top_k} neurons saved to {out_csv}")
    print(df_sorted)
