import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def compute_diff_and_pvalues(emo_all, neu_all):
    # Mean difference
    emo_mean = emo_all.mean(axis=1)  # [L, D]
    neu_mean = neu_all.mean(axis=1)
    diff_matrix = emo_mean - neu_mean

    # Welch's t-test per layer
    L, _, D = emo_all.shape
    pval_matrix = np.zeros((L, D))
    for i in range(L):
        _, p_vals = ttest_ind(emo_all[i], neu_all[i], axis=0, equal_var=False)
        pval_matrix[i] = p_vals
    return diff_matrix, pval_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global analysis of neuron activation differences with heatmap and scatter"
    )
    parser.add_argument("--emotion",     type=str, required=True,
                        help="Emotion name, e.g. anger, joy")
    parser.add_argument("--split",       type=str, default="train",
                        help="Dataset split: train/validation/test")
    parser.add_argument("--emotion_npy", type=str, required=True,
                        help="Path to *_emotion_layers.npy")
    parser.add_argument("--neutral_npy", type=str, required=True,
                        help="Path to *_neutral_layers.npy")
    parser.add_argument("--top_k",       type=int, default=20,
                        help="Number of top neurons per layer to highlight in scatter")
    args = parser.parse_args()
    out_dir = f"results/neuron_stats/{args.emotion}/{args.split}_global"
    os.makedirs(out_dir, exist_ok=True)
    emo_all = np.load(args.emotion_npy)  # [L, N_e, D]
    neu_all = np.load(args.neutral_npy)  # [L, N_n, D]
    L, _, D = emo_all.shape
    print(f"Loaded emotion data {emo_all.shape}, neutral {neu_all.shape}")
    diff_mat, pval_mat = compute_diff_and_pvalues(emo_all, neu_all)
    global_mean_diff = diff_mat.mean(axis=0)  # [D]
    sorted_idx = np.argsort(global_mean_diff)
    sorted_mat = diff_mat[:, sorted_idx]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sorted_mat, aspect='auto', interpolation='bicubic', cmap='coolwarm')
    plt.colorbar(im, label='Mean Difference')
    plt.xlabel('Neuron (sorted by mean difference)')
    plt.ylabel('Layer')
    plt.title('Sorted Neuron-Wise Mean Difference')
    heat_name = os.path.join(out_dir, f"{args.split}_sorted_mean_diff_heatmap.png")
    plt.tight_layout()
    plt.savefig(heat_name)
    plt.close()
    print(f"Saved sorted heatmap to {heat_name}")
    plt.figure(figsize=(8, 6))
    for layer in range(L):
        layer_vals = diff_mat[layer]
        top_idxs = np.argsort(np.abs(layer_vals))[-args.top_k:]
        plt.scatter(top_idxs, np.full_like(top_idxs, layer), c=layer_vals[top_idxs],
                    cmap='coolwarm', edgecolors='k', vmin=diff_mat.min(), vmax=diff_mat.max())
    plt.colorbar(label='Mean Difference')
    plt.xlabel('Neuron Index')
    plt.ylabel('Layer Index')
    plt.title(f'Top {args.top_k} Neurons per Layer - Neuron-Wise Mean Difference')
    scatter_name = os.path.join(out_dir, f"{args.split}_top{args.top_k}_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_name)
    plt.close()
    print(f"Saved scatter plot to {scatter_name}")
    flat = diff_mat.flatten()
    layers_idx, neurons_idx = np.unravel_index(np.arange(L*D), (L, D))
    df = pd.DataFrame({
        'layer': layers_idx + 1,
        'neuron': neurons_idx,
        'mean_diff': flat,
        'abs_diff': np.abs(flat),
        'p_value': pval_mat.flatten()
    })
    df_sorted = df.sort_values('abs_diff', ascending=False)
    csv_name = os.path.join(out_dir, f"{args.split}_global_rank.csv")
    df_sorted.to_csv(csv_name, index=False)
    print(f"Saved global ranking CSV to {csv_name}")
