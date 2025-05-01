import argparse
import numpy as np
from pathlib import Path

def compute_diff_matrix(emo_all: np.ndarray, neu_all: np.ndarray, top_k: int = None, alpha: float = 1.0) -> np.ndarray:
    emo_mean = emo_all.mean(axis=1)  # [L, D]
    neu_mean = neu_all.mean(axis=1)  # [L, D]
    diff = emo_mean - neu_mean       # [L, D]

    if top_k is None:
        return diff * alpha

    diff_top = np.zeros_like(diff)
    num_layers, hidden_size = diff.shape
    for layer_idx in range(num_layers):
        layer_diff = diff[layer_idx]
        top_indices = np.argsort(np.abs(layer_diff))[-top_k:]
        mask = np.zeros(hidden_size, dtype=bool)
        mask[top_indices] = True
        diff_top[layer_idx] = layer_diff * mask

    return diff_top * alpha


def main():
    parser = argparse.ArgumentParser(
        description="Generate steering diff matrix for all layers at once"
    )
    parser.add_argument("--emotion",    type=str, required=True,
                        help="Emotion name (e.g., anger, joy)")
    parser.add_argument("--split",      type=str, required=True,
                        help="Data split (train, val, test)")
    parser.add_argument("--vector_type", choices=["mean", "top_k"], default="top_k",
                        help="Full mean diff or only top_k components")
    parser.add_argument("--top_k",      type=int, default=50,
                        help="Number of top-K neurons to keep per layer when vector_type=top_k")
    parser.add_argument("--alpha",      type=float, default=1.0,
                        help="Scaling factor for diff matrix")
    parser.add_argument("--layers",     type=int, nargs='+', default=None,
                        help="List of 1-based layer indices to apply top_k or mean; other layers zeroed")
    parser.add_argument("--data_dir",   type=str, default="data/processed",
                        help="Root directory for processed data")
    args = parser.parse_args()

    emotion_dir = Path(args.data_dir) / args.emotion
    emo_path = emotion_dir / "hidden_states" / f"{args.split}_emotion_layers.npy"
    neu_path = emotion_dir / "hidden_states" / f"{args.split}_neutral_layers.npy"
    emo_all = np.load(emo_path)
    neu_all = np.load(neu_path)

    top_k = None if args.vector_type == "mean" else args.top_k
    diff_matrix = compute_diff_matrix(emo_all, neu_all, top_k=top_k, alpha=args.alpha)

    if args.layers is not None:
        zero_idxs = [l - 1 for l in args.layers]
        mask_matrix = np.zeros_like(diff_matrix)
        for idx in zero_idxs:
            mask_matrix[idx] = diff_matrix[idx]
        diff_matrix = mask_matrix

    out_dir = emotion_dir / "steering_vector"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.layers is None:
        fname = f"{args.split}_{args.vector_type}_diffmatrix.npy"
    else:
        layers_str = "-".join(str(l) for l in args.layers)
        fname = f"{args.split}_{args.vector_type}_layers{layers_str}_diffmatrix.npy"
    out_path = out_dir / fname
    np.save(out_path, diff_matrix)

    nonzeros = int(np.count_nonzero(diff_matrix))
    print(f"Saved diff matrix to {out_path}")
    print(f"Matrix shape: {diff_matrix.shape}, non-zero elements: {nonzeros}")

if __name__ == "__main__":
    main()
