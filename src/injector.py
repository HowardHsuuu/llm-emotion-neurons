import os
import argparse
import numpy as np

def build_steering_vector(emotion: str, split: str, layer: int, base_dir: str, vector_type: str, top_k: int):
    hs_dir = os.path.join(base_dir, emotion, "hidden_states")
    emo_layers = os.path.join(hs_dir, f"{split}_emotion_layers.npy")
    neu_layers = os.path.join(hs_dir, f"{split}_neutral_layers.npy")
    emo_all = np.load(emo_layers)  # [L, N_e, D]
    neu_all = np.load(neu_layers)  # [L, N_n, D]
    L, N_e, D = emo_all.shape
    emo = emo_all[layer-1]  # [N_e, D]
    neu = neu_all[layer-1]  # [N_n, D]
    diff = emo.mean(axis=0) - neu.mean(axis=0)

    if vector_type == "top_k":
        idxs = np.argsort(np.abs(diff))[-top_k:]
        vec = np.zeros_like(diff)
        vec[idxs] = diff[idxs]
    else:
        vec = diff.copy()

    out_dir = os.path.join(base_dir, emotion, "steering_vector")
    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(
        out_dir,
        f"{split}_layer{layer}_{vector_type}_k{top_k}.npy"
    )
    np.save(vec_path, vec)
    print(f"Steering vector saved to {vec_path}")
    print(f"  norm: {np.linalg.norm(vec):.4f}, nonzeros: {(vec!=0).sum()}/{D}")
    return vec_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build emotion steering vector from multi-layer hidden states"
    )
    parser.add_argument("--emotion",     type=str, required=True,
                        help="Emotion (e.g. anger, joy)")
    parser.add_argument("--split",       type=str, default="train",
                        help="split: train/validation/test")
    parser.add_argument("--layer",       type=int, required=True,
                        help="steering layer index (1-based)")
    parser.add_argument("--base_dir",    type=str, default="data/processed",
                        help="base directory for data")
    parser.add_argument("--vector_type", type=str, default="mean",
                        choices=["mean","top_k"],
                        help="steering vector type: mean or top_k")
    parser.add_argument("--top_k",       type=int, default=50,
                        help="number of top neurons to output")
    args = parser.parse_args()

    build_steering_vector(
        emotion=args.emotion,
        split=args.split,
        layer=args.layer,
        base_dir=args.base_dir,
        vector_type=args.vector_type,
        top_k=args.top_k
    )
