# src/injector.py

import os
import argparse
import numpy as np

def build_steering_vector(emotion: str, split: str, layer: int, base_dir: str, vector_type: str, top_k: int):
    hs_dir = os.path.join(base_dir, emotion, "hidden_states")
    emo_path = os.path.join(hs_dir, f"{split}_emotion.npy")
    neu_path = os.path.join(hs_dir, f"{split}_neutral.npy")
    assert os.path.isfile(emo_path) and os.path.isfile(neu_path), \
        f"Hidden states not found in {hs_dir}"

    emo = np.load(emo_path)  # [N_e, D]
    neu = np.load(neu_path)  # [N_n, D]
    diff = emo.mean(axis=0) - neu.mean(axis=0)
    if vector_type == "top_k":
        # Zero out all but top_k largest-magnitude components
        idxs = np.argsort(np.abs(diff))[-top_k:]
        vec = np.zeros_like(diff)
        vec[idxs] = diff[idxs]
    else:
        vec = diff.copy()
    out_dir = os.path.join(base_dir, emotion, "steering_vector")
    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(out_dir, f"{split}_layer{layer}_{vector_type}_k{top_k}.npy")
    np.save(vec_path, vec)
    print(f"Steering vector saved to {vec_path}")
    print(f"  norm: {np.linalg.norm(vec):.4f}  nonzeros: {(vec!=0).sum()}/{vec.size}")
    return vec_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build emotion steering vector from hidden states"
    )
    parser.add_argument("--emotion",     type=str, required=True,
                        help="Emotion name (e.g. anger, joy)")
    parser.add_argument("--split",       type=str, default="train",
                        help="Which split to use: train/validation/test")
    parser.add_argument("--layer",       type=int, default=-1,
                        help="Layer index (for naming only)")
    parser.add_argument("--base_dir",    type=str, default="data/processed",
                        help="Base directory for processed data")
    parser.add_argument("--vector_type", type=str, default="mean",
                        choices=["mean","top_k"],
                        help="Use full mean diff or top_k sparse")
    parser.add_argument("--top_k",       type=int, default=50,
                        help="If vector_type=top_k, number of neurons to keep")
    args = parser.parse_args()

    build_steering_vector(
        emotion=args.emotion,
        split=args.split,
        layer=args.layer,
        base_dir=args.base_dir,
        vector_type=args.vector_type,
        top_k=args.top_k
    )
