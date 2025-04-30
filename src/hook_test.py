# src/hook_test.py

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def capture_hidden_state(model, tokenizer, prompt, layer_idx, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
    mask = inputs['attention_mask']
    length = mask.sum(dim=1).item()
    return hs[0, length-1, :].cpu()

def add_hook(model, layer_idx, steering_vector, device):
    try:
        block = model.model.layers[layer_idx]
    except AttributeError:
        block = model.transformer.h[layer_idx]
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += steering_vector.to(device)
        return (hidden, output[1]) if isinstance(output, tuple) else hidden
    return block.register_forward_hook(hook)

if __name__ == "__main__":
    model_name = "gpt2"
    layer_idx  = -1
    prompt     = "Testing injection works."
    device     = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device).eval()
    vec_path     = "data/processed/anger/steering_vector/train_layer-1_mean_k50.npy"
    steering_vec = torch.from_numpy(np.load(vec_path)).float()
    h_before = capture_hidden_state(model, tokenizer, prompt, layer_idx, device)
    handle = add_hook(model, layer_idx, steering_vec, device)
    h_after = capture_hidden_state(model, tokenizer, prompt, layer_idx, device)
    handle.remove()
    diff = h_after - h_before
    cos_sim = torch.nn.functional.cosine_similarity(diff, steering_vec.cpu(), dim=0)
    print(f"Cosine similarity between diff and steering vector: {cos_sim.item():.4f}")
    print("First 10 dims of diff:         ", diff[:10].numpy())
    print("First 10 dims of steering_vec:", steering_vec.cpu()[:10].numpy())
