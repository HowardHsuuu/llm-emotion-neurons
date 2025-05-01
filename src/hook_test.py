import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def capture_all_hidden_states(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    mask = inputs['attention_mask']
    length = mask.sum(dim=1).item()
    hs = []
    for i in range(1, len(hidden_states)):
        layer_hs = hidden_states[i][0, length-1, :].cpu()
        hs.append(layer_hs)
    return torch.stack(hs)  # [num_layers, hidden_dim]


def add_hooks_all(model, steering_matrix, device):
    handles = []
    num_layers = steering_matrix.shape[0]
    for layer_idx in range(num_layers):
        try:
            block = model.model.layers[layer_idx]
        except AttributeError:
            block = model.transformer.h[layer_idx]
        vec = steering_matrix[layer_idx].to(device)
        def make_hook(vec):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                hidden[:, -1, :] = hidden[:, -1, :] + vec
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                else:
                    return hidden
            return hook
        handle = block.register_forward_hook(make_hook(vec))
        handles.append(handle)
    return handles


def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    prompt = "Testing injection works across all layers."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device).eval()
    vec_path = "data/processed/anger/steering_vector/train_top_k_diffmatrix.npy"
    steering_matrix = torch.from_numpy(np.load(vec_path)).float()
    h_before = capture_all_hidden_states(model, tokenizer, prompt, device)
    handles = add_hooks_all(model, steering_matrix, device)
    h_after = capture_all_hidden_states(model, tokenizer, prompt, device)
    for handle in handles:
        handle.remove()

    diff = h_after - h_before  # [num_layers, hidden_dim]
    cos_sim_per_layer = torch.nn.functional.cosine_similarity(
        diff, steering_matrix.cpu(), dim=1
    )

    for idx, cs in enumerate(cos_sim_per_layer):
        print(f"Layer {idx:2d} cosine similarity: {cs.item():.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(list(range(len(cos_sim_per_layer))), cos_sim_per_layer.numpy(), marker='o')
    plt.title('Cosine Similarity per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()