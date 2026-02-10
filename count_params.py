from model import OtterLM, Config
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_breakdown(model, config):
    print(f"--- OtterLM Parameter Breakdown (Default Config) ---")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Block Size: {config.block_size}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding Dim: {config.n_embd}")
    print(f"Intermediate Dim: {int(8 * config.n_embd / 3)}")

    total_params = count_parameters(model)

    # Calculate components
    # Embeddings: vocab_size * n_embd
    emb_params = config.vocab_size * config.n_embd

    # Per Layer:
    # LN1: n_embd
    # LN2: n_embd
    # Attn:
    #   Q: n_embd * n_embd
    #   K: n_embd * n_embd (if MHA)
    #   V: n_embd * n_embd (if MHA)
    #   Proj: n_embd * n_embd
    # MLP:
    #   Gate: n_embd * hidden_dim
    #   Up: n_embd * hidden_dim
    #   Down: hidden_dim * n_embd

    # We can iterate through modules to be precise
    emb_params_actual = sum(p.numel() for n, p in model.named_parameters() if 'wte' in n)
    attn_params = sum(p.numel() for n, p in model.named_parameters() if 'c_attn' in n or 'c_proj' in n)
    mlp_params = sum(p.numel() for n, p in model.named_parameters() if 'mlp' in n)
    norm_params = sum(p.numel() for n, p in model.named_parameters() if 'ln_' in n)

    # The LM Head weights are tied to WTE, so they share memory.
    # But usually people count unique parameters.
    # If not tied, head params would be vocab_size * n_embd.

    print(f"\nBreakdown:")
    print(f"  Embeddings (wte): {emb_params_actual:,} ({emb_params_actual/total_params:.1%})")
    print(f"  Attention Layers: {attn_params:,} ({attn_params/total_params:.1%})")
    print(f"  MLP Layers:       {mlp_params:,} ({mlp_params/total_params:.1%})")
    print(f"  Layer Norms:      {norm_params:,} ({norm_params/total_params:.1%})")

    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"Approximate Size: {total_params / 1e6:.1f} Million Parameters")

if __name__ == "__main__":
    config = Config() # Uses default values
    model = OtterLM(config)
    print_parameter_breakdown(model, config)
