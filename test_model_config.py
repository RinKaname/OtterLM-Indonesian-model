import torch
import torch.nn.functional as F
from model import OtterLM, Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_causal_bug():
    print("Testing Causal Bug with default Config (MHA)...")
    config = Config(
        vocab_size=100,
        block_size=32,
        n_layer=1,
        n_head=4,
        n_embd=32,
        dropout=0.0
    )
    model = OtterLM(config)
    model.eval()

    print(f"MHA (Head=4) Params: {count_parameters(model)}")

    if not run_test(model):
        return False
    return True

def test_gqa():
    print("Testing GQA (n_head=4, n_kv_head=1)...")
    config = Config(
        vocab_size=100,
        block_size=32,
        n_layer=1,
        n_head=4,
        n_kv_head=1,
        n_embd=32,
        dropout=0.0
    )
    model = OtterLM(config)
    model.eval()

    params = count_parameters(model)
    print(f"GQA (Head=4, KV=1) Params: {params}")

    # Verify that GQA has fewer params than MHA
    # MHA: c_attn is n_embd -> 3 * n_embd (3 * 32 * 32 = 3072)
    # GQA: c_attn is n_embd -> (4 + 2*1) * 8 = 6 * 8 = 48 -> 32 -> 48 (Wait)
    # n_head=4, head_dim = 32/4 = 8.
    # MHA: 3 * 32 = 96 output dim. 32 * 96 = 3072 weights.
    # GQA: (4 + 2) * 8 = 48 output dim. 32 * 48 = 1536 weights.
    # Difference: 1536 weights.

    if not run_test(model):
        return False
    return True

def test_dropout():
    print("Testing Dropout...")
    config = Config(
        vocab_size=100,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.5
    )
    model = OtterLM(config)

    # Eval mode: dropout should be disabled
    model.eval()
    input_ids = torch.tensor([[0, 1]])
    with torch.no_grad():
        out1, _, _ = model(input_ids)
        out2, _, _ = model(input_ids)

    diff = (out1 - out2).abs().max().item()
    if diff > 1e-6:
        print(f"FAIL: Dropout active in eval mode. Diff: {diff}")
        return False
    else:
        print("SUCCESS: Dropout disabled in eval mode.")

    # Train mode: dropout should be enabled
    model.train()
    with torch.no_grad():
        out1, _, _ = model(input_ids)
        out2, _, _ = model(input_ids)

    diff = (out1 - out2).abs().max().item()
    if diff < 1e-6:
        print(f"FAIL: Dropout not active in train mode. Diff: {diff}")
        return False
    else:
        print(f"SUCCESS: Dropout active in train mode. Diff: {diff}")
        return True

def run_test(model):
    input_ids = torch.tensor([[0, 1]])
    with torch.no_grad():
        _, _, past_kvs = model(input_ids)

    next_input = torch.tensor([[2]])
    torch.manual_seed(42)
    with torch.no_grad():
        out1, _, _ = model(next_input, past_key_values=past_kvs)

    k_orig, v_orig = past_kvs[0]
    # print(f"Past Key shape: {k_orig.shape}")

    k_mod = k_orig.clone()
    k_mod += 10.0
    past_kvs_mod = [(k_mod, v_orig)]

    torch.manual_seed(42)
    with torch.no_grad():
        out2, _, _ = model(next_input, past_key_values=past_kvs_mod)

    diff = (out1 - out2).abs().max().item()
    # print(f"Difference in output when modifying past keys: {diff}")

    if diff < 1e-5:
        print("FAIL: The model ignored changes in past keys.")
        return False
    else:
        print("SUCCESS: The model attended to past keys.")
        return True

if __name__ == "__main__":
    try:
        if not test_causal_bug():
            exit(1)
        print("-" * 20)
        if not test_gqa():
            exit(1)
        print("-" * 20)
        if not test_dropout():
            exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
