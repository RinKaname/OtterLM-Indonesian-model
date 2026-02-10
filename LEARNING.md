# OtterLM Architecture Learning Guide

This document explains the architecture of `OtterLM` (in `model.py`) to help you understand its components, how they interact, and how to modify them for future features or debugging.

OtterLM is a **decoder-only Transformer** model, closely following the architecture of **Llama 2** and **Llama 3**. It is designed for autoregressive text generation.

---

## 1. Configuration (`Config`)

The `Config` dataclass defines the hyperparameters of the model.

*   **`vocab_size`**: Total number of unique tokens in the vocabulary (e.g., 32,000). The embedding layer and output head will have this size.
*   **`block_size`**: Maximum sequence length the model can process (e.g., 2048 tokens). This limits the size of the position embeddings.
*   **`n_layer`**: Number of Transformer blocks (depth of the network). deeper models can learn more complex patterns but are slower.
*   **`n_head`**: Number of attention heads for the Query (Q).
*   **`n_embd`**: Dimensionality of the embeddings and hidden states. Must be divisible by `n_head`.
*   **`n_kv_head`**: Number of attention heads for Keys (K) and Values (V).
    *   If `n_kv_head == n_head`: Standard Multi-Head Attention (MHA).
    *   If `n_kv_head < n_head`: **Grouped Query Attention (GQA)**. This reduces memory usage for the KV cache and speeds up inference.
*   **`dropout`**: Probability of dropping out neurons during training to prevent overfitting (e.g., 0.1).
*   **`rope_theta`**: Base frequency for Rotary Positional Embeddings (RoPE). Higher values allow handling longer contexts better.
*   **`norm_eps`**: Small constant added for numerical stability in RMSNorm (prevention of division by zero).
*   **`use_cache`**: Whether to cache Key/Value states during generation (essential for fast inference).

---

## 2. Normalization (`RMSNorm`)

**Root Mean Square Layer Normalization** is used instead of standard LayerNorm.

*   **Logic:** It normalizes the input `x` by dividing it by the root mean square of its values.
*   **Difference from LayerNorm:** It does *not* subtract the mean (re-centering). It only scales the variance. This is computationally cheaper and often performs better in deep transformers.
*   **Learnable Parameter:** `self.scale` (similar to gamma in LayerNorm) allows the model to rescale the normalized values.

---

## 3. Positional Embeddings (`RoPE`)

OtterLM uses **Rotary Positional Embeddings (RoPE)**, which encode position information by rotating the Query and Key vectors in a high-dimensional space.

### `precompute_rope_freqs`
*   Calculates the rotation frequencies (angles) for each position index `0` to `max_len`.
*   **Output:** Returns `cos` and `sin` values needed for rotation.
*   **Note:** These are precomputed once and cached to save time during forward passes.

### `apply_rotary_emb`
*   Applies the rotation to `q` and `k`.
*   **How it works:** It pairs elements of the embedding vector (e.g., `x[0]` and `x[1]`, `x[2]` and `x[3]`) and rotates them as 2D coordinates.
*   **Important:** This implementation pairs adjacent elements (`::2`, `1::2`), unlike the original Llama paper which pairs the first half with the second half. This is mathematically equivalent but affects compatibility with other pre-trained weights if you try to load them directly.

---

## 4. The Transformer Block (`OtterLMBlock`)

This is the core building block, repeated `n_layer` times. It consists of two main sub-blocks: **Attention** and **Feed-Forward Network (MLP)**.

### A. Attention Mechanism (`_attn_block`)

1.  **Projections:**
    *   The input `x` is projected into Query (`q`), Key (`k`), and Value (`v`) vectors using `self.c_attn`.
    *   **GQA Handling:** If `n_kv_head < n_head`, `k` and `v` have fewer heads than `q`.
2.  **RoPE:**
    *   Rotary embeddings are applied to `q` and `k`. This injects position information *before* attention is computed.
3.  **KV Caching:**
    *   During generation (`past_kv` is not None), the new `k` and `v` are appended to the cached values from previous steps. This prevents re-computing past tokens.
4.  **GQA Expansion:**
    *   If using GQA, the `k` and `v` heads are repeated (`repeat_interleave`) to match the number of `q` heads so that standard dot-product attention can be used.
5.  **Attention Calculation:**
    *   `F.scaled_dot_product_attention` computes `softmax(Q @ K.T / sqrt(d)) @ V`.
    *   **The Critical Logic (`is_causal`):**
        *   **Training (Prefill):** `is_causal=True`. Ensures token `t` can only see tokens `0...t`.
        *   **Decoding (Generation):** `is_causal=False`. When generating the *next* token, it should see *all* past tokens in the cache. Using `True` here would wrongly mask out the history because the query length is 1.
6.  **Output:** The result is projected back to `n_embd` size using `self.c_proj`.

### B. MLP (Feed-Forward) (`_mlp_block`)

OtterLM uses the **SwiGLU** activation function, which is a variant of GLU (Gated Linear Unit).

*   **Structure:**
    *   `gate_proj(x)`: Determines "how much" information passes through.
    *   `up_proj(x)`: Transforms the information.
    *   **SwiGLU:** `down_proj( F.silu(gate) * up )`.
*   **Why:** SwiGLU generally performs better than ReLU or GeLU in LLMs.

---

## 5. The Full Model (`OtterLM`)

This class assembles the components.

1.  **Embeddings (`wte`)**: Converts token IDs into vectors.
2.  **Layers (`h`)**: A list of `OtterLMBlock`s.
3.  **Final Norm (`ln_f`)**: Normalizes the output of the last block.
4.  **LM Head (`lm_head`)**: Projects the final hidden state back to vocabulary size (`vocab_size`) to predict the next token logits.
    *   **Weight Tying:** `self.lm_head.weight = self.transformer.wte.weight`. This shares parameters between input embeddings and output projections, saving memory and often improving performance.

### `generate` Method
*   Implements the autoregressive loop.
*   **Steps:**
    1.  Forward pass to get logits for the last token.
    2.  Apply temperature (scaling) and Top-K filtering (sampling strategy).
    3.  Sample the next token using `torch.multinomial`.
    4.  Append the new token to the sequence and repeat.
*   **Optimization:** It maintains `past_key_values` so each step only processes the *newest* token, not the whole sequence history.

---

## Debugging Tips

*   **Shape Mismatches:** If you change `n_head` or `n_embd`, check the reshape operations in `_attn_block`.
*   **"Blind" Model:** If the model generates gibberish despite low loss, check the `is_causal` flag in `_attn_block`. It must be `False` during decoding steps with cache.
*   **OOM (Out of Memory):** Try reducing `block_size` or switching to GQA (reduce `n_kv_head`).

---

## Model Size & "From Scratch"

You asked: **"Is this model from scratch though, how much parameter my model actually?"**

### 1. Is it from scratch?
**Yes and No.**
*   **Code:** The *architecture implementation* is written from scratch (though heavily inspired by Llama 2). It defines how the math operations (attention, normalization, etc.) are connected.
*   **Weights:** The model is initialized with **random weights** (using `_init_weights`). This means it knows nothing about language initially. You must **train it from scratch** on a large dataset (like the Indonesian Wikipedia you have in `tokenizer.py`) for it to generate meaningful text. It is *not* a pre-trained model like GPT-4 or Llama-2-7b that you just download and run.

### 2. How many parameters?
With the default configuration in `model.py`:
*   Layers: 12
*   Heads: 12
*   Embedding Dim: 768
*   Vocab Size: 32,000

**Total Trainable Parameters: ~109.5 Million** (110M)

**Breakdown:**
*   **Embeddings (Input/Output):** ~24.6M (22.4%) - Stores the meaning of each word.
*   **Attention Layers:** ~28.3M (25.8%) - Captures relationships between words.
*   **MLP Layers:** ~56.6M (51.7%) - Processes information within each token.
*   **Norms:** ~19K (<0.1%) - Stabilizes training.

This is a **small language model** (comparable to GPT-1 or very small distilled models). It is excellent for learning, experimentation, and running on consumer hardware (even CPUs), but will not match the reasoning capabilities of multi-billion parameter models.
