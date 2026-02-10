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

---

## Tokenizer & Vocabulary

You asked: **"What do you think of my tokenizer? Would 32k for Indonesian language sufficient?"**

### Analysis of `otter_tokenizer_id_wiki_32k.json`
I trained your tokenizer on the Indonesian Wikipedia subset and analyzed its performance.

*   **Vocab Size:** 32,000
*   **Fertility Rate:** **~1.19 tokens/word** (on sample text)
    *   *Fertility* measures how many tokens are needed to represent one word on average. Lower is better (1.0 is perfect, meaning 1 word = 1 token).
    *   **Result:** This is **Excellent**.
        *   Common words like `Indonesia`, `adalah`, `negara`, `perekonomian` are single tokens.
        *   Complex words like `diproklamasikan` are split into 3 tokens (`dip`, `roklam`, `asikan`), which is reasonable for a BPE model.

### Is 32k Sufficient?
**Yes, absolutely.**

*   **Why it works:** Indonesian uses the Latin alphabet and has relatively simple morphology compared to some other languages. 32k is the standard size for Llama 2 (which covers many languages). Since you are focusing on *only* Indonesian (or mostly Indonesian), 32k slots are more than enough to store the vast majority of common Indonesian root words and affixes.
*   **Trade-off:**
    *   **32k (Current):**
        *   **Pros:** Smaller model size (Embeddings are ~24M params). Faster training.
        *   **Cons:** Very rare words might be split into more subwords.
    *   **Larger (e.g., 50k+):**
        *   **Pros:** Might capture even more specific scientific or technical terms as single tokens.
        *   **Cons:** Increases model size significantly (going to 64k would add ~24M more parameters to the embedding layer and LM head, making the model ~20% larger without adding "intelligence" layers).

**Recommendation:** Stick with **32k**. It strikes the perfect balance between efficiency and coverage for a mono-lingual or bi-lingual Indonesian model of this size (110M params).

---

## How to Make a Small Model (110M) "Smarter"?

You asked: **"I just wondered how to make model with this scale smarter."**

Making a 110M parameter model punch above its weight class requires specific strategies. Models like **Phi-1 (1.3B)** and **TinyLlama (1.1B)** have shown that small models can be surprisingly capable if trained correctly.

### 1. Data Quality is King ("Textbooks Are All You Need")
The biggest lever you have is **High-Quality Data**.
*   **The Problem:** Web data (CommonCrawl) is noisy. A small model wastes capacity learning useless patterns (HTML tags, ads, bad grammar).
*   **The Solution:**
    *   **Filter Aggressively:** Use only high-quality text (Wikipedia, textbooks, filtered code).
    *   **Synthetic Data:** Use a large model (GPT-4) to generate "textbook-quality" explanations of concepts.
    *   **Instruction Tuning:** Train on Q&A pairs, reasoning tasks, and code snippets.
    *   *Example:* Instead of just reading random sentences, feed it: "Explain photosynthesis to a 5-year-old."

### 2. Over-Training (Beyond Chinchilla)
Standard scaling laws (Chinchilla) suggest training a 110M model on ~2 Billion tokens.
*   **The "TinyLlama" Insight:** Small models continue to improve long after this point.
*   **Strategy:** Train for **10x or 100x longer** (e.g., 20B - 100B tokens). This forces the small model to compress knowledge very efficiently.

### 3. Knowledge Distillation
Instead of learning from raw text, teach your student (110M) to mimic a teacher (e.g., GPT-4 or Llama-3-70B).
*   **How:** Get the teacher's output (logits/probabilities) for a text and train your model to match those probabilities (KL Divergence loss), not just the hard text tokens.
*   **Why:** The teacher provides "soft labels" (e.g., for "The cat sat on the...", the teacher says "mat" (90%) and "rug" (9%)). Learning this distribution provides much more signal than just "mat".

### 4. Curriculum Learning
Start easy, get harder.
1.  **Phase 1:** Train on simple, clean text (children's books, simple Wikipedia).
2.  **Phase 2:** Introduce complex reasoning, code, and scientific papers.
3.  **Phase 3:** Fine-tune on instructions (Q&A).

### 5. Architectural Tweaks (You already have most!)
*   ✅ **GQA:** Helps inference speed and context handling.
*   ✅ **SwiGLU:** Better than ReLU.
*   ✅ **RoPE:** Better position handling.
*   **Deep vs. Wide:** Sometimes, a deeper but narrower model (more layers, smaller embedding dim) reasons better than a wide, shallow one. Your current 12-layer config is a good balance.
