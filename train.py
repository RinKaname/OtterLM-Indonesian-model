import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import OtterLM, Config
from tokenizers import Tokenizer
from datasets import load_dataset

# --- 1. Arguments & Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train OtterLM (110M)")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--block_size", type=int, default=256, help="Context length (smaller for faster training)")
    parser.add_argument("--max_iters", type=int, default=1000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Max learning rate") # High LR for small model
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Min learning rate (10% of max)")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0)")
    parser.add_argument("--tokenizer_path", type=str, default="otter_tokenizer_id_wiki_32k.json")
    return parser.parse_args()

args = parse_args()
print(f"Using device: {args.device}")

# --- 2. Data Loader (Streaming) ---
# We use the same Indonesian Wikipedia dataset as the tokenizer
print("Loading dataset (streaming)...")
# streaming=True returns an IterableDataset. We cycle it to make it infinite.
raw_dataset = load_dataset("HuggingFaceFW/finewiki", name="id", split="train", streaming=True)
raw_dataset = raw_dataset.shuffle(buffer_size=10000, seed=42)

# Load Tokenizer
if not os.path.exists(args.tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}. Run tokenizer.py first!")
tokenizer = Tokenizer.from_file(args.tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

class SmartLoader:
    def __init__(self, dataset, tokenizer, batch_size, block_size):
        self.dataset = dataset
        self.iterator = iter(dataset)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer = []

    def get_batch(self):
        # We need (B, T+1) tokens in total to form X and Y
        # T+1 because Y is shifted by 1
        tokens_per_seq = self.block_size + 1
        total_tokens_needed = self.batch_size * tokens_per_seq

        while len(self.buffer) < total_tokens_needed:
            try:
                ex = next(self.iterator)
            except StopIteration:
                print("Dataset exhausted, resetting iterator...")
                self.iterator = iter(self.dataset)
                ex = next(self.iterator)

            text = ex.get("text", "")
            if text:
                # Encode and add EOS token
                ids = self.tokenizer.encode(text + "</s>").ids
                self.buffer.extend(ids)

        # Extract batch
        chunk = self.buffer[:total_tokens_needed]
        self.buffer = self.buffer[total_tokens_needed:]

        data = torch.tensor(chunk, dtype=torch.long)
        # Reshape to (B, T+1)
        data = data.view(self.batch_size, tokens_per_seq)

        x = data[:, :-1].contiguous().to(args.device)
        y = data[:, 1:].contiguous().to(args.device)
        return x, y

loader = SmartLoader(raw_dataset, tokenizer, args.batch_size, args.block_size)

# --- 3. Model Initialization ---
config = Config(
    vocab_size=vocab_size,
    block_size=args.block_size,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1 # Use dropout for small model training
)
model = OtterLM(config)
model.to(args.device)

# Compile model if requested (requires PyTorch 2.0+)
if args.compile and hasattr(torch, "compile"):
    print("Compiling model...")
    model = torch.compile(model)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# --- 4. Optimizer & Scheduler ---
# We use weight decay only on 2D parameters (weights), not biases or norms.
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': args.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95))

# Cosine Learning Rate Schedule with Warmup
def get_lr(it):
    # 1) Linear warmup
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # 2) If it > max_iters, return min_lr
    if it > args.max_iters:
        return args.min_lr
    # 3) Cosine decay
    decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# --- 5. Training Loop ---
# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda"))

print(f"Starting training for {args.max_iters} iterations...")
model.train()
t0 = time.time()

for iter_num in range(args.max_iters):
    # Determine and set Learning Rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Forward Pass with Gradient Accumulation
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for _ in range(args.grad_accum_steps):
        X, Y = loader.get_batch()

        # Mixed Precision Context
        with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
            logits, loss, _ = model(X, targets=Y)
            # Scale loss by accumulation steps
            loss = loss / args.grad_accum_steps
            loss_accum += loss.item()

        # Backward Pass (scaled)
        scaler.scale(loss).backward()

    # Gradient Clipping (Standard practice: clip norm to 1.0)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer Step (scaled)
    scaler.step(optimizer)
    scaler.update()

    # Logging
    if iter_num % 10 == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000 # ms
        t0 = t1
        # loss_accum is already divided by grad_accum_steps? No.
        # Wait, loss was scaled for backward. loss.item() returns the value.
        # So loss_accum contains sum(loss_i / N). This is the correct average loss.
        print(f"step {iter_num:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | time {dt:.2f}ms")

    # Checkpointing
    if iter_num > 0 and iter_num % args.save_interval == 0:
        checkpoint_path = f"otter_ckpt_{iter_num}.pt"
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)

print("Training complete!")
torch.save(model.state_dict(), "otter_final.pt")
print("Model saved to otter_final.pt")
