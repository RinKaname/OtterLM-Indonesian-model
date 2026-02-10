import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import OtterLM, Config
from tokenizers import Tokenizer
from datasets import load_dataset

# --- 1. Arguments & DDP Setup ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train OtterLM (110M)")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--block_size", type=int, default=512, help="Context length")
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Min learning rate")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--tokenizer_path", type=str, default="otter_tokenizer_id_wiki_32k.json")
    return parser.parse_args()

args = parse_args()

# DDP Configuration
ddp = int(os.environ.get("RANK", -1)) != -1 # Is this a DDP run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # logging only on master
    seed_offset = ddp_rank # ensure different random seed per process
else:
    # Vanilla single-GPU/CPU run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

if master_process:
    print(f"Using device: {device} (World Size: {ddp_world_size})")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on ampere
torch.backends.cudnn.allow_tf32 = True

# --- 2. Data Loader (Streaming) ---
if master_process:
    print("Loading dataset (streaming)...")

# Each rank loads the dataset, but we shuffle differently to ensure different batches
raw_dataset = load_dataset("HuggingFaceFW/finewiki", name="id", split="train", streaming=True)
raw_dataset = raw_dataset.shuffle(buffer_size=10000, seed=42 + seed_offset)

# Load Tokenizer
if not os.path.exists(args.tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}. Run tokenizer.py first!")
tokenizer = Tokenizer.from_file(args.tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

class SmartLoader:
    def __init__(self, dataset, tokenizer, batch_size, block_size, device):
        self.dataset = dataset
        self.iterator = iter(dataset)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer = []
        self.device = device

    def get_batch(self):
        tokens_per_seq = self.block_size + 1
        total_tokens_needed = self.batch_size * tokens_per_seq

        while len(self.buffer) < total_tokens_needed:
            try:
                ex = next(self.iterator)
            except StopIteration:
                if master_process:
                    print("Dataset exhausted, resetting iterator...")
                self.iterator = iter(self.dataset)
                ex = next(self.iterator)

            text = ex.get("text", "")
            if text:
                ids = self.tokenizer.encode(text + "</s>").ids
                self.buffer.extend(ids)

        chunk = self.buffer[:total_tokens_needed]
        self.buffer = self.buffer[total_tokens_needed:]

        data = torch.tensor(chunk, dtype=torch.long)
        data = data.view(self.batch_size, tokens_per_seq)

        x = data[:, :-1].contiguous().to(self.device)
        y = data[:, 1:].contiguous().to(self.device)
        return x, y

loader = SmartLoader(raw_dataset, tokenizer, args.batch_size, args.block_size, device)

# --- 3. Model Initialization ---
config = Config(
    vocab_size=vocab_size,
    block_size=args.block_size,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1
)
model = OtterLM(config)
model.to(device)

if args.compile and hasattr(torch, "compile"):
    if master_process: print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP container
if master_process:
    print(f"Model Parameters: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M")

# --- 4. Optimizer & Scheduler ---
# We use weight decay only on 2D parameters (weights), not biases or norms.
param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': args.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95))

# Cosine Learning Rate Schedule
def get_lr(it):
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    if it > args.max_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# --- 5. Training Loop ---
scaler = torch.cuda.amp.GradScaler(enabled=True)

if master_process:
    print(f"Starting training for {args.max_iters} iterations...")

model.train()
t0 = time.time()

for iter_num in range(args.max_iters):
    # Determine LR
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Forward/Backward
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(args.grad_accum_steps):
        # In DDP, only sync gradients on the last micro-step
        if ddp:
            model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)

        X, Y = loader.get_batch()

        with torch.cuda.amp.autocast():
            logits, loss, _ = model(X, targets=Y)
            loss = loss / args.grad_accum_steps
            loss_accum += loss.item() # This is local loss accumulation

        scaler.scale(loss).backward()

    # Clip Gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Step
    scaler.step(optimizer)
    scaler.update()

    # Logging
    if iter_num % 10 == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000 # ms
        t0 = t1

        # In DDP, we might want to average loss across ranks for logging,
        # but printing rank 0 loss is often good enough for monitoring.
        if master_process:
            print(f"step {iter_num:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | time {dt:.2f}ms")

    # Checkpointing (only on master)
    if master_process and iter_num > 0 and iter_num % args.save_interval == 0:
        checkpoint_path = f"otter_ckpt_{iter_num}.pt"
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(raw_model.state_dict(), checkpoint_path)

if ddp:
    destroy_process_group()

if master_process:
    print("Training complete!")
    torch.save(raw_model.state_dict(), "otter_final.pt")
    print("Model saved to otter_final.pt")
