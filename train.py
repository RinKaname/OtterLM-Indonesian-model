import os
import time
import math
import argparse
import torch
from safetensors.torch import save_file
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
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--tokenizer_path", type=str, default="otter_tokenizer_id_wiki_32k.json")
    return parser.parse_args()

args = parse_args()

# DDP Configuration
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

if master_process:
    print(f"Using device: {device} (World Size: {ddp_world_size})")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 2. Checkpoint Pre-load (to get start_iter for Dataset skip) ---
start_iter = 0
checkpoint = None
if args.resume:
    if master_process:
        print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=device)
    if 'iter_num' in checkpoint:
        start_iter = checkpoint['iter_num'] + 1
        if master_process:
            print(f"Resuming from iteration {start_iter}")

# --- 3. Data Loader (Streaming with Fast-Forward) ---
if master_process:
    print("Loading dataset (streaming)...")

raw_dataset = load_dataset("HuggingFaceFW/finewiki", name="id", split="train", streaming=True)
raw_dataset = raw_dataset.shuffle(buffer_size=10000, seed=42 + seed_offset)

if not os.path.exists(args.tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}. Run tokenizer.py first!")
tokenizer = Tokenizer.from_file(args.tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

class SmartLoader:
    def __init__(self, dataset, tokenizer, batch_size, block_size, device, start_iter=0, grad_accum=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # Calculate how many tokens we processed in previous runs
        tokens_per_batch = batch_size * (block_size + 1)
        total_tokens_processed = start_iter * grad_accum * tokens_per_batch

        # Estimate how many documents to skip (average ~500 tokens per wiki doc)
        docs_to_skip = int(total_tokens_processed / 500)

        if docs_to_skip > 0 and master_process:
            print(f"Fast-forwarding dataset... Skipping approx {docs_to_skip} documents from previous runs.")

        self.iterator = iter(self.dataset.skip(docs_to_skip))
        self.buffer = []

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

loader = SmartLoader(raw_dataset, tokenizer, args.batch_size, args.block_size, device, start_iter, args.grad_accum_steps)

# --- 4. Model Initialization ---
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

raw_model = model

# Restore weights if resuming
if checkpoint is not None:
    state_dict = checkpoint.get('model', checkpoint)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    raw_model.load_state_dict(state_dict)
    if master_process:
        print("Successfully loaded model weights.")

if args.compile and hasattr(torch, "compile"):
    if master_process: print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
    print(f"Model Parameters: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M")

# --- 5. Optimizer & Scheduler ---
param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': args.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95))

scaler = torch.amp.GradScaler('cuda', enabled=True)

if checkpoint is not None:
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

def get_lr(it):
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    if it > args.max_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# --- 6. Training Loop ---
if master_process:
    print(f"Starting training for {args.max_iters} iterations...")

model.train()
t0 = time.time()

for iter_num in range(start_iter, args.max_iters):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(args.grad_accum_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)

        X, Y = loader.get_batch()

        with torch.amp.autocast('cuda'):
            logits, loss, _ = model(X, targets=Y)
            loss = loss / args.grad_accum_steps
            loss_accum += loss.item()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if iter_num % 10 == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000
        t0 = t1

        if master_process:
            print(f"step {iter_num:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | time {dt:.2f}ms")

    if master_process and iter_num > 0 and iter_num % args.save_interval == 0:
        checkpoint_path = f"otter_ckpt_{iter_num}.pt"
        print(f"Saving checkpoint to {checkpoint_path}")
        checkpoint_dict = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'iter_num': iter_num
        }
        torch.save(checkpoint_dict, checkpoint_path)

if ddp:
    destroy_process_group()

if master_process:
    print("Training complete!")
    state_dict = raw_model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.clone()
    save_file(state_dict, "otter_final.safetensors")
    print("Model saved to otter_final.safetensors")
