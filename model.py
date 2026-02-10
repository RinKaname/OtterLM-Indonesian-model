import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class Config:
    vocab_size: int = 32000
    block_size: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int = None
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    use_cache: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()

def precompute_rope_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(q, k, cos, sin):
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q_out = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1).flatten(-2)
    k_out = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1).flatten(-2)
    return q_out, k_out

class OtterLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        # Calculate size for q, k, v projections
        # q: n_head * head_dim
        # k: n_kv_head * head_dim
        # v: n_kv_head * head_dim
        op_size = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, op_size, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.mlp = nn.ModuleDict({
            'gate_proj': nn.Linear(config.n_embd, hidden_dim, bias=False),
            'up_proj': nn.Linear(config.n_embd, hidden_dim, bias=False),
            'down_proj': nn.Linear(hidden_dim, config.n_embd, bias=False),
        })

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, past_kv=None):
        attn_out, new_kv = self._attn_block(self.ln_1(x), cos, sin, past_kv)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self._mlp_block(self.ln_2(x)))
        return x, new_kv

    def _attn_block(self, x, cos, sin, past_kv=None):
        B, T, C = x.size()
        # Split q, k, v
        q, k, v = self.c_attn(x).split([
            self.n_head * self.head_dim,
            self.n_kv_head * self.head_dim,
            self.n_kv_head * self.head_dim
        ], dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE BEFORE caching (never re-rotate cached keys)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        
        new_kv = (k, v) if self.config.use_cache else None
        
        # Repeat K/V for GQA to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # When decoding with past_kv (T=1), we want to attend to all past keys (is_causal=False).
        # When training/prefilling (past_kv=None), we need causal masking (is_causal=True).
        is_causal = past_kv is None
        dropout_p = self.config.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_kv

    def _mlp_block(self, x):
        return self.mlp.down_proj(F.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))

class OtterLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([OtterLMBlock(config) for _ in range(config.n_layer)]),
            'ln_f': RMSNorm(config.n_embd, eps=config.norm_eps),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.block_size, theta=config.rope_theta)
        self.register_buffer("cos", cos.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", sin.unsqueeze(0).unsqueeze(0), persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layer) if module.weight.size(0) == self.config.n_embd else 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None):
        B, T = idx.size()
        start_pos = past_key_values[0][0].size(2) if past_key_values is not None else 0
        
        x = self.transformer.wte(idx)
        cos = self.cos[:, :, start_pos:start_pos+T, :]
        sin = self.sin[:, :, start_pos:start_pos+T, :]
        
        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, kv = block(x, cos, sin, past_kv=pkv)
            new_kvs.append(kv)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        past_kvs = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kvs is not None else idx
            logits, _, past_kvs = self(idx_cond, past_key_values=past_kvs)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
