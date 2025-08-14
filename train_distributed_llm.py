# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================
# Adjust these variables for your multi-GPU setup
NUM_GPUS = 2                    # Number of GPUs to use (change to 4, 8, etc.)
MASTER_PORT = "12355"           # Port for distributed communication
BACKEND = "nccl"                # Use "nccl" for NVIDIA GPUs, "gloo" for CPU
GPU_IDS = [0, 1]                # Specific GPU IDs to use (e.g., [0,1,2,3] for 4 GPUs)

# MODEL SCALING FOR MULTI-GPU
# Adjust batch size and learning rate based on number of GPUs
BASE_BATCH_SIZE = 24            # Base batch size per GPU
BASE_LR = 0.01                  # Base learning rate
SCALE_LR_WITH_GPUS = True       # Whether to scale LR with number of GPUs

# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = BASE_BATCH_SIZE  # per GPU batch size
    max_steps: int = 5000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = BASE_LR * (NUM_GPUS if SCALE_LR_WITH_GPUS else 1)

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class DistributedMuon(torch.optim.Optimizer):
    """Distributed Muon optimizer"""
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                reduce_scatter_futures.append(
                    dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], 
                                       op=dist.ReduceOp.AVG, async_op=True).get_future()
                )

        idx = 0
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather(params_pad[base_i:base_i + world_size], 
                                   params_pad[base_i + rank], async_op=True).get_future()
                )
        torch.futures.collect_all(all_reduce_futures).wait()

class DistAdam(torch.optim.Optimizer):
    """Distributed AdamW optimizer"""
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        
        for group in self.param_groups:
            params: list[torch.Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, 
                                              async_op=True).get_future()
                )
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr']
                state = self.state[p]
                g_slice = grad_slices[idx]
                
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                if wd != 0:
                    p_slice.mul_(1 - lr * wd)
                
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )
        torch.futures.collect_all(all_reduce_futures).wait()

def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache", rank: int = 0):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Only rank 0 loads/saves cache
    if rank == 0:
        if os.path.exists(cache_file):
            print(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
        else:
            print(f"🔄 Processing new data (will cache for future use)")
            
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                                  split="train", streaming=True, token=False)

            texts = []
            for i, item in enumerate(dataset):
                if i >= config.num_documents:
                    break
                texts.append(item["text"][:3000])

            print(f"Loaded {len(texts)} documents")
            print("Tokenizing texts...")
            
            all_tokens = []
            for text in tqdm(texts, desc="Tokenizing"):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

            tokens = all_tokens[:config.max_tokens]
            print(f"Using {len(tokens):,} tokens")
            
            cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"💾 Cached data to {cache_file}")
    
    # Synchronize and broadcast data
    dist.barrier()
    
    if rank == 0:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
    else:
        cached_data = None
    
    cached_data = dist.broadcast_object_list([cached_data])[0] if rank != 0 else cached_data
    
    texts = cached_data['texts']
    tokenizer = cached_data['tokenizer']
    tokens = cached_data['tokens']
    config.vocab_size = tokenizer.vocab_size

    if rank == 0:
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens")
    
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance with distributed reduction"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    # Reduce across all GPUs
    total_loss = torch.tensor(total_loss, device=device)
    total_tokens = torch.tensor(total_tokens, device=device)
    total_correct = torch.tensor(total_correct, device=device)
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)

    avg_loss = total_loss.item() / total_tokens.item()
    accuracy = total_correct.item() / total_tokens.item()
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_distributed_optimizers(model: nn.Module, config: ModelConfig):
    """Setup distributed Muon and DistAdam optimizers"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    rank = dist.get_rank()
    if rank == 0:
        print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"  DistAdam parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = DistributedMuon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = DistAdam(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

class DistributedSampler:
    """Simple distributed sampler"""
    def __init__(self, dataset, rank, world_size, shuffle=True):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(indices)
        
        # Distribute indices across ranks
        indices = indices[self.rank::self.world_size]
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset) // self.world_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with distributed optimizers"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"\n🚀 Training Small model with Distributed Muon optimizer")
        print(f"  🌐 World size: {world_size} GPUs")

    # Initialize model
    set_seed(42 + rank)  # Different seed per rank
    model = MinimalLLM(config)
    device = torch.device('cuda', rank)
    model = model.to(device)

    # Synchronize model parameters across all ranks
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  📊 Total parameters: {total_params:,}")

    # Setup distributed optimizers
    optimizers = setup_distributed_optimizers(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training") if rank == 0 else None

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging (only rank 0)
            if step % 100 == 0 and rank == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                if rank == 0:
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0 and rank == 0:
                pbar.update(100)

    if rank == 0:
        pbar.close()

    training_time = time.time() - start_time
    if rank == 0:
        print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    if rank == 0:
        print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
              f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

def setup_distributed():
    """Setup distributed training environment"""
    # Get distributed training parameters from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", NUM_GPUS))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise RuntimeError(f"Requested {world_size} GPUs but only {available_gpus} available!")
    
    # Set the specific GPU for this process
    if local_rank < len(GPU_IDS):
        gpu_id = GPU_IDS[local_rank]
    else:
        gpu_id = local_rank
    
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    
    # Initialize process group
    dist.init_process_group(backend=BACKEND)
    dist.barrier()
    
    if rank == 0:
        print(f"🔧 Distributed setup: {world_size} GPUs, backend={BACKEND}")
        print(f"🎯 Using GPUs: {GPU_IDS[:world_size]}")
    
    return rank, world_size, local_rank, device

def main():
    rank, world_size, local_rank, device = setup_distributed()
    
    master_process = (rank == 0)
    
    # Check system
    if master_process:
        print(f"🔍 Device: CUDA")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"World size: {world_size} GPUs")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    if master_process:
        print(f"\n📋 Model Configuration:")
        print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
        print(f"   Training: {config.max_steps} steps, batch size {config.batch_size} per GPU")
        print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config, rank=rank)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    
    # Use generator with same seed across all ranks for consistent split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, rank, world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank, world_size, shuffle=False)

    # Create data loaders with samplers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    if master_process:
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"📊 Per GPU: {len(train_sampler)} train, {len(val_sampler)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    if master_process:
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()