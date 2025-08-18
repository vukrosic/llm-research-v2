# =============================================================================
# DISTRIBUTED TRAINING WITH HUGGING FACE ACCELERATE
# =============================================================================
# This version replaces custom distributed training with Accelerate
# Much simpler, more stable, and better performance
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
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
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    # Model architecture - OPTIMIZED FOR RTX 4090s
    d_model: int = 384              # Moderate size to fit memory
    n_heads: int = 8
    num_kv_heads: Optional[int] = None # For Grouped Query Attention (GQA)
    n_layers: int = 6               # Fewer layers to save memory
    d_ff: int = 1536                # Smaller feed-forward
    batch_size: int = 12            # per GPU batch size (can be larger with Accelerate)
    max_steps: int = 2500           # Training steps

    # Training parameters
    gradient_accumulation_steps: int = 4  # Gradient accumulation
    learning_rate: float = 0.001    # Learning rate

    # Data parameters
    max_seq_len: int = 512          # Sequence length
    num_documents: int = 2000       # Number of documents to load
    max_tokens: int = 500000        # Maximum tokens to use

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True            # Mixed precision
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

class MuonOptimizer(torch.optim.Optimizer):
    """Muon optimizer - simplified for Accelerate"""
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                momentum = group['momentum']
                lr = group['lr']
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-lr)

def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    if os.path.exists(cache_file):
        logger.info(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
    else:
        logger.info(f"🔄 Processing new data (will cache for future use)")
        
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

        logger.info(f"Loaded {len(texts)} documents")
        logger.info("Tokenizing texts...")
        
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        tokens = all_tokens[:config.max_tokens]
        logger.info(f"Using {len(tokens):,} tokens")
        
        cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info(f"💾 Cached data to {cache_file}")

    config.vocab_size = cached_data['tokenizer'].vocab_size
    return None, cached_data['tokenizer'], cached_data['tokens']

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
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, num_kv_heads: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else n_heads
        self.d_k = d_model // n_heads

        # Ensure d_model is divisible by num_kv_heads for K and V
        assert d_model % self.num_kv_heads == 0, "d_model must be divisible by num_kv_heads"
        self.d_kv = d_model // self.num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, self.num_kv_heads * self.d_kv * 2, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        Q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        KV = self.kv_proj(x).reshape(batch_size, seq_len, 2, self.num_kv_heads, self.d_kv)
        K, V = KV[:, :, 0].permute(0, 2, 1, 3), KV[:, :, 1].permute(0, 2, 1, 3)

        Q = self.rotary(Q)
        K = self.rotary(K)

        # Repeat K and V heads if num_kv_heads < n_heads (GQA)
        if self.num_kv_heads < self.n_heads:
            K = K.repeat_interleave(self.n_heads // self.num_kv_heads, dim=1)
            V = V.repeat_interleave(self.n_heads // self.num_kv_heads, dim=1)

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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1, num_kv_heads: Optional[int] = None):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, num_kv_heads)
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
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout, config.num_kv_heads)
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

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig, accelerator: Accelerator):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    # Create tensors on the correct device before gathering
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_tokens_tensor = torch.tensor(total_tokens, device=accelerator.device)
    total_correct_tensor = torch.tensor(total_correct, device=accelerator.device)

    # Gather metrics from all processes
    total_loss = accelerator.gather(total_loss_tensor)
    total_tokens = accelerator.gather(total_tokens_tensor)
    total_correct = accelerator.gather(total_correct_tensor)

    # Sum across all processes
    total_loss = total_loss.sum().item()
    total_tokens = total_tokens.sum().item()
    total_correct = total_correct.sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_optimizers(model: nn.Module, config: ModelConfig):
    """Setup optimizers with Muon for linear layers and AdamW for others"""
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

    logger.info(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    logger.info(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = MuonOptimizer(muon_params, lr=config.learning_rate, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.learning_rate*0.1, weight_decay=config.weight_decay)
    
    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader, accelerator: Accelerator):
    """Train the model with Accelerate"""
    
    logger.info(f"\n🚀 Training Small model with Accelerate")
    logger.info(f"  🌐 Device: {accelerator.device}")
    logger.info(f"  🎮 Mixed precision: {accelerator.mixed_precision}")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    
    # Setup optimizers
    optimizers = setup_optimizers(model, config)
    
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

    # Prepare everything with Accelerate
    model, optimizers, train_loader, val_loader, schedulers = accelerator.prepare(
        model, optimizers, train_loader, val_loader, schedulers
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  📊 Total parameters: {total_params:,}")

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    # Only show progress bar on main process
    pbar = tqdm(total=config.max_steps, desc="Training") if accelerator.is_main_process else None

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Forward pass with gradient accumulation
            with accelerator.accumulate(model):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                accelerator.backward(loss)

                # Optimizer step after accumulation
                if accelerator.sync_gradients:
                    # Clip gradients
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Step optimizers
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Step schedulers
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging (only main process)
            if step % 100 == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    perplexity = math.exp(min(loss.item(), 20))

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config, accelerator)
                if accelerator.is_main_process:
                    logger.info(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                              f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                              f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0 and accelerator.is_main_process:
                pbar.update(100)

    if accelerator.is_main_process:
        pbar.close()

    training_time = time.time() - start_time
    if accelerator.is_main_process:
        logger.info(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config, accelerator)
    if accelerator.is_main_process:
        logger.info(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
                  f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

def main():
    """Main training function"""
    
    # Initialize Accelerate
    accelerator = Accelerator(
        mixed_precision="fp16",  # Use mixed precision for better performance
        gradient_accumulation_steps=4,  # Match config
        log_with="tensorboard",  # Optional: log to tensorboard
        project_dir="accelerate_logs"
    )
    
    # Log configuration
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("🤖 DISTRIBUTED LLM TRAINING WITH ACCELERATE")
        logger.info("=" * 80)
        logger.info(f"🔧 Device: {accelerator.device}")
        logger.info(f"🎮 Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"🔧 Num processes: {accelerator.num_processes}")
        logger.info(f"🔧 Process index: {accelerator.process_index}")
        logger.info(f"🎯 Is main process: {accelerator.is_main_process}")
        logger.info("=" * 80)

    # Create config
    config = ModelConfig()
    if accelerator.is_main_process:
        logger.info(f"\n📋 Model Configuration:")
        logger.info(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
        logger.info(f"   Training: {config.max_steps} steps, batch size {config.batch_size} per GPU")
        logger.info(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    if accelerator.is_main_process:
        logger.info(f"🔄 Loading data...")
    texts, tokenizer, tokens = load_and_cache_data(config)
    
    if accelerator.is_main_process:
        logger.info(f"🔄 Creating dataset...")
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    if accelerator.is_main_process:
        logger.info(f"📊 Dataset created with {len(dataset)} samples")

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    
    # Use generator with same seed across all processes for consistent split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    if accelerator.is_main_process:
        logger.info(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader, accelerator)
    total_time = time.time() - start_time

    if accelerator.is_main_process:
        logger.info(f"\n🎉 TRAINING COMPLETED!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"🏆 Final Results:")
        logger.info(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        logger.info(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        logger.info(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")

def run_training():
    """Entry point for training"""
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    run_training()