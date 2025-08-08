#!/usr/bin/env python3
"""
Script to upload trained model checkpoints to Hugging Face Hub
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError
import tempfile
import shutil
from pathlib import Path

# Import model classes from training script
from train_llm import MinimalLLM, ModelConfig

def load_checkpoint(checkpoint_path: str):
    """Load a checkpoint and return model, config, and tokenizer"""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    
    # Load the saved checkpoint
    model_path = os.path.join(checkpoint_path, 'model.pt')
    config_path = os.path.join(checkpoint_path, 'config.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create ModelConfig object
    config = ModelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"✅ Loaded checkpoint from step {step} with loss {loss:.4f}")
    return model, config, tokenizer, step, loss

def create_model_card(config: ModelConfig, step: int, loss: float, repo_name: str):
    """Create a model card for the uploaded model"""
    model_card = f"""---
license: mit
language: en
tags:
- pytorch
- causal-lm
- text-generation
- minimal-llm
library_name: transformers
pipeline_tag: text-generation
---

# {repo_name}

This is a minimal language model trained with a custom implementation using the Muon optimizer.

## Model Details

- **Architecture**: Transformer decoder
- **Parameters**: ~{sum(p.numel() for p in MinimalLLM(config).parameters()):,}
- **Layers**: {config.n_layers}
- **Hidden Size**: {config.d_model}
- **Attention Heads**: {config.n_heads}
- **Feed Forward Size**: {config.d_ff}
- **Vocabulary Size**: {config.vocab_size:,}
- **Max Sequence Length**: {config.max_seq_len}

## Training Details

- **Training Steps**: {step:,}
- **Final Loss**: {loss:.4f}
- **Optimizer**: Muon (MomentUm Orthogonalized by Newton-schulz)
- **Dataset**: SmolLM Corpus (Cosmopedia-v2)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Generate text
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Model Architecture

This model uses a custom minimal transformer implementation with:
- RMSNorm for layer normalization
- SiLU activation function
- Rotary Position Embeddings (RoPE)
- Tied input/output embeddings

## Training Configuration

- Batch Size: {config.batch_size}
- Max Sequence Length: {config.max_seq_len}
- Dropout: {config.dropout}
- Learning Rate: {config.muon_lr}

## Limitations

This is a small experimental model trained for research purposes. It may not perform as well as larger, production-ready models.
"""
    return model_card

def prepare_for_hf_upload(model, tokenizer, config, step, loss, temp_dir):
    """Prepare model files for Hugging Face upload"""
    print("🔧 Preparing model for Hugging Face format...")
    
    # Save model in HF format
    model.save_pretrained(temp_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(temp_dir)
    
    # Create config.json for HF transformers
    hf_config = {
        "architectures": ["MinimalLLM"],
        "auto_map": {
            "AutoModelForCausalLM": "modeling_minimal_llm.MinimalLLM"
        },
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
        "model_type": "minimal_llm",
        "torch_dtype": "float32",
        "transformers_version": "4.36.0"
    }
    
    with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
        json.dump(hf_config, f, indent=2)
    
    # Copy the modeling file
    modeling_code = '''
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Copy the model classes from train_llm.py here
# This is a simplified version - you may need to copy the full implementation
'''
    
    with open(os.path.join(temp_dir, 'modeling_minimal_llm.py'), 'w') as f:
        f.write(modeling_code)
    
    print("✅ Model prepared for upload")

def upload_to_huggingface(checkpoint_path: str, repo_name: str, token: str = None, private: bool = False):
    """Upload a checkpoint to Hugging Face Hub"""
    
    # Load checkpoint
    model, config, tokenizer, step, loss = load_checkpoint(checkpoint_path)
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        print(f"🔍 Checking if repository {repo_name} exists...")
        api.repo_info(repo_name, repo_type="model")
        print(f"✅ Repository {repo_name} exists")
    except RepositoryNotFoundError:
        print(f"🆕 Creating new repository: {repo_name}")
        create_repo(repo_name, token=token, private=private, repo_type="model")
    
    # Prepare files in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Prepare model for HF upload
        prepare_for_hf_upload(model, tokenizer, config, step, loss, temp_dir)
        
        # Create model card
        model_card = create_model_card(config, step, loss, repo_name)
        with open(os.path.join(temp_dir, 'README.md'), 'w') as f:
            f.write(model_card)
        
        # Upload to HF Hub
        print(f"🚀 Uploading to Hugging Face Hub: {repo_name}")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model",
            token=token,
            commit_message=f"Upload checkpoint from step {step}"
        )
    
    print(f"🎉 Successfully uploaded to https://huggingface.co/{repo_name}")
    return f"https://huggingface.co/{repo_name}"

def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """List available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint_step_"):
            try:
                step = int(item.split("_")[-1])
                model_file = os.path.join(checkpoint_path, "model.pt")
                if os.path.exists(model_file):
                    checkpoints.append((step, checkpoint_path))
            except ValueError:
                continue
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="Upload model checkpoints to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Hugging Face repository name (username/model-name)")
    parser.add_argument("--token", type=str, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing checkpoints")
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    if not token and not args.list:
        print("❌ Please provide a Hugging Face token via --token or HF_TOKEN environment variable")
        return
    
    # List checkpoints if requested
    if args.list:
        print(f"📋 Available checkpoints in {args.checkpoint_dir}:")
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print("   No checkpoints found")
        else:
            for step, path in checkpoints:
                print(f"   Step {step:,}: {path}")
        return
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use the latest checkpoint
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(f"❌ No checkpoints found in {args.checkpoint_dir}")
            return
        checkpoint_path = checkpoints[-1][1]
        print(f"🎯 Using latest checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        # Upload to Hugging Face
        url = upload_to_huggingface(checkpoint_path, args.repo_name, token, args.private)
        print(f"\n🎉 Upload completed successfully!")
        print(f"🔗 Model URL: {url}")
        print(f"\n📖 To use your model:")
        print(f"```python")
        print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{args.repo_name}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{args.repo_name}')")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()