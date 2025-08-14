#!/usr/bin/env python3
"""
Simple launcher for 2x RTX 4090 training with memory optimization
"""

import os
import torch

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import and run
from train_distributed_llm import run_novita_4090_training

if __name__ == "__main__":
    print("🚀 Starting 2x RTX 4090 optimized training...")
    run_novita_4090_training()